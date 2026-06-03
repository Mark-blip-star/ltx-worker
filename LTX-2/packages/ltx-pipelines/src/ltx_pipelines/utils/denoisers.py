"""Flat denoiser classes — transformer received at call time, not stored.
Three implementations of the :class:`~ltx_pipelines.utils.types.Denoiser` protocol:
* :class:`SimpleDenoiser` — single transformer call, no guidance.
* :class:`GuidedDenoiser` — static guiders, handles CFG + STG + isolated modality.
* :class:`FactoryGuidedDenoiser` — resolves guiders per-step from sigma.
``GuidedDenoiser`` and ``FactoryGuidedDenoiser`` share the core multi-pass
logic via the module-level :func:`_guided_denoise` function, which batches
all guidance passes into a single transformer call.
"""

import json
import os
import time

import torch

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer import X0Model
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.utils.types import DenoisedLatentResult

_POSITIVE_ONLY_GUIDER = MultiModalGuider(
    params=MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0),
)
"""Guider that only runs the conditioned pass and returns cond unchanged."""


# --- FasterCache-style CFG cache (env-gated, near-lossless spike) -------------
# Guidance deltas (cond - uncond/ptb/mod) vary slowly across steps. On "reuse"
# steps we run ONLY the cond pass (n: 4->1) and reconstruct the skipped passes
# from cached deltas + fresh cond, then apply the unchanged guider formula.
#   LTX_CFG_CACHE=1            enable
#   LTX_CFG_CACHE_INTERVAL=2   full on step%interval==0, reuse otherwise
#   LTX_CFG_CACHE_WARMUP=2     always-full first N steps (cache priming)
_cfg_cache: dict[str, tuple] = {}


def _cfg_cache_enabled() -> bool:
    return os.environ.get("LTX_CFG_CACHE") == "1"


def _cfg_cache_should_reuse(step_index: int) -> bool:
    if not _cfg_cache_enabled():
        return False
    interval = int(os.environ.get("LTX_CFG_CACHE_INTERVAL", "2"))
    warmup = int(os.environ.get("LTX_CFG_CACHE_WARMUP", "2"))
    if interval < 2 or step_index < warmup:
        return False
    # Optional "lo:hi" window — only reuse on MIDDLE steps; early steps build
    # structure and late steps refine detail (both kept full for quality).
    rng = os.environ.get("LTX_CFG_CACHE_RANGE", "")
    if rng:
        try:
            lo, hi = (int(x) for x in rng.split(":"))
            if not (lo <= step_index < hi):
                return False
        except ValueError:
            pass
    return (step_index % interval) != 0


def _cfg_delta(cond: object, val: object) -> torch.Tensor | None:
    if isinstance(cond, torch.Tensor) and isinstance(val, torch.Tensor):
        return cond - val
    return None


def _cfg_recon(cond: object, delta: torch.Tensor | None) -> object:
    if isinstance(cond, torch.Tensor) and delta is not None:
        return cond - delta
    return 0.0


def _mark_cuda_graph_step() -> None:
    marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def _tgate_active(step_index: int) -> bool:
    start = os.environ.get("LTX_TGATE_START_STEP")
    if not start:
        return False
    try:
        return step_index >= int(start)
    except ValueError:
        return False


def _tensor_shape(value: torch.Tensor | None) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def _modality_shapes(modality: object | None) -> dict[str, list[int] | None] | None:
    if modality is None:
        return None
    return {
        "latent": _tensor_shape(getattr(modality, "latent", None)),
        "context": _tensor_shape(getattr(modality, "context", None)),
        "sigma": _tensor_shape(getattr(modality, "sigma", None)),
        "timesteps": _tensor_shape(getattr(modality, "timesteps", None)),
        "positions": _tensor_shape(getattr(modality, "positions", None)),
        "attention_mask": _tensor_shape(getattr(modality, "attention_mask", None)),
    }


def _write_guidance_trace(event: dict[str, object]) -> None:
    trace_file = os.environ.get("LTX_GUIDANCE_TRACE_FILE")
    if not trace_file:
        return
    event.setdefault("run_id", os.environ.get("LTX_GUIDANCE_TRACE_RUN_ID"))
    event.setdefault("stage", os.environ.get("LTX_GUIDANCE_TRACE_STAGE"))
    event.setdefault("attention_backend", os.environ.get("LTX_GUIDANCE_TRACE_ATTENTION_BACKEND"))
    event.setdefault("cuda_graphs_enabled", os.environ.get("LTX_GUIDANCE_TRACE_CUDA_GRAPHS") == "1")
    with open(trace_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _sync_for_trace() -> None:
    if os.environ.get("LTX_GUIDANCE_TRACE_FILE") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _ensure_guider(guider: MultiModalGuider | None) -> MultiModalGuider:
    """Return the guider as-is, or a positive-only guider for absent modalities."""
    return guider if guider is not None else _POSITIVE_ONLY_GUIDER


def _repeat_state(state: LatentState, n: int) -> LatentState:
    """Repeat a ``LatentState`` *n* times along the batch dimension.
    ``(B, ...) → (n*B, ...)`` by tiling the whole tensor n times, so the
    ordering is ``[item0, item1, ..., item0, item1, ...]`` — matching
    ``torch.cat`` of n per-pass contexts.
    """

    def _repeat(t: torch.Tensor) -> torch.Tensor:
        repeats = [1] * t.dim()
        repeats[0] = n
        return t.repeat(repeats)

    return LatentState(
        latent=_repeat(state.latent),
        denoise_mask=_repeat(state.denoise_mask),
        positions=_repeat(state.positions),
        clean_latent=_repeat(state.clean_latent),
        attention_mask=_repeat(state.attention_mask) if state.attention_mask is not None else None,
    )


def _guided_denoise(  # noqa: PLR0913,PLR0915
    transformer: X0Model,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    sigma: torch.Tensor,
    video_guider: MultiModalGuider,
    audio_guider: MultiModalGuider,
    v_context: torch.Tensor | None,
    a_context: torch.Tensor | None,
    *,
    last_denoised_video: torch.Tensor | None,
    last_denoised_audio: torch.Tensor | None,
    step_index: int,
    force_uncond_pass: bool = False,
) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
    """Core guided denoising — batches all guidance passes into one transformer call.
    Collects per-pass contexts first, then builds a single batched Modality
    per present modality via :func:`modality_from_latent_state`.  When wrapped
    with :class:`~ltx_core.batch_split.BatchSplitAdapter`, the transformer may
    split this batch into sequential chunks internally.
    Guiders must not be ``None``. For absent modalities, callers should pass
    :data:`_POSITIVE_ONLY_GUIDER` (via :func:`_ensure_guider`) so that only
    the conditioned pass runs and ``calculate()`` returns cond unchanged.
    """
    if step_index == 0:
        _cfg_cache.clear()
    reuse_cfg = _cfg_cache_should_reuse(step_index) and bool(_cfg_cache)
    v_skip = video_guider.should_skip_step(step_index)
    a_skip = audio_guider.should_skip_step(step_index)

    if v_skip and a_skip:
        video_result = DenoisedLatentResult.result_or_none(denoised=last_denoised_video)
        audio_result = DenoisedLatentResult.result_or_none(denoised=last_denoised_audio)
        return video_result, audio_result

    if video_state is not None and v_context is None:
        raise ValueError("v_context is required when video_state is provided")
    if audio_state is not None and a_context is None:
        raise ValueError("a_context is required when audio_state is provided")
    # Define passes: (name, video_context, audio_context, perturbation_config).
    # Context is None for absent modalities — filtered out during collection.
    _pass = tuple[str, torch.Tensor | None, torch.Tensor | None, PerturbationConfig]
    passes: list[_pass] = [("cond", v_context, a_context, PerturbationConfig.empty())]

    v_needs_neg = video_guider.do_unconditional_generation() or (force_uncond_pass and video_state is not None)
    a_needs_neg = audio_guider.do_unconditional_generation() or (force_uncond_pass and audio_state is not None)
    if (v_needs_neg or a_needs_neg) and not reuse_cfg:
        if v_needs_neg and video_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        if a_needs_neg and audio_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        v_neg = video_guider.negative_context if video_guider.negative_context is not None else v_context
        a_neg = audio_guider.negative_context if audio_guider.negative_context is not None else a_context
        passes.append(("uncond", v_neg, a_neg, PerturbationConfig.empty()))

    stg_perturbations: list[Perturbation] = []
    if video_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
        )
    if audio_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
        )
    if stg_perturbations and not reuse_cfg:
        passes.append(("ptb", v_context, a_context, PerturbationConfig(stg_perturbations)))

    if (
        video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation()
    ) and not reuse_cfg:
        passes.append(
            (
                "mod",
                v_context,
                a_context,
                PerturbationConfig(
                    [
                        Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                        Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                    ]
                ),
            )
        )

    # Collect contexts, repeat states, and build batched modalities.
    pass_names = [name for name, _, _, _ in passes]
    ptb_configs = [ptb for _, _, _, ptb in passes]
    n = len(passes)

    def _batched_sigma(state: LatentState) -> torch.Tensor:
        """Expand scalar sigma to (n * B,) matching the repeated state."""
        return sigma.expand(state.latent.shape[0] * n)

    batched_video = None
    if video_state is not None:
        v_context = torch.cat([vc for _, vc, _, _ in passes], dim=0)
        batched_video = modality_from_latent_state(
            _repeat_state(video_state, n),
            v_context,
            _batched_sigma(video_state),
            enabled=not v_skip,
        )

    batched_audio = None
    if audio_state is not None:
        a_context = torch.cat([ac for _, _, ac, _ in passes], dim=0)
        batched_audio = modality_from_latent_state(
            _repeat_state(audio_state, n),
            a_context,
            _batched_sigma(audio_state),
            enabled=not a_skip,
        )

    trace_enabled = bool(os.environ.get("LTX_GUIDANCE_TRACE_FILE"))
    if trace_enabled:
        _sync_for_trace()
        transformer_start = time.perf_counter()
    _mark_cuda_graph_step()
    all_v, all_a = transformer(
        video=batched_video, audio=batched_audio, perturbations=BatchedPerturbationConfig(ptb_configs)
    )
    if trace_enabled:
        _sync_for_trace()
        _write_guidance_trace(
            {
                "step_index": step_index,
                "sigma": float(sigma.detach().cpu().item() if isinstance(sigma, torch.Tensor) else sigma),
                "cfg_scale": video_guider.params.cfg_scale,
                "stg_scale": video_guider.params.stg_scale,
                "modality_scale": video_guider.params.modality_scale,
                "rescale_scale": video_guider.params.rescale_scale,
                "audio_cfg_scale": audio_guider.params.cfg_scale,
                "audio_stg_scale": audio_guider.params.stg_scale,
                "audio_modality_scale": audio_guider.params.modality_scale,
                "pass_names": pass_names,
                "n_passes": n,
                "batch_shape_before_transformer": {
                    "video": _modality_shapes(batched_video),
                    "audio": _modality_shapes(batched_audio),
                },
                "tgate_active": _tgate_active(step_index),
                "step_seconds": round(time.perf_counter() - transformer_start, 6),
                "guided": True,
            }
        )

    # Split results back and combine via guiders.
    splits_v = list(all_v.chunk(n)) if all_v is not None else [0.0] * n
    splits_a = list(all_a.chunk(n)) if all_a is not None else [0.0] * n
    r = dict(zip(pass_names, zip(splits_v, splits_a, strict=True), strict=True))

    cond_v, cond_a = r["cond"]
    uncond_v, uncond_a = r.get("uncond", (0.0, 0.0))
    ptb_v, ptb_a = r.get("ptb", (0.0, 0.0))
    mod_v, mod_a = r.get("mod", (0.0, 0.0))

    # CFG cache: reconstruct skipped passes from cached deltas, or store deltas on full steps.
    if _cfg_cache_enabled() and not (v_skip and a_skip):
        if reuse_cfg and _cfg_cache:
            if "uncond" in _cfg_cache:
                du = _cfg_cache["uncond"]
                uncond_v, uncond_a = _cfg_recon(cond_v, du[0]), _cfg_recon(cond_a, du[1])
            if "ptb" in _cfg_cache:
                dp = _cfg_cache["ptb"]
                ptb_v, ptb_a = _cfg_recon(cond_v, dp[0]), _cfg_recon(cond_a, dp[1])
            if "mod" in _cfg_cache:
                dm = _cfg_cache["mod"]
                mod_v, mod_a = _cfg_recon(cond_v, dm[0]), _cfg_recon(cond_a, dm[1])
        elif not reuse_cfg:
            if "uncond" in r:
                _cfg_cache["uncond"] = (_cfg_delta(cond_v, uncond_v), _cfg_delta(cond_a, uncond_a))
            if "ptb" in r:
                _cfg_cache["ptb"] = (_cfg_delta(cond_v, ptb_v), _cfg_delta(cond_a, ptb_a))
            if "mod" in r:
                _cfg_cache["mod"] = (_cfg_delta(cond_v, mod_v), _cfg_delta(cond_a, mod_a))

    # Guard each modality on its state: video-only (audio_state=None) must NOT call
    # audio_guider.calculate (cond_a etc. are float 0.0 → .std() crashes). Latent bug
    # exposed by the video-only / audio-off path.
    denoised_video = (
        (last_denoised_video if v_skip else video_guider.calculate(cond_v, uncond_v, ptb_v, mod_v))
        if video_state is not None
        else None
    )
    denoised_audio = (
        (last_denoised_audio if a_skip else audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a))
        if audio_state is not None
        else None
    )
    return (
        DenoisedLatentResult.result_or_none(
            denoised=denoised_video, uncond=uncond_v, cond=cond_v, ptb=ptb_v, mod=mod_v
        ),
        DenoisedLatentResult.result_or_none(
            denoised=denoised_audio, uncond=uncond_a, cond=cond_a, ptb=ptb_a, mod=mod_a
        ),
    )


class SimpleDenoiser:
    """Single transformer call, no guidance.
    Passes ``None`` Modality for absent modalities.
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        os.environ["LTX_CURRENT_STEP_INDEX"] = str(step_index)
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, self.v_context, sigma) if video_state is not None else None
        pos_audio = modality_from_latent_state(audio_state, self.a_context, sigma) if audio_state is not None else None
        trace_enabled = bool(os.environ.get("LTX_GUIDANCE_TRACE_FILE"))
        if trace_enabled:
            _sync_for_trace()
            transformer_start = time.perf_counter()
        _mark_cuda_graph_step()
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        if trace_enabled:
            _sync_for_trace()
            _write_guidance_trace(
                {
                    "step_index": step_index,
                    "sigma": float(sigma.detach().cpu().item() if isinstance(sigma, torch.Tensor) else sigma),
                    "cfg_scale": 1.0,
                    "stg_scale": 0.0,
                    "modality_scale": 1.0,
                    "rescale_scale": 0.0,
                    "audio_cfg_scale": 1.0,
                    "audio_stg_scale": 0.0,
                    "audio_modality_scale": 1.0,
                    "pass_names": ["cond"],
                    "n_passes": 1,
                    "batch_shape_before_transformer": {
                        "video": _modality_shapes(pos_video),
                        "audio": _modality_shapes(pos_audio),
                    },
                    "tgate_active": _tgate_active(step_index),
                    "step_seconds": round(time.perf_counter() - transformer_start, 6),
                    "guided": False,
                }
            )
        return (
            DenoisedLatentResult.result_or_none(denoised=denoised_video),
            DenoisedLatentResult.result_or_none(denoised=denoised_audio),
        )


class GuidedDenoiser:
    """Static guiders — handles CFG + STG + isolated modality.
    Context/guider can be ``None`` for absent modalities (a positive-only
    guider is substituted at call time).
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider: MultiModalGuider | None = None,
        audio_guider: MultiModalGuider | None = None,
        force_uncond_pass: bool = False,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider = video_guider
        self.audio_guider = audio_guider
        self.force_uncond_pass = force_uncond_pass
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        os.environ["LTX_CURRENT_STEP_INDEX"] = str(step_index)
        guided_denoise_result_v, guided_denoise_result_a = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=_ensure_guider(self.video_guider),
            audio_guider=_ensure_guider(self.audio_guider),
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
            force_uncond_pass=self.force_uncond_pass,
        )
        self._last_denoised_video = guided_denoise_result_v.denoised if guided_denoise_result_v is not None else None
        self._last_denoised_audio = guided_denoise_result_a.denoised if guided_denoise_result_a is not None else None
        return guided_denoise_result_v, guided_denoise_result_a


class FactoryGuidedDenoiser:
    """Resolves guiders per-step from sigma, then delegates to shared guided logic."""

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider_factory: MultiModalGuiderFactory | None = None,
        audio_guider_factory: MultiModalGuiderFactory | None = None,
        force_uncond_pass: bool = False,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider_factory = video_guider_factory
        self.audio_guider_factory = audio_guider_factory
        self.force_uncond_pass = force_uncond_pass
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None
        self._sigma_vals_cached: list[float] | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        os.environ["LTX_CURRENT_STEP_INDEX"] = str(step_index)
        if self._sigma_vals_cached is None:
            self._sigma_vals_cached = sigmas.detach().cpu().tolist()
        sigma_val = self._sigma_vals_cached[step_index]

        video_guider = _ensure_guider(
            self.video_guider_factory.build_from_sigma(sigma_val) if self.video_guider_factory else None
        )
        audio_guider = _ensure_guider(
            (self.audio_guider_factory or self.video_guider_factory).build_from_sigma(sigma_val)
            if self.video_guider_factory or self.audio_guider_factory
            else None
        )

        guided_denoise_result_v, guided_denoise_result_a = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=video_guider,
            audio_guider=audio_guider,
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
            force_uncond_pass=self.force_uncond_pass,
        )
        self._last_denoised_video = guided_denoise_result_v.denoised if guided_denoise_result_v is not None else None
        self._last_denoised_audio = guided_denoise_result_a.denoised if guided_denoise_result_a is not None else None
        return guided_denoise_result_v, guided_denoise_result_a
