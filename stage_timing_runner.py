#!/usr/bin/env python3
"""Measure per-stage timing for LTX-2.3 dev two-stage image-to-video."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import gc
import hashlib
import json
import logging
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import nn
from PIL import Image
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

try:
    from transformers import Gemma3ForCausalLM
except ImportError:  # pragma: no cover - depends on transformers version.
    try:
        from transformers.models.gemma3 import Gemma3ForCausalLM
    except ImportError:
        Gemma3ForCausalLM = None

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.blocks import DiffusionStage, PromptEncoder
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import _build_state
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import combined_image_conditionings, generate_enhanced_prompt
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.samplers import (
    euler_denoising_loop,
    gradient_estimating_euler_denoising_loop,
)
from ltx_pipelines.utils.types import ModalitySpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--gemma-prequant-root")
    parser.add_argument("--spatial-upscaler", required=True)
    parser.add_argument("--distilled-lora", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--only", nargs="*", default=["02_cj_car_dusk"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--attention-type", default="flash_attention_3")
    parser.add_argument("--fast-vae", action="store_true")
    parser.add_argument("--tgate-start-step", type=int)
    parser.add_argument("--no-tiling", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--gemma-cache", action="store_true")
    parser.add_argument("--gemma-8bit", action="store_true")
    parser.add_argument("--resident-stages", action="store_true")
    parser.add_argument("--resident-gemma", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--cuda-graph-warmup-calls", type=int, default=1)
    parser.add_argument("--cuda-graph-clone-outputs", action="store_true")
    parser.add_argument("--cuda-graph-profile-timings", action="store_true")
    parser.add_argument("--torch-compile-transformer", action="store_true")
    parser.add_argument("--torch-compile-mode", default="max-autotune")
    parser.add_argument("--torch-compile-dynamic", action="store_true")
    parser.add_argument("--stop-after-stage1", action="store_true")
    parser.add_argument("--decode-stage1-output", action="store_true")
    parser.add_argument("--stage1-png-dir", type=Path)
    parser.add_argument("--stage1-mp4", action="store_true")
    parser.add_argument("--profile-stage1", action="store_true")
    parser.add_argument("--profile-prompt-encoder", action="store_true")
    parser.add_argument("--profile-model-lifecycle", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--guidance-trace-dir", type=Path)
    parser.add_argument("--video-cfg-scale", type=float)
    parser.add_argument("--video-stg-scale", type=float)
    parser.add_argument("--video-modality-scale", type=float)
    parser.add_argument("--video-rescale-scale", type=float)
    parser.add_argument("--audio-cfg-scale", type=float)
    parser.add_argument("--audio-stg-scale", type=float)
    parser.add_argument("--audio-modality-scale", type=float)
    parser.add_argument("--audio-rescale-scale", type=float)
    parser.add_argument("--guidance-window-steps", help="Inclusive step window START:END for all guidance branches.")
    parser.add_argument("--cfg-window-steps", help="Inclusive step window START:END for CFG only.")
    parser.add_argument("--stg-window-steps", help="Inclusive step window START:END for STG only.")
    parser.add_argument("--modality-window-steps", help="Inclusive step window START:END for modality guidance only.")
    parser.add_argument("--encode-crf", type=int, default=19)
    parser.add_argument("--encode-preset", default="veryfast")
    parser.add_argument("--encode-thread-count", type=int, default=0)
    return parser.parse_args()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_allocated_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated() / (1024**3), 3)


def module_resident_gib(module: nn.Module) -> float:
    tensors = list(module.parameters()) + list(module.buffers())
    return round(sum(t.numel() * t.element_size() for t in tensors) / (1024**3), 3)


@contextmanager
def timed(record: dict[str, float], name: str) -> Iterator[None]:
    sync_cuda()
    start = time.perf_counter()
    try:
        yield
    finally:
        sync_cuda()
        record[name] = round(time.perf_counter() - start, 3)


def apply_runtime_env(args: argparse.Namespace) -> None:
    os.environ["LTX_ATTENTION_TYPE"] = args.attention_type
    if args.fast_vae:
        os.environ["LTX_FAST_VAE"] = "1"
    else:
        os.environ.pop("LTX_FAST_VAE", None)
    if args.tgate_start_step is not None:
        os.environ["LTX_TGATE_START_STEP"] = str(args.tgate_start_step)
    else:
        os.environ.pop("LTX_TGATE_START_STEP", None)


def overridden_settings(args: argparse.Namespace, manifest: dict) -> dict:
    settings = dict(manifest["settings"])
    for key in ("width", "height", "frames", "fps"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    settings["duration_seconds"] = round(settings["frames"] / settings["fps"], 4)
    return settings


def parse_step_window(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if ":" not in value:
        raise ValueError(f"Step window must be START:END, got {value!r}")
    start_raw, end_raw = value.split(":", 1)
    start = int(start_raw)
    end = int(end_raw)
    if start < 0 or end < start:
        raise ValueError(f"Invalid step window {value!r}")
    return start, end


def step_in_window(step_index: int, window: tuple[int, int] | None, default: bool = True) -> bool:
    if window is None:
        return default
    return window[0] <= step_index <= window[1]


def override_guider_params(
    params: MultiModalGuiderParams,
    args: argparse.Namespace,
    prefix: str,
) -> MultiModalGuiderParams:
    updates: dict[str, float] = {}
    for cli_name, field_name in (
        (f"{prefix}_cfg_scale", "cfg_scale"),
        (f"{prefix}_stg_scale", "stg_scale"),
        (f"{prefix}_modality_scale", "modality_scale"),
        (f"{prefix}_rescale_scale", "rescale_scale"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            updates[field_name] = value
    # v8.10 Round-2 sweep knobs (env-only; CLI flags above take priority). Bit-exact to v8.9
    # when LTX_STG_BLOCKS / LTX_STG_SCALE unset. Sole STG chokepoint (stage2 = SimpleDenoiser, no guider).
    _stg_blocks_env = os.environ.get("LTX_STG_BLOCKS")
    if _stg_blocks_env is not None and "stg_blocks" not in updates:
        updates["stg_blocks"] = [int(b) for b in json.loads(_stg_blocks_env)]
    _stg_scale_env = os.environ.get("LTX_STG_SCALE")
    if _stg_scale_env is not None and "stg_scale" not in updates:
        updates["stg_scale"] = float(_stg_scale_env)
    return dataclasses.replace(params, **updates) if updates else params


def cond_only_params(params: MultiModalGuiderParams) -> MultiModalGuiderParams:
    return dataclasses.replace(
        params,
        cfg_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
        rescale_scale=0.0,
    )


def guidance_window_summary(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "guidance_window_steps": args.guidance_window_steps,
        "cfg_window_steps": args.cfg_window_steps,
        "stg_window_steps": args.stg_window_steps,
        "modality_window_steps": args.modality_window_steps,
    }


def make_guidance_factory(
    *,
    params: MultiModalGuiderParams,
    negative_context: torch.Tensor | None,
    sigmas: torch.Tensor,
    args: argparse.Namespace,
    prefix: str,
) -> tuple[MultiModalGuiderFactory, dict[str, Any]]:
    base = override_guider_params(params, args, prefix)
    default_window = parse_step_window(args.guidance_window_steps)
    cfg_window = parse_step_window(args.cfg_window_steps) or default_window
    stg_window = parse_step_window(args.stg_window_steps) or default_window
    modality_window = parse_step_window(args.modality_window_steps) or default_window
    windowed = any(window is not None for window in (cfg_window, stg_window, modality_window))

    info: dict[str, Any] = {
        "base": dataclasses.asdict(base),
        "windowed": windowed,
        "windows": guidance_window_summary(args),
        "sigma_schedule": [float(s) for s in sigmas.detach().cpu().tolist()],
    }
    if not windowed:
        return create_multimodal_guider_factory(params=base, negative_context=negative_context), info

    schedule: dict[float, MultiModalGuiderParams] = {}
    rows = []
    sigmas_list = [float(s) for s in sigmas.detach().cpu().tolist()]
    for step_index, sigma in enumerate(sigmas_list):
        cfg_active = step_in_window(step_index, cfg_window)
        stg_active = step_in_window(step_index, stg_window)
        modality_active = step_in_window(step_index, modality_window)
        any_guidance = cfg_active or stg_active or modality_active
        step_params = dataclasses.replace(
            base,
            cfg_scale=base.cfg_scale if cfg_active else 1.0,
            stg_scale=base.stg_scale if stg_active else 0.0,
            modality_scale=base.modality_scale if modality_active else 1.0,
            rescale_scale=base.rescale_scale if any_guidance else 0.0,
        )
        schedule[sigma] = step_params
        rows.append(
            {
                "step_index": step_index,
                "sigma": sigma,
                "cfg_active": cfg_active,
                "stg_active": stg_active,
                "modality_active": modality_active,
                "params": dataclasses.asdict(step_params),
            }
        )
    info["per_step"] = rows
    return MultiModalGuiderFactory.from_dict(schedule, negative_context=negative_context), info


def peak_vram_gib() -> float:
    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_video_frames_png(video: Iterator[torch.Tensor], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    for chunk in video:
        frames = chunk.detach()
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim != 4:
            raise ValueError(
                f"Expected decoded video chunk as [F,H,W,C] or [1,F,H,W,C], got {tuple(frames.shape)}"
            )
        frames_u8 = frames.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
        for frame in frames_u8:
            Image.fromarray(frame).save(out_dir / f"{frame_index:06d}.png")
            frame_index += 1
    return frame_index


class PromptEmbeddingCache:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Any, Any]] = {}
        self.hits = 0
        self.misses = 0

    def key(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image_path: Path,
        seed: int,
        gemma_root: str,
        gemma_8bit: bool,
        gemma_prequant_root: str | None,
    ) -> str:
        payload = {
            "version": 1,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_sha256": sha256_file(image_path),
            "enhance_prompt_seed": seed,
            "gemma_root": str(gemma_root),
            "gemma_8bit": gemma_8bit,
            "gemma_prequant_root": str(gemma_prequant_root) if gemma_prequant_root else None,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get_or_encode(self, key: str, encode: Callable[[], tuple[Any, Any]]) -> tuple[tuple[Any, Any], bool]:
        cached = self._items.get(key)
        if cached is not None:
            self.hits += 1
            return cached, True
        value = encode()
        self._items[key] = value
        self.misses += 1
        return value, False

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._items), "hits": self.hits, "misses": self.misses}


class TimedPromptEncoder:
    def __init__(self, inner: Any, *, keep_resident_text_encoder: bool = False) -> None:
        self._inner = inner
        self.events: list[dict[str, Any]] = []
        self._keep_resident_text_encoder = keep_resident_text_encoder
        self._resident_text_encoder_ctx: Any | None = None
        self._resident_text_encoder: Any | None = None
        self.close_events: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        prompts = list(args[0]) if args else list(kwargs.pop("prompts"))
        enhance_first_prompt = kwargs.pop("enhance_first_prompt", False)
        enhance_prompt_image = kwargs.pop("enhance_prompt_image", None)
        enhance_prompt_seed = kwargs.pop("enhance_prompt_seed", 42)
        if kwargs:
            return self._fallback_call(prompts, enhance_first_prompt, enhance_prompt_image, enhance_prompt_seed, kwargs)

        start = time.perf_counter()
        cuda_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        event: dict[str, Any] = {
            "prompts": len(prompts),
            "enhance_first_prompt": bool(enhance_first_prompt),
            "max_vram_gib_before": round(cuda_before / (1024**3), 3),
        }
        try:
            raw_outputs = self._encode_with_timed_text_encoder(
                prompts,
                event,
                enhance_first_prompt=enhance_first_prompt,
                enhance_prompt_image=enhance_prompt_image,
                enhance_prompt_seed=enhance_prompt_seed,
            )
            return self._process_embeddings_with_timing(raw_outputs, event)
        finally:
            sync_cuda()
            event["seconds"] = round(time.perf_counter() - start, 3)
            event["max_vram_gib_after"] = peak_vram_gib()
            self.events.append(event)

    def _fallback_call(
        self,
        prompts: list[str],
        enhance_first_prompt: bool,
        enhance_prompt_image: str | None,
        enhance_prompt_seed: int,
        extra_kwargs: dict[str, Any],
    ) -> Any:
        start = time.perf_counter()
        try:
            return self._inner(
                prompts,
                enhance_first_prompt=enhance_first_prompt,
                enhance_prompt_image=enhance_prompt_image,
                enhance_prompt_seed=enhance_prompt_seed,
                **extra_kwargs,
            )
        finally:
            sync_cuda()
            self.events.append(
                {
                    "seconds": round(time.perf_counter() - start, 3),
                    "fallback": True,
                    "extra_kwargs": sorted(extra_kwargs),
                    "max_vram_gib_after": peak_vram_gib(),
                }
            )

    def _encode_with_timed_text_encoder(
        self,
        prompts: list[str],
        event: dict[str, Any],
        *,
        enhance_first_prompt: bool,
        enhance_prompt_image: str | None,
        enhance_prompt_seed: int,
    ) -> list[Any]:
        if self._keep_resident_text_encoder and self._resident_text_encoder is not None:
            text_encoder_ctx = self._resident_text_encoder_ctx
            text_encoder = self._resident_text_encoder
            event["resident_cache_hit"] = True
            event["text_encoder_ctx_create_seconds"] = 0.0
            event["text_encoder_enter_seconds"] = 0.0
        else:
            ctx_create_start = time.perf_counter()
            text_encoder_ctx = self._inner._text_encoder_ctx()
            sync_cuda()
            event["text_encoder_ctx_create_seconds"] = round(time.perf_counter() - ctx_create_start, 3)
            builder = getattr(self._inner, "_text_encoder_builder", None)
            builder_event = getattr(builder, "last_build_event", None)
            if builder_event is not None:
                event["text_encoder_builder_event"] = builder_event
            enter_start = time.perf_counter()
            text_encoder = text_encoder_ctx.__enter__()
            sync_cuda()
            event["text_encoder_enter_seconds"] = round(time.perf_counter() - enter_start, 3)
            event["resident_cache_hit"] = False
            if self._keep_resident_text_encoder:
                self._resident_text_encoder_ctx = text_encoder_ctx
                self._resident_text_encoder = text_encoder
        event["text_encoder_resident_weights_gib"] = module_resident_gib(text_encoder)
        event["cuda_allocated_after_text_encoder_load_gib"] = cuda_allocated_gib()
        raw_outputs: list[Any] = []
        try:
            if enhance_first_prompt:
                enhance_start = time.perf_counter()
                prompts = list(prompts)
                prompts[0] = generate_enhanced_prompt(
                    text_encoder,
                    prompts[0],
                    enhance_prompt_image,
                    seed=enhance_prompt_seed,
                )
                sync_cuda()
                event["enhance_first_prompt_seconds"] = round(time.perf_counter() - enhance_start, 3)

            encode_events = []
            for index, prompt in enumerate(prompts):
                encode_start = time.perf_counter()
                raw_outputs.append(text_encoder.encode(prompt))
                sync_cuda()
                encode_events.append(
                    {
                        "index": index,
                        "chars": len(prompt),
                        "seconds": round(time.perf_counter() - encode_start, 3),
                    }
                )
            event["raw_encode_events"] = encode_events
        except BaseException as exc:
            if not self._keep_resident_text_encoder:
                exit_start = time.perf_counter()
                text_encoder_ctx.__exit__(type(exc), exc, exc.__traceback__)
                sync_cuda()
                event["text_encoder_exit_seconds"] = round(time.perf_counter() - exit_start, 3)
            raise

        if self._keep_resident_text_encoder:
            event["text_encoder_exit_seconds"] = 0.0
            sync_cuda()
        else:
            exit_start = time.perf_counter()
            text_encoder_ctx.__exit__(None, None, None)
            sync_cuda()
            event["text_encoder_exit_seconds"] = round(time.perf_counter() - exit_start, 3)
        event["cuda_allocated_after_text_encoder_exit_gib"] = cuda_allocated_gib()
        return raw_outputs

    def _process_embeddings_with_timing(self, raw_outputs: list[Any], event: dict[str, Any]) -> Any:
        build_start = time.perf_counter()
        embeddings_processor = self._inner._build_embeddings_processor()
        sync_cuda()
        event["embeddings_processor_build_seconds"] = round(time.perf_counter() - build_start, 3)
        try:
            process_events = []
            processed_outputs = []
            for index, (hidden_states, mask) in enumerate(raw_outputs):
                process_start = time.perf_counter()
                processed_outputs.append(embeddings_processor.process_hidden_states(hidden_states, mask))
                sync_cuda()
                process_events.append(
                    {
                        "index": index,
                        "seconds": round(time.perf_counter() - process_start, 3),
                    }
                )
            event["embeddings_processor_process_events"] = process_events
            return processed_outputs
        finally:
            release_start = time.perf_counter()
            embeddings_processor.to("meta")
            from ltx_pipelines.utils.helpers import cleanup_memory

            cleanup_memory()
            sync_cuda()
            event["embeddings_processor_release_seconds"] = round(time.perf_counter() - release_start, 3)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def close_resident(self) -> list[dict[str, Any]]:
        if self._resident_text_encoder_ctx is None:
            return list(self.close_events)
        sync_cuda()
        start = time.perf_counter()
        self._resident_text_encoder_ctx.__exit__(None, None, None)
        sync_cuda()
        self.close_events.append(
            {
                "seconds": round(time.perf_counter() - start, 3),
                "cuda_allocated_after_close_gib": cuda_allocated_gib(),
                "peak_vram_after_close_gib": peak_vram_gib(),
            }
        )
        self._resident_text_encoder_ctx = None
        self._resident_text_encoder = None
        return list(self.close_events)


class MetaSafeGemmaTextEncoder(GemmaTextEncoder):
    def to(self, *args: Any, **kwargs: Any) -> "MetaSafeGemmaTextEncoder":
        device = args[0] if args else kwargs.get("device")
        if str(device) == "meta":
            self.model = None
            self.processor = None
            self.tokenizer = None
            return self
        return super().to(*args, **kwargs)


class BitsAndBytesGemmaBuilder:
    def __init__(self, gemma_root: str) -> None:
        self._gemma_root = gemma_root

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_: object,
    ) -> MetaSafeGemmaTextEncoder:
        from ltx_core.utils import find_matching_file

        target_device = device or torch.device("cuda")
        model_folder = find_matching_file(self._gemma_root, "model*.safetensors").parent
        tokenizer_folder = find_matching_file(self._gemma_root, "tokenizer.model").parent
        model = Gemma3ForConditionalGeneration.from_pretrained(
            str(model_folder),
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.bfloat16,
            device_map={"": target_device},
            local_files_only=True,
        )
        tokenizer = LTXVGemmaTokenizer(str(tokenizer_folder), 1024)
        return MetaSafeGemmaTextEncoder(
            model=model,
            tokenizer=tokenizer,
            dtype=dtype or torch.bfloat16,
        ).eval()


class PrequantCausalGemmaBuilder:
    def __init__(self, *, model_root: str, tokenizer_root: str) -> None:
        self._model_root = model_root
        self._tokenizer_root = tokenizer_root
        self.last_build_event: dict[str, Any] | None = None
        init_start = time.perf_counter()
        from ltx_core.utils import find_matching_file

        tokenizer_folder = find_matching_file(self._tokenizer_root, "tokenizer.model").parent
        self._cached_tokenizer: LTXVGemmaTokenizer | None = LTXVGemmaTokenizer(str(tokenizer_folder), 1024)
        self.init_event = {
            "tokenizer_preload_seconds": round(time.perf_counter() - init_start, 3),
        }

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_: object,
    ) -> MetaSafeGemmaTextEncoder:
        if Gemma3ForCausalLM is None:
            raise RuntimeError("Gemma3ForCausalLM is not available in this transformers build")
        from ltx_core.utils import find_matching_file

        target_device = device or torch.device("cuda")
        event: dict[str, Any] = {}
        event["builder_init_event"] = self.init_event
        total_start = time.perf_counter()
        find_start = time.perf_counter()
        model_folder = find_matching_file(self._model_root, "model*.safetensors").parent
        tokenizer_folder = find_matching_file(self._tokenizer_root, "tokenizer.model").parent
        event["find_paths_seconds"] = round(time.perf_counter() - find_start, 3)
        model_start = time.perf_counter()
        model = Gemma3ForCausalLM.from_pretrained(
            str(model_folder),
            torch_dtype=torch.bfloat16,
            device_map={"": target_device},
            local_files_only=True,
        )
        sync_cuda()
        event["from_pretrained_seconds"] = round(time.perf_counter() - model_start, 3)
        tokenizer_start = time.perf_counter()
        if self._cached_tokenizer is None:
            self._cached_tokenizer = LTXVGemmaTokenizer(str(tokenizer_folder), 1024)
            event["tokenizer_cache_hit"] = False
        else:
            event["tokenizer_cache_hit"] = True
        sync_cuda()
        event["tokenizer_seconds"] = round(time.perf_counter() - tokenizer_start, 3)
        wrap_start = time.perf_counter()
        text_encoder = MetaSafeGemmaTextEncoder(
            model=model,
            tokenizer=self._cached_tokenizer,
            dtype=dtype or torch.bfloat16,
        ).eval()
        sync_cuda()
        event["wrap_seconds"] = round(time.perf_counter() - wrap_start, 3)
        event["total_seconds"] = round(time.perf_counter() - total_start, 3)
        self.last_build_event = event
        return text_encoder


class ResidentStageCache:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self.close_events: list[dict[str, Any]] = []

    def key_for(
        self,
        graph_key: str,
        *,
        width: int,
        height: int,
        frames: int,
        fps: float,
        has_video: bool,
        has_audio: bool,
    ) -> str:
        return json.dumps(
            {
                "stage": graph_key,
                "width": width,
                "height": height,
                "frames": frames,
                "fps": fps,
                "has_video": has_video,
                "has_audio": has_audio,
            },
            sort_keys=True,
        )

    def get(
        self,
        stage: DiffusionStage,
        cache_key: str,
        *,
        video_tools: VideoLatentTools | None,
        lifecycle_stats: dict[str, Any] | None,
    ) -> nn.Module:
        entry = self._entries.get(cache_key)
        if entry is not None:
            transformer = entry["transformer"]
            if lifecycle_stats is not None:
                lifecycle_stats["resident_cache_hit"] = True
                lifecycle_stats["resident_cache_key"] = cache_key
                lifecycle_stats["build_load_seconds"] = 0.0
                lifecycle_stats["context_enter_seconds"] = 0.0
                lifecycle_stats["resident_weights_gib"] = module_resident_gib(transformer)
                lifecycle_stats["cuda_allocated_after_load_gib"] = cuda_allocated_gib()
                lifecycle_stats["cuda_peak_after_load_gib"] = peak_vram_gib()
            return transformer

        sync_cuda()
        build_start = time.perf_counter()
        model_ctx = stage.model_context(video_tools=video_tools)
        sync_cuda()
        build_seconds = round(time.perf_counter() - build_start, 3)
        enter_start = time.perf_counter()
        transformer = model_ctx.__enter__()
        sync_cuda()
        context_enter_seconds = round(time.perf_counter() - enter_start, 3)
        self._entries[cache_key] = {
            "model_ctx": model_ctx,
            "transformer": transformer,
            "created_at_peak_gib": peak_vram_gib(),
        }
        if lifecycle_stats is not None:
            lifecycle_stats["resident_cache_hit"] = False
            lifecycle_stats["resident_cache_key"] = cache_key
            lifecycle_stats["build_load_seconds"] = build_seconds
            lifecycle_stats["context_enter_seconds"] = context_enter_seconds
            lifecycle_stats["resident_weights_gib"] = module_resident_gib(transformer)
            lifecycle_stats["cuda_allocated_after_load_gib"] = cuda_allocated_gib()
            lifecycle_stats["cuda_peak_after_load_gib"] = peak_vram_gib()
        return transformer

    def close(self) -> list[dict[str, Any]]:
        for cache_key, entry in list(self._entries.items()):
            sync_cuda()
            start = time.perf_counter()
            entry["model_ctx"].__exit__(None, None, None)
            sync_cuda()
            self.close_events.append(
                {
                    "cache_key": cache_key,
                    "seconds": round(time.perf_counter() - start, 3),
                    "cuda_allocated_after_close_gib": cuda_allocated_gib(),
                    "peak_vram_after_close_gib": peak_vram_gib(),
                }
            )
            del self._entries[cache_key]
        return list(self.close_events)


def _normalize_for_signature(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, torch.Tensor):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if dataclasses.is_dataclass(value):
        return {
            field.name: _normalize_for_signature(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_signature(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_for_signature(item) for key, item in sorted(value.items())}
    return repr(value)


def _tensor_shape(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _graph_entry_description(video: Any, audio: Any, perturbations: Any) -> dict[str, Any]:
    current_step = os.environ.get("LTX_CURRENT_STEP_INDEX")
    start = os.environ.get("LTX_TGATE_START_STEP")
    try:
        tgate_active = start is not None and current_step is not None and int(current_step) >= int(start)
    except ValueError:
        tgate_active = False
    perturbation_items = getattr(perturbations, "perturbations", None)
    if perturbation_items is None:
        perturbation_count = 0
    else:
        try:
            perturbation_count = len(perturbation_items)
        except TypeError:
            perturbation_count = None
    return {
        "step_index": int(current_step) if current_step is not None and current_step.isdigit() else current_step,
        "tgate_active": tgate_active,
        "video_enabled": getattr(video, "enabled", None),
        "audio_enabled": getattr(audio, "enabled", None),
        "video_latent": _tensor_shape(getattr(video, "latent", None)),
        "audio_latent": _tensor_shape(getattr(audio, "latent", None)),
        "video_context": _tensor_shape(getattr(video, "context", None)),
        "audio_context": _tensor_shape(getattr(audio, "context", None)),
        "video_attention_mask": _tensor_shape(getattr(video, "attention_mask", None)),
        "audio_attention_mask": _tensor_shape(getattr(audio, "attention_mask", None)),
        "perturbations": repr(perturbations),
        "perturbation_count": perturbation_count,
    }


def _tensor_like_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    return value


def _clone_dataclass_tensors(value: Any) -> Any:
    if value is None:
        return None
    if not dataclasses.is_dataclass(value):
        raise TypeError(f"Expected dataclass input, got {type(value)!r}")
    return type(value)(
        **{
            field.name: _tensor_like_clone(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    )


def _copy_dataclass_tensors(dst: Any, src: Any) -> None:
    if dst is None or src is None:
        return
    for field in dataclasses.fields(src):
        src_value = getattr(src, field.name)
        dst_value = getattr(dst, field.name)
        if isinstance(src_value, torch.Tensor):
            dst_value.copy_(src_value)


def _clone_output(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.clone() if value is not None else None


_TRTLLM_SCALED_MM_PREWARMED = False


def prewarm_trtllm_scaled_mm() -> None:
    global _TRTLLM_SCALED_MM_PREWARMED
    if _TRTLLM_SCALED_MM_PREWARMED or not torch.cuda.is_available():
        return
    import tensorrt_llm  # noqa: F401, PLC0415 - registers custom torch ops.

    x = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16)
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    qinput, cur_input_scale = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)
    weight = torch.randn((32, 16), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    weight_scale = torch.ones((), device="cuda", dtype=torch.float32)
    torch.ops.trtllm.cublas_scaled_mm(
        qinput,
        weight.t(),
        scale_a=cur_input_scale,
        scale_b=weight_scale,
        bias=None,
        out_dtype=torch.bfloat16,
    )
    sync_cuda()
    _TRTLLM_SCALED_MM_PREWARMED = True


class _CudaGraphEntry:
    def __init__(self, video: Any, audio: Any, perturbations: Any) -> None:
        self.static_video = _clone_dataclass_tensors(video) if video is not None else None
        self.static_audio = _clone_dataclass_tensors(audio) if audio is not None else None
        self.static_perturbations = perturbations
        self.calls = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: tuple[torch.Tensor | None, torch.Tensor | None] | None = None
        self.disabled_error: str | None = None


class CudaGraphTransformerAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        warmup_calls: int = 1,
        clone_outputs: bool = False,
        profile_timings: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self._warmup_calls = warmup_calls
        self._clone_outputs = clone_outputs
        self._profile_timings = profile_timings
        self._entries: dict[str, _CudaGraphEntry] = {}
        self.stats = {
            "eager_calls": 0,
            "captured_calls": 0,
            "replayed_calls": 0,
            "disabled_signatures": 0,
            "eager_seconds": 0.0,
            "capture_seconds": 0.0,
            "replay_seconds": 0.0,
        }
        self.errors: list[str] = []
        self.graph_descriptions: dict[str, dict[str, Any]] = {}

    def _signature(self, video: Any, audio: Any, perturbations: Any) -> str:
        payload = {
            "video": _normalize_for_signature(video),
            "audio": _normalize_for_signature(audio),
            "perturbations": _normalize_for_signature(perturbations),
            "tgate_active": self._tgate_active(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _tgate_active(self) -> bool:
        start = os.environ.get("LTX_TGATE_START_STEP")
        if not start:
            return False
        try:
            current = int(os.environ.get("LTX_CURRENT_STEP_INDEX", "-1"))
            return current >= int(start)
        except ValueError:
            return False

    def _copy_inputs(self, entry: _CudaGraphEntry, video: Any, audio: Any) -> None:
        _copy_dataclass_tensors(entry.static_video, video)
        _copy_dataclass_tensors(entry.static_audio, audio)

    def _prewarm_capture_inputs(self, entry: _CudaGraphEntry) -> None:
        velocity_model = getattr(self._model, "velocity_model", None)
        if velocity_model is None:
            return
        if entry.static_video is not None:
            velocity_model.video_args_preprocessor.prepare(entry.static_video, entry.static_audio)
        if entry.static_audio is not None:
            velocity_model.audio_args_preprocessor.prepare(entry.static_audio, entry.static_video)
        sync_cuda()

    def _profiled_call(self, stat_key: str, fn: Callable[[], Any]) -> Any:
        if not self._profile_timings:
            return fn()
        sync_cuda()
        start = time.perf_counter()
        result = fn()
        sync_cuda()
        self.stats[stat_key] = round(float(self.stats[stat_key]) + time.perf_counter() - start, 6)
        return result

    def _run_eager(self, video: Any, audio: Any, perturbations: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self.stats["eager_calls"] += 1
        return self._profiled_call(
            "eager_seconds",
            lambda: self._model(video=video, audio=audio, perturbations=perturbations),
        )

    def forward(  # noqa: D401
        self,
        video: Any,
        audio: Any,
        perturbations: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not torch.cuda.is_available():
            return self._run_eager(video, audio, perturbations)

        signature = self._signature(video, audio, perturbations)
        entry = self._entries.get(signature)
        if entry is None:
            entry = _CudaGraphEntry(video, audio, perturbations)
            self._entries[signature] = entry
            self.graph_descriptions[signature[:16]] = _graph_entry_description(video, audio, perturbations)

        if entry.disabled_error is not None:
            return self._run_eager(video, audio, perturbations)

        if entry.graph is None:
            if entry.calls < self._warmup_calls:
                entry.calls += 1
                return self._run_eager(video, audio, perturbations)

            self._copy_inputs(entry, video, audio)
            self._prewarm_capture_inputs(entry)
            prewarm_trtllm_scaled_mm()
            graph = torch.cuda.CUDAGraph()
            try:
                def capture_once() -> None:
                    with torch.cuda.graph(graph):
                        entry.output = self._model(
                            video=entry.static_video,
                            audio=entry.static_audio,
                            perturbations=entry.static_perturbations,
                        )

                self._profiled_call("capture_seconds", capture_once)
                entry.graph = graph
                self.stats["captured_calls"] += 1
            except Exception as exc:  # noqa: BLE001 - graph capture fallback is benchmark data.
                entry.disabled_error = f"{type(exc).__name__}: {exc}"
                self.stats["disabled_signatures"] += 1
                self.errors.append(entry.disabled_error)
                raise RuntimeError(f"CUDA graph capture failed: {entry.disabled_error}") from exc

        self._copy_inputs(entry, video, audio)
        self._profiled_call("replay_seconds", entry.graph.replay)
        self.stats["replayed_calls"] += 1
        assert entry.output is not None
        if self._clone_outputs:
            return _clone_output(entry.output[0]), _clone_output(entry.output[1])
        return entry.output

    def close(self) -> None:
        sync_cuda()
        self._entries.clear()
        gc.collect()
        sync_cuda()

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


def _video_tools_for(width: int, height: int, frames: int, fps: float) -> VideoLatentTools:
    pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    return VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)


def run_stage_with_optional_cuda_graphs(  # noqa: PLR0913
    stage: DiffusionStage,
    args: argparse.Namespace,
    graph_stats: dict[str, Any],
    graph_key: str,
    *,
    denoiser: Any,
    sigmas: torch.Tensor,
    noiser: GaussianNoiser,
    width: int,
    height: int,
    frames: int,
    fps: float,
    video: ModalitySpec | None,
    audio: ModalitySpec | None,
    loop: Callable[..., tuple[Any, Any]] | None = None,
    max_batch_size: int = 1,
    lifecycle_stats: dict[str, Any] | None = None,
    resident_stage_cache: ResidentStageCache | None = None,
) -> tuple[Any, Any]:
    if (
        not args.cuda_graphs
        and not args.torch_compile_transformer
        and lifecycle_stats is None
        and resident_stage_cache is None
    ):
        return stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            video=video,
            audio=audio,
            loop=loop,
            max_batch_size=max_batch_size,
        )

    video_tools = _video_tools_for(width, height, frames, fps) if video is not None else None
    model_ctx = None
    if resident_stage_cache is not None:
        cache_key = resident_stage_cache.key_for(
            graph_key,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            has_video=video is not None,
            has_audio=audio is not None,
        )
        transformer = resident_stage_cache.get(
            stage,
            cache_key,
            video_tools=video_tools,
            lifecycle_stats=lifecycle_stats,
        )
    else:
        sync_cuda()
        build_start = time.perf_counter()
        model_ctx = stage.model_context(video_tools=video_tools)
        sync_cuda()
        if lifecycle_stats is not None:
            lifecycle_stats["build_load_seconds"] = round(time.perf_counter() - build_start, 3)
        enter_start = time.perf_counter()
        transformer = model_ctx.__enter__()
        sync_cuda()
        if lifecycle_stats is not None:
            lifecycle_stats["context_enter_seconds"] = round(time.perf_counter() - enter_start, 3)
            lifecycle_stats["resident_cache_hit"] = False
            lifecycle_stats["resident_weights_gib"] = module_resident_gib(transformer)
            lifecycle_stats["cuda_allocated_after_load_gib"] = cuda_allocated_gib()
            lifecycle_stats["cuda_peak_after_load_gib"] = peak_vram_gib()
    try:
        active_transformer = transformer
        if args.torch_compile_transformer:
            compile_start = time.perf_counter()
            active_transformer = torch.compile(
                active_transformer,
                mode=args.torch_compile_mode,
                fullgraph=False,
                dynamic=args.torch_compile_dynamic,
            )
            sync_cuda()
            graph_stats[f"{graph_key}_torch_compile"] = {
                "mode": args.torch_compile_mode,
                "dynamic": args.torch_compile_dynamic,
                "compile_wrap_seconds": round(time.perf_counter() - compile_start, 3),
            }
        graph_transformer = (
            CudaGraphTransformerAdapter(
                active_transformer,
                warmup_calls=args.cuda_graph_warmup_calls,
                clone_outputs=args.cuda_graph_clone_outputs,
                profile_timings=args.cuda_graph_profile_timings,
            )
            if args.cuda_graphs
            else active_transformer
        )
        try:
            sync_cuda()
            denoise_start = time.perf_counter()
            result = stage.run(
                    graph_transformer,
                    denoiser=denoiser,
                    sigmas=sigmas,
                    noiser=noiser,
                    width=width,
                    height=height,
                    frames=frames,
                    fps=fps,
                    video=video,
                    audio=audio,
                    loop=loop,
                    max_batch_size=max_batch_size,
                )
            sync_cuda()
            if lifecycle_stats is not None:
                lifecycle_stats["denoise_seconds"] = round(time.perf_counter() - denoise_start, 3)
                lifecycle_stats["cuda_allocated_after_denoise_gib"] = cuda_allocated_gib()
                lifecycle_stats["cuda_peak_after_denoise_gib"] = peak_vram_gib()
            if args.cuda_graphs:
                graph_stats[graph_key] = {
                    **graph_transformer.stats,
                    "signatures": len(graph_transformer._entries),
                    "graph_descriptions": graph_transformer.graph_descriptions,
                    "errors": graph_transformer.errors[:5],
                }
            return result
        finally:
            if args.cuda_graphs:
                graph_transformer.close()
    finally:
        if resident_stage_cache is not None:
            # Resident transformers would otherwise pin the per-block tgate deltas
            # (several GiB at full res) across stages and requests; teardown used to
            # free them implicitly. Drop them — they are write-before-read per run.
            for m in transformer.modules():
                m.__dict__.pop("_tgate_video_delta", None)
                m.__dict__.pop("_tgate_audio_delta", None)
            sync_cuda()
            if lifecycle_stats is not None:
                lifecycle_stats["teardown_seconds"] = 0.0
                lifecycle_stats["cuda_allocated_after_teardown_gib"] = cuda_allocated_gib()
        else:
            assert model_ctx is not None
            sync_cuda()
            teardown_start = time.perf_counter()
            model_ctx.__exit__(None, None, None)
            sync_cuda()
            if lifecycle_stats is not None:
                lifecycle_stats["teardown_seconds"] = round(time.perf_counter() - teardown_start, 3)
                lifecycle_stats["cuda_allocated_after_teardown_gib"] = cuda_allocated_gib()
    return result


# ---------------------------------------------------------------------------
# CAS (Contrast Adaptive Sharpening) — default-on post-decode crispness pass.
# Operates on the pixel chunks yielded by the VAE decoder ([f,h,w,c] in [0,1],
# on GPU) BEFORE H264 encode. Recovers high-frequency detail lost at the motion
# peak (research 2026-06-17: dominant artifacts = thin-structure dissolve +
# sharpness collapse on fast motion). FidelityFX CAS, "better diagonals" variant.
# Lazy generator => cost lands in the encode block, not decode. Tunable per
# request via settings {cas_amount, cas_mix} or env; defaults applied always.
# ---------------------------------------------------------------------------
_CAS_AMOUNT_DEFAULT = float(os.environ.get("LTX_CAS_AMOUNT", "0.6"))   # sharpness lerp(8->5); 0 disables
_CAS_MIX_DEFAULT = float(os.environ.get("LTX_CAS_MIX", "0.7"))         # out = (1-mix)*orig + mix*cas
_CAS_SUBBATCH = int(os.environ.get("LTX_CAS_SUBBATCH", "16"))          # frames/CAS call (VRAM cap)


def _cas_sharpen_chunk(chunk: torch.Tensor, amount: float, mix: float) -> torch.Tensor:
    """FidelityFX Contrast-Adaptive Sharpen on a decoded pixel chunk [f,h,w,c] in [0,1]."""
    if amount <= 0.0:
        return chunk
    orig_dtype = chunk.dtype
    x = chunk.permute(0, 3, 1, 2).float()  # [f,c,h,w]
    peak = -1.0 / (8.0 - 3.0 * float(amount))  # lerp(8,5,amount)
    outs = []
    for s in range(0, x.shape[0], _CAS_SUBBATCH):
        xb = x[s:s + _CAS_SUBBATCH]
        p = torch.nn.functional.pad(xb, (1, 1, 1, 1), mode="replicate")
        a = p[..., :-2, :-2]; b = p[..., :-2, 1:-1]; c = p[..., :-2, 2:]
        d = p[..., 1:-1, :-2]; e = xb;                f = p[..., 1:-1, 2:]
        g = p[..., 2:, :-2];   h = p[..., 2:, 1:-1];  i = p[..., 2:, 2:]
        mn = torch.minimum(torch.minimum(torch.minimum(d, e), torch.minimum(f, b)), h)
        mn = mn + torch.minimum(mn, torch.minimum(torch.minimum(a, c), torch.minimum(g, i)))
        mx = torch.maximum(torch.maximum(torch.maximum(d, e), torch.maximum(f, b)), h)
        mx = mx + torch.maximum(mx, torch.maximum(torch.maximum(a, c), torch.maximum(g, i)))
        amp = torch.sqrt(torch.clamp(torch.minimum(mn, 2.0 - mx) * torch.reciprocal(mx.clamp_min(1e-5)), 0.0, 1.0))
        w = amp * peak
        sharp = (w * (b + d + f + h) + e) * torch.reciprocal(1.0 + 4.0 * w)
        sharp = ((1.0 - mix) * xb + mix * sharp).clamp_(0.0, 1.0)
        outs.append(sharp)
    out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
    return out.permute(0, 2, 3, 1).to(orig_dtype)


def _maybe_cas(decoded, settings: dict):
    """Wrap the decoded-video iterator (or tensor) with CAS. Default-on; lazy."""
    amount = float(settings.get("cas_amount", _CAS_AMOUNT_DEFAULT))
    mix = float(settings.get("cas_mix", _CAS_MIX_DEFAULT))
    if amount <= 0.0:
        return decoded
    if isinstance(decoded, torch.Tensor):
        return _cas_sharpen_chunk(decoded, amount, mix)

    def _gen():
        for chunk in decoded:
            yield _cas_sharpen_chunk(chunk, amount, mix)

    return _gen()


@torch.inference_mode()
def run_case(
    pipeline: TI2VidTwoStagesPipeline,
    case: dict,
    settings: dict,
    args: argparse.Namespace,
    *,
    repeat_idx: int,
    prompt_cache: PromptEmbeddingCache | None,
    resident_stage_cache: ResidentStageCache | None,
) -> dict:
    timers: dict[str, float] = {}
    graph_stats: dict[str, Any] = {}
    record = {
        "id": case["id"],
        "repeat": repeat_idx,
        "seed": case["seed"],
        "prompt": case["prompt"],
        "timers": timers,
        "prompt_cache_hit": False,
        "prompt_cache_key": None,
        "cuda_graphs": graph_stats,
        "vram_after": {},
        "elapsed_seconds": None,
        "peak_vram_gib": None,
        "output": None,
        "output_sha256": None,
        "error": None,
        "guidance_trace_file": None,
        "guidance_factory": None,
        "model_lifecycle": {},
        "negative_prompt_source": case.get("negative_prompt_source", "default"),
        "effective_decode_noise_scale": None,
        "effective_cas": None,
    }
    trace_file = None
    if args.guidance_trace_dir:
        args.guidance_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = args.guidance_trace_dir / f"{args.label}_{case['id']}_r{repeat_idx}_guidance_steps.jsonl"
        trace_file.write_text("")
        record["guidance_trace_file"] = str(trace_file)
        os.environ["LTX_GUIDANCE_TRACE_FILE"] = str(trace_file)
        os.environ["LTX_GUIDANCE_TRACE_RUN_ID"] = f"{args.label}:{case['id']}:r{repeat_idx}"
        os.environ["LTX_GUIDANCE_TRACE_ATTENTION_BACKEND"] = args.attention_type
        os.environ["LTX_GUIDANCE_TRACE_CUDA_GRAPHS"] = "1" if args.cuda_graphs else "0"
    else:
        for key in (
            "LTX_GUIDANCE_TRACE_FILE",
            "LTX_GUIDANCE_TRACE_RUN_ID",
            "LTX_GUIDANCE_TRACE_ATTENTION_BACKEND",
            "LTX_GUIDANCE_TRACE_CUDA_GRAPHS",
            "LTX_GUIDANCE_TRACE_STAGE",
        ):
            os.environ.pop(key, None)

    generator = torch.Generator(device=pipeline.device).manual_seed(case["seed"])
    noiser = GaussianNoiser(generator=generator)
    dtype = torch.bfloat16
    t2v = bool(settings.get("t2v"))  # v8.17: text-to-video => no conditioning image
    image = None if t2v else ImageConditioningInput(
        path=str(args.inputs_dir / case["file"]),
        frame_idx=0,
        strength=settings["conditioning_strength"],
        crf=settings["conditioning_crf"],
    )
    _imgs = [] if t2v else [image]  # empty => combined_image_conditionings returns [] => pure t2v
    tiling_config = None if args.no_tiling else TilingConfig.default()
    chunks = get_video_chunks_number(settings["frames"], tiling_config or TilingConfig.default())

    def encode_prompt() -> tuple[Any, Any]:
        negative_prompt = case.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        return pipeline.prompt_encoder(
            [case["prompt"], negative_prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=(None if t2v else image.path),
            enhance_prompt_seed=case["seed"],
        )

    prompt_event_start = len(getattr(pipeline.prompt_encoder, "events", []))
    with timed(timers, "prompt_encoder_gemma"):
        if prompt_cache is None:
            ctx_p, ctx_n = encode_prompt()
        else:
            negative_prompt = case.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
            prompt_key = prompt_cache.key(
                prompt=case["prompt"],
                negative_prompt=negative_prompt,
                image_path=(Path(image.path) if image else None),
                seed=case["seed"],
                gemma_root=args.gemma_root,
                gemma_8bit=args.gemma_8bit,
                gemma_prequant_root=args.gemma_prequant_root,
            )
            (ctx_p, ctx_n), cache_hit = prompt_cache.get_or_encode(prompt_key, encode_prompt)
            record["prompt_cache_hit"] = cache_hit
            record["prompt_cache_key"] = prompt_key[:16]
    record["prompt_encoder_events"] = list(getattr(pipeline.prompt_encoder, "events", []))[prompt_event_start:]
    record["vram_after"]["prompt_encoder_gemma"] = peak_vram_gib()
    v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
    v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

    stage_1_output_shape = VideoPixelShape(
        batch=1,
        frames=settings["frames"],
        width=settings["width"] // 2,
        height=settings["height"] // 2,
        fps=settings["fps"],
    )

    # v8.20 meme mode: IC-LoRA reference conditioning (pose/depth/canny control video) rides
    # stage 1 ONLY — stock ICLoraPipeline does the same (stage 2 gets image conditionings only).
    # settings["video_conditioning"] = [(path, strength)]; the reference is VAE-encoded at
    # stage1//downscale resolution by the util itself, using the SAME resident encoder.
    video_conditioning = settings.get("video_conditioning") or []
    reference_downscale = int(settings.get("reference_downscale", 1))

    def _stage_1_conditionings(enc):
        conds = combined_image_conditionings(
            images=_imgs,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
        if video_conditioning:
            from ltx_pipelines.iclora_utils import append_ic_lora_reference_video_conditionings

            append_ic_lora_reference_video_conditionings(
                conds,
                [(str(p), float(s)) for p, s in video_conditioning],
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                num_frames=settings["frames"],
                video_encoder=enc,
                dtype=dtype,
                device=pipeline.device,
                reference_downscale_factor=reference_downscale,
                conditioning_attention_strength=float(settings.get("reference_attention", 1.0)),
                conditioning_attention_mask=None,
                tiling_config=None,
            )
        return conds

    with timed(timers, "stage1_image_conditioner"):
        stage_1_conditionings = pipeline.image_conditioner(_stage_1_conditionings)
    record["vram_after"]["stage1_image_conditioner"] = peak_vram_gib()

    sigmas = LTX2Scheduler().execute(steps=settings["dev_inference_steps"]).to(
        dtype=torch.float32,
        device=pipeline.device,
    )
    video_guider_factory, video_guider_info = make_guidance_factory(
        params=LTX_2_3_PARAMS.video_guider_params,
        negative_context=v_context_n,
        sigmas=sigmas,
        args=args,
        prefix="video",
    )
    audio_guider_factory, audio_guider_info = make_guidance_factory(
        params=LTX_2_3_PARAMS.audio_guider_params,
        negative_context=a_context_n,
        sigmas=sigmas,
        args=args,
        prefix="audio",
    )
    record["guidance_factory"] = {
        "video": video_guider_info,
        "audio": audio_guider_info,
    }
    def execute_stage1() -> tuple[Any, Any]:
        if trace_file:
            os.environ["LTX_GUIDANCE_TRACE_STAGE"] = "stage1"
        return run_stage_with_optional_cuda_graphs(
            pipeline.stage_1,
            args,
            graph_stats,
            "stage1",
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=settings["frames"],
            fps=settings["fps"],
            video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
            audio=None if os.environ.get("LTX_NO_AUDIO") == "1" else ModalitySpec(context=a_context_p),
            loop=(
                gradient_estimating_euler_denoising_loop
                if bool(settings.get("stage1_ge", False))
                else euler_denoising_loop
            ),
            max_batch_size=1,
            lifecycle_stats=record["model_lifecycle"].setdefault("stage1", {})
            if args.profile_model_lifecycle
            else None,
            resident_stage_cache=resident_stage_cache,
        )

    with timed(timers, "stage1_transformer"):
        if args.profile_stage1:
            profile_dir = args.profile_dir or args.outputs_dir / "profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)
            trace_path = profile_dir / f"{args.label}_{case['id']}_r{repeat_idx}_stage1_trace.json"
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as profiler:
                video_state, audio_state = execute_stage1()
            profiler.export_chrome_trace(str(trace_path))
            record["stage1_profiler_trace"] = str(trace_path)
            record["stage1_profiler_cuda_top"] = profiler.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=40,
            )
        else:
            video_state, audio_state = execute_stage1()
    record["vram_after"]["stage1_transformer"] = peak_vram_gib()

    if args.stop_after_stage1:
        if args.decode_stage1_output:
            decoded_video = pipeline.video_decoder(video_state.latent, tiling_config, generator)
            stage1_frame_dir = (
                args.stage1_png_dir
                or args.outputs_dir / f"{args.label}_{case['id']}_r{repeat_idx}_stage1_frames"
            )
            with timed(timers, "stage1_vae_decode_png"):
                frame_count = save_video_frames_png(decoded_video, stage1_frame_dir)
            record["stage1_png_dir"] = str(stage1_frame_dir)
            record["stage1_png_frames"] = frame_count
            record["vram_after"]["stage1_vae_decode_png"] = peak_vram_gib()
            if args.stage1_mp4:
                decoded_video_for_mp4 = pipeline.video_decoder(video_state.latent, tiling_config, generator)
                output_path = args.outputs_dir / f"{args.label}_{case['id']}_r{repeat_idx}_stage1.mp4"
                with timed(timers, "stage1_video_encode_save"):
                    encode_video(
                        video=decoded_video_for_mp4,
                        fps=settings["fps"],
                        audio=None,
                        output_path=str(output_path),
                        video_chunks_number=chunks,
                        crf=args.encode_crf,
                        preset=args.encode_preset,
                        thread_count=args.encode_thread_count,
                    )
                record["output"] = output_path.name
                record["output_sha256"] = sha256_file(output_path)
            else:
                record["output"] = "stage1_png"
        else:
            record["output"] = "stage1_only"
        record["elapsed_seconds"] = round(sum(timers.values()), 3)
        record["wall_seconds"] = record["elapsed_seconds"]
        record["peak_vram_gib"] = peak_vram_gib()
        return record

    with timed(timers, "spatial_upsampler"):
        upscaled_video_latent = pipeline.upsampler(video_state.latent[:1])
    record["vram_after"]["spatial_upsampler"] = peak_vram_gib()

    with timed(timers, "stage2_image_conditioner"):
        stage_2_conditionings = pipeline.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=_imgs,
                height=settings["height"],
                width=settings["width"],
                video_encoder=enc,
                dtype=dtype,
                device=pipeline.device,
            )
        )
    record["vram_after"]["stage2_image_conditioner"] = peak_vram_gib()

    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
    with timed(timers, "stage2_transformer"):
        if trace_file:
            os.environ["LTX_GUIDANCE_TRACE_STAGE"] = "stage2"
        video_state, audio_state = run_stage_with_optional_cuda_graphs(
            pipeline.stage_2,
            args,
            graph_stats,
            "stage2",
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=settings["width"],
            height=settings["height"],
            frames=settings["frames"],
            fps=settings["fps"],
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=None if (os.environ.get("LTX_NO_AUDIO") == "1" or audio_state is None) else ModalitySpec(
                context=a_context_p,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            loop=euler_denoising_loop,
            lifecycle_stats=record["model_lifecycle"].setdefault("stage2", {})
            if args.profile_model_lifecycle
            else None,
            resident_stage_cache=resident_stage_cache,
        )
    record["vram_after"]["stage2_transformer"] = peak_vram_gib()

    with timed(timers, "video_vae_decode"):
        decoded_video = pipeline.video_decoder(
            video_state.latent,
            tiling_config,
            generator,
            decode_noise_scale=settings.get("decode_noise"),
        )
    record["effective_decode_noise_scale"] = pipeline.video_decoder.last_decode_noise_scale
    record["effective_cas"] = {
        "amount": float(settings.get("cas_amount", _CAS_AMOUNT_DEFAULT)),
        "mix": float(settings.get("cas_mix", _CAS_MIX_DEFAULT)),
    }
    decoded_video = _maybe_cas(decoded_video, settings)  # default-on crispness pass, before encode
    record["vram_after"]["video_vae_decode"] = peak_vram_gib()

    if audio_state is not None:
        with timed(timers, "audio_decode"):
            decoded_audio = pipeline.audio_decoder(audio_state.latent)
        record["vram_after"]["audio_decode"] = peak_vram_gib()
    else:
        decoded_audio = None

    output_path = args.outputs_dir / f"{args.label}_{case['id']}_r{repeat_idx}.mp4"
    with timed(timers, "video_encode_save"):
        encode_video(
            video=decoded_video,
            fps=settings["fps"],
            audio=decoded_audio,
            output_path=str(output_path),
            video_chunks_number=chunks,
            crf=args.encode_crf,
            preset=args.encode_preset,
            thread_count=args.encode_thread_count,
        )
    record["output"] = output_path.name
    record["output_sha256"] = sha256_file(output_path)
    record["elapsed_seconds"] = round(sum(timers.values()), 3)
    record["peak_vram_gib"] = peak_vram_gib()
    os.environ.pop("LTX_GUIDANCE_TRACE_STAGE", None)
    return record


def main() -> int:
    args = parse_args()
    apply_runtime_env(args)
    logging.basicConfig(level=logging.INFO)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    timings_path = args.outputs_dir / f"{args.label}_stage_timings.json"

    manifest = json.loads(args.manifest.read_text())
    settings = overridden_settings(args, manifest)
    requested = set(args.only or [])
    cases = [case for case in manifest["inputs"] if not requested or case["id"] in requested]

    results = {
        "label": args.label,
        "mode": "dev",
        "quantization": "fp8-scaled-mm",
        "attention_type": args.attention_type,
        "fast_vae": args.fast_vae,
        "tgate_start_step": args.tgate_start_step,
        "tiling": not args.no_tiling,
        "gemma_cache": args.gemma_cache,
        "gemma_8bit": args.gemma_8bit,
        "gemma_prequant_root": args.gemma_prequant_root,
        "resident_stages": args.resident_stages,
        "resident_gemma": args.resident_gemma,
        "cuda_graphs": args.cuda_graphs,
        "cuda_graph_warmup_calls": args.cuda_graph_warmup_calls,
        "cuda_graph_clone_outputs": args.cuda_graph_clone_outputs,
        "cuda_graph_profile_timings": args.cuda_graph_profile_timings,
        "torch_compile_transformer": args.torch_compile_transformer,
        "torch_compile_mode": args.torch_compile_mode,
        "torch_compile_dynamic": args.torch_compile_dynamic,
        "stop_after_stage1": args.stop_after_stage1,
        "profile_stage1": args.profile_stage1,
        "profile_prompt_encoder": args.profile_prompt_encoder,
        "profile_model_lifecycle": args.profile_model_lifecycle,
        "guidance_trace_dir": str(args.guidance_trace_dir) if args.guidance_trace_dir else None,
        "guidance_overrides": {
            "video_cfg_scale": args.video_cfg_scale,
            "video_stg_scale": args.video_stg_scale,
            "video_modality_scale": args.video_modality_scale,
            "video_rescale_scale": args.video_rescale_scale,
            "audio_cfg_scale": args.audio_cfg_scale,
            "audio_stg_scale": args.audio_stg_scale,
            "audio_modality_scale": args.audio_modality_scale,
            "audio_rescale_scale": args.audio_rescale_scale,
        },
        "guidance_windows": guidance_window_summary(args),
        "encode": {
            "crf": args.encode_crf,
            "preset": args.encode_preset,
            "thread_count": args.encode_thread_count,
        },
        "settings": settings,
        "pipeline_load_seconds": None,
        "post_pipeline_load_peak_vram_gib": None,
        "prompt_encoder_events": [],
        "prompt_cache_stats": None,
        "resident_close_events": {},
        "runs": [],
    }
    timings_path.write_text(json.dumps(results, indent=2))

    quantization = QuantizationPolicy.fp8_scaled_mm(args.checkpoint)
    load_start = time.perf_counter()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint,
        distilled_lora=[
            LoraPathStrengthAndSDOps(args.distilled_lora, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)
        ],
        spatial_upsampler_path=args.spatial_upscaler,
        gemma_root=args.gemma_root,
        loras=[],
        quantization=quantization,
        torch_compile=False,
    )
    if args.gemma_prequant_root:
        pipeline.prompt_encoder = PromptEncoder(
            args.checkpoint,
            args.gemma_root,
            pipeline.dtype,
            pipeline.device,
            text_encoder_builder=PrequantCausalGemmaBuilder(
                model_root=args.gemma_prequant_root,
                tokenizer_root=args.gemma_root,
            ),
        )
    elif args.gemma_8bit:
        pipeline.prompt_encoder = PromptEncoder(
            args.checkpoint,
            args.gemma_root,
            pipeline.dtype,
            pipeline.device,
            text_encoder_builder=BitsAndBytesGemmaBuilder(args.gemma_root),
        )
    if args.profile_prompt_encoder or args.profile_stage1 or args.stop_after_stage1 or args.resident_gemma:
        pipeline.prompt_encoder = TimedPromptEncoder(
            pipeline.prompt_encoder,
            keep_resident_text_encoder=args.resident_gemma,
        )
    sync_cuda()
    results["pipeline_load_seconds"] = round(time.perf_counter() - load_start, 3)
    results["post_pipeline_load_peak_vram_gib"] = peak_vram_gib()
    timings_path.write_text(json.dumps(results, indent=2))
    prompt_cache = PromptEmbeddingCache() if args.gemma_cache else None
    resident_stage_cache = ResidentStageCache() if args.resident_stages else None

    try:
        for repeat_idx in range(1, args.repeat + 1):
            for case in cases:
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    sync_cuda()
                    start = time.perf_counter()
                    record = run_case(
                        pipeline,
                        case,
                        settings,
                        args,
                        repeat_idx=repeat_idx,
                        prompt_cache=prompt_cache,
                        resident_stage_cache=resident_stage_cache,
                    )
                    sync_cuda()
                    record["wall_seconds"] = round(time.perf_counter() - start, 3)
                    logging.info("Completed %s repeat %s in %ss", case["id"], repeat_idx, record["wall_seconds"])
                except Exception as exc:  # noqa: BLE001 - stage timing failures are data.
                    record = {
                        "id": case["id"],
                        "repeat": repeat_idx,
                        "seed": case["seed"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                    results["runs"].append(record)
                    timings_path.write_text(json.dumps(results, indent=2))
                    raise
                results["runs"].append(record)
                prompt_events = getattr(pipeline.prompt_encoder, "events", None)
                if prompt_events is not None:
                    results["prompt_encoder_events"] = prompt_events
                if prompt_cache is not None:
                    results["prompt_cache_stats"] = prompt_cache.stats()
                timings_path.write_text(json.dumps(results, indent=2))
    finally:
        close_events: dict[str, Any] = {}
        if resident_stage_cache is not None:
            close_events["stages"] = resident_stage_cache.close()
        close_resident = getattr(pipeline.prompt_encoder, "close_resident", None)
        if callable(close_resident):
            close_events["gemma"] = close_resident()
        if close_events:
            results["resident_close_events"] = close_events
            timings_path.write_text(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
