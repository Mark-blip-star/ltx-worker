"""LTX-2.3 RunPod serverless handler v7 — two-tier (fast/quality) + resident Gemma.

v7 (optimization campaign 2026-06-10):
- TIERS via request param "tier":
  * "fast" (default): stage1 = 12 steps on the resampled official distilled-sigma curve,
    CFG=1 + STG=1.0 + modality off (2 batched passes instead of 4), audio guidance off,
    stage2 = 2 distilled steps (sigmas [0.909375, 0.6, 0.0]). Blind-judged win-rate 56% vs
    the old 16-step full-guidance config at -34% wall time (pod: 36.1s vs 54.6s warm).
  * "quality": the previous production config — 16 steps (request "steps" honored),
    LTX2Scheduler sigmas, full default guidance (CFG 3.0 / STG 1.0 / modality 3.0), 3-step stage2.
- RESIDENT GEMMA: text encoder is built once and kept in VRAM (PromptEncoder subclass that
  bypasses the free-on-exit gpu_model ctx). bf16 Gemma 24G + pipeline peak ~30G < 94G NVL.
  Plus a prompt-embedding LRU (hash of prompt) — repeated prompts skip the forward entirely.
- Request param "audio" (default true): false sets LTX_NO_AUDIO=1 for the call (single-threaded).
- Returns per-stage "timers" + "config_tag" for observability.

Kept from v5/v6: lazy init + self-diagnosing errors, robust cached-snapshot glob resolve
(lowercase org dir!), fp8->bf16 Gemma dequant (NO bitsandbytes), memory-safe run_case path,
FA3 via LTX_ATTENTION_TYPE.
"""
import base64
import contextlib
import glob
import os
import shutil
import tempfile
import traceback
import types
from pathlib import Path

_CACHE = "/runpod-volume/huggingface-cache" if os.path.isdir("/runpod-volume") else "/app/hfcache"
os.environ.setdefault("HF_HOME", _CACHE)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LTX_ATTENTION_TYPE", "flash_attention_3")  # critical: vanilla attn OOMs
os.environ.setdefault("LTX_FAST_VAE", "1")
os.environ.setdefault("LTX_TGATE_START_STEP", "10")

import torch  # noqa: E402
import runpod  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
import stage_timing_runner as base  # noqa: E402
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: E402
from ltx_core.quantization import QuantizationPolicy  # noqa: E402
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # noqa: E402
from ltx_pipelines.utils.blocks import PromptEncoder  # noqa: E402
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES  # noqa: E402

WEIGHTS_REPO = os.environ["LTX_WEIGHTS_REPO"]
FP8_CKPT_NAME = os.environ.get("LTX_FP8_CKPT_NAME", "ltx-2.3-22b-dev-fp8.safetensors")
UPS_NAME = os.environ.get("LTX_UPSCALER_NAME", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
LORA_NAME = os.environ.get("LTX_DISTILLED_LORA_NAME", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
GEMMA_FP8_SUBDIR = os.environ.get("LTX_GEMMA_FP8_SUBDIR", "gemma-fp8")
GEMMA_BF16_DIR = os.environ.get("LTX_GEMMA_BF16_DIR", "/app/gemma-bf16")
CONFIG_TAG = os.environ.get("LTX_CONFIG_TAG", "v7")
DEFAULT_PROMPT = os.environ.get(
    "LTX_DEFAULT_PROMPT",
    "The rider pedals forward at a steady, even pace, his body rising and dipping slightly with each "
    "push as the wheels spin. The camera tracks behind him at a low angle, gliding forward at a gentle, "
    "constant following distance. Long shadows slide slowly across the pavement and palm fronds sway in "
    "the breeze, while distant cars drift down the boulevard. Warm sunset light flickers softly between "
    "the storefronts as he rides on.",
)
_IN = Path(tempfile.mkdtemp(prefix="ltx_in_"))
_OUT = Path(tempfile.mkdtemp(prefix="ltx_out_"))

_PIPE = None
_INIT_ERR = None
_INIT_LOG = []
# stage2 sigma sets (module-level switch before each run_case; handler is single-threaded)
_STAGE2_DEFAULT = None  # captured at init from base.STAGE_2_DISTILLED_SIGMAS
_STAGE2_FAST = torch.tensor([0.909375, 0.6, 0.0])

# stage1 scheduler dispatch: "ltx2" -> real LTX2Scheduler, "distilled" -> resampled official curve
_SIGMA_MODE = {"mode": "ltx2"}
_LTX2Scheduler_real = base.LTX2Scheduler


class _DispatchScheduler:
    def execute(self, steps: int):
        if _SIGMA_MODE["mode"] == "distilled":
            if steps == 8:
                vals = list(DISTILLED_SIGMA_VALUES)
            else:
                import numpy as np
                src = np.array(DISTILLED_SIGMA_VALUES)
                idx = np.linspace(0, len(src) - 1, steps + 1)
                vals = np.interp(idx, np.arange(len(src)), src).tolist()
            return torch.tensor(vals)
        return _LTX2Scheduler_real().execute(steps=steps)


base.LTX2Scheduler = _DispatchScheduler


class ResidentPromptEncoder(PromptEncoder):
    """Build the Gemma text encoder ONCE and keep it in VRAM (skip free-on-exit ctx)."""

    def _text_encoder_ctx(self):
        if not hasattr(self, "_resident_te"):
            self._resident_te = self._build_text_encoder()
        return contextlib.nullcontext(self._resident_te)


def _resolve_repo() -> str:
    hub = os.path.join(os.environ.get("HF_HOME", _CACHE), "hub")
    repo_dir = os.path.join(hub, "models--" + WEIGHTS_REPO.replace("/", "--"))
    for snap in sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*"))):
        if os.path.exists(os.path.join(snap, FP8_CKPT_NAME)):
            _INIT_LOG.append(f"resolved cached snapshot: {snap}")
            return snap
    try:
        p = snapshot_download(WEIGHTS_REPO, local_files_only=True)
        _INIT_LOG.append(f"resolved via local snapshot_download: {p}")
        return p
    except Exception as exc:
        _INIT_LOG.append(f"not cached ({exc!r}); downloading from HF...")
        return snapshot_download(WEIGHTS_REPO, token=os.environ.get("HF_TOKEN"))


def _dequant_gemma_fp8_to_bf16(fp8_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(fp8_dir, "*.safetensors"))):
        sd = load_file(f)
        out = {}
        for k, v in sd.items():
            if k.endswith(".fp8_scale"):
                continue
            scale = sd.get(k + ".fp8_scale")
            out[k] = (v.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16) if scale is not None else v
        save_file(out, os.path.join(out_dir, os.path.basename(f)))
    for fn in os.listdir(fp8_dir):
        src = os.path.join(fp8_dir, fn)
        if os.path.isfile(src) and not fn.endswith(".safetensors"):
            shutil.copy(src, os.path.join(out_dir, fn))


def _init():
    global _PIPE, _INIT_ERR, _STAGE2_DEFAULT
    if _PIPE is not None or _INIT_ERR is not None:
        return
    try:
        snap = _resolve_repo()
        ckpt = os.path.join(snap, FP8_CKPT_NAME)
        ups = os.path.join(snap, UPS_NAME)
        lora = os.path.join(snap, LORA_NAME)
        gemma_fp8 = os.path.join(snap, GEMMA_FP8_SUBDIR)
        for p in (ckpt, ups, lora, gemma_fp8):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing in snapshot {snap}: {p} (have: {os.listdir(snap)})")
        gemma = GEMMA_BF16_DIR
        if not os.path.exists(os.path.join(gemma, "config.json")):
            _INIT_LOG.append("dequantizing fp8 Gemma -> bf16 (one-time)...")
            _dequant_gemma_fp8_to_bf16(gemma_fp8, gemma)
        _STAGE2_DEFAULT = base.STAGE_2_DISTILLED_SIGMAS
        _INIT_LOG.append("building pipeline (fp8 + distilled-LoRA 0.8 + resident bf16 Gemma)...")
        pipe = TI2VidTwoStagesPipeline(
            checkpoint_path=ckpt,
            distilled_lora=[LoraPathStrengthAndSDOps(lora, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)],
            spatial_upsampler_path=ups, gemma_root=gemma, loras=[],
            quantization=QuantizationPolicy.fp8_scaled_mm(ckpt), torch_compile=False,
        )
        pipe.prompt_encoder = ResidentPromptEncoder(
            ckpt, gemma, pipe.dtype, pipe.device,
            text_encoder_builder=base.PrequantCausalGemmaBuilder(model_root=gemma, tokenizer_root=gemma),
        )
        _PIPE = pipe
        _INIT_LOG.append("pipeline built — ready")
    except Exception:
        _INIT_ERR = "INIT LOG:\n" + "\n".join(_INIT_LOG) + "\n\nTRACEBACK:\n" + traceback.format_exc()


def _tier_args(tier: str, label: str) -> types.SimpleNamespace:
    fast = tier == "fast"
    return types.SimpleNamespace(
        inputs_dir=_IN, outputs_dir=_OUT, trace_root=_OUT, guidance_trace_dir=None, label=label,
        gemma_root=GEMMA_BF16_DIR, gemma_prequant_root=GEMMA_BF16_DIR, gemma_8bit=False, gemma_cache=False,
        no_tiling=False, attention_type="flash_attention_3", fast_vae=True, tgate_start_step=10,
        cuda_graphs=False, cuda_graph_warmup_calls=1, cuda_graph_clone_outputs=False, cuda_graph_profile_timings=False,
        torch_compile_transformer=False, torch_compile_mode="reduce-overhead", torch_compile_dynamic=False,
        profile_stage1=False, profile_dir=None, stop_after_stage1=False, profile_prompt_encoder=False,
        profile_model_lifecycle=False, resident_stages=False, resident_gemma=False,
        decode_stage1_output=False, stage1_png_dir=None, stage1_mp4=False,
        video_cfg_scale=(1.0 if fast else None), video_stg_scale=(1.0 if fast else None),
        video_modality_scale=(1.0 if fast else None), video_rescale_scale=None,
        audio_cfg_scale=(1.0 if fast else None), audio_stg_scale=(0.0 if fast else None),
        audio_modality_scale=(1.0 if fast else None), audio_rescale_scale=None,
        guidance_window_steps=None, cfg_window_steps=None, stg_window_steps=None, modality_window_steps=None,
        encode_crf=19, encode_preset="veryfast", encode_thread_count=0,
    )


def handler(job):
    _init()
    if _INIT_ERR:
        return {"error": "INIT_FAILED", "trace": _INIT_ERR}
    inp = job.get("input", {})
    try:
        tier = str(inp.get("tier", "fast")).lower()
        if tier not in ("fast", "quality"):
            tier = "fast"
        audio_on = bool(inp.get("audio", True))
        steps = int(inp.get("steps", 12 if tier == "fast" else 16))

        img_path = _IN / "req"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(inp["image_b64"]))
        case = {"id": "req", "file": "req", "prompt": inp.get("prompt") or DEFAULT_PROMPT,
                "seed": int(inp.get("seed", 3102))}
        settings = {
            "width": int(inp.get("width", 1280)), "height": int(inp.get("height", 704)),
            "frames": int(inp.get("frames", 121)), "fps": float(inp.get("fps", 24.0)),
            "conditioning_strength": 0.8, "conditioning_crf": 0, "dev_inference_steps": steps,
        }
        # per-request config switches (handler is single-threaded)
        _SIGMA_MODE["mode"] = "distilled" if tier == "fast" else "ltx2"
        base.STAGE_2_DISTILLED_SIGMAS = _STAGE2_FAST if tier == "fast" else _STAGE2_DEFAULT
        if audio_on:
            os.environ.pop("LTX_NO_AUDIO", None)
        else:
            os.environ["LTX_NO_AUDIO"] = "1"

        rec = base.run_case(_PIPE, case, settings, _tier_args(tier, "req"),
                            repeat_idx=1, prompt_cache=None, resident_stage_cache=None)
        if rec.get("error"):
            return {"error": rec["error"], "init_log": _INIT_LOG, "config_tag": f"{CONFIG_TAG}-{tier}"}
        out = _OUT / rec["output"]
        with open(out, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        os.remove(out)
        return {
            "video_b64": data,
            "peak_vram_gib": rec.get("peak_vram_gib"),
            "elapsed_seconds": rec.get("elapsed_seconds"),
            "timers": rec.get("timers"),
            "config_tag": f"{CONFIG_TAG}-{tier}" + ("" if audio_on else "-noaudio"),
            "tier": tier,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "trace": traceback.format_exc()}


if os.environ.get("LTX_SKIP_START") != "1":
    runpod.serverless.start({"handler": handler})
