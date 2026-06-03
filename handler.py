"""LTX-2.3 RunPod serverless handler.
Loads the pipeline ONCE at worker init (residency) + runs a warmup forward
(pre-JIT) so the first real request is warm. Then serves {image_b64, prompt, ...}.
"""
import base64
import os
import tempfile
import traceback

import torch  # noqa: F401  (ensure CUDA init early)
import runpod

from ltx_core.quantization import QuantizationPolicy
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS
from ltx_pipelines.utils.media_io import encode_video

CKPT = os.environ["LTX_FP8_CKPT"]
GEMMA = os.environ["LTX_GEMMA_ROOT"]
UPS = os.environ["LTX_UPSCALER"]
LORA = os.environ["LTX_DISTILLED_LORA"]
os.environ.setdefault("LTX_TGATE_START_STEP", "10")  # conservative default

# ---- load ONCE (residency): models stay resident in VRAM for the worker lifetime ----
print("[init] building pipeline...", flush=True)
PIPE = TI2VidTwoStagesPipeline(
    checkpoint_path=CKPT,
    distilled_lora=[LoraPathStrengthAndSDOps(LORA, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)],
    spatial_upsampler_path=UPS,
    gemma_root=GEMMA,
    loras=[],
    quantization=QuantizationPolicy.fp8_scaled_mm(CKPT),
    torch_compile=False,
)
print("[init] pipeline built", flush=True)


def _generate(image_path, prompt, negative_prompt, seed, width, height, frames, fps, steps):
    images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=0.8, crf=0)]
    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(frames, tiling)
    video, audio = PIPE(
        prompt=prompt,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        seed=seed,
        height=height,
        width=width,
        num_frames=frames,
        frame_rate=fps,
        num_inference_steps=steps,
        video_guider_params=LTX_2_3_PARAMS.video_guider_params,
        audio_guider_params=LTX_2_3_PARAMS.audio_guider_params,
        images=images,
        tiling_config=tiling,
    )
    out = tempfile.mktemp(suffix=".mp4")
    encode_video(
        video=video, fps=fps, audio=audio, output_path=out,
        video_chunks_number=chunks, crf=19, preset="veryfast", thread_count=0,
    )
    return out


# ---- warmup forward (pre-JIT): first REAL request then runs warm ----
_WARMUP_IMG = os.environ.get("LTX_WARMUP_IMG")
if _WARMUP_IMG and os.path.exists(_WARMUP_IMG):
    try:
        print("[init] warmup forward (pre-JIT)...", flush=True)
        _w = _generate(_WARMUP_IMG, "cinematic warmup", DEFAULT_NEGATIVE_PROMPT, 1, 1920, 1088, 121, 24.0, 20)
        os.remove(_w)
        print("[init] warmup done — worker is warm", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[init] warmup skipped:", exc, flush=True)


def handler(job):
    inp = job.get("input", {})
    try:
        img_path = tempfile.mktemp(suffix=".jpg")
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(inp["image_b64"]))
        out = _generate(
            img_path,
            inp["prompt"],
            inp.get("negative_prompt", ""),
            int(inp.get("seed", 0)),
            int(inp.get("width", 1920)),
            int(inp.get("height", 1088)),
            int(inp.get("frames", 121)),
            float(inp.get("fps", 24.0)),
            int(inp.get("steps", 20)),
        )
        with open(out, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return {"video_b64": data}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
