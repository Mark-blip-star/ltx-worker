"""Resident LTX-2.5 distilled worker (phase 2 after Mark's verdict "2.5 wins big").

Same vanilla stack as handler_cleanroom.py, one change: DistilledPipeline is constructed
ONCE and kept in VRAM; per-job params still go through the upstream CLI parser so every
type conversion (image conditioning triplets, hdr, quantization) is exactly theirs.
"""
import base64
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

import runpod

REPO = "Lightricks/LTX-2.5"
MODELS = Path(os.environ.get("LTX25_MODELS_DIR", "/models/ltx-2.5"))
ENHANCER_REPO = "google/gemma-3-12b-it"
ENHANCER_DIR = Path("/models/enhancer")
COMPONENTS = {
    "transformer-path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text-encoder-path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video-vae-path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio-vae-path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial-upsampler-path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

_LOCK = threading.Lock()
_PIPE = None
_PARSER = None
_INIT_S = None


def _mark(msg):
    print(f"[resident] {msg}", flush=True)


def _ensure_weights():
    missing = [rel for rel in COMPONENTS.values() if not (MODELS / rel).exists()]
    if missing:
        _mark(f"downloading {len(missing)} component files...")
        subprocess.run(["hf", "download", REPO, *missing, "--local-dir", str(MODELS)],
                       check=True, timeout=3600)
    if not (ENHANCER_DIR / "config.json").exists():
        _mark("downloading prompt-enhancer gemma...")
        subprocess.run(["hf", "download", ENHANCER_REPO, "--local-dir", str(ENHANCER_DIR)],
                       check=True, timeout=3600)


def _path_argv():
    argv = []
    for flag, rel in COMPONENTS.items():
        argv += [f"--{flag}", str(MODELS / rel)]
    argv += ["--prompt-enhancer-gemma-root", str(ENHANCER_DIR)]
    return argv


def _job_argv(inp, img_path, out_path):
    argv = _path_argv()
    argv += ["--prompt", str(inp.get("prompt", "")),
             "--seed", str(int(inp.get("seed", 42))),
             "--width", str(int(inp.get("width", 1280))),
             "--height", str(int(inp.get("height", 704))),
             "--frame-rate", str(int(inp.get("fps", 24))),
             "--num-frames", str(int(inp.get("frames", 121))),
             "--output-path", str(out_path)]
    if img_path is not None:
        argv += ["--image", str(img_path), "0", "1.0"]
    if bool(inp.get("enhance", True)):
        argv += ["--enhance-prompt"]
    argv += [str(a) for a in inp.get("extra_args", [])]
    return argv


def _init():
    global _PIPE, _PARSER, _INIT_S
    with _LOCK:
        if _PIPE is not None:
            return
        t0 = time.time()
        _ensure_weights()
        t_dl = time.time()
        from ltx_pipelines import distilled as D  # noqa: PLC0415 — after weights exist

        _PARSER = D.add_generated_keyframes_arg(
            D.default_2_stage_distilled_arg_parser(
                params=D.resolve_cli_params(distilled=True), supports_auto_duration=True))
        args = _PARSER.parse_args(_job_argv({}, None, "/tmp/warm.mp4"))
        _mark("building resident pipeline...")
        _PIPE = D.DistilledPipeline(
            model_paths=args.model_paths,
            spatial_upsampler_path=args.spatial_upsampler_path,
            loras=tuple(args.lora) if args.lora else (),
            quantization=args.quantization,
            compilation_config=args.compile,
            offload_mode=args.offload_mode,
            prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
            diffvae_optimization=args.diffvae_optimization,
        )
        _INIT_S = {"download_s": round(t_dl - t0, 1), "build_s": round(time.time() - t_dl, 1)}
        _mark(f"resident ready: {_INIT_S}")


def handler(job):
    import torch  # noqa: PLC0415

    inp = job.get("input") or {}
    try:
        _init()
        from ltx_pipelines import distilled as D  # noqa: PLC0415

        out = Path("/tmp/out.mp4")
        out.unlink(missing_ok=True)
        img_path = None
        if inp.get("image_b64"):
            img_path = Path("/tmp/in.png")
            img_path.write_bytes(base64.b64decode(inp["image_b64"]))
        args = _PARSER.parse_args(_job_argv(inp, img_path, out))

        t0 = time.time()
        with torch.inference_mode():
            hdr = D.resolve_hdr_color_space(images=args.images, hdr=args.hdr)
            vae_dtype = D.vae_dtype_for_hdr(hdr, torch.bfloat16)
            video, audio, num_frames, tiling_config = _PIPE(
                prompt=args.prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.frame_rate,
                images=args.images,
                vae_dtype=vae_dtype,
                color_space=hdr,
                enhance_prompt=args.enhance_prompt,
                enhance_static_cache=args.enhance_static_cache,
                tiling_config=D.AUTO_TILING,
                generated_keyframes=args.num_generated_keyframes,
            )
            D.encode_video(
                video=video, fps=args.frame_rate, audio=audio, output_path=out,
                video_chunks_number=D.get_video_chunks_number(num_frames, tiling_config),
                color_space=hdr,
            )
        gen_s = round(time.time() - t0, 1)
        return {
            "video_b64": base64.b64encode(out.read_bytes()).decode(),
            "gen_seconds": gen_s,
            "init": _INIT_S,
            "build_commit": Path("/BUILD_COMMIT").read_text().strip()[:12] if Path("/BUILD_COMMIT").exists() else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "trace": traceback.format_exc()[-2500:]}


runpod.serverless.start({"handler": handler})
