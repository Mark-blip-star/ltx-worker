"""Thin runpod wrapper around the VANILLA LTX-2.5 distilled CLI. No pipeline surgery:
every generation is `python -m ltx_pipelines.distilled` exactly as Lightricks ships it.

Request: {prompt, seed, width, height, frames?, fps?, image_b64?, enhance?(default true),
          extra_args?: [..]}  ->  {video_b64, seconds, cli_tail, build_commit}
"""
import base64
import json
import os
import subprocess
import time
from pathlib import Path

import runpod

REPO = "Lightricks/LTX-2.5"
MODELS = Path(os.environ.get("LTX25_MODELS_DIR", "/models/ltx-2.5"))
COMPONENTS = {
    "transformer-path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text-encoder-path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video-vae-path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio-vae-path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial-upsampler-path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}
ENHANCER_REPO = "google/gemma-3-12b-it"
ENHANCER_DIR = Path("/models/enhancer")
_READY = False


def _ensure_weights() -> float:
    global _READY
    t0 = time.time()
    if not _READY:
        missing = [rel for rel in COMPONENTS.values() if not (MODELS / rel).exists()]
        if missing:
            cmd = ["hf", "download", REPO, *missing, "--local-dir", str(MODELS)]
            print(f"[cleanroom] downloading {len(missing)} files...", flush=True)
            subprocess.run(cmd, check=True, timeout=3600)
        if not (ENHANCER_DIR / "config.json").exists():
            print("[cleanroom] downloading prompt-enhancer gemma...", flush=True)
            subprocess.run(["hf", "download", ENHANCER_REPO, "--local-dir", str(ENHANCER_DIR)],
                           check=True, timeout=3600)
        _READY = True
    return round(time.time() - t0, 1)


def handler(job):
    inp = job.get("input") or {}
    try:
        dl_s = _ensure_weights()
        out = Path("/tmp/out.mp4")
        out.unlink(missing_ok=True)
        cmd = ["python", "-m", "ltx_pipelines.distilled"]
        for flag, rel in COMPONENTS.items():
            cmd += [f"--{flag}", str(MODELS / rel)]
        cmd += ["--prompt", str(inp.get("prompt", "")),
                "--seed", str(int(inp.get("seed", 42))),
                "--width", str(int(inp.get("width", 1280))),
                "--height", str(int(inp.get("height", 704))),
                "--frame-rate", str(int(inp.get("fps", 24))),
                "--output-path", str(out)]
        if inp.get("frames"):
            cmd += ["--num-frames", str(int(inp["frames"]))]
        if inp.get("image_b64"):
            img = Path("/tmp/in.png")
            img.write_bytes(base64.b64decode(inp["image_b64"]))
            cmd += ["--image", str(img), "0", "1.0"]
        if bool(inp.get("enhance", True)):
            cmd += ["--enhance-prompt", "--prompt-enhancer-gemma-root", str(ENHANCER_DIR)]
        cmd += [str(a) for a in inp.get("extra_args", [])]

        t0 = time.time()
        run = subprocess.run(cmd, cwd="/app/LTX-2", capture_output=True, text=True, timeout=1800)
        tail = (run.stdout + "\n" + run.stderr)[-2500:]
        if run.returncode != 0 or not out.exists():
            return {"error": f"cli exit {run.returncode}", "cli_tail": tail}
        return {
            "video_b64": base64.b64encode(out.read_bytes()).decode(),
            "seconds": round(time.time() - t0, 1),
            "download_seconds": dl_s,
            "cli_tail": tail,
            "build_commit": Path("/BUILD_COMMIT").read_text().strip()[:12] if Path("/BUILD_COMMIT").exists() else None,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc()[-2000:]}


runpod.serverless.start({"handler": handler})
