"""Persistent generation child for the LTX-2.5 resident worker (v9.2 supervisor pattern).

Reads one JSON command per stdin line, answers one JSON line on stdout. All heavy imports
and the vanilla DistilledPipeline live HERE, so a CUDA/native crash kills this process only —
the runpod SDK supervisor survives and reports our stderr.

Commands:
  {"cmd": "init"}                       -> {"ok": true, "init": {...}}
  {"cmd": "gen", "input": {...}, "out": "/tmp/out.mp4"} -> {"ok": true, "gen_s": 12.3}
"""
import faulthandler
import json
import os
import subprocess
import sys
import time
from pathlib import Path

faulthandler.enable(file=sys.stderr)  # native crashes (SIGSEGV/SIGABRT) dump py-stacks to stderr

REPO = os.environ.get("LTX25_WEIGHTS_REPO", "Markooooo/ltx25-prod")  # our mirror (pin-able); env-override for dark/RD
MODELS = Path(os.environ.get("LTX25_MODELS_DIR", "/models/ltx-2.5"))
ENHANCER_DIR = MODELS / "enhancer"  # mirrored into the same repo under enhancer/
COMPONENTS = {
    "transformer-path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text-encoder-path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video-vae-path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio-vae-path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial-upsampler-path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

_PIPE = None
_PARSER = None
_TORCH = None


def log(msg):
    print(f"[child] {msg}", file=sys.stderr, flush=True)


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def ensure_weights():
    missing = [rel for rel in COMPONENTS.values() if not (MODELS / rel).exists()]
    if missing:
        log(f"downloading {len(missing)} component files...")
        subprocess.run(["hf", "download", REPO, *missing, "--local-dir", str(MODELS)],
                       check=True, timeout=3600)
    if not (ENHANCER_DIR / "config.json").exists():
        log("downloading prompt-enhancer gemma (mirror)...")
        subprocess.run(["hf", "download", REPO, "--include", "enhancer/*", "--local-dir", str(MODELS)],
                       check=True, timeout=3600)


def job_argv(inp, img_path, out_path):
    argv = []
    for flag, rel in COMPONENTS.items():
        argv += [f"--{flag}", str(MODELS / rel)]
    argv += ["--prompt-enhancer-gemma-root", str(ENHANCER_DIR),
             "--prompt", str(inp.get("prompt", "")),
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


def _make_resident():
    """v9.4 module residency (quality-neutral): upstream blocks rebuild every model on every
    call and dispose it after (blocks.py: "Blocks build a model on each __call__"). We memoize
    the heavy `_build_*` methods per block instance and no-op `dispose` ONLY on those cached
    modules, so gpu_model()'s exit hook leaves them resident. Light per-call builds that we do
    not memoize (decoders) keep the vendor build/free cycle — no VRAM leaks. Bonus: a stable
    VRAM map gives natten's DiffVAE tile planner honest free-memory numbers.
    """
    import inspect  # noqa: PLC0415

    import ltx_pipelines.utils.blocks as B  # noqa: PLC0415

    names = {"_build_transformer", "_build_text_encoder", "_build_enhancer_text_encoder",
             "_build_embeddings_processor", "_build_encoder"}
    cache = {}

    def residentize(cls, name):
        orig = getattr(cls, name)

        def wrapper(self, *a, **kw):
            key = (id(self), name)
            if key not in cache:
                m = orig(self, *a, **kw)
                try:
                    m.dispose = lambda: None  # gpu_model() exit must not free a resident module
                except AttributeError:
                    pass
                cache[key] = m
                log(f"resident: built+cached {cls.__name__}.{name}")
            return cache[key]

        setattr(cls, name, wrapper)

    patched = []
    for cname, cls in vars(B).items():
        if inspect.isclass(cls):
            for n in names & set(vars(cls)):
                residentize(cls, n)
                patched.append(f"{cname}.{n}")
    log(f"residency patched: {patched}")


def do_init():
    global _PIPE, _PARSER, _TORCH
    t0 = time.time()
    ensure_weights()
    t_dl = time.time()
    log("importing torch + ltx stack...")
    import torch  # noqa: PLC0415
    from ltx_pipelines import distilled as D  # noqa: PLC0415
    _TORCH = torch
    _make_resident()
    try:
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name()
        log(f"GPU: {name} | compute capability sm_{cap[0]}{cap[1]}")
        try:
            import natten  # noqa: PLC0415
            log(f"natten {getattr(natten, '__version__', '?')} importable")
        except Exception as ne:  # noqa: BLE001
            log(f"natten import failed: {ne!r}")
    except Exception as de:  # noqa: BLE001
        log(f"device probe failed: {de!r}")
    log("imports done, building parser...")
    # resolve_cli_params/detect_checkpoint_path read sys.argv of THIS process (that is how the
    # vanilla CLI finds the checkpoint) — feed them the same argv we parse explicitly. Without
    # this they raise SystemExit, which sails past `except Exception` (Mark's worker-log find).
    sys.argv = ["distilled.py"] + job_argv({}, None, "/tmp/warm.mp4")
    _PARSER = D.add_generated_keyframes_arg(
        D.default_2_stage_distilled_arg_parser(
            params=D.resolve_cli_params(distilled=True), supports_auto_duration=True))
    args = _PARSER.parse_args(job_argv({}, None, "/tmp/warm.mp4"))
    log("building resident pipeline...")
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
    log("resident pipeline READY")
    return {"download_s": round(t_dl - t0, 1), "build_s": round(time.time() - t_dl, 1)}


def do_gen(inp, out_path):
    from ltx_pipelines import distilled as D  # noqa: PLC0415

    out = Path(out_path)
    out.unlink(missing_ok=True)
    img_path = None
    if inp.get("image_path"):
        img_path = Path(inp["image_path"])
    args = _PARSER.parse_args(job_argv(inp, img_path, out))
    t0 = time.time()
    with _TORCH.inference_mode():
        hdr = D.resolve_hdr_color_space(images=args.images, hdr=args.hdr)
        vae_dtype = D.vae_dtype_for_hdr(hdr, _TORCH.bfloat16)
        video, audio, num_frames, tiling_config = _PIPE(
            prompt=args.prompt, seed=args.seed, height=args.height, width=args.width,
            num_frames=args.num_frames, frame_rate=args.frame_rate, images=args.images,
            vae_dtype=vae_dtype, color_space=hdr,
            enhance_prompt=args.enhance_prompt, enhance_static_cache=args.enhance_static_cache,
            tiling_config=D.AUTO_TILING, generated_keyframes=args.num_generated_keyframes,
        )
        D.encode_video(video=video, fps=args.frame_rate, audio=audio, output_path=out,
                       video_chunks_number=D.get_video_chunks_number(num_frames, tiling_config),
                       color_space=hdr)
    return {"gen_s": round(time.time() - t0, 1)}


def main():
    log(f"child up, pid={os.getpid()}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "init":
                reply({"ok": True, "init": do_init()})
            elif cmd == "gen":
                r = do_gen(msg.get("input") or {}, msg.get("out") or "/tmp/out.mp4")
                reply({"ok": True, **r})
            elif cmd == "ping":
                reply({"ok": True, "pong": True})
            else:
                reply({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except SystemExit as exc:  # argparse errors raise SystemExit — keep the child alive
            reply({"ok": False, "error": f"argparse exit {exc.code}"})
        except Exception as exc:  # noqa: BLE001
            import traceback
            reply({"ok": False, "error": str(exc)[:300], "trace": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    main()
