"""Persistent generation child for the LTX-2.5 resident worker (v9.2 supervisor pattern).

Reads one JSON command per stdin line, answers one JSON line on stdout. All heavy imports
and the vanilla DistilledPipeline live HERE, so a CUDA/native crash kills this process only —
the runpod SDK supervisor survives and reports our stderr.

Commands:
  {"cmd": "init"}                       -> {"ok": true, "init": {...}}
  {"cmd": "gen", "input": {...}, "out": "/tmp/out.mp4"} -> {"ok": true, "gen_s": 12.3}
  {"cmd": "retake", "input": {...}, "src": "/tmp/retake_src.mp4", "out": "/tmp/out.mp4"}
                                        -> {"ok": true, "gen_s": 9.8}
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
QUANTIZATION = os.environ.get("LTX25_QUANTIZATION", "").strip()  # "", fp8-cast, nvfp4-prequant, ...


def _flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# v9.10 speed knobs (all quality-neutral; defaults = the measured-safe set, compile paths opt-in)
ENHANCE_STATIC_CACHE = _flag("LTX25_ENHANCE_STATIC_CACHE")  # HF static KV cache for the Gemma enhancer
RESIDENT_DECODERS = _flag("LTX25_RESIDENT_DECODERS")  # keep VAE/upsampler/audio decoders on GPU between jobs
WARMUP = _flag("LTX25_WARMUP")  # one prod-shape clip at init so the first user job pays no shape warm-up
DIFFVAE_MODE = os.environ.get("LTX25_DIFFVAE_MODE", "").strip()  # "", chunked_compile, combined_compile
COMPILE = os.environ.get("LTX25_COMPILE", "").strip()  # "", "1" (defaults) or "k=v k=v" CompilationConfig overrides
COMPONENTS = {
    "transformer-path": os.environ.get(
        "LTX25_TRANSFORMER_FILE",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
    "text-encoder-path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video-vae-path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio-vae-path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial-upsampler-path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

_PIPE = None
_PARSER = None
_TORCH = None
_RETAKE_PIPE = None
_RETAKE_PARSER = None


def log(msg):
    print(f"[child] {msg}", file=sys.stderr, flush=True)


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()



def _seeded_models_root():
    """RunPod Cached Models seeds the pinned repo onto the host disk before the container
    starts, but the mount layout is undocumented (2.3 hosts showed both a repo dir at the
    volume root and an HF hub tree). Probe the known shapes and accept only a directory
    holding EVERY component plus the enhancer — anything less falls through to the
    download path unchanged."""
    dirname = "models--" + REPO.replace("/", "--")
    candidates = []
    for root in (Path("/runpod-volume"), Path("/runpod-volume/huggingface-cache/hub")):
        d = root / dirname
        try:
            ref = d / "refs" / "main"
            if ref.is_file():
                candidates.append(d / "snapshots" / ref.read_text().strip())
            snaps = d / "snapshots"
            if snaps.is_dir():
                candidates += sorted(snaps.iterdir(), reverse=True)
        except OSError:
            pass
        candidates.append(d)
    for cand in candidates:
        try:
            if all((cand / rel).is_file() for rel in COMPONENTS.values()) and (
                    cand / "enhancer" / "config.json").is_file():
                return cand
        except OSError:
            continue
    return None


def ensure_weights():
    global MODELS, ENHANCER_DIR
    # An explicit LTX25_MODELS_DIR keeps full manual control (RD override); otherwise a
    # complete seeded copy wins and the virgin worker skips its ~30s download.
    if "LTX25_MODELS_DIR" not in os.environ:
        seeded = _seeded_models_root()
        if seeded is not None:
            MODELS = seeded
            ENHANCER_DIR = MODELS / "enhancer"
            log(f"weights: complete seeded copy at {MODELS} — download skipped")
            return "seeded"
    missing = [rel for rel in COMPONENTS.values() if not (MODELS / rel).exists()]
    if missing:
        log(f"downloading {len(missing)} component files...")
        subprocess.run(["hf", "download", REPO, *missing, "--local-dir", str(MODELS)],
                       check=True, timeout=3600)
    if not (ENHANCER_DIR / "config.json").exists():
        log("downloading prompt-enhancer gemma (mirror)...")
        subprocess.run(["hf", "download", REPO, "--include", "enhancer/*", "--local-dir", str(MODELS)],
                       check=True, timeout=3600)
        return "downloaded"
    return "downloaded" if missing else "cached"


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
    if QUANTIZATION:
        argv += ["--quantization", QUANTIZATION]
    if img_path is not None:
        argv += ["--image", str(img_path), "0", "1.0"]
    if bool(inp.get("enhance", True)):
        argv += ["--enhance-prompt"]
        if ENHANCE_STATIC_CACHE:
            argv += ["--enhance-static-cache"]
    argv += _perf_argv()
    argv += [str(a) for a in inp.get("extra_args", [])]
    return argv


def _perf_argv():
    argv = []
    if DIFFVAE_MODE:
        argv += ["--diffvae-optimization", DIFFVAE_MODE]
    if COMPILE:
        argv += ["--compile"] + ([] if COMPILE in ("1", "true", "on") else COMPILE.split())
    return argv


def retake_argv(inp, src_path, out_path):
    argv = []
    for flag, rel in COMPONENTS.items():
        if flag == "spatial-upsampler-path":
            continue  # retake is single-stage at source resolution; video_editing parser has no upsampler flag
        argv += [f"--{flag}", str(MODELS / rel)]
    argv += ["--prompt-enhancer-gemma-root", str(ENHANCER_DIR),
             "--prompt", str(inp.get("prompt", "")),
             "--seed", str(int(inp.get("seed", 42))),
             "--video-path", str(src_path),
             "--start-time", str(float(inp.get("start_time", 0.0))),
             "--end-time", str(float(inp.get("end_time", 0.0))),
             "--output-path", str(out_path)]
    if QUANTIZATION:
        argv += ["--quantization", QUANTIZATION]
    if bool(inp.get("enhance", False)):  # upstream retake default: the window prompt is used verbatim
        argv += ["--enhance-prompt"]
        if ENHANCE_STATIC_CACHE:
            argv += ["--enhance-static-cache"]
    argv += _perf_argv()
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
    if RESIDENT_DECODERS:
        _make_builders_resident()


_BUILD_CACHE = {}


def _make_builders_resident():
    """v9.10: in this upstream pin the VAE encoder/upsampler, video decoder and audio decoder+vocoder
    are built inline via ``Builder.build()`` inside the blocks' __call__ (no ``_build_*`` hook), so
    every job re-read ~3 GiB of weights from disk and re-materialized the modules. Memoize
    ``SingleGPUModelBuilder.build`` per builder instance (the blocks own their builders for the
    pipeline's lifetime) and neutralize ``dispose`` on the cached module so ``gpu_model()`` leaves
    it resident. The builder itself is pinned in the cache entry so its id() can never be recycled
    onto a different builder. A different device/dtype request (HDR fp32 decode) falls through to a
    real rebuild and evicts the stale entry — correctness over speed on that path.
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder  # noqa: PLC0415

    orig = SingleGPUModelBuilder.build

    def build(self, device=None, dtype=None, **kw):
        key = id(self)
        want = (str(device), str(dtype))
        hit = _BUILD_CACHE.get(key)
        if hit is not None and hit[0] == want:
            return hit[1]
        if hit is not None:
            _BUILD_CACHE.pop(key)
        m = orig(self, device=device, dtype=dtype, **kw)
        try:
            m.dispose = lambda: None
        except AttributeError:
            pass
        _BUILD_CACHE[key] = (want, m, self)
        log(f"resident: builder-cached {type(m).__name__} ({len(_BUILD_CACHE)} modules resident)")
        return m

    SingleGPUModelBuilder.build = build
    log("residency: SingleGPUModelBuilder.build memoized (decoders/upsampler stay on GPU)")


_TIMINGS = {}


def _timed(cls, name, label):
    orig = getattr(cls, name)

    def wrapper(self, *a, **kw):
        t0 = time.time()
        out = orig(self, *a, **kw)
        if hasattr(out, "__next__"):  # iterator: count until exhausted (video decoder yields chunks)
            def it():
                try:
                    yield from out
                finally:
                    _TIMINGS[label] = round(_TIMINGS.get(label, 0.0) + time.time() - t0, 2)
            return it()
        _TIMINGS[label] = round(_TIMINGS.get(label, 0.0) + time.time() - t0, 2)
        return out

    setattr(cls, name, wrapper)


def _install_timers():
    """Per-phase wall clock in the job output (enhance / upsample / decode / audio / mp4) so the
    speed levers can be judged from prod outputs, not guessed."""
    import ltx_pipelines.utils.blocks as B  # noqa: PLC0415
    from ltx_core.text_encoders.gemma.encoders import base_encoder as E  # noqa: PLC0415

    _timed(E.LTXGemmaTextEncoder, "_enhance", "enhance_s")
    for cname, label in (("VideoUpsampler", "upsample_s"), ("VideoDecoder", "decode_s"),
                         ("AudioDecoder", "audio_s"), ("PromptEncoder", "encode_s")):
        cls = getattr(B, cname, None)
        if cls is not None and "__call__" in vars(cls):
            _timed(cls, "__call__", label)
    log("phase timers installed")


def _take_timings():
    t = dict(_TIMINGS)
    _TIMINGS.clear()
    return t


def do_init():
    global _PIPE, _PARSER, _TORCH
    t0 = time.time()
    weights_src = ensure_weights()
    t_dl = time.time()
    log("importing torch + ltx stack...")
    import torch  # noqa: PLC0415
    from ltx_pipelines import distilled as D  # noqa: PLC0415
    _TORCH = torch
    _make_resident()
    try:
        _install_timers()
    except Exception as te:  # noqa: BLE001
        log(f"phase timers unavailable: {te!r}")
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
    info = {"download_s": round(t_dl - t0, 1), "build_s": round(time.time() - t_dl, 1),
            "weights_src": weights_src, "models_dir": str(MODELS),
            "knobs": {"static_cache": ENHANCE_STATIC_CACHE, "resident_decoders": RESIDENT_DECODERS,
                      "warmup": WARMUP, "diffvae_mode": DIFFVAE_MODE or "default", "compile": COMPILE or "off"}}
    if WARMUP:
        # Prod shape (1280x704x121f, enhancer on) — pays the per-shape warm-up, the enhancer's
        # static-cache compile and any --compile/chunked_compile builds once per worker, not on
        # the first user job (measured +6s on prod 22.08 after 17f-only keepalives).
        tw = time.time()
        try:
            r = do_gen({"prompt": "A red kite rises over a windy beach at sunset. Audio: wind, gentle waves.",
                        "seed": 1, "width": 1280, "height": 704, "fps": 24, "frames": 121,
                        "enhance": True}, "/tmp/warm.mp4")
            info["warmup"] = {"ok": True, "gen_s": r.get("gen_s"), "timings": r.get("timings"),
                              "total_s": round(time.time() - tw, 1)}
        except Exception as we:  # noqa: BLE001
            info["warmup"] = {"ok": False, "error": str(we)[:200], "total_s": round(time.time() - tw, 1)}
        log(f"warmup: {info['warmup']}")
        Path("/tmp/warm.mp4").unlink(missing_ok=True)
    return info


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
        te = time.time()
        D.encode_video(video=video, fps=args.frame_rate, audio=audio, output_path=out,
                       video_chunks_number=D.get_video_chunks_number(num_frames, tiling_config),
                       color_space=hdr)
        _TIMINGS["mp4_s"] = round(time.time() - te, 2)  # includes the decode chunks it pulls through
    return {"gen_s": round(time.time() - t0, 1), "timings": _take_timings()}


def _get_retake_pipe(args):
    """RetakePipeline that shares the gen pipeline's resident blocks. Both pipelines build
    their blocks from identical construction args, and blocks only load weights on __call__
    (__init__ stores config), so the duplicates made here are free — swapping in the gen
    pipeline's instances makes the residency cache hit by id() with zero extra VRAM. Retake
    keeps only its own audio_conditioner (gen has none); that encoder builds/frees per call.
    """
    global _RETAKE_PIPE
    if _RETAKE_PIPE is None:
        from ltx_pipelines import retake as R  # noqa: PLC0415

        p = R.RetakePipeline(
            model_paths=args.model_paths,
            loras=tuple(args.lora) if args.lora else (),
            quantization=args.quantization,
            distilled=True,
            compilation_config=args.compile,
            offload_mode=args.offload_mode,
            prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
            diffvae_optimization=args.diffvae_optimization,
        )
        p.prompt_encoder = _PIPE.prompt_encoder
        p.image_conditioner = _PIPE.image_conditioner
        p.stage = _PIPE.stage
        p.video_decoder = _PIPE.video_decoder
        p.audio_decoder = _PIPE.audio_decoder
        _RETAKE_PIPE = p
        log("retake pipeline ready (blocks shared with gen)")
    return _RETAKE_PIPE


def do_retake(inp, src_path, out_path):
    global _RETAKE_PARSER
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number  # noqa: PLC0415
    from ltx_core.types import SpatioTemporalScaleFactors  # noqa: PLC0415
    from ltx_pipelines.utils.args import video_editing_arg_parser  # noqa: PLC0415
    from ltx_pipelines.utils.constants import detect_params  # noqa: PLC0415
    from ltx_pipelines.utils.media_io import (  # noqa: PLC0415
        encode_video, get_videostream_metadata, resolve_hdr_color_space, vae_dtype_for_hdr)

    out = Path(out_path)
    out.unlink(missing_ok=True)
    if _RETAKE_PARSER is None:
        _RETAKE_PARSER = video_editing_arg_parser(distilled=True)
    args = _RETAKE_PARSER.parse_args(retake_argv(inp, src_path, out))
    if args.start_time >= args.end_time:
        raise ValueError(f"start_time ({args.start_time}) must be less than end_time ({args.end_time})")
    # upstream main()'s CLI-stage source validation, verbatim
    video_scale = SpatioTemporalScaleFactors.default()
    src = get_videostream_metadata(args.video_path)
    if (src.frames - 1) % video_scale.time != 0:
        snapped = ((src.frames - 1) // video_scale.time) * video_scale.time + 1
        raise ValueError(f"source frames must satisfy 8k+1; got {src.frames} (nearest {snapped})")
    if src.width % 32 != 0 or src.height % 32 != 0:
        raise ValueError(f"source dims must be multiples of 32; got {src.width}x{src.height}")
    pipe = _get_retake_pipe(args)
    t0 = time.time()
    with _TORCH.inference_mode():
        params = detect_params(args.model_paths.transformer())
        hdr = resolve_hdr_color_space(video_paths=[args.video_path], hdr=args.hdr)
        vae_dtype = vae_dtype_for_hdr(hdr, _TORCH.bfloat16)
        video_iter, audio, tiling_config = pipe(
            video_path=args.video_path,
            prompt=args.prompt,
            start_time=args.start_time,
            end_time=args.end_time,
            seed=args.seed,
            enhance_prompt=args.enhance_prompt,
            enhance_static_cache=args.enhance_static_cache,
            regenerate_video=bool(inp.get("regenerate_video", True)),
            regenerate_audio=bool(inp.get("regenerate_audio", True)),
            video_guider_params=params.video_guider_params,
            audio_guider_params=params.audio_guider_params,
            vae_dtype=vae_dtype,
            color_space=hdr,
            tiling_config=AUTO_TILING,
            max_batch_size=args.max_batch_size,
        )
        encode_video(video=video_iter, fps=int(src.fps), audio=audio, output_path=out,
                     video_chunks_number=get_video_chunks_number(src.frames, tiling_config),
                     color_space=hdr)
    return {"gen_s": round(time.time() - t0, 1), "timings": _take_timings()}


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
            elif cmd == "retake":
                r = do_retake(msg.get("input") or {}, msg.get("src") or "/tmp/retake_src.mp4",
                              msg.get("out") or "/tmp/out.mp4")
                reply({"ok": True, **r})
            elif cmd == "ping":
                reply({"ok": True, "pong": True})
            else:
                reply({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except SystemExit as exc:  # argparse errors raise SystemExit — keep the child alive
            reply({"ok": False, "error": f"argparse exit {exc.code}"})
        except Exception as exc:  # noqa: BLE001
            import traceback
            err = {"ok": False, "error": str(exc)[:300], "trace": traceback.format_exc()[-1500:]}
            try:
                if _TORCH is not None:
                    free_b, total_b = _TORCH.cuda.mem_get_info()
                    err["vram_free_gib"] = round(free_b / 2**30, 1)
                    err["vram_total_gib"] = round(total_b / 2**30, 1)
            except Exception:  # noqa: BLE001
                pass
            reply(err)


if __name__ == "__main__":
    main()
