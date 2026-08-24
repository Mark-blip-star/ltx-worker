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


# v9.10/v9.11 speed knobs; defaults = the lab-verified prod set (22-23.08, H200, 720p/1080p/vertical):
#  - resident decoders + prod-shape warm-up: frames bit-identical to v9.9, no +6s first-job penalty
#  - DiffVAE chunked_compile: decode 7.4->5.5s (720p) / 18.4->13.3s (1080p), PSNR 46 dB vs eager
#  - enhancer static KV cache: OFF — device-side assert on the Gemma-3 enhancer (sliding-window layers)
#  - DiT --compile: OFF — faster (-0.8s/-3.7s) but the same seed renders a different clip (PSNR 17-22 dB)
#  - combined_compile: OFF — recompiles per shape, 1080p got slower
ENHANCE_STATIC_CACHE = _flag("LTX25_ENHANCE_STATIC_CACHE", "0")
# v9.12: the prompt is enhanced on the backend (gpt-4o-mini, same upstream instructions), so the
# worker carries no Gemma-3 enhancer — 23 GB less VRAM and download. enhance:true is ignored.
ENHANCER = _flag("LTX25_ENHANCER", "0")
RESIDENT_DECODERS = _flag("LTX25_RESIDENT_DECODERS")
WARMUP = _flag("LTX25_WARMUP")
DIFFVAE_MODE = os.environ.get("LTX25_DIFFVAE_MODE", "chunked_compile").strip()  # "default" = upstream chunked_eager
if DIFFVAE_MODE == "default":
    DIFFVAE_MODE = ""
COMPILE = os.environ.get("LTX25_COMPILE", "").strip()  # "", "1" (defaults) or "k=v k=v" CompilationConfig overrides
# v9.13: mp4 stage. H100/H200 have no NVENC, so the lever is libx264 settings + knowing the CPU budget.
# "auto" = bench veryfast/superfast/ultrafast at init and keep the first that does >= X264_MIN_FPS on
# THIS host (serverless CPUs are shared: veryfast measured 8-151 fps across hosts, ultrafast 270-430).
X264_PRESET = os.environ.get("LTX25_X264_PRESET", "auto").strip()
X264_THREADS = int(os.environ.get("LTX25_X264_THREADS", "16") or 0)  # ffmpeg auto = 1.5 x 96 CPUs here, which stalls
X264_THREAD_TYPE = os.environ.get("LTX25_X264_THREAD_TYPE", "FRAME").strip().upper()  # FRAME (upstream) | SLICE | AUTO
X264_MIN_FPS = float(os.environ.get("LTX25_X264_MIN_FPS", "60"))  # 121 frames in <= 2 s, hidden behind a 4 s decode
WARMUP_SHAPES = [s for s in os.environ.get("LTX25_WARMUP_SHAPES", "1280x704,704x1280,1920x1088").split(",") if s.strip()]
ENCODE_BENCH = _flag("LTX25_ENCODE_BENCH")  # synthetic libx264 bench at init → init.encode_bench
ENCODE_BENCH_COMBOS = os.environ.get(
    "LTX25_ENCODE_BENCH_COMBOS",
    f"veryfast:{X264_THREADS}:{X264_THREAD_TYPE},superfast:{X264_THREADS}:{X264_THREAD_TYPE},"
    f"ultrafast:{X264_THREADS}:{X264_THREAD_TYPE}").split(",")
_X264 = {"preset": X264_PRESET if X264_PRESET != "auto" else "veryfast"}
COMPONENTS = {
    "transformer-path": os.environ.get(
        "LTX25_TRANSFORMER_FILE",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
    "text-encoder-path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video-vae-path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio-vae-path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial-upsampler-path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

# v9.18: optional adapter LoRAs, fused into the resident transformer at init. Env-only, so ONE
# image serves the plain prod endpoints and a LoRA endpoint. LTX25_LORAS lists files (in
# LTX25_LORA_REPO) as "name.safetensors:strength", comma-separated; unset => argv unchanged =>
# bit-exact to v9.17. Fusion is key-driven and silently skips LoRA keys the transformer does not
# have, so init reports how many modules actually matched (see _lora_report).
LORA_REPO = os.environ.get("LTX25_LORA_REPO", "").strip()
LORA_DIR = Path(os.environ.get("LTX25_LORA_DIR", "/models/loras"))


def _parse_loras(spec):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, strength = item.rpartition(":")
        out.append((name, float(strength)) if sep else (item, 1.0))
    return out


LORAS = _parse_loras(os.environ.get("LTX25_LORAS", ""))

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
                    not ENHANCER or (cand / "enhancer" / "config.json").is_file()):
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
    if ENHANCER and not (ENHANCER_DIR / "config.json").exists():
        log("downloading prompt-enhancer gemma (mirror)...")
        subprocess.run(["hf", "download", REPO, "--include", "enhancer/*", "--local-dir", str(MODELS)],
                       check=True, timeout=3600)
        return "downloaded"
    return "downloaded" if missing else "cached"


def ensure_loras():
    """Pull the adapter files from their own (private) repo into LORA_DIR."""
    if not LORAS:
        return "none"
    missing = [f for f, _ in LORAS if not (LORA_DIR / f).exists()]
    if missing:
        if not LORA_REPO:
            raise RuntimeError(f"LTX25_LORAS wants {missing} but LTX25_LORA_REPO is unset")
        log(f"downloading {len(missing)} LoRA file(s) from {LORA_REPO}...")
        subprocess.run(["hf", "download", LORA_REPO, *missing, "--local-dir", str(LORA_DIR)],
                       check=True, timeout=1800)
        return "downloaded"
    return "cached"


def _lora_report():
    """How many LoRA modules actually land on transformer weights. apply_loras() looks up
    "<model key minus .weight>.lora_A/B.weight" and CONTINUES past every miss, so a LoRA for a
    different architecture fuses nothing and still reports a clean job. Header-only read."""
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(MODELS / COMPONENTS["transformer-path"]), framework="pt") as f:
        prefix = "model.diffusion_model."
        mkeys = {k[len(prefix):] for k in f.keys() if k.startswith(prefix)}
    rep = []
    for fname, strength in LORAS:
        with safe_open(str(LORA_DIR / fname), framework="pt") as f:
            mods = [k[len("diffusion_model."):-len(".lora_A.weight")] for k in f.keys()
                    if k.startswith("diffusion_model.") and k.endswith(".lora_A.weight")]
        matched = sum(1 for m in mods if m + ".weight" in mkeys)
        rep.append({"file": fname, "strength": strength, "modules": len(mods), "matched": matched})
        if matched < len(mods):
            log(f"WARNING: LoRA {fname} matched {matched}/{len(mods)} modules — the rest fuse nothing")
    return rep


def _lora_argv():
    return [a for fname, strength in LORAS
            for a in ("--lora", str(LORA_DIR / fname), str(strength))]


def job_argv(inp, img_path, out_path):
    argv = []
    for flag, rel in COMPONENTS.items():
        argv += [f"--{flag}", str(MODELS / rel)]
    argv += _enhancer_root_argv()
    argv += ["--prompt", str(inp.get("prompt", "")),
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
    if _wants_enhance(inp, default=True):
        argv += ["--enhance-prompt"]
        if ENHANCE_STATIC_CACHE:
            argv += ["--enhance-static-cache"]
    argv += _perf_argv()
    argv += _lora_argv()
    argv += [str(a) for a in inp.get("extra_args", [])]
    return argv


def _enhancer_root_argv():
    return ["--prompt-enhancer-gemma-root", str(ENHANCER_DIR)] if ENHANCER else []


def _wants_enhance(inp, default):
    wanted = bool(inp.get("enhance", default))
    if wanted and not ENHANCER:
        log("enhance requested but LTX25_ENHANCER=0 — prompt used as written")
        return False
    return wanted


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
    argv += _enhancer_root_argv()
    argv += ["--prompt", str(inp.get("prompt", "")),
             "--seed", str(int(inp.get("seed", 42))),
             "--video-path", str(src_path),
             "--start-time", str(float(inp.get("start_time", 0.0))),
             "--end-time", str(float(inp.get("end_time", 0.0))),
             "--output-path", str(out_path)]
    if QUANTIZATION:
        argv += ["--quantization", QUANTIZATION]
    if _wants_enhance(inp, default=False):  # upstream retake default: the window prompt is used verbatim
        argv += ["--enhance-prompt"]
        if ENHANCE_STATIC_CACHE:
            argv += ["--enhance-static-cache"]
    argv += _perf_argv()
    argv += _lora_argv()
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
    lora_src = ensure_loras()
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
    if LORAS:
        try:
            lora_info = _lora_report()
        except Exception as le:  # noqa: BLE001
            lora_info = [{"error": str(le)[:200]}]
        log(f"loras: {lora_info}")
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
            "weights_src": weights_src, "models_dir": str(MODELS), "loras_src": lora_src,
            "knobs": {"enhancer": ENHANCER, "static_cache": ENHANCE_STATIC_CACHE, "resident_decoders": RESIDENT_DECODERS,
                      "warmup": WARMUP, "diffvae_mode": DIFFVAE_MODE or "default", "compile": COMPILE or "off"}}
    if LORAS:
        info["loras"] = lora_info
    info["cpu"] = _cpu_budget()
    if ENCODE_BENCH or X264_PRESET == "auto":
        try:
            info["encode_bench"] = _encode_bench()
        except Exception as be:  # noqa: BLE001
            info["encode_bench"] = {"error": str(be)[:200]}
        log(f"encode bench: {info['encode_bench']}")
        if X264_PRESET == "auto":
            _X264["preset"] = _pick_preset(info["encode_bench"])
    info["x264"] = {"preset": _X264["preset"], "threads": X264_THREADS, "thread_type": X264_THREAD_TYPE}
    log(f"x264: {info['x264']}")
    if WARMUP:
        # Prod shapes (121f) — pays the per-shape warm-up and any chunked_compile/--compile builds
        # once per worker, not on the first user job (measured +6s on prod 22.08 after 17f-only
        # keepalives). LTX25_WARMUP_SHAPES lists WxH; 1920x1088 too when combined_compile is on.
        info["warmup"] = []
        for shape in WARMUP_SHAPES:
            w, h = (int(v) for v in shape.lower().split("x"))
            tw = time.time()
            try:
                r = do_gen({"prompt": "A red kite rises over a windy beach at sunset. Audio: wind, gentle waves.",
                            "seed": 1, "width": w, "height": h, "fps": 24, "frames": 121,
                            "enhance": ENHANCER}, "/tmp/warm.mp4")
                info["warmup"].append({"shape": shape, "ok": True, "gen_s": r.get("gen_s"),
                                       "timings": r.get("timings"), "total_s": round(time.time() - tw, 1)})
            except Exception as we:  # noqa: BLE001
                import traceback
                info["warmup"].append({"shape": shape, "ok": False, "error": str(we)[:200],
                                       "trace": traceback.format_exc()[-1200:], "total_s": round(time.time() - tw, 1)})
                if _cuda_context_poisoned(we):
                    raise
            log(f"warmup {shape}: {info['warmup'][-1]}")
            Path("/tmp/warm.mp4").unlink(missing_ok=True)
    return info


def _pick_preset(bench):
    """First preset in bench order that clears X264_MIN_FPS; the lightest one if none does."""
    last = "ultrafast"
    for combo, r in bench.items():
        preset = combo.split(":")[0]
        last = preset
        if isinstance(r, dict) and r.get("fps", 0) >= X264_MIN_FPS:
            return preset
    return last


def _cpu_budget():
    """How much CPU libx264 really gets: logical CPUs, the affinity mask and the cgroup quota."""
    out = {"cpu_count": os.cpu_count()}
    try:
        out["affinity"] = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        pass
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            out["cgroup_cpu"] = Path(path).read_text().strip()
            break
        except OSError:
            continue
    return out


def _encode_bench(frames=48, width=1280, height=704):
    """libx264 alone, no decoder in the loop: fps per preset/threads/thread-type on THIS host."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    frames_np = np.stack([np.roll(base, shift=i * 7, axis=1) for i in range(frames)])
    video = _TORCH.from_numpy(frames_np).float().div_(255.0).to("cuda")  # [F,H,W,C] in [0,1]
    res = {}
    for combo in ENCODE_BENCH_COMBOS:
        try:
            preset, threads, ttype = combo.strip().split(":")
            t0 = time.time()
            encode_video_fast(video=video, fps=24, audio=None, output_path="/tmp/bench.mp4", video_chunks_number=1,
                              preset=preset, thread_count=int(threads), thread_type=ttype)
            dt = time.time() - t0
            res[combo.strip()] = {"fps": round(frames / dt, 1), "kb": Path("/tmp/bench.mp4").stat().st_size // 1024}
        except Exception as e:  # noqa: BLE001
            res[combo.strip()] = {"error": str(e)[:120]}
    Path("/tmp/bench.mp4").unlink(missing_ok=True)
    return res


def encode_video_fast(video, fps, audio, output_path, video_chunks_number, color_space=None,
                      preset=None, thread_count=None, thread_type=None, crf=19):
    """Upstream ``encode_video`` (media_io/encode.py, pin fd4ded7f) with the x264 threading exposed.
    FRAME threading with ffmpeg's auto thread count (1.5 x 96 CPUs here) stalls on shared serverless
    hosts — measured 8-18 fps at veryfast; SLICE threading / a thread cap / a lighter preset do not.
    HDR (color_space set) keeps the upstream path untouched.
    """
    from pathlib import Path as _P  # noqa: PLC0415

    import av  # noqa: PLC0415
    from ltx_core.color.audio_mux import prepare_audio_stream, validate_audio_waveform, write_audio  # noqa: PLC0415
    from ltx_core.color.yuv import PixelFormat, yuv420p_bt709_converter_  # noqa: PLC0415
    from ltx_pipelines.utils.media_io import encode as UE  # noqa: PLC0415

    preset = preset or _X264["preset"]
    thread_count = X264_THREADS if thread_count is None else thread_count
    thread_type = (thread_type or X264_THREAD_TYPE).upper()
    if color_space is not None:
        return UE.encode_video(video=video, fps=fps, audio=audio, output_path=output_path,
                               video_chunks_number=video_chunks_number, color_space=color_space)
    if audio is not None:
        validate_audio_waveform(audio)
    if isinstance(video, _TORCH.Tensor):
        video = iter([video])
    frame_converter = yuv420p_bt709_converter_

    def convert(chunk):
        return frame_converter(chunk.movedim(-1, -3))

    first_raw = next(video, None)
    if first_raw is None:
        raise ValueError("video is empty; expected at least one frame chunk.")
    first = convert(first_raw)
    if frame_converter.pixel_format == PixelFormat.RGB24:
        height, width = first.shape[-3], first.shape[-2]
    else:
        height, width = first.shape[-2] * 2 // 3, first.shape[-1]
    _P(output_path).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode="w")
    ok = False
    try:
        stream = container.add_stream("libx264", rate=int(fps), options={"crf": str(crf), "preset": preset})
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = thread_count
        stream.codec_context.thread_type = thread_type
        if frame_converter.color_space is not None:
            stream.codec_context.colorspace = frame_converter.color_space.av_colorspace
        if frame_converter.color_range is not None:
            stream.codec_context.color_range = frame_converter.color_range.av_color_range
        audio_stream = prepare_audio_stream(container, audio.sampling_rate) if audio is not None else None

        def cpu_chunks():
            yield first.to("cpu").numpy()
            for chunk in video:
                yield convert(chunk).to("cpu").numpy()

        UE._encode_chunks_threaded(container=container, stream=stream, av_format=frame_converter.pixel_format.av_format,
                                   chunks=cpu_chunks(), progress_total=video_chunks_number)
        if audio is not None:
            write_audio(container, audio_stream, audio)
        ok = True
    finally:
        container.close()
        if not ok:
            _P(output_path).unlink(missing_ok=True)
    return None


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
        encode_video_fast(video=video, fps=args.frame_rate, audio=audio, output_path=out,
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
        encode_video_fast(video=video_iter, fps=int(src.fps), audio=audio, output_path=out,
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
            if _cuda_context_poisoned(exc):
                # A device-side assert / illegal access leaves every later kernel failing in this
                # process (seen 22.08: one bad warm-up, then each job died at torch.Generator).
                # Die now; the supervisor restarts a clean child on the next job.
                err["child_restart"] = True
                reply(err)
                log("CUDA context poisoned — exiting so the supervisor restarts the child")
                sys.stderr.flush()
                os._exit(3)
            reply(err)


def _cuda_context_poisoned(exc):
    msg = str(exc)
    return any(s in msg for s in ("device-side assert", "illegal memory access", "CUDA error",
                                  "unspecified launch failure", "CUBLAS_STATUS_EXECUTION_FAILED"))


if __name__ == "__main__":
    main()
