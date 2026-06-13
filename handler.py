"""LTX-2.3 RunPod serverless handler v8.5 — v8.4 + audio:false default (H-3, owner-approved).

v8.4 (2026-06-11):
- L-5: real StateDictRegistry into the pipeline + ResidentStageCache (graph_key-only keys —
  SingleGPUModelBuilder ignores shapes, stock keys would mint 22.6 GiB per resolution) passed
  into run_case. Kills per-request: stage2 LoRA re-fuse + 22.6 GiB clone, stage1 build/teardown,
  VAE-encoder x3 re-reads, embeddings-processor re-read. Resident floor ~71-73 GiB -> resolution
  gate REJECTS >1280x704x121 (bypass would still OOM: floor + fuse transient + activations).
- L-4: start.sh page-cache prewarm (ckpt-first), handler kills its process group at the first
  REAL job (warm pings let it run). /tmp/prewarm.pid, setsid leader.
- trtllm prewarm REMOVED: measurement showed import tensorrt_llm fails on EVERY boot
  (libpython3.12.so.1.0 unreachable in uv-standalone python) and trtllm_scaled_mm_usable()
  silently latches the torch._scaled_mm fallback — prod has always run the fallback. Kernel
  flip only via offline A/B (ldconfig fix exists, see INIT-SPEEDUP-RESEARCH.md). We log the
  active path at init instead.
- Per-request torch.cuda.reset_peak_memory_stats so peak_vram_gib stays per-request-true.

v8.0-v8.3 (init-speedup campaign 2026-06-11):
- L-2: Gemma fp8 shards stream straight to GPU via safe_open(device="cuda") and dequant there
  (OnGPUFp8GemmaBuilder) — kills the 36 GB read/cast/write disk round-trip (~18-38 s measured).
  Identical math to the old CPU dequant: (w.fp32 * scale.fp32).bf16, attn pinned to "sdpa"
  to match from_pretrained's choice.
- H-1: eager _init() in a daemon thread before runpod.serverless.start; handler serializes
  on _INIT_LOCK (idempotent). Overlaps init with SDK fitness checks. LTX_EAGER_INIT=0 disables.
- L-6: Gemma TE build runs in parallel with the TI2VidTwoStagesPipeline build (2 threads).
- J-1: prewarm_trtllm_scaled_mm() (lazy tensorrt_llm import + fp8 GEMM kernel) inside init.
- H-7: {"input":{"warm":true}} returns after init without generating — keep-warm ping op.
- H-2: "[init] t+XX.XXs phase" timing lines on stdout -> HF worker logs.

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
import json
import os
import signal
import tempfile
import threading
import time
import traceback
import types
from pathlib import Path

_T0 = time.monotonic()


def _mark(phase: str) -> None:
    print(f"[init] t+{time.monotonic() - _T0:8.2f}s {phase}", flush=True)


_CACHE = "/runpod-volume/huggingface-cache" if os.path.isdir("/runpod-volume") else "/app/hfcache"
os.environ.setdefault("HF_HOME", _CACHE)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LTX_ATTENTION_TYPE", "flash_attention_3")  # critical: vanilla attn OOMs
os.environ.setdefault("LTX_FAST_VAE", "1")
os.environ.setdefault("LTX_TGATE_START_STEP", "10")

import torch  # noqa: E402
import runpod  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402
import stage_timing_runner as base  # noqa: E402
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: E402
from ltx_core.loader.registry import StateDictRegistry  # noqa: E402
from ltx_core.quantization import QuantizationPolicy  # noqa: E402
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # noqa: E402
from ltx_pipelines.utils.blocks import PromptEncoder  # noqa: E402
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES  # noqa: E402

_mark("imports_done")

WEIGHTS_REPO = os.environ["LTX_WEIGHTS_REPO"]
FP8_CKPT_NAME = os.environ.get("LTX_FP8_CKPT_NAME", "ltx-2.3-22b-dev-fp8.safetensors")
UPS_NAME = os.environ.get("LTX_UPSCALER_NAME", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
LORA_NAME = os.environ.get("LTX_DISTILLED_LORA_NAME", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
GEMMA_FP8_SUBDIR = os.environ.get("LTX_GEMMA_FP8_SUBDIR", "gemma-fp8")
# Official HQ recipe fuses the distilled LoRA into stage1 too (default strength there: 0.25).
# 0 = off = bit-identical to v8.5 behavior; flipping requires a worker restart (fusion at init).
S1_LORA_STRENGTH = float(os.environ.get("LTX_S1_LORA_STRENGTH", "0"))
# Default stage1 step count for the fast tier; request "steps" still wins. 12 = v8.5 behavior.
FAST_DEFAULT_STEPS = int(os.environ.get("LTX_FAST_DEFAULT_STEPS", "12"))
# Default stage1 sigma grid for the fast tier (JSON list). When set, it replaces the
# resampled-distilled default and pins step count to len-1; a per-request "sigmas" still wins.
# Validated through the same path as per-request sigmas. e.g. A13 anchor-preserving grid.
try:
    FAST_SIGMAS_DEFAULT = json.loads(os.environ["LTX_FAST_SIGMAS"]) if os.environ.get("LTX_FAST_SIGMAS") else None
except (ValueError, TypeError):
    FAST_SIGMAS_DEFAULT = None
CONFIG_TAG = (os.environ.get("LTX_CONFIG_TAG", "v8")
              + (f"-s1_{S1_LORA_STRENGTH:g}" if S1_LORA_STRENGTH > 0 else "")
              + (f"-st{FAST_DEFAULT_STEPS}" if FAST_DEFAULT_STEPS != 12 else "")
              + (f"-fsig{len(FAST_SIGMAS_DEFAULT) - 1}" if FAST_SIGMAS_DEFAULT else ""))
# v8.7 switches. All default-off so an env-clean v8.7 binary is path-identical to v8.6.1
# except the always-on bit-exact changes (resident tail, stream-yield decode), gated by PSNR.
SKIP_NEG_ENCODE = os.environ.get("LTX_SKIP_NEG_ENCODE", "0") == "1"  # bit-exact: prompts encode sequentially
WARMUP_GEN = os.environ.get("LTX_WARMUP_GEN", "0") == "1"  # full prod-shape generation inside init
_S3 = {k: os.environ.get(f"LTX_S3_{k}") for k in ("ENDPOINT", "BUCKET", "KEY", "SECRET")}
S3_ON = all(_S3.values())
RETURN_URL_ONLY = os.environ.get("LTX_RETURN_URL_ONLY", "0") == "1"
_SKIP_NEG_NOW = {"on": False}   # per-request (handler is single-threaded)
_TAIL_RESIDENT = {"done": False}
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
_INIT_LOCK = threading.Lock()
_GEMMA_DIR = None  # resolved gemma-fp8 snapshot subdir (cache-key/meta only after L-2)
_FIRST_JOB_SEEN = False


class _StageKeyedCache(base.ResidentStageCache):
    """L-5: key resident transformers by stage only. SingleGPUModelBuilder.build ignores
    shape/fps/audio kwargs, so one transformer serves every request; the stock key_for
    would mint a new ~22.6 GiB stage2 entry per (w,h,frames,fps,audio) tuple -> OOM."""

    def key_for(self, graph_key, **_):
        return graph_key

    def get(self, stage, cache_key, **kw):
        hit = cache_key in self._entries
        out = super().get(stage, cache_key, **kw)
        # v8.6.1: with the distilled LoRA fused into stage1 too, no live model shares the
        # registry's base SD — drop it after each stage build so at most two transformer
        # copies stay resident (three + activations busts the 94GB H100 NVL).
        if S1_LORA_STRENGTH > 0 and not hit:
            _REGISTRY.clear()
            torch.cuda.empty_cache()
            _INIT_LOG.append(
                f"registry cleared after {cache_key} build; "
                f"cuda_alloc={torch.cuda.memory_allocated() / 2**30:.1f}GiB")
        # v8.7 L-1: once both stage transformers are resident the big build transients are
        # over — now (and only now) pin the small tail models, so their +~3.5 GiB never
        # coexists with a 22.6 GiB fuse transient on the 94 GB H100 NVL.
        if not hit and len(self._entries) >= 2:
            _make_tail_resident()
        return out


_REGISTRY = StateDictRegistry()
_STAGE_CACHE = _StageKeyedCache()
# L-5 residency floor (Gemma 24 + base SD 22.6 + fused stage2 22.6 + small SDs) ≈ 71-73 GiB;
# 1080p activations + a non-cached fuse transient would bust 94 GiB, so larger jobs are rejected.
_MAX_PIXELS = 1280 * 704
_MAX_FRAMES = 121


class _ResidentConditioner:
    """v8.7 L-1: VAE encoder built once; upstream rebuilds + frees it per call."""

    def __init__(self, inner):
        self._enc = inner._build_encoder()

    def __call__(self, fn):
        return fn(self._enc)


class _ResidentUpsampler:
    def __init__(self, inner):
        self._enc = inner._encoder_builder.build(device=inner._device, dtype=inner._dtype).to(inner._device).eval()
        self._ups = inner._upsampler_builder.build(device=inner._device, dtype=inner._dtype).to(inner._device).eval()

    def __call__(self, latent):
        from ltx_core.model.upsampler import upsample_video
        return upsample_video(latent=latent, video_encoder=self._enc, upsampler=self._ups)


class _ResidentDecoder:
    def __init__(self, inner):
        self._dec = inner._decoder_builder.build(device=inner._device, dtype=inner._dtype).to(inner._device).eval()

    def __call__(self, latent, tiling_config=None, generator=None):
        return self._dec.decode_video(latent, tiling_config, generator)


def _make_tail_resident():
    """Swap per-call build-and-free tail blocks for resident ones. Bit-exact: same weights,
    same op order — they just stop being rebuilt and torn down on every job."""
    if _TAIL_RESIDENT["done"] or _PIPE is None:
        return
    try:
        _PIPE.image_conditioner = _ResidentConditioner(_PIPE.image_conditioner)
        _PIPE.upsampler = _ResidentUpsampler(_PIPE.upsampler)
        _PIPE.video_decoder = _ResidentDecoder(_PIPE.video_decoder)
        _TAIL_RESIDENT["done"] = True
        _INIT_LOG.append(
            f"tail resident (vae-enc/upsampler/decoder); cuda_alloc={torch.cuda.memory_allocated() / 2**30:.1f}GiB")
    except Exception as exc:  # noqa: BLE001
        _TAIL_RESIDENT["done"] = True  # don't retry every job
        _INIT_LOG.append(f"tail-resident FAILED, kept per-job path: {exc!r}")


def _s3_put(data: bytes, key: str, ctype: str) -> str:
    """Minimal sigv4 PUT (stdlib only) to DO Spaces / any S3 endpoint. Returns public URL."""
    import datetime as _dt
    import hashlib
    import hmac
    import urllib.request as _ur
    host = f"{_S3['BUCKET']}.{_S3['ENDPOINT']}"
    region = _S3["ENDPOINT"].split(".")[0]
    now = _dt.datetime.now(_dt.timezone.utc)
    amzdate, datestamp = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(data).hexdigest()
    headers = {"host": host, "content-type": ctype, "x-amz-acl": "public-read",
               "x-amz-content-sha256": payload_hash, "x-amz-date": amzdate}
    signed = ";".join(sorted(headers))
    canonical = ("PUT\n/" + key + "\n\n"
                 + "".join(f"{k}:{headers[k]}\n" for k in sorted(headers)) + "\n"
                 + signed + "\n" + payload_hash)
    scope = f"{datestamp}/{region}/s3/aws4_request"
    sts = ("AWS4-HMAC-SHA256\n" + amzdate + "\n" + scope + "\n"
           + hashlib.sha256(canonical.encode()).hexdigest())

    def _hm(k, m):
        return hmac.new(k, m.encode(), hashlib.sha256).digest()

    sig_key = _hm(_hm(_hm(_hm(("AWS4" + _S3["SECRET"]).encode(), datestamp), region), "s3"), "aws4_request")
    signature = hmac.new(sig_key, sts.encode(), hashlib.sha256).hexdigest()
    auth = f"AWS4-HMAC-SHA256 Credential={_S3['KEY']}/{scope}, SignedHeaders={signed}, Signature={signature}"
    req = _ur.Request(f"https://{host}/{key}", data=data, method="PUT", headers={
        "Content-Type": ctype, "x-amz-acl": "public-read",
        "x-amz-content-sha256": payload_hash, "x-amz-date": amzdate, "Authorization": auth})
    _ur.urlopen(req, timeout=120).read()
    return f"https://{host}/{key}"


def _kill_prewarm() -> None:
    """L-4: stop the start.sh page-cache prewarm — a real job owns the disk now."""
    try:
        pid = int(Path("/tmp/prewarm.pid").read_text().strip())
        Path("/tmp/prewarm.pid").unlink(missing_ok=True)  # before killpg: no stale-pid retries
        os.killpg(pid, signal.SIGTERM)  # setsid leader -> reaps the in-flight cat too
        _mark(f"prewarm_killed pid={pid}")
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        pass
# stage2 sigma sets (module-level switch before each run_case; handler is single-threaded)
_STAGE2_DEFAULT = None  # captured at init from base.STAGE_2_DISTILLED_SIGMAS
_STAGE2_FAST = torch.tensor([0.909375, 0.6, 0.0])

# stage1 scheduler dispatch: "ltx2" -> real LTX2Scheduler, "distilled" -> resampled official curve
_SIGMA_MODE = {"mode": "ltx2"}
_LTX2Scheduler_real = base.LTX2Scheduler


class _DispatchScheduler:
    def execute(self, steps: int):
        if _SIGMA_MODE["mode"] == "distilled":
            ov = _SIGMA_MODE.get("override")
            if ov:  # explicit per-request grid (request "sigmas"); validated in handler()
                return torch.tensor(ov, dtype=torch.float32)
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
    """Build the Gemma text encoder ONCE and keep it in VRAM (skip free-on-exit ctx).

    v8.7: the embeddings processor goes resident too (same module, just not freed), and with
    LTX_SKIP_NEG_ENCODE the unused-at-CFG=1 negative prompt is not encoded at all. Both are
    bit-exact for the positive context: upstream encodes prompts SEQUENTIALLY (one
    text_encoder.encode per prompt — no batch interaction), and the guider never reads the
    negative context when cfg_scale == 1.
    """

    def _text_encoder_ctx(self):
        if not hasattr(self, "_resident_te"):
            self._resident_te = self._build_text_encoder()
        return contextlib.nullcontext(self._resident_te)

    def __call__(self, prompts, *, enhance_first_prompt=False, enhance_prompt_image=None,
                 enhance_prompt_seed=42):
        skip_neg = _SKIP_NEG_NOW["on"] and isinstance(prompts, list) and len(prompts) == 2
        if skip_neg:
            prompts = prompts[:1]
        with self._text_encoder_ctx() as text_encoder:
            if enhance_first_prompt:
                from ltx_pipelines.utils.blocks import generate_enhanced_prompt
                prompts = list(prompts)
                prompts[0] = generate_enhanced_prompt(
                    text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed)
            raw_outputs = [text_encoder.encode(p) for p in prompts]
        if not hasattr(self, "_resident_ep"):
            self._resident_ep = self._build_embeddings_processor()
        outs = [self._resident_ep.process_hidden_states(hs, mask) for hs, mask in raw_outputs]
        if skip_neg:
            outs = [outs[0], outs[0]]  # negative ctx is never read at cfg=1
        return outs


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


class OnGPUFp8GemmaBuilder(base.PrequantCausalGemmaBuilder):
    """L-2: stream fp8 shards straight to the GPU and dequant there — no disk round-trip.

    Math is identical to the old CPU dequant ((w.fp32 * scale.fp32).bf16) and the resident
    format is the same bf16, so outputs must match the v7 path bit-for-bit. attn impl is
    pinned to "sdpa" because from_pretrained auto-selects it while a bare constructor
    defaults to "eager" — an unpinned mismatch would silently change encoder numerics.
    """

    def build(self, device=None, dtype=None, **_: object):
        from accelerate import init_empty_weights
        from ltx_core.utils import find_matching_file
        from transformers import Gemma3ForCausalLM

        target = device or torch.device("cuda")
        t0 = time.perf_counter()
        model_folder = find_matching_file(self._model_root, "model*.safetensors").parent
        # config_class.from_pretrained == the exact parsing path from_pretrained uses; it
        # unwraps the composite Gemma3Config (text+vision) into Gemma3TextConfig — a bare
        # AutoConfig here hands the composite to the constructor and crashes on vocab_size.
        cfg = Gemma3ForCausalLM.config_class.from_pretrained(str(model_folder), local_files_only=True)
        cfg._attn_implementation = "sdpa"
        with init_empty_weights(include_buffers=False):
            model = Gemma3ForCausalLM(cfg)
        # The fp8 checkpoint is the full multimodal Gemma3: keys are language_model.* /
        # vision_tower.* / multi_modal_projector.*. from_pretrained strips the prefix and
        # drops non-LM keys for Gemma3ForCausalLM — replicate that here, and skip vision
        # tensors BEFORE fetching them to the GPU. Every tensor (incl. norms/embeddings)
        # carries a per-tensor .fp8_scale companion.
        lm_prefix = "language_model."
        sd, scales = {}, {}
        for shard in sorted(glob.glob(os.path.join(str(model_folder), "*.safetensors"))):
            with safe_open(shard, framework="pt", device=str(target)) as f:
                for key in f.keys():
                    if not key.startswith(lm_prefix):
                        continue
                    name = key[len(lm_prefix):]
                    t = f.get_tensor(key)
                    if name.endswith(".fp8_scale"):
                        scales[name[: -len(".fp8_scale")]] = t
                    else:
                        sd[name] = t
        for k, s in scales.items():
            sd[k] = (sd[k].to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)
        missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
        if unexpected:
            raise RuntimeError(f"gemma load: unexpected keys {list(unexpected)[:5]}")
        model.tie_weights()
        left_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if left_meta:
            raise RuntimeError(f"gemma load: params left on meta after tie: {left_meta[:5]} (missing={list(missing)[:5]})")
        model = model.to(target).eval()
        base.sync_cuda()
        te = base.MetaSafeGemmaTextEncoder(
            model=model, tokenizer=self._cached_tokenizer, dtype=dtype or torch.bfloat16,
        ).eval()
        base.sync_cuda()
        self.last_build_event = {
            "builder": "OnGPUFp8GemmaBuilder",
            "total_seconds": round(time.perf_counter() - t0, 3),
            "missing_keys_tied": list(missing),
        }
        _mark(f"gemma_on_gpu_build_done ({self.last_build_event['total_seconds']}s)")
        return te


def _warmup_generation():
    """v8.7 F-1: one full prod-shape generation inside init. Absorbs cudnn/cublas autotune,
    lazy CUDA init, stage builds (and thus tail residency) before the first real job —
    first-job exec drops from ~24-30s to steady ~12-16s; the cost moves into init where
    the worker was idle anyway. Real jobs serialize behind _INIT_LOCK."""
    import shutil
    import time as _t
    _kill_prewarm()
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warmup.jpg")
    shutil.copyfile(src, _IN / "warmup")
    _SIGMA_MODE["mode"] = "distilled"
    _SIGMA_MODE["override"] = None
    base.STAGE_2_DISTILLED_SIGMAS = _STAGE2_FAST
    os.environ["LTX_NO_AUDIO"] = "1"
    _SKIP_NEG_NOW["on"] = SKIP_NEG_ENCODE
    case = {"id": "warmup", "file": "warmup", "prompt": DEFAULT_PROMPT, "seed": 3102}
    settings = {"width": 1280, "height": 704, "frames": 121, "fps": 24.0,
                "conditioning_strength": 0.8, "conditioning_crf": 0,
                "dev_inference_steps": FAST_DEFAULT_STEPS}
    t0 = _t.time()
    rec = base.run_case(_PIPE, case, settings, _tier_args("fast", "warmup"),
                        repeat_idx=1, prompt_cache=None, resident_stage_cache=_STAGE_CACHE)
    if rec.get("error"):
        raise RuntimeError(f"warmup run_case error: {str(rec['error'])[:300]}")
    out = _OUT / rec["output"]
    if os.path.exists(out):
        os.remove(out)
    _INIT_LOG.append(f"warmup-gen ok in {_t.time() - t0:.1f}s; timers={rec.get('timers')}")


def _init():
    """Idempotent, thread-safe init. Called eagerly from a daemon thread at process start
    (H-1) and again by every handler invocation (no-op once done; blocks while in-flight)."""
    global _PIPE, _INIT_ERR, _STAGE2_DEFAULT, _GEMMA_DIR
    with _INIT_LOCK:
        if _PIPE is not None or _INIT_ERR is not None:
            return
        try:
            _mark("init_start")
            snap = _resolve_repo()
            _mark("snapshot_resolved")
            ckpt = os.path.join(snap, FP8_CKPT_NAME)
            ups = os.path.join(snap, UPS_NAME)
            lora = os.path.join(snap, LORA_NAME)
            gemma_fp8 = os.path.join(snap, GEMMA_FP8_SUBDIR)
            for p in (ckpt, ups, lora, gemma_fp8):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"missing in snapshot {snap}: {p} (have: {os.listdir(snap)})")
            _GEMMA_DIR = gemma_fp8
            _STAGE2_DEFAULT = base.STAGE_2_DISTILLED_SIGMAS
            gemma_builder = OnGPUFp8GemmaBuilder(model_root=gemma_fp8, tokenizer_root=gemma_fp8)

            def _build_pipe():
                _mark("pipeline_build_start")
                p = TI2VidTwoStagesPipeline(
                    checkpoint_path=ckpt,
                    distilled_lora=[LoraPathStrengthAndSDOps(lora, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)],
                    spatial_upsampler_path=ups, gemma_root=gemma_fp8, loras=[],
                    quantization=QuantizationPolicy.fp8_scaled_mm(ckpt), torch_compile=False,
                    registry=_REGISTRY,
                    distilled_lora_stage_1=(
                        [LoraPathStrengthAndSDOps(lora, S1_LORA_STRENGTH, LTXV_LORA_COMFY_RENAMING_MAP)]
                        if S1_LORA_STRENGTH > 0 else None),
                )
                _mark("pipeline_build_done")
                return p

            # L-6: Gemma streams to GPU while the transformer SD is read from disk.
            # LTX_PARALLEL_INIT=0 serializes the loads (rollback knob: concurrent Gemma +
            # LoRA-fusion spike is the known OOM pattern on smaller cards).
            from concurrent.futures import ThreadPoolExecutor
            if os.environ.get("LTX_PARALLEL_INIT", "1") == "1":
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_te = ex.submit(gemma_builder.build, device=torch.device("cuda"), dtype=torch.bfloat16)
                    f_pipe = ex.submit(_build_pipe)
                    pipe = f_pipe.result()
                    te = f_te.result()
            else:
                pipe = _build_pipe()
                te = gemma_builder.build(device=torch.device("cuda"), dtype=torch.bfloat16)

            # Measurement 2026-06-11: import tensorrt_llm fails on every boot (libpython3.12
            # unreachable in uv-standalone python) and prod has always run torch._scaled_mm.
            # Log the active path; the kernel flip is gated behind the offline A/B.
            from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
            _mark(f"scaled_mm_path={'trtllm' if trtllm_scaled_mm_usable() else 'torch_fallback'}")

            enc = ResidentPromptEncoder(
                ckpt, gemma_fp8, pipe.dtype, pipe.device,
                registry=_REGISTRY, text_encoder_builder=gemma_builder,
            )
            enc._resident_te = te  # pre-seed: first encode skips the build entirely
            pipe.prompt_encoder = enc
            _PIPE = pipe
            _INIT_LOG.append("pipeline built — ready")
            if WARMUP_GEN:
                try:
                    _warmup_generation()
                    _mark("warmup_gen_done")
                except Exception as exc:  # noqa: BLE001
                    _INIT_LOG.append(f"warmup-gen FAILED (non-fatal): {exc!r}")
                    _mark("warmup_gen_failed")
            _mark("ready")
        except Exception:
            _INIT_ERR = "INIT LOG:\n" + "\n".join(_INIT_LOG) + "\n\nTRACEBACK:\n" + traceback.format_exc()
            _mark("init_FAILED")


def _tier_args(tier: str, label: str) -> types.SimpleNamespace:
    fast = tier == "fast"
    return types.SimpleNamespace(
        inputs_dir=_IN, outputs_dir=_OUT, trace_root=_OUT, guidance_trace_dir=None, label=label,
        gemma_root=_GEMMA_DIR, gemma_prequant_root=_GEMMA_DIR, gemma_8bit=False, gemma_cache=False,
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
    global _FIRST_JOB_SEEN
    if not _FIRST_JOB_SEEN:
        _FIRST_JOB_SEEN = True
        _mark("first_job_received")
    _init()
    if _INIT_ERR:
        return {"error": "INIT_FAILED", "trace": _INIT_ERR}
    inp = job.get("input", {})
    if inp.get("warm"):  # H-7: keep-warm ping — init done, no generation; prewarm keeps running
        return {"warm": "ok", "config_tag": CONFIG_TAG}
    _kill_prewarm()  # L-4: a real generation owns the disk from here
    try:
        tier = str(inp.get("tier", "fast")).lower()
        if tier not in ("fast", "quality"):
            tier = "fast"
        audio_on = bool(inp.get("audio", False))  # H-3 (owner-approved 2026-06-11): audio off by default, −3.3s
        steps = int(inp.get("steps", FAST_DEFAULT_STEPS if tier == "fast" else 16))
        sig = inp.get("sigmas")  # explicit stage1 grid, fast tier only (sigma-sweep lever)
        if sig is None and tier == "fast" and FAST_SIGMAS_DEFAULT is not None:
            sig = list(FAST_SIGMAS_DEFAULT)  # env default (e.g. A13); validated below like per-request
        if sig is not None:
            if tier != "fast":
                return {"error": "sigmas supported on fast tier only", "config_tag": f"{CONFIG_TAG}-{tier}"}
            try:
                sig = [float(x) for x in sig]
            except (TypeError, ValueError):
                return {"error": "sigmas must be a list of floats", "config_tag": f"{CONFIG_TAG}-{tier}"}
            ok = (6 <= len(sig) <= 34 and abs(sig[-1]) < 1e-9 and 0.9 <= sig[0] <= 1.0
                  and all(sig[i] > sig[i + 1] for i in range(len(sig) - 1)))
            if not ok:
                return {"error": "bad sigmas: need 0.9<=first<=1.0, strictly descending, last 0, len 6..34",
                        "config_tag": f"{CONFIG_TAG}-{tier}"}
            sig[-1] = 0.0
            steps = len(sig) - 1

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
        # L-5 gate: resident floor ~71-73 GiB; larger jobs would OOM even uncached
        # (floor stays + their own 22.6 GiB fuse transient + ~2.3x activations).
        if settings["width"] * settings["height"] > _MAX_PIXELS or settings["frames"] > _MAX_FRAMES:
            return {"error": "resolution_gated",
                    "max": "1280x704x121 (L-5 resident-cache VRAM floor)",
                    "config_tag": f"{CONFIG_TAG}-{tier}"}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # keep peak_vram_gib per-request-true
        # per-request config switches (handler is single-threaded)
        _SIGMA_MODE["mode"] = "distilled" if tier == "fast" else "ltx2"
        _SIGMA_MODE["override"] = sig if tier == "fast" else None
        _SKIP_NEG_NOW["on"] = SKIP_NEG_ENCODE and tier == "fast" and not audio_on
        base.STAGE_2_DISTILLED_SIGMAS = _STAGE2_FAST if tier == "fast" else _STAGE2_DEFAULT
        if audio_on:
            os.environ.pop("LTX_NO_AUDIO", None)
        else:
            os.environ["LTX_NO_AUDIO"] = "1"

        rec = base.run_case(_PIPE, case, settings, _tier_args(tier, "req"),
                            repeat_idx=1, prompt_cache=None, resident_stage_cache=_STAGE_CACHE)
        if rec.get("error"):
            return {"error": rec["error"], "init_log": _INIT_LOG, "config_tag": f"{CONFIG_TAG}-{tier}"}
        out = _OUT / rec["output"]
        with open(out, "rb") as f:
            raw = f.read()
        os.remove(out)
        resp = {
            "peak_vram_gib": rec.get("peak_vram_gib"),
            "elapsed_seconds": rec.get("elapsed_seconds"),
            "timers": rec.get("timers"),
            "config_tag": (f"{CONFIG_TAG}-{tier}" + ("" if audio_on else "-noaudio")
                           + (f"-sig{steps}" if sig else "")),
            "tier": tier,
        }
        if S3_ON:
            try:
                import uuid
                url = _s3_put(raw, f"videos/{uuid.uuid4().hex}.mp4", "video/mp4")
                resp["video_url"] = url
                resp["video_cdn_url"] = url.replace(".digitaloceanspaces.com", ".cdn.digitaloceanspaces.com")
            except Exception as exc:  # noqa: BLE001
                resp["upload_error"] = str(exc)[:300]
        if not (RETURN_URL_ONLY and resp.get("video_url")):
            resp["video_b64"] = base64.b64encode(raw).decode()
        return resp
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "trace": traceback.format_exc()}


if os.environ.get("LTX_SKIP_START") != "1":
    if os.environ.get("LTX_EAGER_INIT", "1") == "1":
        threading.Thread(target=_init, daemon=True, name="eager-init").start()
    runpod.serverless.start({"handler": handler})
