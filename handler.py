"""LTX-2.3 RunPod serverless handler v8.20 — v8.19 + meme motion-control mode (Union IC-LoRA).

v8.20 (2026-07-17):
- MEME MODE (env LTX_MEME=1): request {image_b64, reference_video_b64} => DWPose skeleton of the
  reference video rides the Union IC-LoRA as reference conditioning; first frame = the character
  image; LTX audio off, the meme's own soundtrack is remuxed onto the output. The Union LoRA is
  fused via the existing LTX_EXTRA_LORA_* lever (its reference_downscale_factor is read from the
  safetensors metadata). Dims must be divisible by 128; frames follow the meme (cap 241 @24fps).
  Non-meme endpoints (LTX_MEME unset) are bit-exact: no dwpose import, no new request fields.

v8.19 (2026-07-16):
- Terminate the phrase handed to the upsampler. Upstream's enhance_t2v frames the user turn as
  "user prompt: <text>" with no terminator, so Gemma completes the open phrase instead of (or
  before) answering: "football player" -> "and a football on a field." + EOS. Closing it fixes
  the rewrite on every seed measured (8/8 clean vs 1/8 junk-only + 2/8 junk-prefixed).
- Belts, because sampling stays stochastic and a bad rewrite REPLACES the user's prompt:
  re-roll a degenerate (<40 word) draw, then fall back to the raw prompt; strip leading junk.

v8.18 (2026-07-16):
- t2v jobs now run the in-worker upsampler too: text-only Gemma enhance_t2v with the official
  LTX t2v system prompt (raw short prompts at a fixed seed collapsed into one prior scene —
  the "man chopping food" bug). Env LTX_ENHANCE_T2V=0 is the kill-switch (i2v path untouched);
  LTX_ENHANCE_T2V_SYSTEM_PROMPT overrides the shipped prompt. Enhancement is seeded from the
  request seed so re-rolls explore different interpretations.

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
import math
import os
import re
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
from request_config import (  # noqa: E402
    resolve_boolean,
    resolve_decode_noise,
    resolve_negative_prompt,
    resolve_optional_number,
)
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: E402
from ltx_core.loader.registry import StateDictRegistry  # noqa: E402
from ltx_core.quantization import QuantizationPolicy  # noqa: E402
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # noqa: E402
from ltx_pipelines.utils.blocks import PromptEncoder  # noqa: E402
from ltx_pipelines.utils.constants import (  # noqa: E402
    DEFAULT_NEGATIVE_PROMPT,
    DISTILLED_SIGMA_VALUES,
    LTX_2_3_PARAMS,
)

# v8.10 F1 (NATIVE-QUALITY-PLAN root-cause fix): gradient-estimating Euler on stage1.
# 2nd-order AB2 velocity correction reuses the cached previous-step velocity (NO extra
# transformer call => 0 wall-time) to stop 1st-order Euler from chording the curved motion
# band — the root of the snap (frozen->ramp on fountain/mage) + fast-motion blur (samurai).
# The selected loop is passed to stage 1 explicitly. Stage 2 always receives the
# stock Euler loop, avoiding the old process-global blocks-module monkeypatch.
# The endpoint env remains the default; per-request "stage1_ge" can override it.
_STAGE1_GE_DEFAULT = os.environ.get("LTX_STAGE1_GE") == "1"
_CFG_CACHE_ENV = os.environ.get("LTX_CFG_CACHE") == "1"  # same: an endpoint-level LTX_CFG_CACHE=1 must
# survive requests that omit the per-request "cfg_cache" flag (else the per-request block below would
# pop it on every such request and the env default never engages — the v8.13 30s-instead-of-Q5 bug).
# Cache-tuning env defaults (interval/warmup/range), captured so per-request knobs restore them when
# omitted (single-threaded handler => without this a prior request's tuning leaks to the next).
_CFG_CACHE_INTERVAL_ENV = os.environ.get("LTX_CFG_CACHE_INTERVAL")
_CFG_CACHE_WARMUP_ENV = os.environ.get("LTX_CFG_CACHE_WARMUP")
_CFG_CACHE_RANGE_ENV = os.environ.get("LTX_CFG_CACHE_RANGE")
_DETAIL_DAEMON_ENV = os.environ.get("LTX_DETAIL_DAEMON")  # preserve env default across requests (v8.15)

_mark("imports_done")

WEIGHTS_REPO = os.environ["LTX_WEIGHTS_REPO"]
FP8_CKPT_NAME = os.environ.get("LTX_FP8_CKPT_NAME", "ltx-2.3-22b-dev-fp8.safetensors")
UPS_NAME = os.environ.get("LTX_UPSCALER_NAME", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
LORA_NAME = os.environ.get("LTX_DISTILLED_LORA_NAME", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
GEMMA_FP8_SUBDIR = os.environ.get("LTX_GEMMA_FP8_SUBDIR", "gemma-fp8")
# Official HQ recipe fuses the distilled LoRA into stage1 too (default strength there: 0.25).
# 0 = off = bit-identical to v8.5 behavior; flipping requires a worker restart (fusion at init).
S1_LORA_STRENGTH = float(os.environ.get("LTX_S1_LORA_STRENGTH", "0"))
# v8.15 QUALITY EXPERIMENT — optional EXTRA LoRA (e.g. VBVR motion/reasoning), fused at init alongside
# the distilled LoRA via the same proven path. All unset/0 => no extra LoRA => bit-exact to v8.14.
# Downloaded from its own HF repo at init only when LTX_EXTRA_LORA_REPO is set. Independent s1/s2 strength.
EXTRA_LORA_REPO = os.environ.get("LTX_EXTRA_LORA_REPO") or None      # e.g. LiconStudio/Ltx2.3-VBVR-lora-I2V
EXTRA_LORA_FILE = os.environ.get("LTX_EXTRA_LORA_FILE") or None      # e.g. Ltx2.3-Licon-VBVR-I2V-390K-R32.safetensors
EXTRA_LORA_S1 = float(os.environ.get("LTX_EXTRA_LORA_S1", "0"))      # stage1 strength
EXTRA_LORA_S2 = float(os.environ.get("LTX_EXTRA_LORA_S2", "0"))      # stage2 refiner strength
# Default stage1 step count for the fast tier; request "steps" still wins. 12 = v8.5 behavior.
FAST_DEFAULT_STEPS = int(os.environ.get("LTX_FAST_DEFAULT_STEPS", "12"))
# Default stage1 sigma grid for the fast tier (JSON list). When set, it replaces the
# resampled-distilled default and pins step count to len-1; a per-request "sigmas" still wins.
# Validated through the same path as per-request sigmas. e.g. A13 anchor-preserving grid.
try:
    FAST_SIGMAS_DEFAULT = json.loads(os.environ["LTX_FAST_SIGMAS"]) if os.environ.get("LTX_FAST_SIGMAS") else None
except (ValueError, TypeError):
    FAST_SIGMAS_DEFAULT = None
# v8.8 prompt-enhancement (in-worker Gemma-3 i2v rewriter). All OFF by default => prod bit-exact.
# LTX_ENHANCE_VISION=1 loads the FULL Gemma3ForConditionalGeneration (vision_tower + projector)
#   instead of the text-only Gemma3ForCausalLM, so enhance_i2v (image-aware) works. Costs VRAM+init.
# LTX_ENHANCE=1 makes enhancement the per-request default; request {"enhance": true/false} always wins.
# LTX_ENHANCE_SYSTEM_PROMPT overrides the shipped gemma_i2v system prompt (e.g. physics-momentum).
ENHANCE_VISION = os.environ.get("LTX_ENHANCE_VISION", "0") == "1"
ENHANCE_DEFAULT = os.environ.get("LTX_ENHANCE", "0") == "1"
ENHANCE_SYS = os.environ.get("LTX_ENHANCE_SYSTEM_PROMPT") or None
ENHANCE_MAX_TOKENS = int(os.environ.get("LTX_ENHANCE_MAX_TOKENS", "400"))
# v8.18 t2v enhancement (text-only Gemma rewrite; no image involved). Rides the same ENHANCE_DEFAULT/
# ENHANCE_VISION gates as i2v (full Gemma has the lm_head that generate() needs). LTX_ENHANCE_T2V=0
# disables ONLY the t2v rewrite (env-level rollback without an image rollback).
ENHANCE_T2V = os.environ.get("LTX_ENHANCE_T2V", "1") == "1"
ENHANCE_T2V_SYS = os.environ.get("LTX_ENHANCE_T2V_SYSTEM_PROMPT") or None
# v8.20 MEME MODE (motion control): request carries reference_video_b64 + image_b64; the worker
# extracts a DWPose skeleton video and rides it through the Union IC-LoRA reference conditioning.
# OFF by default => the regular video endpoint stays bit-exact and never loads dwpose/onnxruntime.
# The Union LoRA itself is fused via the existing LTX_EXTRA_LORA_* lever (same proven fuse path);
# a meme endpoint = same image, env: LTX_MEME=1 + LTX_EXTRA_LORA_REPO/FILE/S1/S2.
MEME_MODE = os.environ.get("LTX_MEME", "0") == "1"
MEME_MAX_FRAMES = int(os.environ.get("LTX_MEME_MAX_FRAMES", "241"))  # 10s @ 24fps, 8k+1
DWPOSE_REPO = os.environ.get("LTX_DWPOSE_REPO", "RedHash/DWPose")
_DWPOSE = {"det": None}       # built at init when LTX_MEME=1 (single-threaded handler)
_REF_DOWNSCALE = {"factor": 1}  # read from the extra LoRA's safetensors metadata at init
# Degenerate-rewrite guard (see _enhance_prompt_t2v): re-roll a too-short rewrite, then give up
# and use the raw prompt. 40 words is well under a real rewrite (~125-210) and well over the
# junk draws (2-6). Each retry costs ~8s of upsampler on an otherwise ~25-45s job.
ENHANCE_T2V_TRIES = int(os.environ.get("LTX_ENHANCE_T2V_TRIES", "2"))
ENHANCE_T2V_MIN_WORDS = int(os.environ.get("LTX_ENHANCE_T2V_MIN_WORDS", "40"))
# v8.22: retarget the meme skeleton into the character photo's composition (see
# _pose_control_video). Per-request "retarget": false for A/B; env kill-switch here.
MEME_RETARGET = os.environ.get("LTX_MEME_RETARGET", "1") == "1"
# v8.23 FACE RESTORE: pose-driven regeneration cannot preserve a specific real face — frame 0
# is the photo, later frames reinvent it. Post-pass swaps the source face back onto every
# frame (insightface buffalo_l + inswapper_128). OFF by default: inswapper's license is
# research/non-commercial — enabling in prod is the owner's call. Cartoons auto-skip (no face).
FACE_RESTORE = os.environ.get("LTX_FACE_RESTORE", "0") == "1"
FACE_SWAP_REPO = os.environ.get("LTX_FACE_SWAP_REPO", "ezioruan/inswapper_128.onnx")
FACE_SWAP_FILE = os.environ.get("LTX_FACE_SWAP_FILE", "inswapper_128.onnx")
_FACE = {"app": None, "swapper": None}
# v8.9 CAS (Contrast-Adaptive Sharpen) — DEFAULT-ON post-decode crispness pass (research 2026-06-17:
#   dominant artifacts = thin-structure dissolve + sharpness collapse on fast motion). Applied to the
#   pixel chunks before H264 encode in stage_timing_runner (~+0.6s). Tunable; request {"cas_amount":0}
#   disables for A/B. Defaults live in stage_timing_runner (_CAS_AMOUNT_DEFAULT / _CAS_MIX_DEFAULT).
CAS_AMOUNT_DEFAULT = float(os.environ.get("LTX_CAS_AMOUNT", "0.6"))
CAS_MIX_DEFAULT = float(os.environ.get("LTX_CAS_MIX", "0.7"))
CONFIG_TAG = (os.environ.get("LTX_CONFIG_TAG", "v8")
              + (f"-s1_{S1_LORA_STRENGTH:g}" if S1_LORA_STRENGTH > 0 else "")
              + (f"-xlora{EXTRA_LORA_S1:g}_{EXTRA_LORA_S2:g}" if EXTRA_LORA_REPO else "")
              + (f"-st{FAST_DEFAULT_STEPS}" if FAST_DEFAULT_STEPS != 12 else "")
              + (f"-fsig{len(FAST_SIGMAS_DEFAULT) - 1}" if FAST_SIGMAS_DEFAULT else "")
              + (f"-cas{CAS_AMOUNT_DEFAULT:g}" if CAS_AMOUNT_DEFAULT > 0 else ""))
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
# v8.10: env-overridable so a test deploy can lift the gate for C5 (1408x768) / H200-1080p
# WITHOUT a rebuild. Unset => bit-identical default (1280*704 / 121). VRAM caveat still holds:
# only lift on a card with headroom (H200), never on the 94GB H100 NVL.
_MAX_PIXELS = int(os.environ.get("LTX_MAX_PIXELS", str(1280 * 704)))
_MAX_FRAMES = int(os.environ.get("LTX_MAX_FRAMES", "121"))


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
        self._default_decode_noise_scale = float(self._dec.decode_noise_scale)
        self.last_decode_noise_scale = self._default_decode_noise_scale

    def __call__(
        self,
        latent,
        tiling_config=None,
        generator=None,
        decode_noise_scale=None,
    ):
        # The stock VideoDecoder builds a fresh decoder per call, so an omitted
        # request override naturally returns to the checkpoint default. This
        # resident wrapper must reset explicitly or one request's override
        # would leak into the next request.
        resolved = (
            self._default_decode_noise_scale
            if decode_noise_scale is None
            else float(decode_noise_scale)
        )
        if not math.isfinite(resolved) or not 0.0 <= resolved <= 0.25:
            raise ValueError(
                "decode_noise_scale must be finite and between 0 and 0.25"
            )
        self._dec.decode_noise_scale = resolved
        self.last_decode_noise_scale = resolved
        return self._dec.decode_video(latent, tiling_config, generator)


# Min free VRAM (GiB) required to KEEP the tail models resident. Resident tail (~3.5 GiB) that
# also stays live through decode raises the generation peak; on the 94 GB H100 NVL that tips a
# cold first-job over the ceiling (observed: 93.3 GiB peak -> OOM). H200 (141 GB) has ample room.
# So pin only when headroom is large; otherwise fall back to the per-job build-and-free tail
# (v8.6.1 behavior, NVL-safe). Both paths are bit-exact — same weights, same op order.
TAIL_RESIDENT_MIN_FREE_GIB = float(os.environ.get("LTX_TAIL_RESIDENT_MIN_FREE_GIB", "45"))


def _make_tail_resident():
    """Pin tail blocks resident ONLY when VRAM headroom is large (H200). Bit-exact either way."""
    if _TAIL_RESIDENT["done"] or _PIPE is None:
        return
    _TAIL_RESIDENT["done"] = True  # decide once; never thrash per job
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        free_gib = free_b / 2**30
        if free_gib < TAIL_RESIDENT_MIN_FREE_GIB:
            _INIT_LOG.append(
                f"tail NOT pinned: free={free_gib:.1f}GiB < {TAIL_RESIDENT_MIN_FREE_GIB}GiB "
                f"(total={total_b / 2**30:.0f}GiB) — per-job tail keeps peak NVL-safe")
            return
        _PIPE.image_conditioner = _ResidentConditioner(_PIPE.image_conditioner)
        _PIPE.upsampler = _ResidentUpsampler(_PIPE.upsampler)
        _PIPE.video_decoder = _ResidentDecoder(_PIPE.video_decoder)
        _INIT_LOG.append(
            f"tail resident (free was {free_gib:.1f}GiB); cuda_alloc={torch.cuda.memory_allocated() / 2**30:.1f}GiB")
    except Exception as exc:  # noqa: BLE001
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
# v8.10 S1 lever: stage2 fast-tier sigmas overridable via env LTX_STAGE2_SIGMAS (JSON list) so a
# sweep can test the +1-step grid [0.909375, 0.725, 0.421875, 0.0] WITHOUT a rebuild. Unset =>
# bit-identical default [0.909375, 0.6, 0.0]. Consumed wholesale (only [0] indexed for noise_scale);
# step count = len(sigmas)-1.
try:
    _stage2_env = json.loads(os.environ["LTX_STAGE2_SIGMAS"]) if os.environ.get("LTX_STAGE2_SIGMAS") else None
except (ValueError, TypeError):
    _stage2_env = None
if _stage2_env is not None:
    _vals = [float(x) for x in _stage2_env]
    if not (len(_vals) >= 2 and abs(_vals[-1]) < 1e-9
            and all(_vals[i] > _vals[i + 1] for i in range(len(_vals) - 1))):
        raise ValueError(f"LTX_STAGE2_SIGMAS must be strictly descending with last==0: {_vals}")
    _STAGE2_FAST = torch.tensor(_vals)
else:
    _STAGE2_FAST = torch.tensor([0.909375, 0.6, 0.0])
# v8.10: augment CONFIG_TAG with the new knobs (defined here, after _STAGE2_FAST; no NameError)
CONFIG_TAG += ((f"-mpx{_MAX_PIXELS}" if _MAX_PIXELS != 1280 * 704 else "")
               + (f"-mfr{_MAX_FRAMES}" if _MAX_FRAMES != 121 else "")
               + (f"-s2len{len(_STAGE2_FAST)}" if len(_STAGE2_FAST) != 3 else ""))

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
        # v8.20.2: hosts WITHOUT console-pinned Cached Models hit this fallback, but hub freezes
        # HF_HUB_OFFLINE into `constants` AT IMPORT TIME — popping the env var at call time does
        # nothing (v8.20.1 mistake; log_uploader dodges it by scrubbing before import in a fresh
        # process). Flip the constant itself for the fetch, then restore the offline fast path.
        import huggingface_hub.constants as _hfc

        prev_flag = _hfc.HF_HUB_OFFLINE
        prev_env = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        _hfc.HF_HUB_OFFLINE = False
        try:
            return snapshot_download(WEIGHTS_REPO, token=os.environ.get("HF_TOKEN"))
        finally:
            _hfc.HF_HUB_OFFLINE = prev_flag
            for k, v in prev_env.items():
                if v is not None:
                    os.environ[k] = v


class _VisionGemmaEncoder(base.MetaSafeGemmaTextEncoder):
    """Full multimodal Gemma (for enhance_i2v). encode() routes through .language_model directly
    so text-encoding stays bit-exact with the text-only Gemma3ForCausalLM path (same submodule,
    same weights, same forward) — enabling vision must NOT change generation output."""

    def encode(self, text, padding_side="left"):  # noqa: ARG002
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        outputs = self.model.model.language_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states, attention_mask


class OnGPUFp8GemmaBuilder(base.PrequantCausalGemmaBuilder):
    """L-2: stream fp8 shards straight to the GPU and dequant there — no disk round-trip.

    Math is identical to the old CPU dequant ((w.fp32 * scale.fp32).bf16) and the resident
    format is the same bf16, so outputs must match the v7 path bit-for-bit. attn impl is
    pinned to "sdpa" because from_pretrained auto-selects it while a bare constructor
    defaults to "eager" — an unpinned mismatch would silently change encoder numerics.
    """

    def build(self, device=None, dtype=None, **_: object):
        if ENHANCE_VISION:
            return self._build_with_vision(device, dtype)
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

    def _build_with_vision(self, device=None, dtype=None):
        """v8.8: load the FULL Gemma3ForConditionalGeneration (vision_tower + projector + lm_head)
        so enhance_i2v (image-aware prompt rewrite) works. Key mapping + computed buffers mirror
        LTX-2's encoder_configurator (GEMMA_LLM_KEY_OPS + create_and_populate). encode() is
        overridden to route through .language_model directly => bit-exact with the text-only path."""
        from accelerate import init_empty_weights
        from ltx_core.utils import find_matching_file
        from transformers import AutoImageProcessor, Gemma3Config, Gemma3ForConditionalGeneration, Gemma3Processor
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX

        target = device or torch.device("cuda")
        t0 = time.perf_counter()
        model_folder = find_matching_file(self._model_root, "model*.safetensors").parent
        gcfg = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
        gcfg._attn_implementation = "sdpa"
        with init_empty_weights(include_buffers=False):
            model = Gemma3ForConditionalGeneration(gcfg)

        def remap(k):
            if k.startswith("language_model.model."):
                return "model.language_model." + k[len("language_model.model."):]
            if k.startswith("vision_tower."):
                return "model.vision_tower." + k[len("vision_tower."):]
            if k.startswith("multi_modal_projector."):
                return "model.multi_modal_projector." + k[len("multi_modal_projector."):]
            return None  # drop language_model.lm_head.* etc; lm_head is duplicated from embed below

        sd, scales = {}, {}
        for shard in sorted(glob.glob(os.path.join(str(model_folder), "*.safetensors"))):
            with safe_open(shard, framework="pt", device=str(target)) as f:
                for key in f.keys():
                    is_scale = key.endswith(".fp8_scale")
                    base_key = key[: -len(".fp8_scale")] if is_scale else key
                    nk = remap(base_key)
                    if nk is None:
                        continue
                    t = f.get_tensor(key)
                    (scales if is_scale else sd)[nk] = t
        for k, s in scales.items():
            if k in sd:
                sd[k] = (sd[k].to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)
        emb = sd.get("model.language_model.embed_tokens.weight")
        if emb is not None:
            sd["lm_head.weight"] = emb
        missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
        if unexpected:
            raise RuntimeError(f"enhance-gemma unexpected keys: {list(unexpected)[:8]}")
        # computed buffers (mirror encoder_configurator.create_and_populate)
        v_model = model.model.vision_tower.vision_model
        l_model = model.model.language_model
        tcfg = model.config.text_config
        dim = getattr(tcfg, "head_dim", tcfg.hidden_size // tcfg.num_attention_heads)
        local_rope = 1.0 / (tcfg.rope_local_base_freq ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float) / dim))
        inv_freqs, _ = ROPE_INIT_FUNCTIONS[tcfg.rope_scaling["rope_type"]](tcfg)
        plen = len(v_model.embeddings.position_ids[0])
        v_model.embeddings.register_buffer("position_ids", torch.arange(plen, dtype=torch.long).unsqueeze(0))
        l_model.embed_tokens.register_buffer("embed_scale", torch.tensor(tcfg.hidden_size ** 0.5))
        l_model.rotary_emb_local.register_buffer("inv_freq", local_rope)
        l_model.rotary_emb.register_buffer("inv_freq", inv_freqs)
        left_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if left_meta:
            raise RuntimeError(f"enhance-gemma params left on meta: {left_meta[:5]} (missing={list(missing)[:5]})")
        model = model.to(target).eval()
        # The slim image ships NO C compiler. Gemma3.generate() defaults to a static/"hybrid" KV
        # cache whose forward is torch.compile'd (Triton -> needs cc) and crashes with
        # "Failed to find C compiler". Force the eager dynamic cache so enhance never compiles.
        # (Our video generation never hits this — it runs FA3 + torch._scaled_mm, torch_compile=False.)
        model.generation_config.cache_implementation = "dynamic"
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True  # belt: any stray compile falls back to eager
        except Exception:  # noqa: BLE001
            pass
        base.sync_cuda()
        proc_root = str(find_matching_file(self._model_root, "preprocessor_config.json").parent)
        image_processor = AutoImageProcessor.from_pretrained(proc_root, local_files_only=True)
        processor = Gemma3Processor(image_processor=image_processor, tokenizer=self._cached_tokenizer.tokenizer)
        te = _VisionGemmaEncoder(model=model, tokenizer=self._cached_tokenizer, processor=processor,
                                 dtype=dtype or torch.bfloat16).eval()
        base.sync_cuda()
        self.last_build_event = {"builder": "OnGPUFp8GemmaBuilder+vision",
                                 "total_seconds": round(time.perf_counter() - t0, 3),
                                 "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
        _mark(f"gemma_vision_build_done ({self.last_build_event['total_seconds']}s, "
              f"{self.last_build_event['vram_gib']}GiB)")
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
                "dev_inference_steps": FAST_DEFAULT_STEPS,
                "stage1_ge": _STAGE1_GE_DEFAULT}
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

            # optional extra quality LoRA (e.g. VBVR), from its own HF repo
            extra_lora_path = None
            if EXTRA_LORA_REPO and EXTRA_LORA_FILE:
                from huggingface_hub import hf_hub_download
                # v8.16 fix: RunPod cached-model workers force HF_HUB_OFFLINE on, which blocked this
                # download (v8.15 init crash). Scrub it for the fetch (same trick as log_uploader.py).
                os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
                extra_lora_path = hf_hub_download(EXTRA_LORA_REPO, EXTRA_LORA_FILE)
                _INIT_LOG.append(f"extra LoRA {EXTRA_LORA_REPO}/{EXTRA_LORA_FILE} s1={EXTRA_LORA_S1} s2={EXTRA_LORA_S2}")
                # v8.20: IC-LoRAs carry their reference downscale in safetensors metadata (ref0.5 => 2).
                from ltx_pipelines.iclora_utils import read_lora_reference_downscale_factor
                _REF_DOWNSCALE["factor"] = read_lora_reference_downscale_factor(extra_lora_path)
                _INIT_LOG.append(f"reference_downscale_factor={_REF_DOWNSCALE['factor']}")

            if MEME_MODE:  # v8.20: DWPose onnx pair for the meme motion-control preprocessor
                from huggingface_hub import hf_hub_download
                os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
                det_onnx = hf_hub_download(DWPOSE_REPO, "yolox_l.onnx")
                pose_onnx = hf_hub_download(DWPOSE_REPO, "dw-ll_ucoco_384.onnx")
                from dwpose_vendor import DWposeDetector
                _DWPOSE["det"] = DWposeDetector(model_det=det_onnx, model_pose=pose_onnx, device="cuda")
                # v8.23.1: ORT falls back to CPU silently when cuDNN is missing — surface which
                # provider actually won so a slow pose pass is diagnosable from the job response.
                _DWPOSE["provider"] = _DWPOSE["det"].pose_estimation.session_det.get_providers()[0]
                _INIT_LOG.append(f"dwpose ready (meme mode, {_DWPOSE['provider']})")
                if FACE_RESTORE:  # v8.23: source-face swap-back pass (see FACE_RESTORE note)
                    import huggingface_hub.constants as _hfc23
                    _prev23 = _hfc23.HF_HUB_OFFLINE
                    _hfc23.HF_HUB_OFFLINE = False
                    try:
                        swap_onnx = hf_hub_download(FACE_SWAP_REPO, FACE_SWAP_FILE)
                    finally:
                        _hfc23.HF_HUB_OFFLINE = _prev23
                    os.environ.setdefault("INSIGHTFACE_HOME", "/runpod-volume/insightface")
                    from insightface.app import FaceAnalysis
                    from insightface.model_zoo import get_model
                    fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    fa.prepare(ctx_id=0, det_size=(640, 640))
                    _FACE["app"] = fa
                    _FACE["swapper"] = get_model(swap_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    _FACE["provider"] = next(iter(fa.models.values())).session.get_providers()[0]
                    _INIT_LOG.append(f"face-restore ready (buffalo_l + inswapper, {_FACE['provider']})")

            def _build_pipe():
                _mark("pipeline_build_start")
                _map = LTXV_LORA_COMFY_RENAMING_MAP
                stage2_loras = [LoraPathStrengthAndSDOps(lora, 0.8, _map)]
                stage1_loras = ([LoraPathStrengthAndSDOps(lora, S1_LORA_STRENGTH, _map)]
                                if S1_LORA_STRENGTH > 0 else [])
                if extra_lora_path:  # fuse VBVR/etc alongside the distilled LoRA on the requested stages
                    if EXTRA_LORA_S2 > 0:
                        stage2_loras.append(LoraPathStrengthAndSDOps(extra_lora_path, EXTRA_LORA_S2, _map))
                    if EXTRA_LORA_S1 > 0:
                        stage1_loras.append(LoraPathStrengthAndSDOps(extra_lora_path, EXTRA_LORA_S1, _map))
                p = TI2VidTwoStagesPipeline(
                    checkpoint_path=ckpt,
                    distilled_lora=stage2_loras,
                    spatial_upsampler_path=ups, gemma_root=gemma_fp8, loras=[],
                    quantization=QuantizationPolicy.fp8_scaled_mm(ckpt), torch_compile=False,
                    registry=_REGISTRY,
                    distilled_lora_stage_1=(stage1_loras or None),
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


def _enhance_prompt(prompt: str, img_path, max_new_tokens=None, system_prompt=None) -> str:
    """In-worker Gemma-3 i2v prompt rewrite (image-aware). Requires ENHANCE_VISION (full Gemma).
    max_new_tokens / system_prompt: per-request overrides (v8.16) — let the autonomous loop A/B the
    upsampler directive (e.g. static-camera single-action) WITHOUT a rebuild. Falls back to env then default."""
    from PIL import Image
    te = _PIPE.prompt_encoder._resident_te
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = 896 / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    sysp = system_prompt or ENHANCE_SYS or te.default_gemma_i2v_system_prompt
    mnt = int(max_new_tokens) if max_new_tokens else ENHANCE_MAX_TOKENS
    return te.enhance_i2v(prompt, img, system_prompt=sysp, max_new_tokens=mnt).strip()


# Leading junk the upsampler can emit (v8.19): _enhance slices the output at the PADDED input
# length, so up to 7 leading tokens are eaten and a bare ". " / "*" is left behind; and a
# completion fragment can still slip in front of the real rewrite. Every real rewrite opens with
# a capital ("Style: ...", "A young ...", "In a medium close-up ..."), so a lowercase opening
# sentence is junk by construction. Bounded quantifiers — this runs on model output.
_LEADING_PUNCT_RE = re.compile(r'^[\s.,;:*"\'\-—]+')
# The inner alternation lets a fragment cross punctuation inside quotes (knight says "For the
# kingdom!"); LAZY so it ends at the FIRST sentence boundary — greedy ate the real answer too.
_LEADING_FRAGMENT_RE = re.compile(r'^[a-z](?:[^.!?]|[.!?](?=["\')])){0,200}?[.!?]+["\')\]]*\s+(?=[A-Z*])')


def _strip_leading_junk(text: str) -> str:
    out = _LEADING_PUNCT_RE.sub("", text)
    out = _LEADING_FRAGMENT_RE.sub("", out)
    return _LEADING_PUNCT_RE.sub("", out).strip()


def _enhance_prompt_t2v(prompt: str, seed: int, max_new_tokens=None, system_prompt=None) -> str:
    """In-worker Gemma-3 t2v prompt rewrite (v8.18, text-only — no image in the request).
    Seeded from the request seed so identical prompts with different seeds get different
    rewrites. Same per-request overrides as i2v (enhance_max_tokens / enhance_system_prompt).

    v8.19 TERMINATOR: upstream's enhance_t2v frames the user turn as "user prompt: <text>" with
    NO terminator, leaving an open phrase — so Gemma COMPLETES it before (or instead of)
    answering: "football player" -> "and a football on a field." + EOS (6 words), or
    "red dragon" -> "and knight fighting in a forest, knight says \"For the kingdom!\"" glued in
    front of the real rewrite, dragging a knight into the video. enhance_i2v does not show this
    — its framing ends in "." — and closing the phrase here fixes it: measured on the exact seed
    that returned 6 words of junk, "football player." returns a clean 162-word rewrite.
    Retry guard stays as a belt: sampling is stochastic, and a fragment REPLACES the user's
    prompt, so a too-short draw is worse than no rewrite at all."""
    te = _PIPE.prompt_encoder._resident_te
    sysp = system_prompt or ENHANCE_T2V_SYS or te.default_gemma_t2v_system_prompt
    mnt = int(max_new_tokens) if max_new_tokens else ENHANCE_MAX_TOKENS
    closed = prompt.strip()
    if closed and closed[-1] not in ".!?…\"')":
        closed += "."
    for attempt in range(ENHANCE_T2V_TRIES):
        out = te.enhance_t2v(closed, system_prompt=sysp, max_new_tokens=mnt,
                             seed=seed + attempt * 7919).strip()  # prime stride => independent draws
        out = _strip_leading_junk(out)
        if len(out.split()) >= ENHANCE_T2V_MIN_WORDS:
            return out
        _INIT_LOG.append(f"t2v enhance degenerate ({len(out.split())}w) on try {attempt + 1}: {out[:80]!r}")
    return prompt  # raw prompt beats a fragment that would replace it


def _meme_frames(src_path, fps: float) -> int:
    """Frame budget from the reference video: min(ref duration, cap) @ fps, snapped to 8k+1."""
    import cv2
    cap = cv2.VideoCapture(str(src_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    duration = (count / src_fps) if count > 0 else 5.0
    n = int(duration * fps)
    n = max(49, min(n, MEME_MAX_FRAMES))
    return (n - 1) // 8 * 8 + 1


def _pose_control_video(src_path, char_path, width: int, height: int, n_frames: int, fps: float,
                        retarget: bool = True):
    """Reference video -> DWPose skeleton control video (v8.20 meme mode; v8.22 retargeting).

    v8.21: only the DOMINANT person is rendered (meme audiences materialised otherwise).
    v8.22 RETARGETING: the skeleton is scaled+anchored INTO the character photo's composition.
    Without it, frame-0 conditioning (the photo) and the skeleton (meme's position/scale)
    contradict each other and the model resolves the conflict with a hard scene cut ~0.5s in,
    repainting the character and background (prod posts 27278-27280). WAN-Animate ships the
    same alignment step natively. Affine = uniform scale matching body-bbox heights + feet
    bottom-center anchor; fallback chain: char skeleton -> yolox person bbox -> no retarget.

    Returns (path, retarget_mode)."""
    import cv2
    import numpy as np

    det = _DWPOSE["det"]
    est = det.pose_estimation

    def _fit_to_canvas(img):
        fh, fw = img.shape[:2]
        target_ar = width / height
        if fw / fh > target_ar:
            cw = max(2, int(fh * target_ar))
            x0 = (fw - cw) // 2
            img = img[:, x0:x0 + cw]
        else:
            ch = max(2, int(fw / target_ar))
            y0 = (fh - ch) // 2
            img = img[y0:y0 + ch, :]
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def _dominant_raw(rgb):
        """(candidates[1,134,2], scores[1,134]) of the most prominent person, else None."""
        candidates, scores = est(rgb)
        if candidates.shape[0] == 0:
            return None
        best, best_val = -1, -1.0
        for i in range(candidates.shape[0]):
            body, sc = candidates[i, :18], scores[i, :18]
            vis = sc > 0.3
            if vis.sum() < 4:
                continue
            bxs, bys = body[vis, 0], body[vis, 1]
            val = float((bxs.max() - bxs.min()) * (bys.max() - bys.min()) * sc[vis].mean())
            if val > best_val:
                best, best_val = i, val
        if best < 0:
            return None
        return candidates[best:best + 1].copy(), scores[best:best + 1].copy()

    def _body_bbox(cand, sc):
        body, s = cand[0, :18], sc[0, :18]
        vis = s > 0.3
        bxs, bys = body[vis, 0], body[vis, 1]
        return float(bxs.min()), float(bys.min()), float(bxs.max()), float(bys.max())

    cap = cv2.VideoCapture(str(src_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise ValueError("reference video decoded to zero frames")

    # Detect once per SOURCE frame (output frames repeat sources when ref fps < target).
    raw_cache: dict = {}

    def _raw_for(idx):
        if idx not in raw_cache:
            canvas = _fit_to_canvas(frames[idx])
            raw_cache[idx] = _dominant_raw(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        return raw_cache[idx]

    sample_idx = [min(int(round(i / fps * src_fps)), len(frames) - 1) for i in range(n_frames)]

    # --- retarget affine (v8.22) ---
    s_scale, tx, ty, mode = 1.0, 0.0, 0.0, "off"
    if retarget:
        char_bgr = cv2.imread(str(char_path))
        if char_bgr is not None:
            char_canvas = _fit_to_canvas(char_bgr)
            char_rgb = cv2.cvtColor(char_canvas, cv2.COLOR_BGR2RGB)
            ref_boxes = []
            for idx in sorted(set(sample_idx[: max(1, len(sample_idx) // 2)])):
                raw = _raw_for(idx)
                if raw is not None:
                    ref_boxes.append(_body_bbox(*raw))
            if ref_boxes:
                rx0, ry0, rx1, ry1 = [float(v) for v in np.median(np.array(ref_boxes), axis=0)]
                anchor = None
                char_raw = _dominant_raw(char_rgb)
                if char_raw is not None:
                    cx0, cy0, cx1, cy1 = _body_bbox(*char_raw)
                    if (cy1 - cy0) > 8:
                        anchor, mode = (cx0, cy0, cx1, cy1), "skeleton"
                if anchor is None:  # cartoons DWPose can miss; yolox person bbox still often fires
                    from dwpose_vendor.body_estimation.detector import inference_detector
                    boxes = inference_detector(est.session_det, char_rgb)
                    if boxes is not None and len(boxes):
                        b = max(boxes, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]))
                        anchor, mode = (float(b[0]), float(b[1]), float(b[2]), float(b[3])), "bbox"
                if anchor is not None and (ry1 - ry0) > 8:
                    cx0, cy0, cx1, cy1 = anchor
                    s_scale = float(np.clip((cy1 - cy0) / (ry1 - ry0), 0.4, 2.5))
                    tx = (cx0 + cx1) / 2.0 - s_scale * (rx0 + rx1) / 2.0
                    ty = cy1 - s_scale * ry1  # feet stay where the character stands
                else:
                    mode = "off"

    from dwpose_vendor.draw import draw_openpose

    out_path = _IN / "pose_ctrl.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        raw = _raw_for(sample_idx[i])
        if raw is None:
            vw.write(blank)
            continue
        cand, sc = raw[0].copy(), raw[1].copy()
        if mode != "off":
            cand[..., 0] = cand[..., 0] * s_scale + tx
            cand[..., 1] = cand[..., 1] * s_scale + ty
        pose = det._format_pose(cand, sc, width, height)
        frame_rgb = draw_openpose(pose, height=height, width=width,
                                  include_hands=True, include_face=True)
        vw.write(cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR))
    vw.release()
    return out_path, mode


def _face_restore(video_bytes: bytes, char_path) -> tuple:
    """v8.23: swap the SOURCE face back onto every generated frame. Pose-driven regeneration
    reinvents a real person's face after frame 0; this restores pixel-level identity while
    keeping LTX's motion/lighting. Returns (bytes, status). No face on either side => no-op."""
    import io

    import av
    import cv2
    import numpy as np

    fa, swapper = _FACE["app"], _FACE["swapper"]
    src = cv2.imread(str(char_path))
    if src is None:
        return video_bytes, "skipped_bad_source"
    src_faces = fa.get(src)
    if not src_faces:
        return video_bytes, "skipped_no_source_face"  # cartoons land here by design
    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    vin = av.open(io.BytesIO(video_bytes))
    vstream = vin.streams.video[0]
    fps = float(vstream.average_rate) if vstream.average_rate else 24.0
    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    ostream = out.add_stream("libx264", rate=round(fps))
    ostream.width = vstream.codec_context.width
    ostream.height = vstream.codec_context.height
    ostream.pix_fmt = "yuv420p"
    ostream.options = {"crf": "18", "preset": "medium"}
    swapped = 0
    try:
        for frame in vin.decode(vstream):
            bgr = frame.to_ndarray(format="bgr24")
            faces = fa.get(bgr)
            if faces:
                tgt = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                bgr = swapper.get(bgr, tgt, src_face, paste_back=True)
                swapped += 1
            new = av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format="bgr24")
            for packet in ostream.encode(new):
                out.mux(packet)
        for packet in ostream.encode():
            out.mux(packet)
    finally:
        out.close()
        vin.close()
    if swapped == 0:
        return video_bytes, "skipped_no_target_faces"
    return buf.getvalue(), f"applied_{swapped}f"


def _mux_ref_audio(video_bytes: bytes, ref_path, duration_s: float) -> bytes:
    """Copy the reference video's audio track onto the generated clip (packet remux, no
    re-encode). Meme mode generates with LTX audio OFF; the meme's own sound is the product."""
    import io

    import av

    with av.open(str(ref_path)) as probe:
        if not probe.streams.audio:
            return video_bytes

    vin = av.open(io.BytesIO(video_bytes))
    ain = av.open(str(ref_path))
    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    try:
        # av >= 12 renamed template-based stream copy; add_stream(template=) raises
        # "takes at least 1 positional argument" on the version the image ships.
        if hasattr(out, "add_stream_from_template"):
            ov = out.add_stream_from_template(vin.streams.video[0])
            oa = out.add_stream_from_template(ain.streams.audio[0])
        else:
            ov = out.add_stream(template=vin.streams.video[0])
            oa = out.add_stream(template=ain.streams.audio[0])
        for p in vin.demux(vin.streams.video[0]):
            if p.dts is None:
                continue
            p.stream = ov
            out.mux(p)
        for p in ain.demux(ain.streams.audio[0]):
            if p.dts is None:
                continue
            if p.pts is not None and float(p.pts * p.time_base) > duration_s:
                break
            p.stream = oa
            out.mux(p)
    finally:
        out.close()
        vin.close()
        ain.close()
    return buf.getvalue()


def _effective_cfg_cache() -> dict:
    """Describe the values the denoiser will read for this request."""
    interval_raw = os.environ.get("LTX_CFG_CACHE_INTERVAL", "2")
    warmup_raw = os.environ.get("LTX_CFG_CACHE_WARMUP", "2")
    try:
        interval = int(interval_raw)
        interval_valid = True
    except (TypeError, ValueError):
        interval = None
        interval_valid = False
    try:
        warmup = int(warmup_raw)
        warmup_valid = True
    except (TypeError, ValueError):
        warmup = None
        warmup_valid = False
    range_raw = os.environ.get("LTX_CFG_CACHE_RANGE") or None
    range_effective = None
    range_valid = True
    if range_raw:
        try:
            lo, hi = (int(x) for x in range_raw.split(":"))
            range_effective = {"start": lo, "end_exclusive": hi}
        except (TypeError, ValueError):
            # denoisers.py also treats an unparsable range as no window.
            range_valid = False
    return {
        "enabled": os.environ.get("LTX_CFG_CACHE") == "1",
        "interval": interval,
        "interval_valid": interval_valid,
        "configured_interval": interval_raw,
        "warmup_steps": warmup,
        "warmup_valid": warmup_valid,
        "configured_warmup": warmup_raw,
        "range": range_effective,
        "range_valid": range_valid,
        "configured_range": range_raw,
    }


def _effective_tgate() -> dict:
    start_raw = os.environ.get("LTX_TGATE_START_STEP")
    try:
        start_step = int(start_raw) if start_raw is not None else None
        valid = True
    except (TypeError, ValueError):
        start_step = None
        valid = False
    return {
        "configured": start_raw is not None,
        "enabled": start_step is not None,
        "start_step": start_step,
        "configured_start_step": start_raw,
        "valid": valid,
    }


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
    if inp.get("enhance_probe"):  # diagnostics: confirm vision-Gemma loaded + sample rewrite + VRAM
        if not ENHANCE_VISION:
            return {"enhance_probe": "LTX_ENHANCE_VISION not enabled", "config_tag": CONFIG_TAG}
        try:
            pp = _IN / "probe"
            with open(pp, "wb") as f:
                f.write(base64.b64decode(inp["image_b64"]))
            out = _enhance_prompt(inp.get("prompt", "two fighters fight"), pp)
            return {"enhance_probe": "ok", "enhanced": out,
                    "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                    "gemma_build": getattr(_PIPE.prompt_encoder._resident_te, "model", None) is not None,
                    "config_tag": CONFIG_TAG}
        except Exception:  # noqa: BLE001
            return {"enhance_probe": "FAILED", "trace": traceback.format_exc()}
    _kill_prewarm()  # L-4: a real generation owns the disk from here
    try:
        tier = str(inp.get("tier", os.environ.get("LTX_DEFAULT_TIER", "fast"))).lower()
        if tier not in ("fast", "quality"):
            tier = "fast"
        try:
            negative_prompt, negative_prompt_source = resolve_negative_prompt(
                inp, DEFAULT_NEGATIVE_PROMPT
            )
            decode_noise = resolve_decode_noise(inp)
            ge_on = resolve_boolean(inp, "stage1_ge", _STAGE1_GE_DEFAULT)
            audio_on = resolve_boolean(inp, "audio", False)
            enhance_requested = resolve_boolean(inp, "enhance", ENHANCE_DEFAULT)
            cfg_cache_on = resolve_boolean(inp, "cfg_cache", _CFG_CACHE_ENV)
            cas_amount = resolve_optional_number(
                inp, "cas_amount", minimum=0.0, maximum=1.0
            )
            cas_mix = resolve_optional_number(
                inp, "cas_mix", minimum=0.0, maximum=1.0
            )
        except ValueError as exc:
            return {
                "error": "invalid_request",
                "detail": str(exc),
                "config_tag": f"{CONFIG_TAG}-{tier}",
            }
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

        # v8.20: reference_video_b64 present => MEME MODE (motion control): i2v first-frame
        # conditioning from image_b64 + DWPose skeleton of the reference video through the
        # Union IC-LoRA. Requires an endpoint provisioned with LTX_MEME=1 + the fused IC-LoRA.
        meme_mode = bool(inp.get("reference_video_b64") or inp.get("reference_video_url"))
        ref_src = None
        if meme_mode:
            if not (MEME_MODE and _DWPOSE.get("det") is not None):
                return {"error": "meme mode not enabled on this endpoint (needs LTX_MEME=1 + LTX_EXTRA_LORA_* union IC-LoRA)",
                        "config_tag": f"{CONFIG_TAG}-{tier}"}
            if _REF_DOWNSCALE["factor"] == 1 or not EXTRA_LORA_REPO:
                return {"error": "meme mode endpoint misconfigured: union IC-LoRA not fused (LTX_EXTRA_LORA_* unset)",
                        "config_tag": f"{CONFIG_TAG}-{tier}"}
            if not inp.get("image_b64"):
                return {"error": "meme mode requires image_b64 (the character image)",
                        "config_tag": f"{CONFIG_TAG}-{tier}"}
            # v8.22.1: bake EXIF orientation into the character pixels. The backend does this
            # via sharp already, but a raw phone photo sent straight to the API arrives rotated
            # for cv2/the conditioning reader (both ignore EXIF) and the clip comes out sideways.
            try:
                import io as _io

                from PIL import Image as _PILImage
                from PIL import ImageOps as _PILImageOps

                _img = _PILImage.open(_io.BytesIO(base64.b64decode(inp["image_b64"])))
                _img = _PILImageOps.exif_transpose(_img).convert("RGB")
                _buf = _io.BytesIO()
                _img.save(_buf, format="PNG")
                inp["image_b64"] = base64.b64encode(_buf.getvalue()).decode()
            except Exception as exc:  # noqa: BLE001 — a bad EXIF tag must not kill the job
                _INIT_LOG.append(f"exif transpose skipped: {exc!r}")
            # v8.20.4: reference by URL is the product path — RunPod /run caps payloads at 10MB
            # and a 10s meme reference + character image in base64 blows past it (pair B did).
            ref_src = _IN / "ref_src.mp4"
            if inp.get("reference_video_b64"):
                with open(ref_src, "wb") as f:
                    f.write(base64.b64decode(inp["reference_video_b64"]))
            else:
                import urllib.request as _rq
                with _rq.urlopen(inp["reference_video_url"], timeout=60) as r, open(ref_src, "wb") as f:
                    f.write(r.read())

        # v8.17: image_b64 OPTIONAL. Absent => TEXT-TO-VIDEO (no conditioning image); present => i2v.
        # Same LTX-2.3 model handles both; t2v just passes empty image-conditionings (combined_image_
        # conditionings(images=[]) -> no conditioning) and skips the vision-enhancer (which needs an image).
        t2v = not inp.get("image_b64")
        img_path = _IN / "req"
        if not t2v:
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(inp["image_b64"]))
        case = {
            "id": "req",
            "file": "req",
            "prompt": inp.get("prompt") or DEFAULT_PROMPT,
            "negative_prompt": negative_prompt,
            "negative_prompt_source": negative_prompt_source,
            "seed": int(inp.get("seed", 3102)),
        }
        raw_prompt = case["prompt"]
        enhanced_prompt = None
        # Prompt upsampler is the ALWAYS-ON product default (env LTX_ENHANCE=1 => ENHANCE_DEFAULT=True;
        # the frontend never sends `enhance`, so it always runs). v8.14 keeps a per-request `enhance:false`
        # purely as an INTERNAL A/B affordance: a paired sampler A/B (e.g. cfg-cache) needs an identical
        # prompt across arms, impossible while Gemma samples non-deterministically — so tests bypass it and
        # feed a fixed pre-enhanced prompt. Not exposed in the product UI; default stays on.
        enhance_seconds = None  # Phase-0 instrumentation: the ~6-8s upsampler was never measured per-call
        enhance_eligible = ENHANCE_VISION and (ENHANCE_T2V or not t2v)
        if enhance_requested and enhance_eligible:  # t2v rewrite text-only; i2v stays vision-based
            try:
                _t_enh = time.time()
                if t2v:
                    enhanced_prompt = _enhance_prompt_t2v(raw_prompt, case["seed"],
                                                          inp.get("enhance_max_tokens"),
                                                          inp.get("enhance_system_prompt"))
                else:
                    enhanced_prompt = _enhance_prompt(raw_prompt, img_path, inp.get("enhance_max_tokens"),
                                                      inp.get("enhance_system_prompt"))
                enhance_seconds = round(time.time() - _t_enh, 2)
                case["prompt"] = enhanced_prompt
            except Exception as exc:  # noqa: BLE001 — graceful fallback to raw prompt
                _INIT_LOG.append(f"enhance failed, raw used: {exc!r}")
        settings = {
            "width": int(inp.get("width", 1280)), "height": int(inp.get("height", 704)),
            "frames": int(inp.get("frames", 121)), "fps": float(inp.get("fps", 24.0)),
            "conditioning_strength": float(inp.get("conditioning_strength", 0.8)),  # v8.16 per-request:
            "conditioning_crf": 0, "dev_inference_steps": steps,  # lower => subject freer to move (un-freeze)
            "t2v": t2v,  # v8.17: text-to-video (no conditioning image)
        }
        # CAS crispness pass (default-on). Per-request override for A/B; omit in prod => env/default.
        if cas_amount is not None:
            settings["cas_amount"] = cas_amount
        if cas_mix is not None:
            settings["cas_mix"] = cas_mix
        if decode_noise is not None:
            settings["decode_noise"] = decode_noise
        pose_seconds = None
        if meme_mode:
            # v8.21: output orientation must follow the REFERENCE video (the pose canvas), not
            # the character photo. A landscape photo + portrait meme made the backend pick
            # 1152x640, the portrait skeleton got its head/legs center-cropped away and the
            # generation invented a giant torso (prod post 27278). Swap dims to match the ref.
            import cv2 as _cv2
            _rc = _cv2.VideoCapture(str(ref_src))
            _rw = _rc.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0
            _rh = _rc.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0
            _rc.release()
            if _rw and _rh and ((_rh > _rw) != (settings["height"] > settings["width"])):
                settings["width"], settings["height"] = settings["height"], settings["width"]
                _INIT_LOG.append(f"meme dims swapped to follow reference orientation ({int(_rw)}x{int(_rh)})")
            # The reference is VAE-encoded at stage1//downscale resolution => full-res dims must
            # be divisible by 128 (empirically: 576x1024 fails inside the VAE, 640x1152 works).
            if settings["width"] % 128 or settings["height"] % 128:
                return {"error": "meme mode needs width/height divisible by 128 (reference VAE grid)",
                        "config_tag": f"{CONFIG_TAG}-{tier}"}
            if "frames" not in inp:  # follow the meme's own length up to the cap
                settings["frames"] = _meme_frames(ref_src, settings["fps"])
            audio_on = False  # LTX audio off; the meme's own soundtrack is muxed back post-encode
            # Identity anchor: the character image must survive 5-10s of pose-driven motion.
            # Measured on pair A seed 42: 0.8 (i2v default) lost the sunglasses, 0.95 kept them.
            if "conditioning_strength" not in inp:
                settings["conditioning_strength"] = 0.95
            _t_pose = time.time()
            pose_path, retarget_mode = _pose_control_video(
                ref_src, img_path, settings["width"], settings["height"],
                settings["frames"], settings["fps"],
                retarget=bool(inp.get("retarget", MEME_RETARGET)),
            )
            pose_seconds = round(time.time() - _t_pose, 2)
            settings["video_conditioning"] = [(str(pose_path), float(inp.get("reference_strength", 0.9)))]
            settings["reference_downscale"] = _REF_DOWNSCALE["factor"]
            if "reference_attention" in inp:
                settings["reference_attention"] = float(inp["reference_attention"])

        # L-5 gate: resident floor ~71-73 GiB; larger jobs would OOM even uncached
        # (floor stays + their own 22.6 GiB fuse transient + ~2.3x activations).
        if settings["width"] * settings["height"] > _MAX_PIXELS or settings["frames"] > _MAX_FRAMES:
            return {"error": "resolution_gated",
                    "max": f"{_MAX_PIXELS}px x {_MAX_FRAMES}f (L-5 resident-cache VRAM floor)",
                    "config_tag": f"{CONFIG_TAG}-{tier}"}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # keep peak_vram_gib per-request-true
        # per-request config switches (handler is single-threaded)
        _SIGMA_MODE["mode"] = "distilled" if tier == "fast" else "ltx2"
        _SIGMA_MODE["override"] = sig if tier == "fast" else None
        # F1 (NATIVE-QUALITY): gradient-estimating Euler on stage1 — per-request, 0 wall-time.
        settings["stage1_ge"] = ge_on
        # F2: per-request stage2 sigmas (re-noise/refine grid); else env/default _STAGE2_FAST.
        _stage2 = _STAGE2_FAST
        if tier == "fast" and "stage2_sigmas" in inp:
            v2 = [float(x) for x in inp["stage2_sigmas"]]
            if len(v2) >= 2 and abs(v2[-1]) < 1e-9 and all(v2[i] > v2[i + 1] for i in range(len(v2) - 1)):
                _stage2 = torch.tensor(v2)
        base.STAGE_2_DISTILLED_SIGMAS = _stage2 if tier == "fast" else _STAGE2_DEFAULT
        # v8.15: per-request Detail-Daemon (sampler hook reads LTX_DETAIL_DAEMON*); restore env default
        # when omitted so it never leaks across requests. Prod (no flag, no env) => off => bit-exact.
        if "detail_daemon" in inp:
            os.environ["LTX_DETAIL_DAEMON"] = str(float(inp["detail_daemon"]))
            for _ek, _ik in (("LTX_DETAIL_DAEMON_START", "detail_daemon_start"),
                             ("LTX_DETAIL_DAEMON_END", "detail_daemon_end")):
                if _ik in inp:
                    os.environ[_ek] = str(int(inp[_ik]))
                else:
                    os.environ.pop(_ek, None)
        elif not _DETAIL_DAEMON_ENV:
            os.environ.pop("LTX_DETAIL_DAEMON", None)
        if audio_on:
            os.environ.pop("LTX_NO_AUDIO", None)
        else:
            os.environ["LTX_NO_AUDIO"] = "1"
        # GUIDANCE (ablation/tier): per-request CFG/modality on the fast tier (costs an uncond pass).
        # cfg>1 restores classifier-free guidance (drives motion on under-animated scenes like the mage).
        # cfg_cache amortizes the uncond pass (LTX_CFG_CACHE in denoisers.py): ~+30-40% vs +100%.
        targs = _tier_args(tier, "req")
        guided = False
        if "cfg" in inp:
            targs.video_cfg_scale = float(inp["cfg"]); guided = float(inp["cfg"]) > 1.0
        if "modality" in inp:
            targs.video_modality_scale = float(inp["modality"])
        # The negative context may only be elided when CFG is actually 1. A
        # per-request cfg>1 must encode the caller's negative prompt.
        effective_video_cfg = (
            float(targs.video_cfg_scale)
            if targs.video_cfg_scale is not None
            else float(LTX_2_3_PARAMS.video_guider_params.cfg_scale)
        )
        _SKIP_NEG_NOW["on"] = (
            SKIP_NEG_ENCODE
            and tier == "fast"
            and not audio_on
            and effective_video_cfg == 1.0
            and negative_prompt_source == "default"
        )
        if cfg_cache_on:
            os.environ["LTX_CFG_CACHE"] = "1"
        else:
            os.environ.pop("LTX_CFG_CACHE", None)
        # per-request cache tuning (interval/warmup/range) for A/B; restore endpoint env default when
        # a request omits one so tuning never leaks across requests (handler is single-threaded).
        for _k, _v, _d in (("LTX_CFG_CACHE_INTERVAL", inp.get("cfg_cache_interval"), _CFG_CACHE_INTERVAL_ENV),
                           ("LTX_CFG_CACHE_WARMUP", inp.get("cfg_cache_warmup"), _CFG_CACHE_WARMUP_ENV),
                           ("LTX_CFG_CACHE_RANGE", inp.get("cfg_cache_range"), _CFG_CACHE_RANGE_ENV)):
            if _v is not None:
                os.environ[_k] = str(_v)
            elif _d is not None:
                os.environ[_k] = _d
            else:
                os.environ.pop(_k, None)

        cfg_cache_effective = _effective_cfg_cache()
        tgate_effective = _effective_tgate()
        rec = base.run_case(_PIPE, case, settings, targs,
                            repeat_idx=1, prompt_cache=None, resident_stage_cache=_STAGE_CACHE)
        if rec.get("error"):
            return {"error": rec["error"], "init_log": _INIT_LOG, "config_tag": f"{CONFIG_TAG}-{tier}"}
        out = _OUT / rec["output"]
        with open(out, "rb") as f:
            raw = f.read()
        os.remove(out)
        face_status = None
        if (meme_mode and FACE_RESTORE and _FACE.get("swapper") is not None
                and bool(inp.get("face_restore", True))):
            try:
                _t_face = time.time()
                raw, face_status = _face_restore(raw, img_path)
                face_status = f"{face_status} ({round(time.time() - _t_face, 1)}s)"
            except Exception as exc:  # noqa: BLE001 — the un-restored clip still beats a failed job
                face_status = f"face_restore_failed: {str(exc)[:120]}"
        mux_status = None
        if meme_mode and bool(inp.get("preserve_source_audio", True)):
            try:
                raw = _mux_ref_audio(raw, ref_src, settings["frames"] / settings["fps"])
                mux_status = "source_audio_muxed"
            except Exception as exc:  # noqa: BLE001 — video without audio beats a failed job
                mux_status = f"mux_failed: {str(exc)[:120]}"
        cas_effective = rec.get("effective_cas") or {
            "amount": float(settings.get("cas_amount", CAS_AMOUNT_DEFAULT)),
            "mix": float(settings.get("cas_mix", CAS_MIX_DEFAULT)),
        }
        resp = {
            "peak_vram_gib": rec.get("peak_vram_gib"),
            "elapsed_seconds": rec.get("elapsed_seconds"),
            "timers": rec.get("timers"),
            "config_tag": (f"{CONFIG_TAG}-{tier}" + ("-meme" if meme_mode else "")
                           + ("" if audio_on else "-noaudio")
                           + (f"-sig{steps}" if sig else "") + ("-enh" if enhanced_prompt else "")
                           + ("-ge" if ge_on else "") + (f"-s2_{len(_stage2)}" if "stage2_sigmas" in inp else "")
                           + (f"-dn{inp['decode_noise']}" if "decode_noise" in inp else "")
                           + (f"-cfg{inp['cfg']}" if "cfg" in inp else "") + ("-cfgcache" if inp.get("cfg_cache") else "")
                           + (f"-mod{inp['modality']}" if "modality" in inp else "")
                           + (f"-dd{inp['detail_daemon']}" if "detail_daemon" in inp else "")),
            "tier": tier,
            "effective_config": {
                "sampler": {
                    "stage1": "gradient_estimating_euler" if ge_on else "euler",
                    "stage1_ge": ge_on,
                    "stage2": "euler",
                },
                "cfg_cache": cfg_cache_effective,
                "tgate": tgate_effective,
                "cas": {
                    **cas_effective,
                    "enabled": float(cas_effective["amount"]) > 0.0,
                },
                "decode_noise": {
                    "scale": rec.get("effective_decode_noise_scale"),
                    "source": "request" if decode_noise is not None else "decoder_default",
                },
                "enhance": {
                    "requested": enhance_requested,
                    "eligible": enhance_eligible,
                    "applied": enhanced_prompt is not None,
                },
                "audio": {"enabled": audio_on},
                "tier": tier,
                "shape": {
                    "width": settings["width"],
                    "height": settings["height"],
                    "frames": settings["frames"],
                    "fps": settings["fps"],
                },
                "negative_prompt": {
                    "source": negative_prompt_source,
                    "characters": len(negative_prompt),
                    "encoded": not _SKIP_NEG_NOW["on"],
                    "cfg_scale": effective_video_cfg,
                    "applied_to_guidance": effective_video_cfg != 1.0,
                },
            },
        }
        if meme_mode:
            resp["pose_seconds"] = pose_seconds
            resp["frames"] = settings["frames"]
            resp["retarget"] = retarget_mode
            resp["onnx_provider"] = {"pose": _DWPOSE.get("provider"), "face": _FACE.get("provider")}
            if face_status:
                resp["face_restore"] = face_status
            if mux_status:
                resp["audio"] = mux_status
        if enhanced_prompt is not None:
            resp["raw_prompt"] = raw_prompt
            resp["enhanced_prompt"] = enhanced_prompt
            # Phase-0 instrumentation: real per-call upsampler cost + output size. enhance_words near
            # ~300 (≈400-token cap) means the cap binds and trimming to 256 would save time; well below
            # means EOS already stops short and the A1 trim lever is a no-op. enhance_seconds is THE
            # measurement that was previously only owner-estimated (~6-8s).
            resp["enhance_seconds"] = enhance_seconds
            resp["enhance_words"] = len(enhanced_prompt.split())
            resp["enhance_chars"] = len(enhanced_prompt)
            resp["enhance_max_tokens"] = ENHANCE_MAX_TOKENS
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
