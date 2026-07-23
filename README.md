# ltx-worker — LTX-2.3 serverless worker (RunPod GitHub build)
RunPod builds this Dockerfile on their infra: venv (torch 2.7.1+cu126) + FA3 (from source, sm_90a)
+ weights baked (HF_TOKEN build arg). Handler: residency + warmup-forward.
Set build env: HF_TOKEN. GPU: H100 (any DC). Enable FlashBoot + scale-to-zero.

## Lossless regional compile canary

Regional `torch.compile` is disabled by default. The first guarded rollout is
stage 1 only and uses Inductor's default mode (no CUDA Graphs). Enable it only
on an isolated H200 dark canary until the paired parity matrix passes:

```text
LTX_REGIONAL_COMPILE=1
LTX_REGIONAL_COMPILE_STAGE1=1
LTX_REGIONAL_COMPILE_STAGE2=0
LTX_REGIONAL_COMPILE_MODE=default
```

Enabling the canary makes the full generation warmup mandatory and fail-closed.
`stage2` and `reduce-overhead` are rejected in this version. Compiler artifacts
default to `/tmp/ltx-torchinductor-cache`; set `TORCHINDUCTOR_CACHE_DIR` before
worker start to use another location. Dynamic-shape guards remain enabled.
