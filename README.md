# ltx-worker — LTX-2.3 serverless worker (RunPod GitHub build)
RunPod builds this Dockerfile on their infra: venv (torch 2.7.1+cu126) + FA3 (from source, sm_90a)
+ weights baked (HF_TOKEN build arg). Handler: residency + warmup-forward.
Set build env: HF_TOKEN. GPU: H100 (any DC). Enable FlashBoot + scale-to-zero.
