# LTX-2.3 serverless worker — RunPod GitHub-build (builds on RunPod infra).
# FA3 pulled as a prebuilt wheel + weights downloaded at build time (no big files in git).
# Build ARG HF_TOKEN required (set in RunPod endpoint build env).
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=EAGER
WORKDIR /app

# 1. Code (our patched LTX-2 + handler)
COPY LTX-2 /app/LTX-2
COPY handler.py /app/handler.py
COPY warmup.jpg /app/warmup.jpg

# 2. Frozen venv (torch 2.7.1+cu126). Bootstrap uv if the base image lacks it.
RUN (command -v uv >/dev/null 2>&1 || pip install --no-cache-dir -q uv) && \
    cd /app/LTX-2 && uv sync --frozen --extra fp8-trtllm
ENV VENV=/app/LTX-2/.venv PATH="/app/LTX-2/.venv/bin:${PATH}"

# 3. FlashAttention-3 — prebuilt wheel (sm_90a, torch 2.7.1+cu126), pulled from public release.
#    Avoids nvcc dependency, CUDA-version skew, and ~40-min source compile on RunPod infra.
#    (abi3 wheel → installs on CPython 3.10+; runtime warmup validates the import on Hopper.)
RUN curl -fsSL -o /tmp/fa3.whl \
      https://github.com/Mark-blip-star/ltx-fa3/releases/download/v1/flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl && \
    uv pip install --python "$VENV/bin/python" /tmp/fa3.whl && \
    uv pip install --python "$VENV/bin/python" runpod && rm -f /tmp/fa3.whl

# 4. Weights baked (downloaded on RunPod build infra; HF_TOKEN via build ARG)
ARG HF_TOKEN
RUN python3 -m venv /opt/hf && /opt/hf/bin/pip install -q -U "huggingface_hub[cli]" && \
    HF_TOKEN="$HF_TOKEN" /opt/hf/bin/hf download Lightricks/LTX-2.3-fp8 ltx-2.3-22b-dev-fp8.safetensors --local-dir /app/models/fp8 && \
    HF_TOKEN="$HF_TOKEN" /opt/hf/bin/hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors ltx-2.3-22b-distilled-lora-384-1.1.safetensors --local-dir /app/models/base && \
    HF_TOKEN="$HF_TOKEN" /opt/hf/bin/hf download google/gemma-3-12b-it --local-dir /app/models/gemma

# 5. Entrypoint
ENV LTX_FP8_CKPT=/app/models/fp8/ltx-2.3-22b-dev-fp8.safetensors \
    LTX_GEMMA_ROOT=/app/models/gemma \
    LTX_UPSCALER=/app/models/base/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    LTX_DISTILLED_LORA=/app/models/base/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
    LTX_WARMUP_IMG=/app/warmup.jpg
CMD ["/app/LTX-2/.venv/bin/python", "/app/handler.py"]
