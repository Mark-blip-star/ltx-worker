# LTX-2.3 serverless worker — RunPod GitHub-build (builds on RunPod infra).
# FA3 pulled as a prebuilt wheel + weights downloaded at build time (no big files in git).
# Build ARG HF_TOKEN required (set in RunPod endpoint build env).
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=EAGER
WORKDIR /app

# 1. Code (our patched LTX-2 + handler)
COPY LTX-2 /app/LTX-2
COPY request_config.py /app/request_config.py
COPY regional_compile_config.py /app/regional_compile_config.py
COPY handler.py /app/handler.py
COPY warmup.jpg /app/warmup.jpg

# 2. Frozen venv (torch 2.7.1+cu126). Bootstrap uv if the base image lacks it.
RUN (command -v uv >/dev/null 2>&1 || pip install --no-cache-dir -q uv) && \
    cd /app/LTX-2 && uv sync --frozen --extra fp8-trtllm
ENV VENV=/app/LTX-2/.venv PATH="/app/LTX-2/.venv/bin:${PATH}"

# 3. FlashAttention-3 — prebuilt wheel (sm_90a, torch 2.7.1+cu126), pulled from public release.
#    Avoids nvcc dependency, CUDA-version skew, and ~40-min source compile on RunPod infra.
#    --no-deps: deps (torch/einops) already satisfied by the frozen sync; don't let uv touch the
#    frozen env (that resolution was the build failure). set -eux makes any failure self-evident.
RUN set -eux; \
    "$VENV/bin/python" --version; \
    mkdir -p /tmp/fa3; \
    curl -fSL -o /tmp/fa3/flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl \
      "https://github.com/Mark-blip-star/ltx-fa3/releases/download/v1/flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl"; \
    ls -l /tmp/fa3/; \
    uv pip install --python "$VENV/bin/python" --no-deps /tmp/fa3/flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl; \
    uv pip install --python "$VENV/bin/python" runpod; \
    rm -rf /tmp/fa3

# 4. NO weights baked. RunPod github-build has a HARD 30-min limit, and exporting a
#    baked-weights image (~60GB) blows past it (layer export alone ~17 min). So ALL weights
#    (LTX = public, Gemma = gated) are fetched at RUNTIME in handler.py via the runtime
#    HF_TOKEN env. FlashBoot then snapshots the loaded models, so warm/restored starts skip it.

# 5. Entrypoint
ENV LTX_FP8_CKPT=/app/models/fp8/ltx-2.3-22b-dev-fp8.safetensors \
    LTX_GEMMA_ROOT=/app/models/gemma \
    LTX_UPSCALER=/app/models/base/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    LTX_DISTILLED_LORA=/app/models/base/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
    LTX_WARMUP_IMG=/app/warmup.jpg
CMD ["/app/LTX-2/.venv/bin/python", "/app/handler.py"]
