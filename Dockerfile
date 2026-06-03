# LTX-2.3 serverless worker — RunPod GitHub-build (builds on RunPod infra).
# FA3 built from source + weights downloaded at build time (no big files in git).
# Build ARG HF_TOKEN required (set in RunPod endpoint build env).
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=EAGER
WORKDIR /app

# 1. Code (our patched LTX-2 + handler)
COPY LTX-2 /app/LTX-2
COPY handler.py /app/handler.py
COPY warmup.jpg /app/warmup.jpg

# 2. Frozen venv (torch 2.7.1+cu126)
RUN cd /app/LTX-2 && uv sync --frozen --extra fp8-trtllm
ENV VENV=/app/LTX-2/.venv PATH="/app/LTX-2/.venv/bin:${PATH}"

# 3. FlashAttention-3 from source (pinned commit, sm_90a, fwd-only) — built on RunPod infra
RUN git clone https://github.com/Dao-AILab/flash-attention.git /tmp/fa && \
    cd /tmp/fa && git checkout f82d0dc6d69bfb80f319a6b8909d94e60c2fb7b1 && \
    git submodule update --init --recursive && \
    uv pip install --python "$VENV/bin/python" -U setuptools wheel ninja packaging && \
    cd /tmp/fa/hopper && \
    TORCH_CUDA_ARCH_LIST="9.0a" MAX_JOBS=16 FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
        "$VENV/bin/python" setup.py install && \
    "$VENV/bin/python" -c "import flash_attn_interface; print('FA3 ok')" && \
    uv pip install --python "$VENV/bin/python" runpod && rm -rf /tmp/fa

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
