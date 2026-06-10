#!/usr/bin/env bash
# Worker entrypoint: run handler with ALL output teed to a log that is shipped to a private
# HF repo every 15s (self-service worker logs — RunPod exposes none via API).
WID="${RUNPOD_POD_ID:-unknown}"
LOG=/tmp/worker.log
: > "$LOG"
{
  echo "=== boot $(date -u +%FT%TZ) worker=$WID ==="
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
  echo "--- /runpod-volume ---"
  ls /runpod-volume 2>&1
  ls /runpod-volume/huggingface-cache/hub 2>&1 | head -5
  df -h / /runpod-volume 2>&1 | tail -3
  echo "--- env (scrubbed) ---"
  env | grep -vE 'TOKEN|KEY|SECRET|PASSWORD' | sort
  echo "=== starting handler ==="
} >> "$LOG" 2>&1

( while true; do
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE /app/LTX-2/.venv/bin/python /app/log_uploader.py "$WID" "$LOG" >/dev/null 2>&1
    sleep 15
  done ) &

final_upload() {
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE /app/LTX-2/.venv/bin/python /app/log_uploader.py "$WID" "$LOG" >/dev/null 2>&1
}
trap final_upload EXIT

cd /app
/app/LTX-2/.venv/bin/python -u /app/handler.py 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
echo "=== handler exited code=$code $(date -u +%FT%TZ) ===" >> "$LOG"
final_upload
exit "$code"
