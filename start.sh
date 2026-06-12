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
} >> "$LOG" 2>&1

# Log shipper starts FIRST: anything below (incl. a hung probe) must never block telemetry.
( while true; do
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE /app/LTX-2/.venv/bin/python /app/log_uploader.py "$WID" "$LOG" >/dev/null 2>&1
    sleep 15
  done ) &

# Snapshot resolution in the MAIN shell — both the probes and the prewarm need it.
HUBDIR=$(timeout 15 find /runpod-volume/huggingface-cache/hub -maxdepth 1 -type d -iname "models--*ltx-worker-weights*" 2>/dev/null | head -1)
SNAP=""
if [ -n "$HUBDIR" ] && [ -f "$HUBDIR/refs/main" ]; then
  SNAP="$HUBDIR/snapshots/$(cat "$HUBDIR/refs/main")"
fi
[ -d "$SNAP" ] || SNAP=$(ls -dt "$HUBDIR"/snapshots/*/ 2>/dev/null | head -1)
SNAP="${SNAP%/}"

# A0 probes (H-5, INIT-SPEEDUP-RESEARCH.md) — background + per-command timeouts.
( {
  echo "--- probes (A0: disk/RAM/staging) ---"
  echo "[probe] staged=$([ -d "$SNAP" ] && echo 1 || echo 0) snap=$SNAP"
  [ -d "$SNAP" ] && echo "[probe] snap_newest_mtime=$(timeout 20 find "$SNAP/" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1) now=$(date +%s)"
  free -g | awk 'NR==2{print "[probe] ram_total_gb="$2" available_gb="$7}'
  echo "[probe] cgroup_mem_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo na)"
  DDFILE=$(ls -S "$SNAP"/gemma-fp8/*.safetensors 2>/dev/null | head -1)
  if [ -n "$DDFILE" ]; then
    DD=$(timeout 45 dd if="$DDFILE" of=/dev/null bs=64M count=16 iflag=direct 2>&1 | tail -1)
    [ -n "$DD" ] || DD=$(timeout 45 dd if="$DDFILE" of=/dev/null bs=64M count=16 2>&1 | tail -1)
    echo "[probe] nvme_read: $DD"
  fi
  echo "[probe] done"
} >> "$LOG" 2>&1 ) &

# L-4: page-cache prewarm, ckpt first (its lazy re-read inside the FIRST GEN is the measured
# slow-host tail). setsid leader -> handler kills the whole group at the first real job.
if [ -d "$SNAP" ]; then
  setsid bash -c '
    SNAP="$1"
    AVAIL=$(free -g | awk "NR==2{print \$7}")
    if [ "${AVAIL:-0}" -lt 80 ]; then echo "[prewarm] skipped available_gb=$AVAIL"; exit 0; fi
    echo "[prewarm] start available_gb=$AVAIL"
    for f in \
        "$SNAP/${LTX_FP8_CKPT_NAME:-ltx-2.3-22b-dev-fp8.safetensors}" \
        "$SNAP/${LTX_GEMMA_FP8_SUBDIR:-gemma-fp8}"/*.safetensors \
        "$SNAP/${LTX_DISTILLED_LORA_NAME:-ltx-2.3-22b-distilled-lora-384-1.1.safetensors}" \
        "$SNAP/${LTX_UPSCALER_NAME:-ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"; do
      [ -f "$f" ] || continue
      nice -n 19 ionice -c3 cat "$f" > /dev/null 2>&1 || true
      echo "[prewarm] cached $(basename "$f") $(date -u +%T)"
    done
    echo "[prewarm] done $(date -u +%FT%TZ)"
  ' _ "$SNAP" >> "$LOG" 2>&1 &
  echo $! > /tmp/prewarm.pid
fi
echo "=== starting handler ===" >> "$LOG"

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
