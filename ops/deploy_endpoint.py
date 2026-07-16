#!/usr/bin/env python3
"""Deploy an ltx-worker-slim tag to the prod serverless endpoint.

    python3 ops/deploy_endpoint.py <TAG> <CONFIG_TAG>   # e.g. v8.19 v8.19-something

Playbook section 4: PATCH the template (full env dict — a partial PATCH wipes the rest),
then drain so warm workers stop serving the old image. Aborts if jobs are live rather than
purging real user work.
Auth: RUNPOD_VIDEO_KEY env wins; `op` fallback needs desktop biometrics (fails headless).
Headless key: decrypt provider_runtime_settings.RUNPOD_VIDEO_API_KEY on the prod droplet."""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

TEMPLATE_ID = "y9podaeiut"
ENDPOINT_ID = "g9mihjeqiml41t"
if len(sys.argv) != 3:
    sys.exit(__doc__)
IMAGE = f"ghcr.io/mark-blip-star/ltx-worker-slim:{sys.argv[1]}"
CONFIG_TAG = sys.argv[2]

KEY = os.environ.get("RUNPOD_VIDEO_KEY") or subprocess.run(
    ["op", "read", "op://Yallery/runpod-account-1/credential"],
    capture_output=True, text=True, check=True).stdout.strip()
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def call(method: str, url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers=HDRS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url} -> {e.code}: {e.read().decode()[:400]}") from e


# 1. No live jobs => safe to drain.
health = call("GET", f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health")
jobs = health.get("jobs", {})
if jobs.get("inProgress", 0) or jobs.get("inQueue", 0):
    sys.exit(f"ABORT: live jobs on endpoint: {jobs}")
print("queue empty:", jobs, flush=True)

# 2. Template: current env + image/tag bump (PATCH needs the FULL env dict).
tpl = call("GET", f"https://rest.runpod.io/v1/templates/{TEMPLATE_ID}")
env = dict(tpl.get("env") or {})
print("current image:", tpl.get("imageName"), "| env keys:", len(env), flush=True)
env["LTX_CONFIG_TAG"] = CONFIG_TAG
env["LTX_ENHANCE_T2V"] = "1"
call("PATCH", f"https://rest.runpod.io/v1/templates/{TEMPLATE_ID}",
     {"imageName": IMAGE, "env": env})
tpl2 = call("GET", f"https://rest.runpod.io/v1/templates/{TEMPLATE_ID}")
print("patched image:", tpl2.get("imageName"), "| LTX_CONFIG_TAG:", (tpl2.get("env") or {}).get("LTX_CONFIG_TAG"), flush=True)

# 3. Drain so warm workers stop serving the old image (purge-queue is a no-op here: queue is empty).
call("POST", f"https://api.runpod.ai/v2/{ENDPOINT_ID}/purge-queue", {})
call("PATCH", f"https://rest.runpod.io/v1/endpoints/{ENDPOINT_ID}", {"workersMax": 0})
for i in range(60):
    time.sleep(5)
    w = call("GET", f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health").get("workers", {})
    alive = sum(w.get(k, 0) for k in ("idle", "ready", "running", "initializing"))
    if alive == 0:
        print(f"drained after {(i + 1) * 5}s", flush=True)
        break
else:
    print("WARN: workers still alive after 300s, proceeding (they die with workersMax=0)", flush=True)

# Restoring workersMax can silently not reach the job API: REST reports workersMax:5 while /run
# still 409s with ENDPOINT_PAUSED (max_workers=0). Re-PATCH and probe until it accepts work.
for attempt in range(6):
    call("PATCH", f"https://rest.runpod.io/v1/endpoints/{ENDPOINT_ID}", {"workersMax": 5})
    time.sleep(10)
    try:
        probe = call("POST", f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run", {"input": {"warm": True}})
    except RuntimeError as e:
        if "ENDPOINT_PAUSED" in str(e) or "409" in str(e):
            print(f"endpoint still paused, re-patching (attempt {attempt + 1})", flush=True)
            continue
        raise
    print("endpoint accepting work; warm ping:", probe.get("id"), flush=True)
    break
else:
    sys.exit("ABORT: endpoint still paused after 6 attempts — PATCH workersMax by hand")
print("DEPLOY_OK", flush=True)
