#!/usr/bin/env python3
"""Create the LTX meme endpoint (template + endpoint) on the video RunPod account.
Env is cloned from the LIVE prod template (y9podaeiut) so every v8.19 tuning knob carries
over, then meme mode + the Union IC-LoRA fuse are switched on via env."""
import json
import os
import urllib.request

REST = "https://rest.runpod.io/v1"
KEY = os.environ["RUNPOD_VIDEO_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
import sys
IMAGE = f"ghcr.io/mark-blip-star/ltx-worker-slim:{sys.argv[1] if len(sys.argv) > 1 else 'v8.20'}"
SRC_TEMPLATE = "y9podaeiut"


def req(path, data=None, method="GET"):
    r = urllib.request.Request(REST + path, data=json.dumps(data).encode() if data else None,
                               headers=H, method=method)
    return json.loads(urllib.request.urlopen(r, timeout=60).read() or b"{}")


src = req(f"/templates/{SRC_TEMPLATE}")
env = dict(src.get("env") or {})
env.update({
    "LTX_CONFIG_TAG": "v8.20-meme",
    "LTX_MEME": "1",
    "LTX_EXTRA_LORA_REPO": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    "LTX_EXTRA_LORA_FILE": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "LTX_EXTRA_LORA_S1": "1.0",
    "LTX_EXTRA_LORA_S2": "1.0",
})
print("cloned env keys:", len(env), "| src image was:", src.get("imageName"))

# The ghcr image is PRIVATE: without the registry auth the workers can't pull and sit
# "unhealthy" while jobs rot IN_QUEUE forever (hit on 2026-07-17). Clone it from the source.
tpl = req("/templates", {"name": "ltx-meme-worker", "imageName": IMAGE,
                         "containerDiskInGb": src.get("containerDiskInGb") or 100,
                         "containerRegistryAuthId": src.get("containerRegistryAuthId"),
                         "env": env, "isServerless": True}, "POST")
tid = tpl.get("id")
print("TEMPLATE:", tid)

ep = req("/endpoints", {"name": "ltx-meme-ep", "templateId": tid,
                        "gpuTypeIds": ["NVIDIA H200"], "workersMin": 0, "workersMax": 2,
                        "idleTimeout": 30, "flashboot": True, "scalerType": "REQUEST_COUNT",
                        "executionTimeoutMs": 1200000, "computeType": "GPU"}, "POST")
print("ENDPOINT:", ep.get("id"), "| name:", ep.get("name"))
json.dump({"template_id": tid, "endpoint_id": ep.get("id")}, open("meme_endpoint.json", "w"))
print("NOTE: weights repo Cached-Models pinning is console-only; without it the first boot per host pulls ~50GB itself.")
print("CREATE_DONE")
