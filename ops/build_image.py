#!/usr/bin/env python3
"""Build ghcr.io/mark-blip-star/ltx-worker-slim:<NEW_TAG> on a throwaway DO droplet.

    python3 ops/build_image.py <PREV_TAG> <NEW_TAG>     # e.g. v8.18 v8.19

Follows ~/Desktop/LTX-WORKER-OPS-PLAYBOOK.md section 3: build on a droplet (a Mac can't),
cache from the previous tag (else the venv rebuilds from scratch), arm a self-destruct
backstop, and delete the droplet in `finally` no matter what."""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

SETTINGS = "/Users/mac/Projects/DiroLabs/.claude/settings.local.json"
SRC = "/Users/mac/Projects/ltx23-samp-benchmark-20260526/deploy/gh/"
IMAGE = "ghcr.io/mark-blip-star/ltx-worker-slim"
if len(sys.argv) != 3:
    sys.exit(__doc__)
PREV_TAG, NEW_TAG = sys.argv[1], sys.argv[2]
SSH_KEY_ID = 56895626
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR"]

env = json.load(open(SETTINGS))["env"]
GH_TOKEN = env["GITHUB_TOKEN"]
# The DiroLabs DO_API_TOKEN is dead (401); the live token is in 1Password.
DO_TOKEN = subprocess.run(["op", "read", "op://Yallery/digitalocean-yallery/credential"],
                          capture_output=True, text=True, check=True).stdout.strip()


def do_api(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"https://api.digitalocean.com/v2{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {DO_TOKEN}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        raise RuntimeError(f"DO {method} {path} -> {e.code}: {detail}") from e


def ssh(ip: str, cmd: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, f"root@{ip}", cmd],
                          capture_output=True, text=True, timeout=timeout)


print("account check:", do_api("GET", "/account")["account"]["status"], flush=True)

droplet_id = None
try:
    created = None
    for region, size in [("fra1", "s-8vcpu-16gb-amd"), ("ams3", "s-8vcpu-16gb-amd"),
                         ("fra1", "s-8vcpu-16gb"), ("ams3", "s-8vcpu-16gb")]:
        try:
            created = do_api("POST", "/droplets", {
                "name": f"ltx-build-{NEW_TAG}", "region": region, "size": size,
                "image": "ubuntu-24-04-x64", "ssh_keys": [SSH_KEY_ID], "tags": ["ltx-build"],
            })
            print(f"droplet requested: {region}/{size}", flush=True)
            break
        except RuntimeError as e:
            if "422" in str(e):
                print(f"unavailable {region}/{size}, trying next", flush=True)
                continue
            raise
    if not created:
        sys.exit("no droplet size/region available")
    droplet_id = created["droplet"]["id"]
    print("droplet id:", droplet_id, flush=True)

    ip = None
    for _ in range(60):
        time.sleep(10)
        d = do_api("GET", f"/droplets/{droplet_id}")["droplet"]
        if d["status"] == "active":
            ip = next(n["ip_address"] for n in d["networks"]["v4"] if n["type"] == "public")
            break
    if not ip:
        sys.exit("droplet never became active")
    print("droplet ip:", ip, flush=True)

    for i in range(40):
        if ssh(ip, "true", timeout=20).returncode == 0:
            break
        time.sleep(10)
    else:
        sys.exit("ssh never came up")
    print("ssh ready", flush=True)

    # Self-destruct backstop in case this local process dies mid-build.
    ssh(ip, "nohup bash -c 'sleep 5400; curl -s -X DELETE -H \"Authorization: Bearer "
            + DO_TOKEN + f"\" https://api.digitalocean.com/v2/droplets/{droplet_id}' "
            ">/dev/null 2>&1 & echo backstop_armed")
    print("backstop armed", flush=True)

    # Fresh droplets hold the dpkg lock for a while (cloud-init/unattended-upgrades) — wait + retry.
    r = ssh(ip, "cloud-init status --wait >/dev/null 2>&1; "
                "for i in 1 2 3 4; do curl -fsSL https://get.docker.com | sh && break || sleep 30; done; "
                "docker --version", timeout=1200)
    if r.returncode != 0 or "Docker version" not in r.stdout:
        sys.exit(f"docker install failed:\nSTDOUT: {r.stdout[-800:]}\nSTDERR: {r.stderr[-800:]}")
    print("docker:", r.stdout.strip().splitlines()[-1], flush=True)

    ssh(ip, "mkdir -p /build/gh")
    r = subprocess.run(["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS),
                        "--exclude", ".git", "--exclude", "__pycache__", "--exclude", "handler_v2.py",
                        SRC, f"root@{ip}:/build/gh/"], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        sys.exit(f"rsync failed: {r.stderr[-500:]}")
    print("rsync done", flush=True)

    r = ssh(ip, f"echo '{GH_TOKEN}' | docker login ghcr.io -u mark-blip-star --password-stdin")
    if r.returncode != 0:
        sys.exit(f"ghcr login failed: {r.stderr[-300:]}")
    print("ghcr login ok", flush=True)

    build_cmd = (
        "set -o pipefail; cd /build/gh && docker buildx build -f Dockerfile.slim --platform linux/amd64 "
        f"--provenance=false --cache-from type=registry,ref={IMAGE}:{PREV_TAG} "
        f"--cache-to type=inline -t {IMAGE}:{NEW_TAG} --push . 2>&1 | tail -40 "
        "&& echo PUSH_MARKER_OK"
    )
    r = ssh(ip, build_cmd, timeout=3600)
    print("=== build tail ===", flush=True)
    print(r.stdout[-3000:], flush=True)
    if r.returncode != 0 or "PUSH_MARKER_OK" not in r.stdout:
        sys.exit(f"BUILD FAILED (exit {r.returncode}, marker={'PUSH_MARKER_OK' in r.stdout})")

    r = ssh(ip, f"docker buildx imagetools inspect {IMAGE}:{NEW_TAG} | head -5")
    print("=== pushed image ===", flush=True)
    print(r.stdout, flush=True)
    print("BUILD_OK", flush=True)
finally:
    if droplet_id:
        try:
            do_api("DELETE", f"/droplets/{droplet_id}")
            print(f"droplet {droplet_id} deleted", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"DROPLET DELETE FAILED ({droplet_id}): {exc} — backstop will reap it", flush=True)
