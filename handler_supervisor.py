"""v9.2 supervisor: thin runpod handler + persistent generation child (resident_child.py).

The SDK process never imports torch. The child holds the resident LTX-2.5 pipeline; if it
dies (segfault, OOM, SystemExit), the supervisor reports the child's stderr tail in the job
response and restarts it on the next job.
"""
import base64
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import runpod

CHILD = Path("/app/resident_child.py")
ERRLOG = Path("/tmp/child.err")
_STATE = {"proc": None, "q": None, "init": None}

# v9.20 guard, see cuda_preflight.py: a host whose driver predates the image's CUDA runtime used to
# announce itself ready and fail every job it took. The probe runs in its own short-lived process
# so this one keeps its promise of never importing torch.
PREFLIGHT = Path("/app/cuda_preflight.py")
PREFLIGHT_ENABLED = os.environ.get("LTX25_PREFLIGHT", "1").strip().lower() in ("1", "true", "on")
PREFLIGHT_TIMEOUT_S = int(os.environ.get("LTX25_PREFLIGHT_TIMEOUT_S", "300") or 300)

# v9.13: the finished clip goes straight to Spaces from here (4-9 MB of base64 through RunPod's
# status API and a second upload from the droplet were ~1.5-2 s of every clip). Output carries
# the object keys + probe metadata; anything missing or failing falls back to video_b64.
SPACES = {k: os.environ.get(f"SPACES_{k}", "").strip() for k in ("REGION", "BUCKET", "ACCESS_KEY", "SECRET_KEY")}
SPACES_ENABLED = all(SPACES.values()) and os.environ.get("LTX25_DIRECT_UPLOAD", "1").strip().lower() in ("1", "true", "on")
VIDEO_FOLDER = os.environ.get("LTX25_UPLOAD_FOLDER", "octoai_videos").strip("/")
_S3 = {"client": None}


def _s3():
    if _S3["client"] is None:
        import boto3  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415

        _S3["client"] = boto3.client(
            "s3", region_name=SPACES["REGION"], endpoint_url=f"https://{SPACES['REGION']}.digitaloceanspaces.com",
            aws_access_key_id=SPACES["ACCESS_KEY"], aws_secret_access_key=SPACES["SECRET_KEY"],
            config=Config(retries={"max_attempts": 3}, connect_timeout=5, read_timeout=60))
    return _S3["client"]


MAX_SOURCE_BYTES = 64 * 1024 * 1024


def _fetch_source(url, dest):
    """Download a retake source (http/https only, capped) straight into the container."""
    import urllib.request  # noqa: PLC0415

    if not url.startswith(("http://", "https://")):
        raise ValueError("video_url must be http(s)")
    req = urllib.request.Request(url, headers={"User-Agent": "YEngineWorker/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_SOURCE_BYTES:
                raise ValueError(f"source larger than {MAX_SOURCE_BYTES} bytes")
            out.write(chunk)
    if total == 0:
        raise ValueError("empty source")
    return total


def _probe(path):
    """width/height/has_audio the backend would otherwise ffprobe on the droplet."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                          "-of", "json", str(path)], capture_output=True, text=True, timeout=30).stdout
    meta = {"width": None, "height": None, "has_audio": False}
    for s in json.loads(out or "{}").get("streams", []):
        if s.get("codec_type") == "video" and meta["width"] is None:
            meta["width"], meta["height"] = s.get("width"), s.get("height")
        elif s.get("codec_type") == "audio":
            meta["has_audio"] = True
    return meta


def _upload_clip(out):
    """PUT clip + first-frame poster under the backend's own key layout; returns output fields."""
    t0 = time.time()
    data = Path(out).read_bytes()
    asset_id = uuid.uuid4().hex
    video_key = f"{VIDEO_FOLDER}/{asset_id}.mp4"
    preview_key = f"{VIDEO_FOLDER}/{asset_id}_preview.jpg"
    preview = Path("/tmp/preview.jpg")
    preview.unlink(missing_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-frames:v", "1", "-q:v", "3", str(preview)],
                   capture_output=True, timeout=60)
    meta = _probe(out)
    s3 = _s3()
    s3.put_object(Bucket=SPACES["BUCKET"], Key=video_key, Body=data, ContentType="video/mp4", ACL="public-read",
                  CacheControl="public, max-age=31536000, immutable")
    if preview.is_file():
        s3.put_object(Bucket=SPACES["BUCKET"], Key=preview_key, Body=preview.read_bytes(), ContentType="image/jpeg",
                      ACL="public-read", CacheControl="public, max-age=31536000, immutable")
    else:
        preview_key = None
    return {"video_key": video_key, "preview_key": preview_key, "byte_length": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "upload_s": round(time.time() - t0, 2), **meta}


def _reader(proc, q):
    for line in proc.stdout:
        q.put(line)
    q.put(None)  # EOF marker


def _stderr_tail(n=2500):
    try:
        return ERRLOG.read_text(errors="replace")[-n:]
    except OSError:
        return ""


def _child_alive():
    p = _STATE["proc"]
    return p is not None and p.poll() is None


def _start_child():
    ERRLOG.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [sys.executable, str(CHILD)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=open(ERRLOG, "ab"),
        cwd="/app/LTX-2", text=True, bufsize=1,
    )
    q = queue.Queue()
    threading.Thread(target=_reader, args=(proc, q), daemon=True).start()
    _STATE.update(proc=proc, q=q, init=None)
    print(f"[supervisor] child started pid={proc.pid}", flush=True)


def _rpc(msg, timeout):
    p, q = _STATE["proc"], _STATE["q"]
    p.stdin.write(json.dumps(msg) + "\n")
    p.stdin.flush()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = q.get(timeout=5)
        except queue.Empty:
            if not _child_alive():
                return {"ok": False, "error": "child died", "stderr_tail": _stderr_tail()}
            continue
        if line is None:
            return {"ok": False, "error": "child EOF", "stderr_tail": _stderr_tail()}
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            print("[supervisor] non-json from child:", line[:200], flush=True)
    return {"ok": False, "error": f"child rpc timeout {timeout}s", "stderr_tail": _stderr_tail()}


def _preflight_or_die():
    """Fail closed: anything short of a working GPU exits before the SDK accepts a single job."""
    if not PREFLIGHT_ENABLED:
        print("[preflight] disabled by LTX25_PREFLIGHT", flush=True)
        return
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(PREFLIGHT)], capture_output=True, text=True,
                           timeout=PREFLIGHT_TIMEOUT_S, cwd="/app/LTX-2")
        ok = r.returncode == 0
        report = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        ok, report = False, f"probe timed out after {PREFLIGHT_TIMEOUT_S}s"
    except Exception as exc:  # noqa: BLE001
        ok, report = False, f"probe could not run: {exc!r}"[:300]
    print(f"[preflight] {'ok' if ok else 'FAILED'} in {time.time() - t0:.1f}s: {report}", flush=True)
    if not ok:
        print("[preflight] exiting non-zero so the platform replaces this worker", flush=True)
        sys.stdout.flush()
        sys.exit(1)


def handler(job):
    inp = job.get("input") or {}
    try:
        if not _child_alive():
            _start_child()
        if _STATE["init"] is None:
            r = _rpc({"cmd": "init"}, timeout=3600)
            if not r.get("ok"):
                return {"error": "init failed", **{k: r[k] for k in ("error", "trace") if r.get(k)},
                        "stderr_tail": r.get("stderr_tail") or _stderr_tail(1500)}
            init = r.get("init") or {}
            if init.get("fatal"):
                # Second line of defence behind the preflight: init itself proved the GPU is
                # unusable. Die without answering — RunPod requeues the job onto another worker,
                # which beats burning it here and keeping the bad host in the pool.
                print(f"[preflight] init reported fatal: {init['fatal']} — exiting", flush=True)
                sys.stdout.flush()
                os._exit(1)
            _STATE["init"] = init

        out = "/tmp/out.mp4"
        task = str(inp.get("task") or "gen").lower()
        if task == "retake":
            src = Path("/tmp/retake_src.mp4")
            if inp.get("video_url"):
                # v9.17: the source comes by URL — base64 in /run hit RunPod's request cap on big
                # clips (an 18 MB 720p source became 24 MB of JSON → 400/502, 23.08).
                try:
                    _fetch_source(str(inp["video_url"]), src)
                except Exception as fe:  # noqa: BLE001
                    return {"error": f"retake source fetch failed: {str(fe)[:200]}"}
            elif inp.get("video_b64"):
                src.write_bytes(base64.b64decode(inp["video_b64"]))
            else:
                return {"error": "retake requires video_url or video_b64 (source video)"}
            r_inp = dict(inp)
            r_inp.pop("video_b64", None)
            r_inp.pop("video_url", None)
            r_inp.pop("task", None)
            r = _rpc({"cmd": "retake", "input": r_inp, "src": str(src), "out": out}, timeout=1800)
        else:
            gen_inp = dict(inp)
            if inp.get("image_b64"):
                img = Path("/tmp/in.png")
                img.write_bytes(base64.b64decode(inp["image_b64"]))
                gen_inp.pop("image_b64", None)
                gen_inp["image_path"] = str(img)
            r = _rpc({"cmd": "gen", "input": gen_inp, "out": out}, timeout=1800)
        if not r.get("ok"):
            return {"error": r.get("error", "gen failed"),
                    **{k: r[k] for k in ("trace", "vram_free_gib", "vram_total_gib", "child_restart") if r.get(k)},
                    "stderr_tail": r.get("stderr_tail") or _stderr_tail(1500),
                    "init": _STATE["init"]}
        result = {"gen_seconds": r.get("gen_s"), "timings": r.get("timings"), "init": _STATE["init"],
                  "build_commit": Path("/BUILD_COMMIT").read_text().strip()[:12] if Path("/BUILD_COMMIT").exists() else None}
        if inp.get("discard_output"):  # keepalive: nothing to deliver, nothing to store
            return result
        if SPACES_ENABLED and not inp.get("return_b64"):
            try:
                result.update(_upload_clip(out))
                return result
            except Exception as ue:  # noqa: BLE001
                print(f"[supervisor] direct upload failed, returning base64: {ue!r}", flush=True)
                result["upload_error"] = str(ue)[:200]
        result["video_b64"] = base64.b64encode(Path(out).read_bytes()).decode()
        return result
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc()[-2000:],
                "stderr_tail": _stderr_tail()}


_preflight_or_die()
runpod.serverless.start({"handler": handler})
