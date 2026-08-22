"""v9.2 supervisor: thin runpod handler + persistent generation child (resident_child.py).

The SDK process never imports torch. The child holds the resident LTX-2.5 pipeline; if it
dies (segfault, OOM, SystemExit), the supervisor reports the child's stderr tail in the job
response and restarts it on the next job.
"""
import base64
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import runpod

CHILD = Path("/app/resident_child.py")
ERRLOG = Path("/tmp/child.err")
_STATE = {"proc": None, "q": None, "init": None}


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
            _STATE["init"] = r.get("init")

        out = "/tmp/out.mp4"
        task = str(inp.get("task") or "gen").lower()
        if task == "retake":
            if not inp.get("video_b64"):
                return {"error": "retake requires video_b64 (source video)"}
            src = Path("/tmp/retake_src.mp4")
            src.write_bytes(base64.b64decode(inp["video_b64"]))
            r_inp = dict(inp)
            r_inp.pop("video_b64", None)
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
        data = Path(out).read_bytes()
        return {"video_b64": base64.b64encode(data).decode(),
                "gen_seconds": r.get("gen_s"), "timings": r.get("timings"), "init": _STATE["init"],
                "build_commit": Path("/BUILD_COMMIT").read_text().strip()[:12] if Path("/BUILD_COMMIT").exists() else None}
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc()[-2000:],
                "stderr_tail": _stderr_tail()}


runpod.serverless.start({"handler": handler})
