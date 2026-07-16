#!/usr/bin/env python3
"""Post-deploy verification for the t2v upsampler path.

    RUNPOD_VIDEO_KEY=... python3 ops/verify_t2v.py [expected-tag]   # default: v8.19

Gates:
  1. config_tag carries the expected tag  -> the new image is actually serving
  2. the seeds that broke come back clean -> no degenerate/junk-prefixed rewrites
  3. enhance:false is bit-exact old MD5   -> the deploy touched only the upsampler
  4. i2v enhance_probe still rewrites     -> vision path not regressed

seeds 0 and 7 on "football player"/"red dragon" are not arbitrary: they are the draws that
reproduced the v8.18 upsampler artifact (see _enhance_prompt_t2v). Keep them.
"""
import base64
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

EP = "https://api.runpod.ai/v2/g9mihjeqiml41t"
KEY = os.environ["RUNPOD_VIDEO_KEY"]
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
EXPECT_TAG = sys.argv[1] if len(sys.argv) > 1 else "v8.19"
# "football player" @ seed 0 with NO enhancement: the original kitchen video. Pinning it
# proves a deploy changed only the upsampler and never touched the sampler.
OLD_KITCHEN_MD5 = "fb8c550e261152be666ba6ff75f16268"
WARMUP_JPG = str(pathlib.Path(__file__).resolve().parent.parent / "warmup.jpg")
fails = []


def call(method, url, body=None, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers=HDRS, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(inp, tries=110):
    job = call("POST", f"{EP}/run", {"input": inp})["id"]
    for _ in range(tries):
        time.sleep(6)
        r = call("GET", f"{EP}/status/{job}")
        if r.get("status") == "COMPLETED":
            return r["output"]
        if r.get("status") in ("FAILED", "CANCELLED"):
            return {"error": json.dumps(r)[:200]}
    return {"error": "timeout"}


def t2v(label, prompt, seed, **extra):
    out = run({"prompt": prompt, "width": 1280, "height": 704, "frames": 121, "fps": 24,
               "audio": True, "tier": "quality", "seed": seed, **extra})
    if out.get("error"):
        fails.append(f"{label}: {out['error']}")
        return f"  [{label}] ERROR {out['error'][:90]}"
    vb = out.pop("video_b64", None)
    md5 = hashlib.md5(base64.b64decode(vb)).hexdigest() if vb else None
    tag = out.get("config_tag", "")
    ep = (out.get("enhanced_prompt") or "").strip()
    w = out.get("enhance_words") or 0
    lines = [f"  [{label}] tag={tag[:22]} words={w} md5={md5[:8] if md5 else '-'}"]
    if ep:
        lines.append(f"      head: {ep[:100]}")
    if EXPECT_TAG not in tag:
        fails.append(f"{label}: stale image, want {EXPECT_TAG}, got tag={tag}")
    if extra.get("enhance") is False:
        if md5 != OLD_KITCHEN_MD5:
            fails.append(f"{label}: sampler drifted, md5={md5} want={OLD_KITCHEN_MD5}")
        else:
            lines.append("      BIT-EXACT OK (sampler untouched)")
    else:
        if w < 40:
            fails.append(f"{label}: degenerate rewrite ({w} words): {ep[:60]!r}")
        first = ep.split()[0] if ep else ""
        if first[:1].islower():
            fails.append(f"{label}: junk prefix survived: {ep[:60]!r}")
    return "\n".join(lines)


print("=== t2v on the seeds that broke (bare prompts, as the app sends them) ===", flush=True)
cases = [("football-s0", "football player", 0), ("football-s7", "football player", 7),
         ("dragon-s0", "red dragon", 0), ("ua-dragon", "дракон", 0)]
with ThreadPoolExecutor(max_workers=4) as ex:
    for line in ex.map(lambda a: t2v(*a), cases):
        print(line, flush=True)

print("=== regression gates ===", flush=True)
print(t2v("noenh-bitexact", "football player", 0, enhance=False), flush=True)

img = base64.b64encode(open(WARMUP_JPG, "rb").read()).decode()
probe = call("POST", f"{EP}/runsync", {"input": {"enhance_probe": True,
                                                 "prompt": "the man walks forward",
                                                 "image_b64": img}}, timeout=600)
pout = probe.get("output") or {}
enh = (pout.get("enhanced") or "").strip()
print(f"  [i2v-probe] {pout.get('enhance_probe')} | words={len(enh.split())}", flush=True)
print(f"      head: {enh[:100]}", flush=True)
if pout.get("enhance_probe") != "ok" or len(enh.split()) < 40:
    fails.append(f"i2v probe regressed: {str(pout)[:120]}")

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("ALL GATES PASSED")
