"""v9.20 fail-closed GPU probe: runs once before the SDK is allowed to accept jobs.

2026-09-08 prod incident. RunPod handed the video endpoints H200 hosts whose NVIDIA driver was
12.8 while this image needs CUDA 13, so torch could not initialise CUDA at all. Nothing in the
worker treated that as fatal: the compile-cache key degraded to `…-na` (no SM detected), the x264
bench returned an error, and natten's DiffVAE tile planner computed `usable_bytes=0` — every job
died with "Cannot fit a DiffVAE decode tile under the memory budget", an error that reads like a
resolution problem and is nothing of the sort. The worker stayed in the pool and kept taking jobs;
one such worker ate ten of them in an hour. The endpoint's `allowedCudaVersions` filter was set to
["13.0"] the whole time and did not prevent it.

A worker that cannot use its GPU must exit before it is ready, not serve. Exit code 1 makes the
platform replace it, which is the only outcome that ends with a healthy host taking the traffic.
"""
import json
import sys

# Ampere and up. The prod fleet is Hopper (sm90); the floor exists to catch "no GPU at all",
# not to gate on architecture — a real capability check is what separates a live GPU from `-na`.
MIN_CAPABILITY = (8, 0)


def probe():
    out = {"ok": False}
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"torch import failed: {exc!r}"[:300]
        return out

    out["torch"] = torch.__version__
    out["cuda_runtime"] = torch.version.cuda
    try:
        if not torch.cuda.is_available():
            out["error"] = "torch.cuda.is_available() is False"
            return out
        cap = torch.cuda.get_device_capability()
        out["sm"] = f"sm{cap[0]}{cap[1]}"
        out["device"] = torch.cuda.get_device_name()
        if cap < MIN_CAPABILITY:
            out["error"] = f"compute capability {out['sm']} below sm{MIN_CAPABILITY[0]}{MIN_CAPABILITY[1]}"
            return out

        # An allocation and a real kernel: on the 12.8 hosts the driver complaint surfaces at the
        # first launch, not at is_available().
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        (a @ a).sum().item()
        torch.cuda.synchronize()

        free_b, total_b = torch.cuda.mem_get_info()
        out["vram_free_gib"] = round(free_b / 2**30, 1)
        out["vram_total_gib"] = round(total_b / 2**30, 1)
        if free_b <= 0:
            # The exact number the tile planner divides by; zero here means every decode fails.
            out["error"] = "mem_get_info reports no free VRAM"
            return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out

    out["ok"] = True
    return out


if __name__ == "__main__":
    report = probe()
    print(json.dumps(report), flush=True)
    sys.exit(0 if report.get("ok") else 1)
