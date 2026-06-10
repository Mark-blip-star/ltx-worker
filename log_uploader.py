"""Upload the worker log to the private HF logs repo (read back via HF API).

Runs as a separate process with HF_HUB_OFFLINE scrubbed (RunPod forces it on cached-model
workers, which would otherwise block the upload).
"""
import os
import sys

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import HfApi  # noqa: E402

wid, path = sys.argv[1], sys.argv[2]
HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
    path_or_fileobj=path,
    path_in_repo=f"logs/{wid}.log",
    repo_id=os.environ.get("LTX_LOGS_REPO", "Markooooo/ltx-worker-logs"),
    repo_type="model",
    commit_message=f"log {wid}",
)
