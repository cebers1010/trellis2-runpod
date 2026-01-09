import os
import sys
from huggingface_hub import snapshot_download

MODEL_REPO = "microsoft/TRELLIS.2-4B"
LOCAL_DIR = "./trellis-2-4b-model"

def download_model():
    print(f"Starting download of {MODEL_REPO} to {LOCAL_DIR}...", flush=True)
    try:
        # snapshot_download usually handles progress bars, but in CI/Docker they might be hidden.
        # We ensure the download happens and catch errors.
        path = snapshot_download(
            repo_id=MODEL_REPO, 
            local_dir=LOCAL_DIR,
            resume_download=True
        )
        print(f"Model downloaded successfully to {path}", flush=True)
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    download_model()
