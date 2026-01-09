import os
from huggingface_hub import snapshot_download

MODEL_REPO = "microsoft/TRELLIS.2-4B"
LOCAL_DIR = "./trellis-2-4b-model"

print(f"Downloading {MODEL_REPO} to {LOCAL_DIR}...")
try:
    snapshot_download(repo_id=MODEL_REPO, local_dir=LOCAL_DIR)
    print("Model downloaded successfully.")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)
