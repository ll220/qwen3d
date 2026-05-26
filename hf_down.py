from huggingface_hub import snapshot_download
import shutil
import os

# Clear cache for this specific model
cache_dir = "[path-to-cache-dir]"
model_cache_path = "~/models--Qwen--Qwen2.5-VL-3B-Instruct"
if os.path.exists(model_cache_path):
    shutil.rmtree(model_cache_path)

# Re-download the model
snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", force_download=True)