import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

t0 = time.time()
path = snapshot_download(
    repo_id="bert-base-chinese",
    allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"],
    max_workers=4,
)
print("downloaded to:", path)
print("elapsed: %.1f s" % (time.time() - t0))
for f in sorted(os.listdir(path)):
    fp = os.path.join(path, f)
    print("  %-28s %10.2f MB" % (f, os.path.getsize(fp) / 1024 / 1024))
