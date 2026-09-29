# -*- coding: utf-8 -*-
"""
任务四：知识图谱 —— 数据下载脚本
=================================
从 RotatE 官方仓库（DeepGraphLearning/KnowledgeGraphEmbedding）拉取标准数据集：

    data/<name>/train.txt / valid.txt / test.txt
    data/<name>/entities.dict / relations.dict

本机通过代理访问外网时 urllib 容易卡住，因此优先调用 curl，失败再回退 urllib。

用法：
    python download_data.py                       # 下载 countries_S1 + wn18rr
    python download_data.py --datasets wn18rr
"""
import argparse
import os
import shutil
import subprocess
import time
import urllib.request

BASE = ("https://raw.githubusercontent.com/DeepGraphLearning/"
        "KnowledgeGraphEmbedding/master/data")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(CODE_DIR, "..", "data"))

FILES = ["train.txt", "valid.txt", "test.txt", "entities.dict", "relations.dict"]


def _curl(url, path):
    exe = shutil.which("curl") or "curl"
    cmd = [exe, "-sL", "--retry", "4", "--retry-delay", "2", "-m", "600",
           "--speed-limit", "1024", "--speed-time", "180", "-o", path, url]
    return subprocess.run(cmd).returncode == 0


def _urllib(url, path):
    with urllib.request.urlopen(url, timeout=180) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 16)
    return True


def download(url, path, retries=6):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("  [skip] %s (%d bytes)"
              % (os.path.basename(path), os.path.getsize(path)))
        return path
    for i in range(1, retries + 1):
        for tag, fn in (("curl", _curl), ("urllib", _urllib)):
            try:
                print("  [get ] %s via %s (第 %d 次)" % (url, tag, i), flush=True)
                if fn(url, path) and os.path.exists(path) and os.path.getsize(path) > 0:
                    print("  [ok  ] %s (%d bytes)"
                          % (os.path.basename(path), os.path.getsize(path)),
                          flush=True)
                    return path
            except Exception as e:
                print("  [fail] %s: %s" % (tag, str(e)[:120]), flush=True)
            if os.path.exists(path):
                os.remove(path)
            time.sleep(2)
    raise RuntimeError("download failed: " + url)


def fetch_dataset(name):
    print("=== %s ===" % name, flush=True)
    d = os.path.join(DATA_ROOT, name)
    for fn in FILES:
        download("%s/%s/%s" % (BASE, name, fn), os.path.join(d, fn))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["countries_S1", "wn18rr"])
    args = ap.parse_args()
    for n in args.datasets:
        fetch_dataset(n)
    print("完成")


if __name__ == "__main__":
    main()
