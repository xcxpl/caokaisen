# -*- coding: utf-8 -*-
"""
任务三：图分类 —— 批量实验编排
===============================
  stage1  池化方法对比：{MUTAG, PROTEINS, IMDB-BINARY, ogbg-molhiv}
                        × {GCN,GAT,SAGE,GIN} × {AvgPool, MaxPool, MinPool}
  stage2  网络层数消融：MUTAG × 4 模型 × layers ∈ {2, 3, 4}（AvgPool）
  stage3  学习率消融：  MUTAG × 4 模型 × lr ∈ {0.1, 0.01, 0.001}（AvgPool）
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASET_INFO, get_device  # noqa: E402
from run import MODELS, run_single  # noqa: E402

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(CODE_DIR, "..", "..", "results"))

POOLS = ["avg", "max", "min"]
DATASETS = ["MUTAG", "PROTEINS", "IMDB-BINARY", "ogbg-molhiv"]


def already_done(path, key_cols, row):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty:
        return False
    mask = pd.Series(True, index=df.index)
    for c in key_cols:
        if c not in df.columns:
            return False
        mask &= (df[c].astype(str) == str(row[c]))
    return bool(mask.any())


def append_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


def stage1():
    print("\n" + "=" * 78)
    print("STAGE 1  池化方法对比：数据集 × 模型 × {AvgPool, MaxPool, MinPool}")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task3_pooling.csv")
    keys = ["dataset", "model", "pool"]
    rows = []
    for ds in DATASETS:
        for model in MODELS:
            for pool in POOLS:
                row = dict(dataset=ds, model=model, pool=pool)
                if already_done(path, keys, row):
                    print(f"  [skip] {ds}/{model}/{pool}", flush=True)
                    continue
                print(f"  [run ] {ds}/{model}/{pool}", flush=True)
                try:
                    r = run_single(dataset=ds, model=model, pool=pool)
                except Exception as e:  # noqa: BLE001
                    print(f"  [FAIL] {type(e).__name__}: {str(e)[:150]}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                m = r.get("metric_name", "test_acc")
                print(f"         {m}={r[m]:.4f} train={r['train_time_s']:.1f}s "
                      f"per_ep={r['time_per_epoch_s']:.3f}s gpu={r['peak_gpu_mb']:.0f}MB",
                      flush=True)
                rows.append(r)
                append_csv(path, [r])
                torch.cuda.empty_cache()
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage2():
    print("\n" + "=" * 78)
    print("STAGE 2  网络层数消融（MUTAG, AvgPool）")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task3_layers.csv")
    keys = ["dataset", "model", "pool", "num_layers"]
    rows = []
    for model in MODELS:
        for layers in [2, 3, 4]:
            row = dict(dataset="MUTAG", model=model, pool="avg", num_layers=layers)
            if already_done(path, keys, row):
                print(f"  [skip] {model}/L={layers}", flush=True)
                continue
            print(f"  [run ] MUTAG/{model}/L={layers}", flush=True)
            r = run_single(dataset="MUTAG", model=model, pool="avg", layers=layers)
            print(f"         test_acc={r['test_acc']:.4f}", flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage3():
    print("\n" + "=" * 78)
    print("STAGE 3  学习率消融（MUTAG, AvgPool）")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task3_lr.csv")
    keys = ["dataset", "model", "pool", "lr"]
    rows = []
    for model in MODELS:
        for lr in [0.1, 0.01, 0.001]:
            row = dict(dataset="MUTAG", model=model, pool="avg", lr=lr)
            if already_done(path, keys, row):
                print(f"  [skip] {model}/lr={lr}", flush=True)
                continue
            print(f"  [run ] MUTAG/{model}/lr={lr}", flush=True)
            r = run_single(dataset="MUTAG", model=model, pool="avg", lr=lr)
            print(f"         test_acc={r['test_acc']:.4f}", flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    print(f"device = {get_device()}")
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    fn = {1: stage1, 2: stage2, 3: stage3}
    for s in args.stages:
        fn[s]()
    print(f"\n全部完成，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
