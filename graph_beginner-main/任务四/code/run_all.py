# -*- coding: utf-8 -*-
"""
任务四：知识图谱补全 —— 批量实验编排
=====================================
  stage1  主对比：{wn18rr, countries_S1} × {TransE, RotatE, ConvE}
  stage2  嵌入维度消融：countries_S1 × {TransE, RotatE} × dim ∈ {50, 100, 200}
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

DATASETS = ["countries_S1", "wn18rr"]


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
    print("STAGE 1  主对比：数据集 × {TransE, RotatE, ConvE}")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task4_kge.csv")
    keys = ["dataset", "model"]
    rows = []
    for ds in DATASETS:
        for model in MODELS:
            row = dict(dataset=ds, model=model)
            if already_done(path, keys, row):
                print(f"  [skip] {ds}/{model}", flush=True)
                continue
            print(f"  [run ] {ds}/{model}", flush=True)
            try:
                r = run_single(dataset=ds, model=model)
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {type(e).__name__}: {str(e)[:150]}", flush=True)
                torch.cuda.empty_cache()
                continue
            print(f"         MRR={r['test_mrr']:.4f} H@1={r['test_hits@1']:.4f} "
                  f"H@10={r['test_hits@10']:.4f} "
                  f"per_ep={r['time_per_epoch_s']:.3f}s", flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage2():
    print("\n" + "=" * 78)
    print("STAGE 2  嵌入维度消融（countries_S1）")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task4_dim.csv")
    keys = ["dataset", "model", "dim"]
    rows = []
    for model in ["transe", "rotate"]:
        for dim in [50, 100, 200]:
            row = dict(dataset="countries_S1", model=model, dim=dim)
            if already_done(path, keys, row):
                print(f"  [skip] {model}/dim={dim}", flush=True)
                continue
            print(f"  [run ] countries_S1/{model}/dim={dim}", flush=True)
            r = run_single(dataset="countries_S1", model=model, dim=dim,
                           epochs=200)
            print(f"         MRR={r['test_mrr']:.4f} H@10={r['test_hits@10']:.4f}",
                  flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", type=int, default=[1, 2])
    args = ap.parse_args()
    print(f"device = {get_device()}")
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    fn = {1: stage1, 2: stage2}
    for s in args.stages:
        fn[s]()
    print(f"\n全部完成，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
