# -*- coding: utf-8 -*-
"""
任务二：链路预测 —— 批量实验编排
=================================
  stage1  主对比：{Cora, Citeseer, Amazon} × {GCN,GAT,SAGE,GIN} × {全图, 子图采样}
  stage2  学习率消融：Cora × 4 模型 × lr ∈ {0.1, 0.01, 0.001}
  stage3  网络层数消融：Cora × 4 模型 × layers ∈ {1, 2, 3}
  stage4  采样参数消融：Cora × GCN × batch_size ∈ {256,1024,4096} × num_neighbors ∈ {5,10,20}
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_device  # noqa: E402
from run import DATA_ROOT, MODELS, run_single  # noqa: E402

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(CODE_DIR, "..", "..", "results"))

MAIN_SETS = ["Cora", "Citeseer", "Amazon"]
AMAZON_SPLITS = 10          # Amazon 全图训练：10 个公开随机划分
AMAZON_SAMPLE_SPLITS = 3    # Amazon 子图采样训练：3 个划分（4GB 显存下的折中）
EPOCHS_OVERRIDE = {"Amazon": 100}
# Amazon 采样训练的超参：大 batch + 小邻居数，显存友好且每轮仅 ~1.5s
AMAZON_SAMPLE_CFG = dict(batch_size=16384, num_neighbors=5)


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


def stage1(quick):
    print("\n" + "=" * 78)
    print("STAGE 1  主对比：数据集 × 模型 × 训练模式")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task2_main.csv")
    keys = ["dataset", "model", "mode", "split_id"]
    rows = []
    for ds in MAIN_SETS:
        for model in MODELS:
            for mode in ["full", "sample"]:
                if quick:
                    splits = [0]
                elif ds == "Amazon":
                    splits = list(range(AMAZON_SAMPLE_SPLITS if mode == "sample"
                                          else AMAZON_SPLITS))
                else:
                    splits = [0]
                cfg = AMAZON_SAMPLE_CFG if (ds == "Amazon" and mode == "sample") else {}
                for sid in splits:
                    row = dict(dataset=ds, model=model, mode=mode, split_id=sid)
                    if already_done(path, keys, row):
                        print(f"  [skip] {ds}/{model}/{mode}/s{sid}", flush=True)
                        continue
                    print(f"  [run ] {ds}/{model}/{mode}/s{sid}", flush=True)
                    try:
                        r = run_single(dataset=ds, model=model, mode=mode,
                                       epochs=EPOCHS_OVERRIDE.get(ds),
                                       split_id=sid, verbose=False, **cfg)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [FAIL] {type(e).__name__}: {str(e)[:160]}", flush=True)
                        torch.cuda.empty_cache()
                        continue
                    print(f"         test_auc={r['test_auc']:.4f} "
                          f"test_ap={r['test_ap']:.4f} train={r['train_time_s']:.1f}s "
                          f"per_ep={r['time_per_epoch_s']:.3f}s", flush=True)
                    rows.append(r)
                    append_csv(path, [r])
                    torch.cuda.empty_cache()
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage2():
    print("\n" + "=" * 78)
    print("STAGE 2  学习率消融（Cora, 全图训练）")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task2_lr.csv")
    keys = ["dataset", "model", "mode", "lr", "num_layers"]
    rows = []
    for model in MODELS:
        for lr in [0.1, 0.01, 0.001]:
            row = dict(dataset="Cora", model=model, mode="full", lr=lr, num_layers=2)
            if already_done(path, keys, row):
                print(f"  [skip] {model}/lr={lr}", flush=True)
                continue
            print(f"  [run ] Cora/{model}/lr={lr}", flush=True)
            r = run_single(dataset="Cora", model=model, mode="full",
                           lr=lr, layers=2, epochs=100)
            print(f"         test_auc={r['test_auc']:.4f} test_ap={r['test_ap']:.4f}",
                  flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage3():
    print("\n" + "=" * 78)
    print("STAGE 3  网络层数消融（Cora, 全图训练）")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task2_layers.csv")
    keys = ["dataset", "model", "mode", "lr", "num_layers"]
    rows = []
    for model in MODELS:
        for layers in [1, 2, 3]:
            row = dict(dataset="Cora", model=model, mode="full", lr=0.01,
                       num_layers=layers)
            if already_done(path, keys, row):
                print(f"  [skip] {model}/layers={layers}", flush=True)
                continue
            print(f"  [run ] Cora/{model}/layers={layers}", flush=True)
            r = run_single(dataset="Cora", model=model, mode="full",
                           lr=0.01, layers=layers, epochs=100)
            print(f"         test_auc={r['test_auc']:.4f} test_ap={r['test_ap']:.4f}",
                  flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage4():
    print("\n" + "=" * 78)
    print("STAGE 4  采样参数消融（Cora, GCN, 子图采样训练）")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task2_sampler.csv")
    keys = ["dataset", "model", "mode", "batch_size", "num_neighbors"]
    rows = []
    for bs in [256, 1024, 4096]:
        for nb in [5, 10, 20]:
            row = dict(dataset="Cora", model="gcn", mode="sample",
                       batch_size=bs, num_neighbors=nb)
            if already_done(path, keys, row):
                print(f"  [skip] bs={bs}/nb={nb}", flush=True)
                continue
            print(f"  [run ] Cora/gcn/sample bs={bs} nb={nb}", flush=True)
            r = run_single(dataset="Cora", model="gcn", mode="sample",
                           batch_size=bs, num_neighbors=nb, epochs=100)
            print(f"         test_auc={r['test_auc']:.4f} "
                  f"per_ep={r['time_per_epoch_s']:.3f}s", flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print(f"device = {get_device()}")
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    fn = {1: lambda: stage1(args.quick), 2: stage2, 3: stage3, 4: stage4}
    for s in args.stages:
        fn[s]()
    print(f"\n全部完成，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
