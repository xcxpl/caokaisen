# -*- coding: utf-8 -*-
"""
任务一：节点分类 —— 批量实验编排
=================================
分 4 个阶段：
  stage1  主对比：{Cora, Citeseer, Amazon} × {GCN, GAT, SAGE, GIN} × {全图, 子图采样}
  stage2  学习率消融：Cora × 4 模型 × lr ∈ {0.1, 0.01, 0.001}
  stage3  网络层数消融：Cora × 4 模型 × layers ∈ {1, 2, 3}
  stage4  采样参数消融：Cora × GCN × batch_size ∈ {256,1024,4096} × num_neighbors ∈ {5,10,20}

结果写入 results/ 下的 csv（追加 + 断点续跑）。
用法：
    python run_all.py                 # 全部阶段
    python run_all.py --stages 2 3    # 指定阶段
    python run_all.py --quick         # 快速模式（Amazon 只跑 1 个划分、epoch 减半）
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_device, get_masks, load_dataset  # noqa: E402
from run import DATA_ROOT, MODELS, run_single  # noqa: E402

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(CODE_DIR, "..", "..", "results"))

MAIN_SETS = ["Cora", "Citeseer", "Amazon"]
# 大图作为“采样收益最明显”的数据集，减少 epoch 以控制总耗时
EPOCHS_OVERRIDE = {"Amazon": 100}
# 子图采样训练的 batch_size（种子节点数）。
# 注意：Cora/Citeseer/Amazon 的公开划分训练节点只有 140/120/200 个，
# 若 batch_size 取 1024 则「每轮只有 1 个 mini-batch」，采样训练会退化成
# 「一次全批量更新」，与全图训练的时间对比就失去意义。这里统一取 64，
# 保证每个数据集每轮都有 2~4 个真实的 mini-batch。
SAMPLE_BATCH = {"Cora": 64, "Citeseer": 64, "Amazon": 64}


def n_splits_of(dataset):
    info = load_dataset(dataset, os.path.join(DATA_ROOT, dataset))
    _, _, _, k = get_masks(info["data"], 0)
    return k


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
    path = os.path.join(RESULTS, "task1_main.csv")
    keys = ["dataset", "model", "mode", "split_id"]
    rows = []
    for ds in MAIN_SETS:
        splits = [0] if (quick or ds != "Amazon") else list(range(n_splits_of(ds)))
        for model in MODELS:
            for mode in ["full", "sample"]:
                for sid in splits:
                    row = dict(dataset=ds, model=model, mode=mode, split_id=sid)
                    if already_done(path, keys, row):
                        print(f"  [skip] {ds}/{model}/{mode}/s{sid}", flush=True)
                        continue
                    print(f"  [run ] {ds}/{model}/{mode}/s{sid}", flush=True)
                    kw = {}
                    if mode == "sample":
                        kw["batch_size"] = SAMPLE_BATCH.get(ds, 64)
                        kw["num_neighbors"] = 10
                    try:
                        r = run_single(dataset=ds, model=model, mode=mode,
                                       epochs=EPOCHS_OVERRIDE.get(ds),
                                       split_id=sid, verbose=False, **kw)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [FAIL] {type(e).__name__}: {str(e)[:160]}", flush=True)
                        torch.cuda.empty_cache()
                        continue
                    print(f"         test_acc={r['test_acc']:.4f} "
                          f"train={r['train_time_s']:.1f}s "
                          f"per_ep={r['time_per_epoch_s']:.3f}s gpu={r['peak_gpu_mb']:.0f}MB",
                          flush=True)
                    rows.append(r)
                    append_csv(path, [r])
                    torch.cuda.empty_cache()
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage2():
    print("\n" + "=" * 78)
    print("STAGE 2  学习率消融（Cora, 全图训练）")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task1_lr.csv")
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
                           lr=lr, layers=2, epochs=200)
            print(f"         test_acc={r['test_acc']:.4f}", flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage3():
    print("\n" + "=" * 78)
    print("STAGE 3  网络层数消融（Cora, 全图训练）")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task1_layers.csv")
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
                           lr=0.01, layers=layers, epochs=200)
            print(f"         test_acc={r['test_acc']:.4f}", flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage4():
    print("\n" + "=" * 78)
    print("STAGE 4  采样参数消融（Cora, GCN, 子图采样训练）")
    print("=" * 78, flush=True)
    # 注意：Cora 公开划分只有 140 个训练节点，batch_size 必须 < 140 才能产生
    # 多个 mini-batch，否则每轮只有 1 个 batch，batch_size 对结果没有任何影响。
    path = os.path.join(RESULTS, "task1_sampler.csv")
    keys = ["dataset", "model", "mode", "batch_size", "num_neighbors"]
    rows = []
    for bs in [16, 32, 64]:
        for nb in [5, 10, 20]:
            row = dict(dataset="Cora", model="gcn", mode="sample",
                       batch_size=bs, num_neighbors=nb)
            if already_done(path, keys, row):
                print(f"  [skip] bs={bs}/nb={nb}", flush=True)
                continue
            print(f"  [run ] Cora/gcn/sample bs={bs} nb={nb}", flush=True)
            r = run_single(dataset="Cora", model="gcn", mode="sample",
                           batch_size=bs, num_neighbors=nb, epochs=200)
            print(f"         test_acc={r['test_acc']:.4f} "
                  f"per_ep={r['time_per_epoch_s']:.3f}s", flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def stage5():
    """
    补充实验：Amazon 上 GIN 全图训练的学习率敏感性。
    stage1 显示 GIN 在 Amazon 全图训练下准确率仅 ~0.24（而采样训练 ~0.70），
    这里通过扫描学习率判断是「优化问题」还是「表达能力问题」。
    """
    print("\n" + "=" * 78)
    print("STAGE 5  补充：Amazon × GIN × 全图训练 × lr ∈ {1e-2, 1e-3, 1e-4}")
    print("=" * 78, flush=True)
    path = os.path.join(RESULTS, "task1_gin_amazon_lr.csv")
    keys = ["dataset", "model", "mode", "lr", "split_id"]
    rows = []
    for lr in [1e-2, 1e-3, 1e-4]:
        for sid in [0, 1]:
            row = dict(dataset="Amazon", model="gin", mode="full", lr=lr,
                       split_id=sid)
            if already_done(path, keys, row):
                print(f"  [skip] lr={lr}/s{sid}", flush=True)
                continue
            print(f"  [run ] Amazon/gin/full lr={lr} s{sid}", flush=True)
            r = run_single(dataset="Amazon", model="gin", mode="full",
                           lr=lr, split_id=sid, epochs=100)
            print(f"         test_acc={r['test_acc']:.4f} val_acc={r['val_acc']:.4f}",
                  flush=True)
            rows.append(r)
            append_csv(path, [r])
    print(f"  -> {path} ({len(rows)} 条新记录)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"device = {device}")
    if torch.cuda.is_available():
        print(f"gpu    = {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    fn = {1: lambda: stage1(args.quick), 2: stage2, 3: stage3, 4: stage4,
          5: stage5}
    for s in args.stages:
        fn[s]()
    print(f"\n全部完成，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
