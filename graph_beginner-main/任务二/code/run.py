# -*- coding: utf-8 -*-
"""
任务二：图上的链路预测 —— 单组实验入口
======================================
4 个模型（gcn / gat / sage / gin）× 2 种训练模式（full 全图 / sample 子图采样）
× 3 个数据集（Cora / Citeseer / Amazon）。

边划分：85% 训练 / 5% 验证 / 10% 测试（负采样比例 1:1），评估指标 ROC-AUC、AP。

用法示例
--------
python run.py --dataset Cora --model gcn --mode full
python run.py --dataset Amazon --model sage --mode sample --all-splits
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (DATASET_CFG, DATASET_INFO, LinkPredictor, Timer,  # noqa: E402
                    get_device, load_dataset, make_link_split,
                    peak_gpu_mem_mb, reset_gpu_mem, set_seed,
                    train_link_full, train_link_sampled)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(CODE_DIR, "..", "data"))
MODELS = ["gcn", "gat", "sage", "gin"]


def run_single(dataset="Cora", model="gcn", mode="full", epochs=None, lr=None,
               layers=2, hidden=None, dropout=0.5, weight_decay=5e-4,
               batch_size=1024, num_neighbors=10, seed=42, split_id=0,
               device=None, verbose=False, data_root=DATA_ROOT):
    cfg = DATASET_CFG.get(dataset, dict(epochs=100, lr=0.01, hidden=64, gat_heads=8))
    epochs = int(epochs if epochs is not None else cfg["epochs"])
    lr = float(lr if lr is not None else cfg["lr"])
    hidden = int(hidden if hidden is not None else cfg["hidden"])
    gat_heads = int(cfg.get("gat_heads", 8))

    set_seed(seed)
    device = device or get_device()

    info = load_dataset(dataset, data_root)
    data = info["data"].clone()          # 避免污染进程内缓存（缓存始终在 CPU）

    # 同一数据集的不同 split_id 用不同随机种子做边划分，避免完全重复
    train_data, val_data, test_data = make_link_split(data, seed=seed + split_id)

    net = LinkPredictor(info["num_features"], hidden, num_layers=layers,
                        kind=model, dropout=dropout, gat_heads=gat_heads)
    n_params = sum(p.numel() for p in net.parameters())

    reset_gpu_mem()
    t_all = Timer()
    with t_all:
        if mode == "full":
            val_auc, test_auc, test_ap, loop_t, ep_times = train_link_full(
                net, train_data, val_data, test_data, device,
                epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=verbose)
        elif mode == "sample":
            val_auc, test_auc, test_ap, loop_t, ep_times = train_link_sampled(
                net, train_data, val_data, test_data, device,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, num_neighbors=num_neighbors, verbose=verbose)
        else:
            raise ValueError(f"未知训练模式: {mode}")

    pure_train = float(sum(ep_times))
    return {
        "task": "link_prediction",
        "dataset": dataset,
        "model": model,
        "mode": mode,
        "split_id": split_id,
        "num_layers": layers,
        "hidden": hidden,
        "gat_heads": gat_heads if model == "gat" else "",
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size if mode == "sample" else int(data.num_nodes),
        "num_neighbors": num_neighbors if mode == "sample" else "",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "train_edges": int(train_data.edge_label_index.size(1)),
        "val_edges": int(val_data.edge_label_index.size(1)),
        "test_edges": int(test_data.edge_label_index.size(1)),
        "n_params": int(n_params),
        "val_auc": round(float(val_auc), 4),
        "test_auc": round(float(test_auc), 4),
        "test_ap": round(float(test_ap), 4),
        "train_time_s": round(pure_train, 2),
        "loop_time_s": round(float(loop_t), 2),
        "time_per_epoch_s": round(pure_train / max(len(ep_times), 1), 4),
        "wall_time_s": round(float(t_all.t), 2),
        "peak_gpu_mb": round(float(peak_gpu_mem_mb()), 1),
        "seed": seed,
    }


def main():
    p = argparse.ArgumentParser(description="任务二 链路预测")
    p.add_argument("--dataset", default="Cora", choices=list(DATASET_INFO))
    p.add_argument("--model", default="gcn", choices=MODELS)
    p.add_argument("--mode", default="full", choices=["full", "sample"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-neighbors", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-id", type=int, default=0)
    p.add_argument("--all-splits", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    n_splits = 10 if args.dataset == "Amazon" else 1
    split_ids = list(range(n_splits)) if args.all_splits else [args.split_id]

    rows = []
    for sid in split_ids:
        print(f"[run] {args.dataset} | {args.model} | {args.mode} | split {sid}", flush=True)
        row = run_single(dataset=args.dataset, model=args.model, mode=args.mode,
                         epochs=args.epochs, lr=args.lr, layers=args.layers,
                         hidden=args.hidden, dropout=args.dropout,
                         weight_decay=args.weight_decay,
                         batch_size=args.batch_size,
                         num_neighbors=args.num_neighbors,
                         seed=args.seed, split_id=sid, device=args.device,
                         verbose=args.verbose)
        print("  -> " + json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)

    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df = pd.DataFrame(rows)
        if os.path.exists(args.out):
            df.to_csv(args.out, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
