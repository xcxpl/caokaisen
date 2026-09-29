# -*- coding: utf-8 -*-
"""
任务一：节点分类 —— 单组实验入口
================================
支持 4 个模型（gcn / gat / sage / gin）× 2 种训练模式（full 全图 / sample 子图采样）
× 3 个数据集（Cora / Citeseer / Amazon）。

用法示例
--------
# 全图训练：Cora + GCN
python run.py --dataset Cora --model gcn --mode full

# 子图采样训练：Amazon + GraphSAGE，10 个划分全部跑
python run.py --dataset Amazon --model sage --mode sample --all-splits

# 学习率 / 层数消融
python run.py --dataset Cora --model gcn --mode full --lr 0.005 --layers 3
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (DATASET_CFG, DATASET_INFO, NodeClassifier, Timer,  # noqa: E402
                    get_device, get_masks, load_dataset,
                    measure_inference_time, peak_gpu_mem_mb, reset_gpu_mem,
                    set_seed, train_full_graph, train_sampled_graph)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(CODE_DIR, "..", "data"))
MODELS = ["gcn", "gat", "sage", "gin"]


def run_single(dataset="Cora", model="gcn", mode="full", epochs=None, lr=None,
               layers=2, hidden=None, dropout=0.5, weight_decay=5e-4,
               batch_size=1024, num_neighbors=10, seed=42, split_id=0,
               device=None, verbose=False, data_root=DATA_ROOT):
    """跑一组配置，返回指标 dict。"""
    cfg = DATASET_CFG.get(dataset, dict(epochs=200, lr=0.01, hidden=64, gat_heads=8))
    epochs = int(epochs if epochs is not None else cfg["epochs"])
    lr = float(lr if lr is not None else cfg["lr"])
    hidden = int(hidden if hidden is not None else cfg["hidden"])
    gat_heads = int(cfg.get("gat_heads", 8))

    set_seed(seed)
    device = device or get_device()

    info = load_dataset(dataset, data_root)
    data = info["data"].clone()          # 避免污染进程内缓存（缓存始终在 CPU）
    train_mask, val_mask, test_mask, n_splits = get_masks(data, split_id)

    net = NodeClassifier(info["num_features"], hidden, info["num_classes"],
                         num_layers=layers, kind=model, dropout=dropout,
                         gat_heads=gat_heads)
    n_params = sum(p.numel() for p in net.parameters())

    reset_gpu_mem()
    t_all = Timer()
    with t_all:
        if mode == "full":
            val_acc, test_acc, train_time, ep_times = train_full_graph(
                net, data, train_mask, val_mask, test_mask, device,
                epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=verbose)
        elif mode == "sample":
            val_acc, test_acc, train_time, ep_times = train_sampled_graph(
                net, data, train_mask, val_mask, test_mask, device,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, num_neighbors=num_neighbors, verbose=verbose)
        else:
            raise ValueError(f"未知训练模式: {mode}")

    infer_ms = measure_inference_time(net, data, device)

    pure_train = float(sum(ep_times))   # 纯训练时间（不含评估/验证）
    return {
        "task": "node_classification",
        "dataset": dataset,
        "model": model,
        "mode": mode,                     # full = 全图训练, sample = 子图采样训练
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
        "n_params": int(n_params),
        "val_acc": round(float(val_acc), 4),
        "test_acc": round(float(test_acc), 4),
        "train_time_s": round(pure_train, 2),
        "loop_time_s": round(float(train_time), 2),
        "time_per_epoch_s": round(pure_train / max(len(ep_times), 1), 4),
        "wall_time_s": round(float(t_all.t), 2),
        "infer_time_ms": round(float(infer_ms), 2),
        "peak_gpu_mb": round(float(peak_gpu_mem_mb()), 1),
        "seed": seed,
    }


def main():
    p = argparse.ArgumentParser(description="任务一 节点分类")
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
    p.add_argument("--all-splits", action="store_true",
                   help="对该数据集的所有公开划分各跑一次并汇总")
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None, help="把结果 csv 追加写入该文件")
    args = p.parse_args()

    if args.all_splits:
        info = load_dataset(args.dataset, os.path.join(DATA_ROOT, args.dataset))
        _, _, _, n_splits = get_masks(info["data"], 0)
        split_ids = list(range(n_splits))
    else:
        split_ids = [args.split_id]

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
