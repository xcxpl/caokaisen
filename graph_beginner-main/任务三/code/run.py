# -*- coding: utf-8 -*-
"""
任务三：图分类 —— 单组实验入口
==============================
数据集：MUTAG / PROTEINS / IMDB-BINARY（TUDataset）+ ogbg-molhiv（OGB，替代 ZINC）
模型：  GCN / GAT / GraphSAGE / GIN
池化：  AvgPooling / MaxPooling / MinPooling

用法：
    python run.py --dataset MUTAG --model gin --pool avg
    python run.py --dataset PROTEINS --model gcn --pool max --epochs 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (DATASET_CFG, DATASET_INFO, POOLINGS, GraphClassifier,  # noqa: E402
                    Timer, get_device, load_graph_dataset, peak_gpu_mem_mb,
                    reset_gpu_mem, set_seed, train_graph_clf)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(CODE_DIR, "..", "data"))
MODELS = ["gcn", "gat", "sage", "gin"]


def run_single(dataset="MUTAG", model="gin", pool="avg", epochs=None, lr=None,
               layers=3, hidden=None, dropout=0.5, weight_decay=5e-4,
               batch_size=None, seed=42, device=None, verbose=False,
               data_root=DATA_ROOT):
    from torch_geometric.loader import DataLoader

    cfg = DATASET_CFG[dataset]
    epochs = int(epochs if epochs is not None else cfg["epochs"])
    lr = float(lr if lr is not None else cfg["lr"])
    hidden = int(hidden if hidden is not None else cfg["hidden"])
    batch_size = int(batch_size if batch_size is not None else cfg["batch_size"])
    task = cfg["task"]

    set_seed(seed)
    device = device or get_device()

    tr, va, te, in_dim, out_dim, task = load_graph_dataset(
        dataset, data_root, seed=seed)

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    net = GraphClassifier(in_dim, hidden, out_dim, num_layers=layers, kind=model,
                          pool=pool, dropout=dropout,
                          gat_heads=4 if dataset == "ogbg-molhiv" else 8)
    n_params = sum(p.numel() for p in net.parameters())

    reset_gpu_mem()
    t_all = Timer()
    with t_all:
        best_val, best_test, loop_t, ep_times = train_graph_clf(
            net, train_loader, val_loader, test_loader, device, task=task,
            epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=verbose)

    pure = float(sum(ep_times))
    metric = "test_acc" if task == "cls" else "test_auc"
    row = {
        "task": "graph_classification",
        "dataset": dataset,
        "model": model,
        "pool": pool,
        "num_layers": layers,
        "hidden": hidden,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_graphs_train": len(tr),
        "n_graphs_val": len(va),
        "n_graphs_test": len(te),
        "n_params": int(n_params),
        "val_metric": round(float(best_val), 4),
        "metric_name": metric,
        # 统一写两列（不适用的一列留空），保证同一 csv 内所有行 schema 一致，
        # 避免「molhiv 的 AUC 被塞进 test_acc 列」这种错位。
        "test_acc": round(float(best_test), 4) if metric == "test_acc" else "",
        "test_auc": round(float(best_test), 4) if metric == "test_auc" else "",
        "train_time_s": round(pure, 2),
        "loop_time_s": round(float(loop_t), 2),
        "time_per_epoch_s": round(pure / max(len(ep_times), 1), 4),
        "wall_time_s": round(float(t_all.t), 2),
        "peak_gpu_mb": round(float(peak_gpu_mem_mb()), 1),
        "seed": seed,
    }
    return row


def main():
    p = argparse.ArgumentParser(description="任务三 图分类")
    p.add_argument("--dataset", default="MUTAG", choices=list(DATASET_INFO))
    p.add_argument("--model", default="gin", choices=MODELS)
    p.add_argument("--pool", default="avg", choices=POOLINGS)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"[run] {args.dataset} | {args.model} | pool={args.pool}", flush=True)
    row = run_single(dataset=args.dataset, model=args.model, pool=args.pool,
                     epochs=args.epochs, lr=args.lr, layers=args.layers,
                     hidden=args.hidden, dropout=args.dropout,
                     weight_decay=args.weight_decay, batch_size=args.batch_size,
                     seed=args.seed, device=args.device, verbose=args.verbose)
    print("  -> " + json.dumps(row, ensure_ascii=False), flush=True)

    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df = pd.DataFrame([row])
        header = not os.path.exists(args.out)
        df.to_csv(args.out, mode="a", header=header, index=False,
                  encoding="utf-8-sig")
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
