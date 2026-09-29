# -*- coding: utf-8 -*-
"""
任务四：知识图谱补全 —— 单组实验入口
====================================
模型：TransE / RotatE / ConvE
数据集：wn18rr（主）、countries_S1（小规模快速验证）

用法：
    python run.py --dataset wn18rr --model transe
    python run.py --dataset countries_S1 --model conve --epochs 60
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (DATASET_INFO, KGData, build_model, evaluate_filtered,  # noqa: E402
                    get_device, peak_gpu_mem_mb, reset_gpu_mem, set_seed,
                    train_kge)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(CODE_DIR, "..", "data"))
MODELS = ["transe", "rotate", "conve"]


def run_single(dataset="wn18rr", model="transe", epochs=None, batch_size=None,
               lr=None, dim=None, neg_num=1, seed=42, device=None,
               verbose=False, data_root=DATA_ROOT):
    set_seed(seed)
    device = device or get_device()

    data = KGData(dataset, data_root)
    if verbose:
        print("  " + data.describe(), flush=True)

    net, cfg = build_model(model, data, dim=dim)
    n_params = sum(p.numel() for p in net.parameters())

    reset_gpu_mem()
    train_time, ep_times = train_kge(
        net, data, device, epochs=epochs, batch_size=batch_size, lr=lr,
        neg_num=neg_num, verbose=verbose)

    # 在验证集上挑最优 epoch 不便于实现，这里直接报告训练结束后的测试指标
    res_val = evaluate_filtered(net, data.valid, data, device)
    res_test = evaluate_filtered(net, data.test, data, device)

    return {
        "task": "knowledge_graph",
        "dataset": dataset,
        "model": model,
        "dim": cfg["dim"],
        "lr": float(lr if lr is not None else cfg["lr"]),
        "epochs": int(epochs if epochs is not None else cfg["epochs"]),
        "batch_size": int(batch_size if batch_size is not None else cfg["batch_size"]),
        "neg_num": neg_num,
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
        "train_triples": len(data.train),
        "valid_triples": len(data.valid),
        "test_triples": len(data.test),
        "n_params": int(n_params),
        "valid_mrr": round(res_val["mrr"], 4),
        "valid_hits@10": round(res_val["hits@10"], 4),
        "test_mrr": round(res_test["mrr"], 4),
        "test_hits@1": round(res_test["hits@1"], 4),
        "test_hits@3": round(res_test["hits@3"], 4),
        "test_hits@10": round(res_test["hits@10"], 4),
        "train_time_s": round(float(train_time), 2),
        "time_per_epoch_s": round(float(train_time) / max(len(ep_times), 1), 4),
        "peak_gpu_mb": round(float(peak_gpu_mem_mb()), 1),
        "seed": seed,
    }


def main():
    p = argparse.ArgumentParser(description="任务四 知识图谱补全")
    p.add_argument("--dataset", default="wn18rr", choices=list(DATASET_INFO))
    p.add_argument("--model", default="transe", choices=MODELS)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--neg-num", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"[run] {args.dataset} | {args.model}", flush=True)
    row = run_single(dataset=args.dataset, model=args.model, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr, dim=args.dim,
                     neg_num=args.neg_num, seed=args.seed, device=args.device,
                     verbose=args.verbose)
    print("  -> " + json.dumps(row, ensure_ascii=False), flush=True)
    print(f"  MRR={row['test_mrr']:.4f} H@1={row['test_hits@1']:.4f} "
          f"H@3={row['test_hits@3']:.4f} H@10={row['test_hits@10']:.4f}", flush=True)

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
