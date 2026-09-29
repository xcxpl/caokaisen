# -*- coding: utf-8 -*-
"""把 results/*.csv 里的关键数字压缩成便于写报告的摘要。"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def show(df, keys, cols, title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    g = df.groupby(keys, dropna=False)[cols].agg(["mean", "std"])
    print(g.round(4).to_string())


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"[missing] {name}")
        return None
    df = pd.read_csv(p)
    df = df.drop_duplicates(subset=["dataset", "model", "mode", "split_id"]
                            if "mode" in df.columns else
                            ["dataset", "model", "pool"] if "pool" in df.columns
                            else ["dataset", "model"], keep="last")
    return df


def main():
    df = read("task1_main.csv")
    if df is not None:
        show(df, ["dataset", "model", "mode"],
             ["test_acc", "time_per_epoch_s", "train_time_s", "peak_gpu_mb"],
             "任务一 主对比（测试准确率 / 每轮耗时 / 训练耗时 / 显存）")
        tot = df.pivot_table(index=["dataset", "model"], columns="mode",
                             values="time_per_epoch_s", aggfunc="mean")
        if {"full", "sample"} <= set(tot.columns):
            tot["speedup"] = tot["full"] / tot["sample"]
            print("\n--- 采样训练相对全图训练的「每轮加速比」---")
            print(tot.round(3).to_string())

    for n in ["task1_lr.csv", "task1_layers.csv", "task1_sampler.csv"]:
        d = read(n)
        if d is not None:
            k = [c for c in ["lr", "num_layers", "batch_size", "num_neighbors"]
                 if c in d.columns and d[c].notna().any()]
            show(d, ["model"] + k, ["test_acc"], f"{n}")

    df = read("task2_main.csv")
    if df is not None:
        show(df, ["dataset", "model", "mode"],
             ["test_auc", "test_ap", "time_per_epoch_s", "train_time_s"],
             "任务二 主对比（AUC / AP / 每轮耗时）")
        tot = df.pivot_table(index=["dataset", "model"], columns="mode",
                             values="time_per_epoch_s", aggfunc="mean")
        if {"full", "sample"} <= set(tot.columns):
            tot["speedup"] = tot["full"] / tot["sample"]
            print("\n--- 采样训练相对全图训练的「每轮加速比」---")
            print(tot.round(3).to_string())

    for n in ["task2_lr.csv", "task2_layers.csv", "task2_sampler.csv"]:
        d = read(n)
        if d is not None:
            k = [c for c in ["lr", "num_layers", "batch_size", "num_neighbors"]
                 if c in d.columns and d[c].notna().any()]
            show(d, ["model"] + k, ["test_auc"], f"{n}")

    df = read("task3_pooling.csv")
    if df is not None:
        df["metric"] = df.get("test_acc", pd.Series(index=df.index)).fillna(
            df.get("test_auc", pd.Series(index=df.index)))
        show(df, ["dataset", "model", "pool"], ["metric", "time_per_epoch_s"],
             "任务三 池化方法对比")
        piv = df.pivot_table(index=["dataset", "model"], columns="pool",
                             values="metric", aggfunc="mean")
        print("\n--- 每种池化在各 (数据集, 模型) 上的排名（1 最好）---")
        print(piv.rank(axis=1, ascending=False).to_string())
        print("\n--- 各数据集上 3 种池化的平均表现 ---")
        print(df.groupby(["dataset", "pool"])["metric"].mean().round(4).to_string())

    for n in ["task3_layers.csv", "task3_lr.csv"]:
        d = read(n)
        if d is not None:
            k = [c for c in ["num_layers", "lr"] if c in d.columns]
            show(d, ["model"] + k, ["test_acc"], f"{n}")

    df = read("task4_kge.csv")
    if df is not None:
        show(df, ["dataset", "model"],
             ["test_mrr", "test_hits@1", "test_hits@3", "test_hits@10",
              "time_per_epoch_s"], "任务四 模型对比")
    d = read("task4_dim.csv")
    if d is not None:
        show(d, ["model", "dim"], ["test_mrr", "test_hits@10"], "task4_dim.csv")


if __name__ == "__main__":
    main()
