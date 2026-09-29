# -*- coding: utf-8 -*-
"""对 results/*.csv 去重（保留最后一次运行的结果）并排序，便于统计与复现。"""
import glob
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")

KEYS = {
    "task1_main.csv": ["dataset", "model", "mode", "split_id"],
    "task1_lr.csv": ["dataset", "model", "mode", "lr", "num_layers"],
    "task1_layers.csv": ["dataset", "model", "mode", "lr", "num_layers"],
    "task1_sampler.csv": ["dataset", "model", "mode", "batch_size", "num_neighbors"],
    "task2_main.csv": ["dataset", "model", "mode", "split_id"],
    "task2_lr.csv": ["dataset", "model", "mode", "lr", "num_layers"],
    "task2_layers.csv": ["dataset", "model", "mode", "lr", "num_layers"],
    "task2_sampler.csv": ["dataset", "model", "mode", "batch_size", "num_neighbors"],
    "task3_pooling.csv": ["dataset", "model", "pool", "num_layers", "lr"],
    "task3_layers.csv": ["dataset", "model", "pool", "num_layers"],
    "task3_lr.csv": ["dataset", "model", "pool", "lr"],
    "task4_kge.csv": ["dataset", "model", "dim", "lr", "epochs"],
    "task4_dim.csv": ["dataset", "model", "dim"],
}


def main():
    for path in sorted(glob.glob(os.path.join(RES, "task*.csv"))):
        name = os.path.basename(path)
        df = pd.read_csv(path)
        n0 = len(df)
        keys = [k for k in KEYS.get(name, ["dataset", "model"]) if k in df.columns]
        df = df.drop_duplicates(subset=keys, keep="last")
        df = df.sort_values(keys).reset_index(drop=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{name}: {n0} -> {len(df)} 行 (keys={keys})")


if __name__ == "__main__":
    main()
