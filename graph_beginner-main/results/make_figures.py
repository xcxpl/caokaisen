# -*- coding: utf-8 -*-
"""
结果可视化：读取 results/*.csv，生成 report/figures/*.png
用法：python results/make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "report", "figures")
os.makedirs(FIG, exist_ok=True)

MODELS = ["gcn", "gat", "sage", "gin"]
COLORS = {"full": "#4C72B0", "sample": "#DD8452"}
POOL_COLORS = {"avg": "#4C72B0", "max": "#DD8452", "min": "#55A868"}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df if not df.empty else None


def agg(df, keys, cols):
    g = df.groupby(keys, dropna=False)[cols].agg(["mean", "std"]).reset_index()
    g.columns = [("_".join([str(c) for c in col if c]) if isinstance(col, tuple)
                  else str(col)) for col in g.columns]
    return g


# ---------------------------------------------------------------- 任务一/二
def fig_full_vs_sample(fname, metric, ylabel, title, out):
    df = load(fname)
    if df is None:
        return
    g = agg(df, ["dataset", "model", "mode"], [metric, "time_per_epoch_s",
                                               "train_time_s"])
    datasets = list(dict.fromkeys(g["dataset"]))
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.6 * len(datasets), 7.2))
    if len(datasets) == 1:
        axes = axes.reshape(2, 1)
    x = np.arange(len(MODELS))
    w = 0.36
    for j, ds in enumerate(datasets):
        sub = g[g["dataset"] == ds]
        for k, (m, key) in enumerate([("full", "full"), ("sample", "sample")]):
            vals, errs = [], []
            for mo in MODELS:
                r = sub[(sub["model"] == mo) & (sub["mode"] == m)]
                vals.append(float(r[metric + "_mean"].iloc[0]) if len(r) else np.nan)
                errs.append(float(r[metric + "_std"].iloc[0])
                            if len(r) and not pd.isna(r[metric + "_std"].iloc[0]) else 0.0)
            axes[0, j].bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=3,
                           color=COLORS[m], label="全图训练" if m == "full" else "子图采样训练")
        axes[0, j].set_title(f"{ds}")
        axes[0, j].set_xticks(x)
        axes[0, j].set_xticklabels([m.upper() for m in MODELS])
        axes[0, j].set_ylabel(ylabel)
        axes[0, j].grid(axis="y", alpha=0.3)
        axes[0, j].legend(fontsize=8)

        for k, (m, mkey) in enumerate([("full", "full"), ("sample", "sample")]):
            vals = []
            for mo in MODELS:
                r = sub[(sub["model"] == mo) & (sub["mode"] == m)]
                vals.append(float(r["time_per_epoch_s_mean"].iloc[0]) if len(r) else np.nan)
            axes[1, j].bar(x + (k - 0.5) * w, vals, w, color=COLORS[m])
        axes[1, j].set_yscale("log")
        axes[1, j].set_xticks(x)
        axes[1, j].set_xticklabels([m.upper() for m in MODELS])
        axes[1, j].set_ylabel("每轮耗时 (s, log)")
        axes[1, j].grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, out))
    plt.close(fig)
    print("saved", out)


def fig_ablation_line(fname, xcol, metric, ylabel, title, out, xlabel=None):
    df = load(fname)
    if df is None:
        return
    g = agg(df, ["model", xcol], [metric])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mo in MODELS:
        sub = g[g["model"] == mo].sort_values(xcol)
        if len(sub) == 0:
            continue
        ax.errorbar(sub[xcol], sub[metric + "_mean"],
                    yerr=sub[metric + "_std"].fillna(0),
                    marker="o", capsize=3, label=mo.upper())
    ax.set_xlabel(xlabel or xcol)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if xcol in ("lr",):
        ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, out))
    plt.close(fig)
    print("saved", out)


def fig_sampler_heatmap(fname, metric, title, out):
    df = load(fname)
    if df is None:
        return
    g = agg(df, ["batch_size", "num_neighbors"], [metric, "time_per_epoch_s"])
    for key, tag in [(metric, "acc"), ("time_per_epoch_s", "time")]:
        piv = g.pivot(index="batch_size", columns="num_neighbors",
                      values=key + "_mean")
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("num_neighbors")
        ax.set_ylabel("batch_size")
        ax.set_title(f"{title} — {('指标' if tag=='acc' else '每轮耗时(s)')}")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center",
                        va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, out.replace(".png", f"_{tag}.png")))
        plt.close(fig)
        print("saved", out.replace(".png", f"_{tag}.png"))


# ------------------------------------------------------------------- 任务三
def fig_pooling():
    df = load("task3_pooling.csv")
    if df is None:
        return
    if "test_acc" not in df.columns:
        df["test_acc"] = np.nan
    if "test_auc" not in df.columns:
        df["test_auc"] = np.nan
    df["metric"] = df["test_acc"].fillna(df["test_auc"])
    g = agg(df, ["dataset", "model", "pool"], ["metric", "time_per_epoch_s"])
    datasets = list(dict.fromkeys(g["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.8),
                             squeeze=False)
    x = np.arange(len(MODELS))
    w = 0.26
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        sub = g[g["dataset"] == ds]
        for k, pool in enumerate(["avg", "max", "min"]):
            vals = []
            for mo in MODELS:
                r = sub[(sub["model"] == mo) & (sub["pool"] == pool)]
                vals.append(float(r["metric_mean"].iloc[0]) if len(r) else np.nan)
            ax.bar(x + (k - 1) * w, vals, w,
                   color=POOL_COLORS[pool],
                   label={"avg": "AvgPooling", "max": "MaxPooling",
                          "min": "MinPooling"}[pool])
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in MODELS])
        ax.set_ylabel("测试指标 (acc / AUC)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("不同池化方法 × 不同 GNN 的图分类性能", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "task3_pooling.png"))
    plt.close(fig)
    print("saved task3_pooling.png")


# ------------------------------------------------------------------- 任务四
def fig_kge():
    df = load("task4_kge.csv")
    if df is None:
        return
    datasets = list(dict.fromkeys(df["dataset"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(datasets))
    w = 0.26
    models = ["transe", "rotate", "conve"]
    for k, mo in enumerate(models):
        vals = []
        for ds in datasets:
            r = df[(df["dataset"] == ds) & (df["model"] == mo)]
            vals.append(float(r["test_mrr"].iloc[0]) if len(r) else np.nan)
        axes[0].bar(x + (k - 1) * w, vals, w, label=mo.upper())
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("测试 MRR")
    axes[0].set_title("KGE 模型 MRR 对比")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    for k, mo in enumerate(models):
        vals = []
        for ds in datasets:
            r = df[(df["dataset"] == ds) & (df["model"] == mo)]
            vals.append(float(r["test_hits@10"].iloc[0]) if len(r) else np.nan)
        axes[1].bar(x + (k - 1) * w, vals, w, label=mo.upper())
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel("测试 Hits@10")
    axes[1].set_title("KGE 模型 Hits@10 对比")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "task4_kge.png"))
    plt.close(fig)
    print("saved task4_kge.png")

    d2 = load("task4_dim.csv")
    if d2 is not None:
        g = agg(d2, ["model", "dim"], ["test_mrr"])
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for mo in ["transe", "rotate"]:
            sub = g[g["model"] == mo].sort_values("dim")
            if len(sub):
                ax.plot(sub["dim"], sub["test_mrr_mean"], marker="o",
                        label=mo.upper())
        ax.set_xlabel("嵌入维度 dim")
        ax.set_ylabel("测试 MRR")
        ax.set_title("嵌入维度对 MRR 的影响 (countries_S1)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "task4_dim.png"))
        plt.close(fig)
        print("saved task4_dim.png")


def dump_tables():
    """把关键结果汇总成 markdown 表格，供报告引用。"""
    out = []
    spec = [
        ("task1_main.csv", ["dataset", "model", "mode"],
         ["test_acc", "time_per_epoch_s", "train_time_s", "peak_gpu_mb"]),
        ("task1_lr.csv", ["dataset", "model", "lr"], ["test_acc"]),
        ("task1_layers.csv", ["dataset", "model", "num_layers"], ["test_acc"]),
        ("task1_sampler.csv", ["batch_size", "num_neighbors"],
         ["test_acc", "time_per_epoch_s"]),
        ("task2_main.csv", ["dataset", "model", "mode"],
         ["test_auc", "test_ap", "time_per_epoch_s", "train_time_s", "peak_gpu_mb"]),
        ("task2_lr.csv", ["dataset", "model", "lr"], ["test_auc"]),
        ("task2_layers.csv", ["dataset", "model", "num_layers"], ["test_auc"]),
        ("task2_sampler.csv", ["batch_size", "num_neighbors"],
         ["test_auc", "time_per_epoch_s"]),
        ("task3_pooling.csv", ["dataset", "model", "pool"],
         ["test_acc", "test_auc", "time_per_epoch_s"]),
        ("task3_layers.csv", ["model", "num_layers"], ["test_acc"]),
        ("task3_lr.csv", ["model", "lr"], ["test_acc"]),
        ("task4_kge.csv", ["dataset", "model"],
         ["test_mrr", "test_hits@1", "test_hits@3", "test_hits@10", "time_per_epoch_s"]),
        ("task4_dim.csv", ["model", "dim"], ["test_mrr", "test_hits@10"]),
    ]
    for fname, keys, cols in spec:
        df = load(fname)
        if df is None:
            continue
        have = [c for c in cols if c in df.columns]
        if not have:
            continue
        sub = df[keys + have].copy()
        for c in have:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        g = sub.groupby(keys, dropna=False)[have].agg(["mean", "std"]).reset_index()
        g.columns = [("_".join([str(c) for c in col if c]) if isinstance(col, tuple)
                      else str(col)) for col in g.columns]
        out.append(f"\n### {fname}\n")
        out.append(g.round(4).to_markdown(index=False))
    if out:
        p = os.path.join(RES, "summary_tables.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(out))
        print("saved", p)


if __name__ == "__main__":
    fig_full_vs_sample("task1_main.csv", "test_acc", "测试准确率",
                       "任务一 节点分类：全图训练 vs 子图采样训练", "task1_main.png")
    fig_ablation_line("task1_lr.csv", "lr", "test_acc", "测试准确率",
                      "任务一 学习率影响 (Cora)", "task1_lr.png")
    fig_ablation_line("task1_layers.csv", "num_layers", "test_acc", "测试准确率",
                      "任务一 网络层数影响 (Cora)", "task1_layers.png", "层数")
    fig_sampler_heatmap("task1_sampler.csv", "test_acc",
                        "任务一 采样参数 (Cora/GCN)", "task1_sampler.png")

    fig_full_vs_sample("task2_main.csv", "test_auc", "测试 AUC",
                       "任务二 链路预测：全图训练 vs 子图采样训练", "task2_main.png")
    fig_ablation_line("task2_lr.csv", "lr", "test_auc", "测试 AUC",
                      "任务二 学习率影响 (Cora)", "task2_lr.png")
    fig_ablation_line("task2_layers.csv", "num_layers", "test_auc", "测试 AUC",
                      "任务二 网络层数影响 (Cora)", "task2_layers.png", "层数")
    fig_sampler_heatmap("task2_sampler.csv", "test_auc",
                        "任务二 采样参数 (Cora/GCN)", "task2_sampler.png")

    fig_pooling()
    fig_ablation_line("task3_layers.csv", "num_layers", "test_acc", "测试准确率",
                      "任务三 网络层数影响 (MUTAG/AvgPool)", "task3_layers.png", "层数")
    fig_ablation_line("task3_lr.csv", "lr", "test_acc", "测试准确率",
                      "任务三 学习率影响 (MUTAG/AvgPool)", "task3_lr.png")
    fig_kge()
    dump_tables()
    print("全部图表完成 ->", FIG)
