"""任务一 M5：可视化注意力热图（展示全部 n_heads=4 个头 + 平均）。

对正面/负面/长句三个样本，取最后一层，把 4 个头的 (T, T) 注意力矩阵画成 2x2 网格，
并列出 4 个头平均后的 CLS 行注意力（看模型最关注哪些词元）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.model import load_for_eval


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
CKPT = ROOT / "ckpt" / "best.pt"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MAX_CHARS = 40


def get_sample(df, keywords, label=None, longest=False):
    pool = df
    if label is not None:
        pool = df[df["label"] == label]
    if keywords:
        kw_pool = pool[pool["text"].apply(lambda t: any(k in str(t) for k in keywords))]
        if len(kw_pool):
            pool = kw_pool
    if longest:
        idx = pool["text"].str.len().idxmax()
    else:
        medium = pool[pool["text"].str.len().between(15, MAX_CHARS)]
        idx = medium.index[0] if len(medium) else pool.index[0]
    text = str(pool.loc[idx, "text"])
    return text[:MAX_CHARS], int(pool.loc[idx, "label"])


def plot_heads(model, tokenize_fn, text, label, fname, layer):
    ids = tokenize_fn(text)
    with torch.no_grad():
        logits = model(ids.unsqueeze(0))
        pred = int(logits.argmax(dim=-1).item())
    model.eval()
    model(ids.unsqueeze(0))
    attn = model.blocks[layer].attn.attention_weights[0].detach().numpy()  # (H, T, T)
    H = attn.shape[0]
    T = min(attn.shape[1], MAX_CHARS + 1)
    attn = attn[:, :T, :T]
    chars = ["<cls>"] + list(text[: T - 1])
    chars += [""] * (T - len(chars))
    avg_cls = attn[:, 0, :].mean(axis=0)  # 4 个头平均的 CLS 行

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for hi in range(H):
        ax = axes[hi // 2][hi % 2]
        im = ax.imshow(attn[hi], cmap="viridis", aspect="auto")
        ax.set_title(f"head {hi}")
        step = max(1, T // 24)
        ax.set_xticks(range(0, T, step))
        ax.set_xticklabels(chars[0:T:step], rotation=90, fontsize=6)
        ax.set_yticks(range(0, T, step))
        ax.set_yticklabels(chars[0:T:step], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"文本: {text}\n真实标签={'正面' if label == 1 else '负面'}  "
        f"预测={'正面' if pred == 1 else '负面'}  |  layer={layer}, n_heads={H}",
        fontsize=11,
    )
    out = FIG / fname
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")

    top = avg_cls.argsort()[::-1][:6]
    print("  平均CLS行 top6: " + "  ".join(f"{chars[int(p)]}({avg_cls[int(p)]:.2f})" for p in top))
    return pred


def main():
    FIG.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "validation.parquet")
    model, tokenize_fn = load_for_eval(str(CKPT))
    print(f"模型: n_heads={model.blocks[0].attn.n_heads}, "
          f"n_layers={len(model.blocks)}, d_model={model.d_model}")
    last_layer = len(model.blocks) - 1

    pos_text, pos_label = get_sample(df, ["不错", "好", "喜欢"], label=1)
    neg_text, neg_label = get_sample(df, ["失望", "差", "不好"], label=0)
    long_text, long_label = get_sample(df, [], longest=True)
    plot_heads(model, tokenize_fn, pos_text, pos_label, "attn_positive.png", last_layer)
    plot_heads(model, tokenize_fn, neg_text, neg_label, "attn_negative.png", last_layer)
    plot_heads(model, tokenize_fn, long_text, long_label, "attn_long.png", last_layer)


if __name__ == "__main__":
    main()
