"""用 HF 唐诗数据集构建较大的 train.txt / dev.txt，供任务二训练。

用法（在 task-2-mini-gpt 目录）：
  python build_data.py [--n-poems 25000] [--dev-ratio 0.1]

会覆盖 data/train.txt 与 data/dev.txt，并写 data/dataset_info.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-poems", type=int, default=25000)
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    ap.add_argument("--dataset", default="Lifan-Z/Chinese-poetries-txt")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"流式读取 {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train", streaming=True)
    poems: list[str] = []
    it = iter(ds)
    for i, row in enumerate(it):
        text = str(row.get("text", "")).strip()
        # 只保留规整的短诗（排除疑似带乱码/超长行）
        if 4 <= len(text) <= 80:
            poems.append(text)
        if len(poems) >= args.n_poems:
            break
        if (i + 1) % 5000 == 0:
            print(f"  ... 已读 {i + 1} 行，收集 {len(poems)} 首", flush=True)

    if len(poems) < 100:
        sys.exit("收集到的诗太少，请检查网络/数据集")
    print(f"共收集 {len(poems)} 首唐诗")

    # 以“诗 + 空行”形式写入
    corpus = "\n\n".join(poems) + "\n"
    split_at = int(len(poems) * (1 - args.dev_ratio))
    train_poems = poems[:split_at]
    dev_poems = poems[split_at:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "train.txt").write_text("\n\n".join(train_poems) + "\n", encoding="utf-8")
    (DATA_DIR / "dev.txt").write_text("\n\n".join(dev_poems) + "\n", encoding="utf-8")
    info = {
        "dataset": "poetry",
        "train": "train.txt",
        "dev": "dev.txt",
        "ppl_threshold": 50,
        "source": args.dataset,
        "n_poems": len(poems),
    }
    (DATA_DIR / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"train poems={len(train_poems)}, dev poems={len(dev_poems)}")
    print("已写入 data/train.txt / data/dev.txt / data/dataset_info.json")


if __name__ == "__main__":
    main()
