"""加载训练好的 mini-GPT 进行文本生成。

用法（在 task-2-mini-gpt 目录下）：
    python generate.py --prompt "床前明月光"
    python generate.py --prompt "春眠不觉晓" --top-k 1 --max-new-tokens 60
    python generate.py --prompt "白日依山尽" --temperature 1.1 --top-p 0.95

说明：
    - greedy：--top-k 1 或 --temperature 0
    - 不传 top-k / top-p / temperature 时走默认采样（top_k=50, top_p=0.95, temperature=0.8）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import load_for_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="床前明月光", help="生成提示（中文）")
    ap.add_argument("--max-new-tokens", type=int, default=48, help="生成的新 token 数")
    ap.add_argument("--top-k", type=int, default=50, help="top-k；1 表示 greedy")
    ap.add_argument("--top-p", type=float, default=0.95, help="nucleus top-p")
    ap.add_argument("--temperature", type=float, default=0.8, help="采样温度，<=0 视为 greedy")
    ap.add_argument("--ckpt", default="ckpt/best.pt", help="模型权重路径（tokenizer.json 须与其同目录）")
    ap.add_argument("--num-samples", type=int, default=1, help="重复采样次数，观察多样性")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)

    try:
        model, tokenizer = load_for_eval(args.ckpt)
    except Exception as e:  # 未找到 ckpt 等
        sys.exit(f"加载失败：{e}\n请先运行训练得到 ckpt/best.pt 与 ckpt/tokenizer.json")

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        sys.exit(f"提示无法被分词：{args.prompt!r}")

    print(f"模型：{args.ckpt}（vocab={tokenizer.vocab_size}, block={model.block_size}）")
    print(f"参数：top_k={args.top_k}, top_p={args.top_p}, temperature={args.temperature}, n={args.max_new_tokens}")
    print("-" * 60)

    for i in range(args.num_samples):
        ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        text = tokenizer.decode(ids)
        label = f"样例 {i + 1}" if args.num_samples > 1 else "生成结果"
        print(f"[{label}]")
        print(text)
        print()


if __name__ == "__main__":
    main()
