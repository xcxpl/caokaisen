"""任务二训练脚本。

流程：
1. 在 data/train.txt 上训练字节级 BPE，保存 ckpt/tokenizer.json。
2. 用 BPE 编码 train/dev。
3. 搭建 MiniGPT（n_vocab = tokenizer.vocab_size），next-token prediction。
4. AdamW + warmup/cosine + gradient clipping，监控 dev 困惑度，保存最优 ckpt/best.pt。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.tokenizer import BPETokenizer
from src.model import MiniGPT


TASK_ROOT = Path(__file__).resolve().parent
DATA_DIR = TASK_ROOT / "data"
CKPT_DIR = TASK_ROOT / "ckpt"
TOK_PATH = CKPT_DIR / "tokenizer.json"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_batches(ids: list[int], block_size: int, batch_size: int, device: torch.device, drop_last: bool = True):
    """把 token 流切成 (input, target) 批。每个样本长度 block_size，stride = block_size。"""
    n = len(ids)
    xs, ys = [], []
    for i in range(0, n - block_size, block_size):
        xs.append(ids[i : i + block_size])
        ys.append(ids[i + 1 : i + block_size + 1])
    if not drop_last and n >= block_size + 1:
        # 追加末尾一段（若不足则忽略，保长度一致）
        i = n - block_size - 1
        if i > 0:
            xs.append(ids[i : i + block_size])
            ys.append(ids[i + 1 : i + block_size + 1])
    if not xs:
        return []
    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)
    # 随机划分为若干批
    perm = torch.randperm(len(xs))
    batches = []
    for k in range(0, len(xs) - len(xs) % batch_size, batch_size):
        idx = perm[k : k + batch_size]
        batches.append((xs[idx].to(device), ys[idx].to(device)))
    return batches


@torch.no_grad()
def evaluate_ppl(model: MiniGPT, tokenizer: BPETokenizer, dev_path: Path, block_size: int, device: torch.device) -> float:
    model.eval()
    text = dev_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)[:4096]
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(ids) - 1), block_size):
        window = ids[i : i + block_size + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(chunk)
        nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        n_tok += chunk.size(1) - 1
    if n_tok == 0:
        return float("inf")
    return math.exp(nll / n_tok)


def get_lr(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def sample(model: MiniGPT, tokenizer: BPETokenizer, prompt: str, device: torch.device, n: int = 60):
    ids = tokenizer.encode(prompt)
    out = model.generate(ids, max_new_tokens=n, top_k=50, top_p=0.95, temperature=0.8)
    return tokenizer.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log-interval", type=int, default=20)     # 每 N 步打印一次 train loss
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--early-ppl", type=float, default=20.0)   # dev ppl 低于此值即提前保存最优并停止
    ap.add_argument("--patience", type=int, default=15)        # dev 困惑度连续 N 次评测无提升则早停
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[device] {device}")

    train_path = DATA_DIR / "train.txt"
    dev_path = DATA_DIR / "dev.txt"
    if not train_path.exists() or not dev_path.exists():
        sys.exit("缺少 data/train.txt 或 data/dev.txt，请先运行 python data/download.py")

    # 1. 训练/加载 BPE
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    reuse_ok = TOK_PATH.exists()
    if reuse_ok:
        candidate = BPETokenizer.from_pretrained(TOK_PATH)
        if candidate.vocab_size != args.vocab_size:
            reuse_ok = False
    if reuse_ok:
        tokenizer = candidate
        print(f"[tokenizer] 复用 ckpt/tokenizer.json (vocab={tokenizer.vocab_size})")
    else:
        raw = train_path.read_text(encoding="utf-8")
        tokenizer = BPETokenizer()
        tokenizer.train(raw, args.vocab_size)
        tokenizer.save(TOK_PATH)
        print(f"[tokenizer] 训练完成 vocab_size={tokenizer.vocab_size}")
    n_vocab = tokenizer.vocab_size

    # 2. 编码（带缓存，避免重复开销）
    cache_key = f"{train_path.stat().st_size}_{n_vocab}"
    train_ids_path = DATA_DIR / f"train_ids_{cache_key}.json"
    if train_ids_path.exists():
        train_ids = json.loads(train_ids_path.read_text(encoding="utf-8"))
        print(f"[data] 复用缓存 train tokens={len(train_ids)}")
    else:
        train_ids = tokenizer.encode(train_path.read_text(encoding="utf-8"))
        train_ids_path.write_text(json.dumps(train_ids), encoding="utf-8")
        print(f"[data] train tokens={len(train_ids)}")

    # 3. 模型
    config = {
        "n_vocab": n_vocab,
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "dropout": args.dropout,
        "rope_base": 10000.0,
    }
    model = MiniGPT(**config).to(device)
    model.special_ids = tokenizer.special_token_ids
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    batches = build_batches(train_ids, args.block_size, args.batch_size, device)
    if not batches:
        sys.exit("训练样本过少，请减小 block_size 或检查语料。")
    print(f"[data] batches/epoch≈{len(batches)}")

    best_ppl = float("inf")
    no_improve = 0
    step = 0
    start = time.time()
    model.train()
    stop = False
    while step < args.max_steps and not stop:
        for xb, yb in batches:
            # 每批更新 lr
            lr = get_lr(step, args.warmup, args.max_steps, args.lr, args.lr * 0.1)
            for g in opt.param_groups:
                g["lr"] = lr

            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % args.log_interval == 0:
                print(f"[step {step:5d}] train_loss={loss.item():.4f} train_ppl={math.exp(loss.item()):.2f}", flush=True)

            if step % args.eval_interval == 0:
                add_token_ppl = math.exp(loss.item())
                dev_ppl = evaluate_ppl(model, tokenizer, dev_path, args.block_size, device)
                tag = "★" if dev_ppl < best_ppl else " "
                print(
                    f"[step {step:5d}] train_loss={loss.item():.4f} train_ppl={add_token_ppl:.2f} "
                    f"dev_ppl={dev_ppl:.2f} lr={lr:.2e} {tag}", flush=True
                )
                if dev_ppl < best_ppl:
                    best_ppl = dev_ppl
                    no_improve = 0
                    torch.save(
                        {"model": model.state_dict(), "config": config, "best_ppl": best_ppl},
                        CKPT_DIR / "best.pt",
                    )
                else:
                    no_improve += 1
                if dev_ppl < args.early_ppl:
                    print(f"[early-stop] dev_ppl {dev_ppl:.2f} < {args.early_ppl}")
                    stop = True
                    break
                if no_improve >= args.patience:
                    print(f"[early-stop] dev_ppl 连续 {args.patience} 次无提升，best={best_ppl:.2f}")
                    stop = True
                    break
            if step >= args.max_steps:
                stop = True
                break

    # 若因步数耗尽结束，确保保存最优
    if (CKPT_DIR / "best.pt").exists():
        print(f"[saved] {CKPT_DIR / 'best.pt'}, best_ppl={best_ppl:.2f}")
    else:
        print(f"[save] 未触发早停，保存当前模型")
        torch.save({"model": model.state_dict(), "config": config, "best_ppl": best_ppl}, CKPT_DIR / "best.pt")

    # 最终采样演示
    model.eval()
    for prompt in ["床前明月光", "君不见黄河之水天上来", "春眠不觉晓"]:
        print("\n>>", prompt)
        print(sample(model, tokenizer, prompt, device, n=48))


if __name__ == "__main__":
    main()
