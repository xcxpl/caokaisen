"""Supervised fine-tuning of Qwen2.5-0.5B with hand-written LoRA."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chat import build_labels, format_messages
from src.data_utils import moss_record_to_messages, sample_records
from src.lora import inject_lora, count_trainable_parameters
from src.model_utils import load_base_model, load_tokenizer, save_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "moss-sft" / "moss-003-sft-no-tools.jsonl.zip"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "ckpt" / "sft"))
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def build_sft_examples(tokenizer, records, max_length: int) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    stats = {
        "records_seen": 0,
        "records_skipped": 0,
        "total_tokens": 0,
        "supervised_tokens": 0,
        "max_tokens": 0,
    }
    for record in records:
        stats["records_seen"] += 1
        try:
            messages = moss_record_to_messages(record)
            text = format_messages(messages)
            encoded = tokenizer(text, add_special_tokens=True)
            ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            labels = build_labels(ids, messages, tokenizer=tokenizer)
            ids = ids[:max_length]
            labels = labels[:max_length]
            supervised = int((labels != -100).sum())
            if supervised < 8:
                stats["records_skipped"] += 1
                continue
        except Exception:
            stats["records_skipped"] += 1
            continue

        examples.append(
            {
                "input_ids": ids,
                "labels": labels,
                "length": int(ids.numel()),
                "supervised": supervised,
            }
        )
        stats["total_tokens"] += int(ids.numel())
        stats["supervised_tokens"] += supervised
        stats["max_tokens"] = max(stats["max_tokens"], int(ids.numel()))
    return examples, stats


def make_scheduler(optimizer, max_steps: int) -> LambdaLR:
    warmup = max(1, int(max_steps * 0.03))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, schedule)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"SFT data not found: {data_path}\n"
            "Download moss-003-sft-no-tools.jsonl.zip first."
        )

    tokenizer = load_tokenizer(args.model)
    records = sample_records(
        data_path,
        limit=args.max_samples,
        shuffle=True,
        seed=args.seed,
        max_scan=args.max_scan,
    )
    examples, stats = build_sft_examples(tokenizer, records, args.max_length)
    if not examples:
        raise RuntimeError("No usable SFT examples after preprocessing")
    print(json.dumps({"preprocessing": stats, "usable": len(examples)}, ensure_ascii=False))

    model = load_base_model(args.model, device=device, dtype=torch.bfloat16)
    inject_lora(
        model,
        target_modules=args.target_modules.split(","),
        r=args.rank,
        alpha=args.alpha,
        lora_dtype=torch.float32,
    )
    if args.gradient_checkpointing:
        from src.model_utils import enable_gradient_checkpointing

        enable_gradient_checkpointing(model)
    model.train()
    model.config.use_cache = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = make_scheduler(optimizer, args.max_steps)
    trainable_count, total_count, ratio = count_trainable_parameters(model)
    print(
        json.dumps(
            {
                "trainable": trainable_count,
                "total": total_count,
                "ratio": round(ratio, 6),
                "device": str(device),
            },
            ensure_ascii=False,
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    log_stream = log_path.open("w", encoding="utf-8")

    started = time.time()
    step = 0
    micro_step = 0
    running_loss = 0.0
    running_tokens = 0
    final_loss = None
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for example in examples:
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            labels = example["labels"].unsqueeze(0).to(device)

            logits = model(input_ids=input_ids).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            (loss / args.grad_accum).backward()
            running_loss += float(loss.detach().cpu())
            running_tokens += int((shift_labels != -100).sum().cpu())
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                final_loss = running_loss / args.grad_accum

                if step % args.log_every == 0 or step == 1:
                    elapsed = time.time() - started
                    record = {
                        "step": step,
                        "loss": round(final_loss, 5),
                        "lr": round(float(scheduler.get_last_lr()[0]), 8),
                        "tokens": running_tokens,
                        "elapsed_sec": round(elapsed, 2),
                        "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1)
                        if torch.cuda.is_available()
                        else 0,
                    }
                    print(json.dumps(record, ensure_ascii=False))
                    log_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_stream.flush()

                running_loss = 0.0
                running_tokens = 0
                if step >= args.max_steps:
                    break
            del logits, shift_logits, shift_labels, loss

    log_stream.close()
    metadata = {
        "stage": "sft",
        "base_model": str(args.model),
        "data": str(data_path),
        "max_samples": args.max_samples,
        "max_scan": args.max_scan,
        "usable_examples": len(examples),
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "final_loss": round(final_loss or 0.0, 6),
        "duration_sec": round(time.time() - started, 2),
        "preprocessing": stats,
    }
    save_adapter(model, output_dir, metadata=metadata)
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved SFT LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
