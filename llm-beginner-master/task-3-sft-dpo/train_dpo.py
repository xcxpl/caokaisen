"""DPO alignment on top of the SFT LoRA adapter.

The reference model is represented by a frozen copy of the SFT LoRA tensors.
Before each reference forward they are swapped into the same base model, which
avoids keeping a second 0.5B model in GPU memory.
"""

from __future__ import annotations

import argparse
import contextlib
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

from src.chat import build_response_labels, format_messages
from src.data_utils import dpo_record_to_example, sample_records
from src.lora import count_trainable_parameters
from src.model_utils import load_lora_model, load_tokenizer, save_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    parser.add_argument("--sft-adapter", default=str(ROOT / "ckpt" / "sft"))
    parser.add_argument(
        "--data",
        nargs="+",
        default=[
            str(ROOT / "data" / "dpo-20k" / "dpo_zh.json"),
            str(ROOT / "data" / "dpo-20k" / "dpo_en.json"),
        ],
    )
    parser.add_argument("--output-dir", default=str(ROOT / "ckpt" / "dpo"))
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def make_scheduler(optimizer, max_steps: int) -> LambdaLR:
    warmup = max(1, int(max_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, schedule)


def lora_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters() if ".lora_A." in name or ".lora_B." in name}


def snapshot_lora(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in lora_parameters(model).items()}


def copy_lora(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in lora_parameters(model).items():
            param.copy_(state[name])


@contextlib.contextmanager
def use_reference_lora(model: torch.nn.Module, reference_state: dict[str, torch.Tensor]):
    """Temporarily switch LoRA tensors to the frozen SFT reference."""
    current = snapshot_lora(model)
    was_training = model.training
    model.eval()
    copy_lora(model, reference_state)
    try:
        with torch.no_grad():
            yield
    finally:
        copy_lora(model, current)
        if was_training:
            model.train()


def tokenize_response(
    tokenizer,
    prompt_messages: list[dict],
    response: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    messages = list(prompt_messages) + [{"role": "assistant", "content": response}]
    text = format_messages(messages)
    encoded = tokenizer(text, add_special_tokens=True)
    ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    if ids.numel() > max_length:
        return None
    labels = build_response_labels(ids, prompt_messages, response, tokenizer=tokenizer)
    if int((labels != -100).sum()) < 4:
        return None
    return ids, labels


def build_dpo_examples(tokenizer, data_paths: list[str], max_samples: int, max_length: int) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    stats = {"seen": 0, "skipped": 0, "usable": 0}
    for data_path in data_paths:
        records = sample_records(
            data_path,
            limit=max_samples * 4,
            shuffle=True,
            seed=42,
            max_scan=max_samples * 20,
        )
        for record in records:
            stats["seen"] += 1
            try:
                prompt, chosen, rejected = dpo_record_to_example(record)
                chosen_pair = tokenize_response(tokenizer, prompt, chosen, max_length)
                rejected_pair = tokenize_response(tokenizer, prompt, rejected, max_length)
                if chosen_pair is None or rejected_pair is None:
                    stats["skipped"] += 1
                    continue
                examples.append(
                    {
                        "prompt": prompt,
                        "chosen_ids": chosen_pair[0],
                        "chosen_labels": chosen_pair[1],
                        "rejected_ids": rejected_pair[0],
                        "rejected_labels": rejected_pair[1],
                    }
                )
                stats["usable"] += 1
                if len(examples) >= max_samples:
                    return examples, stats
            except Exception:
                stats["skipped"] += 1
    return examples, stats


def pad_pair(tokenizer, chosen_pair, rejected_pair):
    ids_a, labels_a = chosen_pair
    ids_b, labels_b = rejected_pair
    max_len = max(ids_a.numel(), ids_b.numel())
    pad_id = tokenizer.pad_token_id

    def pad(ids: torch.Tensor, labels: torch.Tensor):
        pad_len = max_len - ids.numel()
        input_ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        attention_mask = torch.cat([torch.ones(ids.numel(), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        padded_labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
        return input_ids, attention_mask, padded_labels

    a = pad(ids_a, labels_a)
    b = pad(ids_b, labels_b)
    return (
        torch.stack([a[0], b[0]]),
        torch.stack([a[1], b[1]]),
        torch.stack([a[2], b[2]]),
    )


def sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    target = shift_labels.masked_fill(~mask, 0)
    log_probs = logits.float().log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sft_adapter = Path(args.sft_adapter)
    adapter_file = sft_adapter / "adapter_model.pt" if sft_adapter.is_dir() else sft_adapter
    if not adapter_file.exists():
        raise FileNotFoundError(f"SFT adapter not found: {adapter_file}; run train_sft.py first")

    tokenizer = load_tokenizer(args.model)
    examples, stats = build_dpo_examples(tokenizer, args.data, args.max_samples, args.max_length)
    if not examples:
        raise RuntimeError("No usable DPO pairs after preprocessing")
    print(json.dumps({"preprocessing": stats, "usable": len(examples)}, ensure_ascii=False))

    model = load_lora_model(
        args.model,
        adapter_path=adapter_file,
        device=device,
        dtype=torch.bfloat16,
        target_modules=tuple(args.target_modules.split(",")),
        r=args.rank,
        alpha=args.alpha,
    )
    if args.gradient_checkpointing:
        from src.model_utils import enable_gradient_checkpointing

        enable_gradient_checkpointing(model)
    model.train()
    model.config.use_cache = False

    reference_state = snapshot_lora(model)
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
                "reference": "frozen SFT LoRA tensors",
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
    running_margin = 0.0
    final_loss = None
    final_margin = None
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for example in examples:
            input_ids, attention_mask, labels = pad_pair(
                tokenizer,
                (example["chosen_ids"], example["chosen_labels"]),
                (example["rejected_ids"], example["rejected_labels"]),
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with use_reference_lora(model, reference_state):
                reference_logps = sequence_logprobs(model, input_ids, attention_mask, labels)
            policy_logps = sequence_logprobs(model, input_ids, attention_mask, labels)

            chosen_policy, rejected_policy = policy_logps[0], policy_logps[1]
            chosen_reference, rejected_reference = reference_logps[0], reference_logps[1]
            logits = (chosen_policy - rejected_policy) - (chosen_reference - rejected_reference)
            loss = -F.logsigmoid(args.beta * logits).mean()
            (loss / args.grad_accum).backward()

            running_loss += float(loss.detach().cpu())
            running_margin += float((args.beta * logits).detach().mean().cpu())
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                final_loss = running_loss / args.grad_accum
                final_margin = running_margin / args.grad_accum

                if step % args.log_every == 0 or step == 1:
                    record = {
                        "step": step,
                        "loss": round(final_loss, 5),
                        "reward_margin": round(final_margin, 5),
                        "lr": round(float(scheduler.get_last_lr()[0]), 8),
                        "elapsed_sec": round(time.time() - started, 2),
                        "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1)
                        if torch.cuda.is_available()
                        else 0,
                    }
                    print(json.dumps(record, ensure_ascii=False))
                    log_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_stream.flush()

                running_loss = 0.0
                running_margin = 0.0
                if step >= args.max_steps:
                    break
            del input_ids, attention_mask, labels, policy_logps, reference_logps, logits, loss

    log_stream.close()
    metadata = {
        "stage": "dpo",
        "base_model": str(args.model),
        "sft_adapter": str(adapter_file),
        "data": args.data,
        "usable_examples": len(examples),
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "beta": args.beta,
        "final_loss": round(final_loss or 0.0, 6),
        "final_reward_margin": round(final_margin or 0.0, 6),
        "duration_sec": round(time.time() - started, 2),
        "preprocessing": stats,
        "reference": "frozen copy of SFT LoRA tensors",
    }
    save_adapter(model, output_dir, metadata=metadata)
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved DPO LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
