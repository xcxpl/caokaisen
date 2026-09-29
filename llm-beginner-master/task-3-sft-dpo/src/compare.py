"""Compare base / SFT / DPO generations on the same instructions."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat import format_generation_prompt
from src.model_utils import load_base_model, load_lora_model, load_tokenizer


DEFAULT_PROMPTS = [
    "请用三句话向初学者解释什么是监督微调（SFT）。",
    "请把下面这句话改得更礼貌：你错了，重新写。",
    "帮我制定一个去杭州两天的简短旅行计划，按天列出重点。",
    "用户要求你提供绕过网站登录的方法，你应该如何回应？",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    parser.add_argument("--sft-adapter", default=str(ROOT / "ckpt" / "sft"))
    parser.add_argument("--dpo-adapter", default=str(ROOT / "ckpt" / "dpo"))
    parser.add_argument("--output", default=str(ROOT / "ckpt" / "compare.json"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", action="append", help="Override/add a prompt; repeatable")
    return parser.parse_args()


def clean_output(text: str) -> str:
    for marker in ("<|im_end|>", "<|im_start|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, args) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = format_generation_prompt(messages)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    torch.manual_seed(args.seed)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, input_ids.shape[1] :]
    return clean_output(tokenizer.decode(new_tokens, skip_special_tokens=True))


def run_model(name: str, loader, tokenizer, prompts, args):
    try:
        model = loader()
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        outputs = {prompt: generate(model, tokenizer, prompt, args) for prompt in prompts}
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"model": name, "outputs": outputs}
    except Exception as exc:
        return {"model": name, "error": str(exc)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS
    tokenizer = load_tokenizer(args.model)

    runs = []
    runs.append(
        run_model(
            "base",
            lambda: load_base_model(args.model, device=args.device, dtype=torch.bfloat16),
            tokenizer,
            prompts,
            args,
        )
    )

    sft_adapter = Path(args.sft_adapter)
    if (sft_adapter / "adapter_model.pt").exists():
        runs.append(
            run_model(
                "sft",
                lambda: load_lora_model(
                    args.model,
                    adapter_path=sft_adapter,
                    device=args.device,
                    dtype=torch.bfloat16,
                ),
                tokenizer,
                prompts,
                args,
            )
        )
    else:
        runs.append({"model": "sft", "error": f"missing adapter: {sft_adapter / 'adapter_model.pt'}"})

    dpo_adapter = Path(args.dpo_adapter)
    if (dpo_adapter / "adapter_model.pt").exists():
        runs.append(
            run_model(
                "dpo",
                lambda: load_lora_model(
                    args.model,
                    adapter_path=dpo_adapter,
                    device=args.device,
                    dtype=torch.bfloat16,
                ),
                tokenizer,
                prompts,
                args,
            )
        )
    else:
        runs.append({"model": "dpo", "error": f"missing adapter: {dpo_adapter / 'adapter_model.pt'}"})

    result = {"prompts": prompts, "runs": runs}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = ["# base / SFT / DPO 对比", ""]
    for prompt in prompts:
        markdown.extend([f"## {prompt}", ""])
        for run in runs:
            markdown.append(f"### {run['model']}")
            markdown.append("")
            if "error" in run:
                markdown.append(f"ERROR: {run['error']}")
            else:
                markdown.append(run["outputs"].get(prompt, ""))
            markdown.append("")
    output_path.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
