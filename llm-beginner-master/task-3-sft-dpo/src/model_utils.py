"""Model loading and adapter helpers that fit a 4 GB laptop GPU.

The normal ``from_pretrained`` path needs a large CPU/Windows page-file
working set on this machine.  These helpers build the model on the ``meta``
device and copy safetensors weights directly to the target device, avoiding a
full CPU copy of the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .lora import inject_lora, load_lora, lora_state_dict


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _copy_safetensors_to_model(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    """Copy tensors one-by-one so the checkpoint is never duplicated on CPU."""
    model_state = model.state_dict()
    loaded: set[str] = set()
    # ``safe_open(device="cuda")`` can fail with invalid-storage errors after
    # several allocations on low-memory Windows drivers.  Read from the mmap
    # on CPU and move one tensor at a time instead; peak GPU use is one tensor.
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        keys.sort(key=lambda key: int(torch.tensor(handle.get_slice(key).get_shape()).prod()))
        for index, key in enumerate(keys):
            if key not in model_state:
                continue
            target = model_state[key]
            tensor = handle.get_tensor(key)
            if tuple(target.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"Shape mismatch for {key}: model={tuple(target.shape)} checkpoint={tuple(tensor.shape)}"
                )
            target.copy_(tensor.to(dtype=target.dtype, device=target.device))
            loaded.add(key)
            del tensor
            if device.type == "cuda" and index % 16 == 0:
                torch.cuda.empty_cache()

    # Tied weights such as lm_head.weight are intentionally absent from the
    # checkpoint because they share storage with token embeddings.
    missing = [key for key in model_state if key not in loaded and "lm_head.weight" not in key]
    if missing:
        raise RuntimeError(f"Missing checkpoint tensors: {missing[:10]}")


def load_base_model(
    model_path: str | Path,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
) -> torch.nn.Module:
    """Load Qwen from local safetensors directly onto ``device``."""
    model_path = Path(model_path)
    checkpoint = model_path / "model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing {checkpoint}; run data/download.py first")

    device = resolve_device(device)
    config = AutoConfig.from_pretrained(str(model_path))
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model = model.to_empty(device=device)
    model = model.to(dtype=dtype)
    _copy_safetensors_to_model(model, checkpoint, device)
    if getattr(model.config, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        # The safetensors checkpoint omits lm_head.weight when embeddings are
        # tied.  Re-tie it after loading, otherwise the output head remains
        # uninitialized and the loss cannot backpropagate into the model body.
        model.tie_weights()
    model.eval()
    return model


def load_lora_model(
    model_path: str | Path,
    adapter_path: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """Load base weights and optionally an existing LoRA adapter."""
    model = load_base_model(model_path, device=device, dtype=dtype)
    inject_lora(
        model,
        target_modules=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
        lora_dtype=torch.float32,
    )
    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        if adapter_path.is_dir():
            adapter_path = adapter_path / "adapter_model.pt"
        load_lora(model, adapter_path)
    return model


def save_adapter(
    model: torch.nn.Module,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    filename: str = "adapter_model.pt",
) -> Path:
    """Save a LoRA state dict plus JSON metadata into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    if not state:
        raise ValueError("No LoRA tensors found; inject LoRA before saving")

    torch.save(state, output_dir / filename)
    config = {
        "format": "task3-handwritten-lora",
        "target_modules": ["q_proj", "v_proj"],
        "r": 8,
        "alpha": 16.0,
        "lora_scaling": 2.0,
        "trainable_tensors": len(state),
    }
    if metadata:
        config.update(metadata)
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def load_adapter(model: torch.nn.Module, adapter_dir: str | Path) -> torch.nn.Module:
    adapter_dir = Path(adapter_dir)
    adapter_file = adapter_dir / "adapter_model.pt" if adapter_dir.is_dir() else adapter_dir
    load_lora(model, adapter_file)
    model.eval()
    return model


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, float | int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "ratio": trainable / total if total else 0.0,
    }
