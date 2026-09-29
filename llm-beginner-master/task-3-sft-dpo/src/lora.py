"""A small, dependency-free LoRA implementation.

The implementation intentionally does not depend on ``peft``.  It replaces
selected ``torch.nn.Linear`` modules with a wrapper that keeps the original
linear layer frozen and learns two low-rank matrices.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank residual branch."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        lora_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)!r}")
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.merged = False
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Training LoRA in fp32 is more stable than fp16 AdamW while the frozen
        # base weights can remain fp16/bf16.
        if lora_dtype is None:
            base_dtype = base_layer.weight.dtype
            lora_dtype = torch.float32 if base_dtype in (torch.float16, torch.bfloat16) else base_dtype
        self.lora_A.to(device=base_layer.weight.device, dtype=lora_dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=lora_dtype)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.merged:
            return base_out

        lora_input = x.to(dtype=self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(lora_input)))
        return base_out + (lora_out * self.scaling).to(dtype=base_out.dtype)

    def merge(self) -> None:
        """Merge ``scaling * B @ A`` into the frozen base weight in-place."""
        if self.merged:
            return
        with torch.no_grad():
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.base_layer.weight.add_(delta.to(dtype=self.base_layer.weight.dtype))
        self.merged = True


def _matches_target(name: str, targets: Sequence[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return any(name == target or leaf == target or name.endswith(f".{target}") for target in targets)


def _replace_linears(module: nn.Module, targets: Sequence[str], factory) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and _matches_target(child_name, targets):
            setattr(module, child_name, factory(child))
            replaced += 1
        else:
            replaced += _replace_linears(child, targets, factory)
    return replaced


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    lora_dtype: torch.dtype | None = None,
) -> nn.Module:
    """Inject trainable LoRA branches into selected linear layers.

    Args:
        model: Hugging Face causal language model.
        target_modules: Linear module names or suffixes, e.g. ``q_proj``.
        r: Low-rank dimension.
        alpha: LoRA scaling numerator.  Effective scale is ``alpha / r``.
        dropout: Optional dropout on the LoRA input.
        lora_dtype: Dtype for A/B.  Defaults to fp32 when the base layer is
            fp16/bf16, otherwise it follows the base layer.

    Returns:
        The same model object, modified in place.
    """
    targets = list(target_modules)
    if not targets:
        raise ValueError("target_modules must not be empty")

    # Freeze every pre-existing parameter first.  Newly-created LoRA parameters
    # below remain trainable by default.
    for param in model.parameters():
        param.requires_grad = False

    def factory(base_layer: nn.Linear) -> LoRALinear:
        return LoRALinear(
            base_layer,
            r=r,
            alpha=alpha,
            dropout=dropout,
            lora_dtype=lora_dtype,
        )

    replaced = _replace_linears(model, targets, factory)
    if replaced == 0:
        raise ValueError(f"No linear layers matched target_modules={targets!r}")
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA branches into their base weights and remove the wrappers."""

    def merge_children(module: nn.Module) -> None:
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                setattr(module, child_name, child.base_layer)
            else:
                merge_children(child)

    merge_children(model)
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA A/B tensors from a model state dict."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_A." in key or ".lora_B." in key
    }


def save_lora(model: nn.Module, path: str | Path) -> None:
    """Save LoRA tensors and metadata to a ``.pt`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    if not state:
        raise ValueError("No LoRA parameters found; call inject_lora first")
    torch.save(state, path)


def load_lora(model: nn.Module, path: str | Path, strict: bool = False) -> nn.Module:
    """Load LoRA A/B tensors into an already-injected model."""
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA load mismatch. missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int, float]:
    """Return ``(trainable, total, ratio)``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total if total else 0.0
    return trainable, total, ratio
