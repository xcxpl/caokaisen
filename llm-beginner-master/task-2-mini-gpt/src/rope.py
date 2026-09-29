"""旋转位置编码（RoPE / RoFormer）。

只对 Q、K 施加旋转，V 不旋转。采用 LLaMA 风格前后折半（half-split）的配对方式，
与解码逻辑保持一致。freq 基底默认 base=10000。
"""

from __future__ import annotations

import math

import torch


def precompute_inv_freq(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """计算 inv_freq = base^{-2i/d}，形状 [head_dim // 2]。"""
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def get_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """根据绝对位置（可以带缓存偏移）计算 cos/sin，形状 [T, head_dim//2]。"""
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [T, d//2]
    return angles.cos(), angles.sin()


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 RoPE 应用到 Q、K（前后折半配对）。

    q/k: [B, H, T, D]；cos/sin: [T, D//2]。
    """
    d = q.size(-1)
    hd = d // 2
    cos = cos.to(q.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd]
    sin = sin.to(q.dtype).unsqueeze(0).unsqueeze(0)

    q1, q2 = q[..., :hd], q[..., hd:]
    k1, k2 = k[..., :hd], k[..., hd:]

    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot
