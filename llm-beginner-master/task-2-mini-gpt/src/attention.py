"""Causal multi-head attention，支持 RoPE 与 KV cache。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary, get_cos_sin, precompute_inv_freq


class CausalSelfAttention(nn.Module):
    """因果多头注意力。

    关键点：
    - RoPE 只作用在 Q、K 上。
    - 支持 KV cache：传入 (k_cache, v_cache) 时在序列维（dim=2）上拼接，
      新 token 的 RoPE 位置要接在缓存长度之后（否则角度错误导致与全量前向不一致）。
    - causal mask 按当前总长度动态构建，避免超出训练长度时越界。
    """

    def __init__(self, n_embd: int, n_head: int, rope_base: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.rope_base = rope_base

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.inv_freq = precompute_inv_freq(self.head_dim, rope_base)
        self.register_buffer("inv_freq_buf", self.inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 位置偏移 = 当前缓存长度；RoPE 位置必须从缓存长度起算
        kv_len = 0
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            kv_len = k_cache.size(2)

        positions = torch.arange(kv_len, kv_len + T, device=x.device, dtype=torch.long)
        cos, sin = get_cos_sin(positions, self.inv_freq_buf)
        q, k = apply_rotary(q, k, cos, sin)

        # KV cache：在序列维（dim=2）拼接
        if kv_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        total_t = k.size(2)
        # 因果掩码：query 绝对位置 >= key 绝对位置。query 是“当前新增的 T 个”，
        # key 是“缓存 + 当前”的 total_t 个；二者数量在增量解码时不同，不能直接 tril。
        query_pos = torch.arange(kv_len, kv_len + T, device=x.device)
        key_pos = torch.arange(0, total_t, device=x.device)
        mask = (query_pos[:, None] >= key_pos[None, :]).view(1, 1, T, total_t)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # [B,H,T,D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer decoder block：attention + FFN + residual + LayerNorm（Pre-LN）。"""

    def __init__(self, n_embd: int, n_head: int, rope_base: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, rope_base=rope_base, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, cache
