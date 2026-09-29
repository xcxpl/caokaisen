"""Decoder-only mini-GPT：RoPE + KV cache + 采样生成。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .attention import Block
from .sampling import sample_logits


class MiniGPT(nn.Module):
    """极简 decoder-only 语言模型。

    接口约定（与 eval/run.py 对齐）：
        forward(ids, kv_cache=None, return_cache=False)
        generate(prompt_ids, max_new_tokens, top_k, top_p, temperature)
        block_size / max_seq_len 属性
        load_for_eval(ckpt_path) -> (model, tokenizer)
    """

    def __init__(
        self,
        n_vocab: int,
        block_size: int = 256,
        n_embd: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.block_size = block_size
        self.max_seq_len = block_size        # 自检按 block_size 切窗；同时用作 RoPE 语义上限
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.rope_base = rope_base

        self.wte = nn.Embedding(n_vocab, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, rope_base=rope_base, dropout=dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_vocab, bias=False)

        self.apply(self._init_weights)
        # 供采样屏蔽特殊 token；load_for_eval 时由 tokenizer 注入
        self.special_ids: set[int] = set()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        ids: torch.Tensor,
        kv_cache: tuple | None = None,
        return_cache: bool = False,
    ):
        """前向传播。

        ids: [B, T] long。
        - return_cache=False：返回 logits [B, T, V]。
        - return_cache=True ：返回 (logits, per-layer cache)。
          首步传 kv_cache=None 会从零建立缓存；后续步骤传上一轮返回的 cache。
        """
        B, T = ids.size()
        x = self.drop(self.wte(ids))

        if kv_cache is None:
            kv_cache = [None] * self.n_layer
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, kv_cache[i])
            new_cache.append(c)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]

        if return_cache:
            return logits, tuple(new_cache)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> list[int]:
        """自回归生成，返回包含 prompt 在内的完整 id 序列。"""
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if ids.numel() == 0:
            return []
        cache = None
        with torch.no_grad():
            logits, cache = self(ids, kv_cache=cache, return_cache=True)
            last = logits[:, -1:]
            for _ in range(max_new_tokens):
                bad = self.special_ids if self.special_ids else None
                nxt = sample_logits(
                    last, top_k=top_k, top_p=top_p, temperature=temperature, bad_ids=bad
                )
                prompt_ids.append(nxt)
                ids = torch.tensor([[nxt]], dtype=torch.long, device=device)
                logits, cache = self(ids, kv_cache=cache, return_cache=True)
                last = logits[:, -1:]
        return prompt_ids


def load_for_eval(ckpt_path: str | Path) -> tuple[MiniGPT, object]:
    """从 ckpt/best.pt 与同目录 tokenizer.json 加载模型与分词器。"""
    from .tokenizer import BPETokenizer

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    model = MiniGPT(**config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok_path = ckpt_path.parent / "tokenizer.json"
    tokenizer = BPETokenizer.from_pretrained(str(tok_path))
    model.special_ids = tokenizer.special_token_ids
    return model, tokenizer
