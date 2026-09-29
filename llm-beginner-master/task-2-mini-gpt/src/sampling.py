"""采样策略：greedy / top-k / top-p / temperature。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_logits(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float = 1.0,
    bad_ids: set[int] | None = None,
) -> int:
    """从 [.., V] logits 中采样一个 token id。

    支持 greedy（temperature=0 或 top_k=1）、top-k、top-p（nucleus）、temperature。
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]     # 取最后一个 position： [B, V]
    if logits.dim() == 2:
        logits = logits[0]            # 只支持 batch=1
    logits = logits.float()

    if temperature is None:
        temperature = 1.0
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    # top-k：保留概率最高的 k 个，其余置 -inf
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.numel())
        thresh = torch.topk(logits, k)[0][-1]
        logits[logits < thresh] = float("-inf")

    # top-p（nucleus）：按概率降序累积到 p，保留这一批，其余置 -inf
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=0)
        keep = cum <= top_p
        # 至少保留一个 token（避免出现全 -inf）
        keep[..., 0] = True
        keep_ids = sorted_idx[keep]
        mask = torch.full_like(logits, float("-inf"))
        mask[keep_ids] = logits[keep_ids]
        logits = mask

    # 屏蔽特殊 token（可选）
    if bad_ids:
        for bad in bad_ids:
            if 0 <= bad < logits.numel():
                logits[bad] = float("-inf")

    # 防全 -inf（极端情况退回 argmax）
    if torch.all(torch.isinf(logits)):
        return int(torch.argmax(logits).item())

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())
