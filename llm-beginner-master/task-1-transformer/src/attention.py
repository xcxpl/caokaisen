import math
import torch
import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None, return_attention=False):
    """缩放点积注意力 (scaled dot-product attention)。

    公式：attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    参数
    ----
    Q, K, V : torch.Tensor, 形状 (B, H, T, D)
    mask : torch.Tensor(bool) 或 None
        与 (B, H, T, T) 可广播的形状；``True`` 表示该位置被屏蔽（不参与 attention）。
        常见：padding mask 形状 (B, 1, 1, T)；causal mask 形状 (T, T)。
    return_attention : bool
        若为 True，额外返回注意力权重矩阵 (B, H, T, T)，用于可视化。
    返回
    ----
    输出 (B, H, T, D)；若 return_attention 为 True，则返回 (输出, 注意力权重)。
    """
    d_k = Q.size(-1)
    # 1) Q @ K^T，得到注意力分数 (B, H, T, T)
    scores = Q @ K.transpose(-2, -1)
    # 2) 除以 sqrt(d_k) 做缩放。注意：是用 d_k，不是 d_model。
    scores = scores / math.sqrt(d_k)

    # 3) 屏蔽：把被屏蔽位置填成很大的负数。注意千万别用“乘以 0”，那样 softmax 之后概率还会泄漏。
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    # 4) 在最后一个维度（key 维度 = K 序列长度）上做 softmax。
    attn = torch.softmax(scores, dim=-1)

    # 5) 加权求和：attn @ V
    out = attn @ V

    if return_attention:
        return out, attn
    return out


class MultiHeadAttention(nn.Module):
    """多头注意力：把 d_model 切成 n_heads 份，各 head 独立算注意力，再拼回去。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 三个独立的投影矩阵（Q/K/V 不共享，符合标准做法）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        # 记录最近一次前向的注意力权重，方便画热图
        self.attention_weights = None

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # 投影并切分 head：
        #   (B, T, d_model) -> (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, return_attention=True)
        self.attention_weights = attn

        # 合并 head：
        #   (B, n_heads, T, d_k) -> (B, T, n_heads, d_k) -> (B, T, d_model)
        # 注意 reshape 前要先 .contiguous()
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
