"""Transformer 编码器块：注意力 + 前馈网络 + 残差 + LayerNorm。"""
import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """一层 Transformer encoder block。

    采用 Pre-LayerNorm（先归一化再做子层，再把结果加回残差）：
        x = x + Dropout(Attention(LN1(x)))
        x = x + Dropout(FFN(LN2(x)))
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),                      # 中间激活
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 第一处残差：自注意力
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # 第二处残差：前馈网络
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
