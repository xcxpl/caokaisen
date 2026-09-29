"""用堆叠的 TransformerBlock 搭一个文本分类器。"""
import torch
import torch.nn as nn

from .block import TransformerBlock


class TransformerClassifier(nn.Module):
    """一个用于二分类的 Transformer encoder 分类器。

    输入 id 序列 (B, T)；在序列开头加 CLS（由 tokenize_fn 负责），
    最后一层用 CLS 位置的向量做分类。
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 num_classes=2, max_len=128, pad_id=0, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # 可学习的位置编码（绝对位置编码，可选改进：换 RoPE）
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, ids, attention_mask=None):
        B, T = ids.shape
        x = self.token_emb(ids) + self.pos_emb[:, :T, :]
        x = self.dropout(x)

        if attention_mask is None:
            # 默认用 padding mask：把 PAD token 的位置（作为 key）屏蔽掉。
            attention_mask = (ids == self.pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        cls = x[:, 0]          # 取 CLS 位置
        return self.fc(cls)    # (B, num_classes)


def load_for_eval(ckpt_path):
    """从 checkpoint 重建分类器与分词器，供自检脚本调用。

    返回 (model, tokenize_fn)，其中
        tokenize_fn(text: str) -> LongTensor 形状 (T,)
        model(ids) 接受形状 (B, T) 的 id 张量，返回 (B, num_classes) 的 logits
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    vocab = ckpt["vocab"]

    model = TransformerClassifier(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    unk_id = vocab.get("<unk>", 1)
    cls_id = vocab.get("<cls>", 2)
    max_len = config["max_len"]

    def tokenize_fn(text: str) -> torch.Tensor:
        # 序列开头加 CLS，截断到 max_len
        ids = [cls_id] + [vocab.get(ch, unk_id) for ch in text]
        ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)

    return model, tokenize_fn
