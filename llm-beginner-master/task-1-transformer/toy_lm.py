"""任务一 M4：用手写 attention + causal mask 跑一个迷你因果语言模型。

用唐诗做 next-token prediction，验证"未来词元不泄漏"。
"""
import math
from pathlib import Path

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT.parent / "poetryFromTang.txt"


def build_vocab(text):
    chars = sorted(set(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def causal_mask(T, device=None):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class CausalBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        T = x.size(1)
        x = x + self.attn(self.norm1(x), mask=causal_mask(T, x.device))
        x = x + self.ffn(self.norm2(x))
        return x


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=64, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([CausalBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + self.pos[:, :T, :]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def get_batch(text_ids, batch_size, seq_len, device):
    n = len(text_ids)
    starts = torch.randint(0, n - seq_len - 1, (batch_size,))
    x = torch.stack([text_ids[s : s + seq_len] for s in starts])
    y = torch.stack([text_ids[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = CORPUS.read_text(encoding="utf-8")
    vocab = build_vocab(text)
    itos = {v: k for k, v in vocab.items()}
    text_ids = torch.tensor([vocab.get(c, vocab["<unk>"]) for c in text])

    model = ToyCausalLM(len(vocab)).to(device)
    print(f"语料长度 {len(text)} 字符, vocab {len(vocab)}, device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq_len, batch_size = 48, 32
    total = 300
    for step in range(1, total + 1):
        x, y = get_batch(text_ids, batch_size, seq_len, device)
        opt.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d} loss={loss.item():.4f}")

    # 用因果模型生成一小段（汉字种子 + 温度采样）
    model.eval()
    import random
    cjk = [v for v in itos if len(itos[v]) == 1 and ord(itos[v]) > 0x4E00]
    seed = random.choice(cjk)
    generated = [seed]
    start = torch.tensor([[seed]], device=device)
    temperature = 0.9
    with torch.no_grad():
        for _ in range(40):
            logits = model(start)
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            generated.append(nxt)
            start = torch.tensor([generated[-seq_len:]], device=device)
    print("生成示例: " + "".join(itos[i] for i in generated))


if __name__ == "__main__":
    main()
