"""在 ChnSentiCorp 上训练手写的 Transformer 分类器（任务一 M3）。"""
import math
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model import TransformerClassifier


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "ckpt" / "best.pt"


def build_vocab(texts):
    """基于训练语料构造字符级词表（<pad>/<unk>/<cls> 占前 3 个 id）。"""
    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
    for text in texts:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    """句子 -> id 序列，开头加 CLS 并截断。"""
    ids = [vocab["<cls>"]] + [vocab.get(ch, vocab["<unk>"]) for ch in text]
    return ids[:max_len]


class TextDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.texts = [encode(t, vocab, max_len) for t in df["text"]]
        self.labels = df["label"].astype("int64").tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.tensor(self.texts[i], dtype=torch.long), self.labels[i]


def collate(batch):
    """按 batch 内最大长度做 padding，并返回 padding 掩码依据（PAD=0）。"""
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    maxlen = max(t.size(0) for t in texts)
    padded = torch.zeros(len(texts), maxlen, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i, : t.size(0)] = t
    return padded, labels


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_df = pd.read_parquet(DATA / "train.parquet")
    val_df = pd.read_parquet(DATA / "validation.parquet")

    vocab = build_vocab(train_df["text"].tolist())
    vocab_size = len(vocab)
    print(f"vocab_size: {vocab_size}  (训练样本: {len(train_df)}, 验证: {len(val_df)})")

    config = dict(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        num_classes=2,
        max_len=128,
        pad_id=vocab["<pad>"],
        dropout=0.1,
    )

    train_ds = TextDataset(train_df, vocab, config["max_len"])
    val_ds = TextDataset(val_df, vocab, config["max_len"])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    model = TransformerClassifier(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    epochs = 10
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # 余弦退火：从 1 降到 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 3
    bad_epochs = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            global_step += 1

        val_acc = evaluate(model, val_loader, device)
        avg_loss = running_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:2d}] loss={avg_loss:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0
            CKPT.parent.mkdir(exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "config": config, "vocab": vocab},
                CKPT,
            )
            print(f"  -> 新最佳 dev acc {best_acc:.4f}，已保存到 {CKPT}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("早停触发。")
                break

    print(f"\n训练完成，最佳 dev acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
