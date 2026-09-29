"""手写字节级 BPE tokenizer（不使用 tiktoken / sentencepiece）。

设计要点：
- 采用 GPT-2 风格 byte-level：先用 bytes_to_unicode 把每个字节映射成唯一 unicode 字符，
  使 BPE 能作用在任意 UTF-8 文本（含中文）上，且 decode 时统一按字节还原，可跨多字节边界。
- 词表 = 特殊 token（<pad>/<bos>/<eos>/<unk>）+ 256 个字节符 + 学习到的 merge 结果。
- encode/decode 可逆（对 UTF-8 文本 round-trip）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# 按“非空白串 / 空白串”分段：保全部字节（round-trip 不变），且每段很短，
# 使 BPE 的 merge 扫描成本从 O(merges × 全文) 降到 O(每段 × merges)，编码大幅提速。
_SPLIT_RE = re.compile(r"\S+|\s+")


def bytes_to_unicode() -> dict[int, str]:
    """把 0..255 的字节映射到唯一可见 unicode 字符（GPT-2 的做法）。

    ASCII 可打印部分保持不变，其余字节映射到 256+n 的码位，保证每个字节
    都能被当作一个独立 "字符" 参与 BPE。返回 {byte: char}。
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """返回相邻 token 对集合。"""
    pairs = set()
    for a, b in zip(word, word[1:]):
        pairs.add((a, b))
    return pairs


class BPETokenizer:
    """手写字节级 BPE 分词器。

    接口约定（与 eval/run.py 对齐）：
        encode(text) -> List[int]
        decode(ids)  -> str
        vocab_size   -> int
        from_pretrained(path) -> BPETokenizer
    """

    def __init__(self) -> None:
        self.byte_encoder: dict[int, str] = bytes_to_unicode()
        self.byte_decoder: dict[str, int] = {v: k for k, v in self.byte_encoder.items()}
        self.encoder: dict[str, int] = {}
        self.decoder: dict[int, str] = {}
        self.special_tokens: list[str] = []
        self.merges: list[tuple[str, str]] = []          # 按训练顺序
        self.bpe_ranks: dict[tuple[str, str], int] = {}  # pair -> rank（越小优先级越高）

        # 初始词表：特殊 token 在前，256 个字节符随后。
        for s in SPECIAL_TOKENS:
            self._add_token(s)
            self.special_tokens.append(s)
        for b in range(256):
            self._add_token(self.byte_encoder[b])

    # ------------------------------------------------------------------ 基础
    def _add_token(self, token: str) -> None:
        if token not in self.encoder:
            self.encoder[token] = len(self.encoder)
            self.decoder[len(self.decoder)] = token

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def special_token_ids(self) -> set[int]:
        return {self.encoder[s] for s in self.special_tokens if s in self.encoder}

    # ------------------------------------------------------------------ 训练
    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """在 text 上学习 BPE merge，直到词表达到 vocab_size 或没有可合并对。

        以文本行作为 "词"（word）进行统计：中文无空格，一行即一个词。
        """
        words: list[list[str]] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            word = [self.byte_encoder[b] for b in line.encode("utf-8")]
            if word:
                words.append(word)

        target_merges = vocab_size - self.vocab_size
        for m in range(target_merges):
            # 统计相邻对频率
            pair_freq: dict[tuple[str, str], int] = {}
            for word in words:
                for a, b in zip(word, word[1:]):
                    key = (a, b)
                    pair_freq[key] = pair_freq.get(key, 0) + 1
            if not pair_freq:
                break
            # 取频率最高的一对；并列时按字典序决定性打平（保证可复现）
            best = max(pair_freq, key=lambda p: (pair_freq[p], p))

            merged = best[0] + best[1]
            self._add_token(merged)
            self.merges.append(best)
            self.bpe_ranks[best] = m

            # 在全部 "词" 中合并该对
            new_words: list[list[str]] = []
            for word in words:
                new_word: list[str] = []
                i = 0
                n = len(word)
                while i < n:
                    if i < n - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

            if verbose and (m + 1) % max(1, target_merges // 10) == 0:
                print(f"[BPE] merges {m + 1}/{target_merges}, vocab={self.vocab_size}")

        if verbose:
            print(f"[BPE] done: merges={len(self.merges)}, vocab_size={self.vocab_size}")

    # ------------------------------------------------------------------ 编码
    def encode(self, text: str) -> list[int]:
        """文本 -> token id 列表。"""
        if not text:
            return []
        ids: list[int] = []
        for chunk in _SPLIT_RE.findall(text):
            word = [self.byte_encoder[b] for b in chunk.encode("utf-8")]
            for t in self._bpe(word):
                ids.append(self.encoder[t])
        return ids

    def _bpe(self, word: list[str]) -> list[str]:
        """对字节符序列应用 BPE merge（按训练 rank 优先级）。"""
        if len(word) <= 1:
            return word
        word = tuple(word)
        while True:
            pairs = get_pairs(word)
            if not pairs:
                break
            # 找 rank 最小（最优先）的可合并对
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            a, b = best
            new_word: list[str] = []
            n = len(word)
            i = 0
            while i < n:
                if i < n - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        return list(word)

    # ------------------------------------------------------------------ 解码
    def decode(self, ids: list[int]) -> str:
        """token id 列表 -> 文本（逐 id 展开为字节后统一按 UTF-8 还原）。"""
        bytes_out = bytearray()
        for i in ids:
            token = self.decoder.get(i)
            if token is None:
                continue
            for ch in token:
                b = self.byte_decoder.get(ch)
                if b is not None:
                    bytes_out.append(b)
        return bytes_out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ 持久化
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "BPETokenizer",
            "special_tokens": self.special_tokens,
            "merges": [list(m) for m in self.merges],
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BPETokenizer":
        tok = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "BPETokenizer":
            raise ValueError(f"不是 BPETokenizer 词表：{path}")

        # 重建特殊 token + 字节基表 + merges，必须与 train() 顺序严格一致。
        tok.special_tokens = payload["special_tokens"]
        tok.encoder = {}
        tok.decoder = {}
        for s in tok.special_tokens:
            tok._add_token(s)
        for b in range(256):
            tok._add_token(tok.byte_encoder[b])
        tok.merges = [tuple(m) for m in payload["merges"]]
        tok.bpe_ranks = {pair: rank for rank, pair in enumerate(tok.merges)}
        # 关键：merge 产出的 token 也必须重新登记到词表，否则 encode 会 KeyError。
        for pair in tok.merges:
            tok._add_token(pair[0] + pair[1])
        return tok
