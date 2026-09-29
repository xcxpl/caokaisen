"""轻量 BM25 词法检索，用于与 BGE 向量检索做混合召回。

中文没有天然空格，这里同时使用：
    * 英文/数字单词
    * 单个汉字
    * 相邻汉字 bigram
这样既能匹配 FlashAttention / GraphSAGE / GAT 等术语，也能匹配中文短语。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence

_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    text = str(text).lower()
    tokens = _LATIN_RE.findall(text)
    cjk = _CJK_RE.findall(text)
    tokens.extend(cjk)
    tokens.extend("".join(cjk[i:i + 2]) for i in range(len(cjk) - 1))
    return tokens


def ranks_from_scores(scores: Sequence[float]) -> List[int]:
    """把分数转成 1-based 排名。分数相同按索引稳定排序。"""
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    return ranks


class BM25Index:
    """内存 BM25（k1=1.5, b=0.75），适合千级 chunk 的轻量混合检索。"""

    def __init__(
        self,
        documents: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = [str(document) for document in documents]
        self.tokenized = [tokenize(document) for document in self.documents]
        self.doc_len = [len(tokens) for tokens in self.tokenized]
        self.n_docs = len(self.documents)
        self.avgdl = (
            sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        )
        self.k1 = float(k1)
        self.b = float(b)

        self.inverted: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        document_freq: Counter[str] = Counter()
        for index, tokens in enumerate(self.tokenized):
            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                self.inverted[term].append((index, freq))
                document_freq[term] += 1
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def score(self, query: str) -> List[float]:
        scores = [0.0] * self.n_docs
        if not self.n_docs or not self.avgdl:
            return scores
        for term in tokenize(query):
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for index, term_freq in postings:
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[index] / self.avgdl
                )
                scores[index] += idf * term_freq * (self.k1 + 1) / denominator
        return scores
