"""检索：BGE 向量召回 + BM25 词法召回（RRF 混合）。

默认返回 hybrid 结果；``mode="dense"`` 可复现纯向量检索，
``mode="lexical"`` 可只看 BM25。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .indexer import (
        BGEEmbedder,
        DEFAULT_INDEX_DIR,
        DEFAULT_MODEL_PATH,
        load_index,
    )
    from .lexical import BM25Index, ranks_from_scores
except ImportError:  # pragma: no cover - 直接脚本运行时的回退
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.indexer import (
        BGEEmbedder,
        DEFAULT_INDEX_DIR,
        DEFAULT_MODEL_PATH,
        load_index,
    )
    from src.lexical import BM25Index, ranks_from_scores

# 同一进程内复用索引/模型；eval/run.py 的 M3 与 M4 会分别实例化 Retriever。
_ARTIFACT_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def _load_artifacts(
    index_dir: str | Path,
    model_path: str | Path,
    device: Optional[str],
) -> Dict[str, Any]:
    key = (str(Path(index_dir).resolve()), str(Path(model_path).resolve()), str(device))
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached
    index, chunks, manifest = load_index(index_dir)
    embedder = BGEEmbedder(
        model_path=model_path,
        device=device,
        batch_size=64,
        max_length=512,
    )
    lexical = BM25Index([chunk.get("text", "") for chunk in chunks])
    artifacts = {
        "index": index,
        "chunks": chunks,
        "manifest": manifest,
        "embedder": embedder,
        "lexical": lexical,
    }
    _ARTIFACT_CACHE[key] = artifacts
    return artifacts


class Retriever:
    """BGE query embedding + FAISS 内积检索。"""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
    ) -> None:
        artifacts = _load_artifacts(index_dir, model_path, device)
        self.index = artifacts["index"]
        self.chunks = artifacts["chunks"]
        self.manifest = artifacts["manifest"]
        self.embedder = artifacts["embedder"]
        self.lexical = artifacts["lexical"]

    def _dense_scores(self, query: str) -> List[float]:
        """对所有 chunk 计算 dense 分数，保证 RRF 排名完整。"""
        query_embedding = self.embedder.encode([query], is_query=True, normalize=True)
        scores, dense_indices = self.index.search(query_embedding, len(self.chunks))
        dense_scores = [0.0] * len(self.chunks)
        for score, index in zip(scores[0], dense_indices[0]):
            if index >= 0:
                dense_scores[int(index)] = float(score)
        return dense_scores

    def retrieve(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """返回 top-k chunk，每个 dict 含 text / score / source。

        Args:
            mode: ``hybrid``（默认） / ``dense`` / ``lexical``。
            dense_weight / lexical_weight / rrf_k: RRF 融合参数。
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if not self.chunks:
            return []
        k = max(1, min(int(k), len(self.chunks)))
        mode = (mode or "hybrid").lower()

        dense_scores = self._dense_scores(query) if mode in {"hybrid", "dense"} else [0.0] * len(self.chunks)
        lexical_scores = self.lexical.score(query) if mode in {"hybrid", "lexical"} else [0.0] * len(self.chunks)
        dense_ranks = ranks_from_scores(dense_scores) if mode in {"hybrid", "dense"} else [0] * len(self.chunks)
        lexical_ranks = ranks_from_scores(lexical_scores) if mode in {"hybrid", "lexical"} else [0] * len(self.chunks)

        if mode == "dense":
            final_scores = dense_scores
        elif mode == "lexical":
            final_scores = lexical_scores
        else:
            final_scores = [
                dense_weight / (rrf_k + dense_ranks[index])
                + lexical_weight / (rrf_k + lexical_ranks[index])
                for index in range(len(self.chunks))
            ]

        order = sorted(range(len(self.chunks)), key=lambda index: final_scores[index], reverse=True)
        results: List[Dict[str, Any]] = []
        for index in order[:k]:
            chunk = self.chunks[index]
            results.append({
                "text": chunk["text"],
                "score": float(final_scores[index]),
                "dense_score": float(dense_scores[index]),
                "lexical_score": float(lexical_scores[index]),
                "dense_rank": int(dense_ranks[index]) if dense_ranks[index] else None,
                "lexical_rank": int(lexical_ranks[index]) if lexical_ranks[index] else None,
                "source": chunk.get("source", {}),
                "chunk_id": chunk.get("chunk_id", int(index)),
                "section": chunk.get("section", ""),
                "rank": len(results) + 1,
            })
        return results

    def retrieve_dense(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """纯向量检索，供消融实验使用。"""
        return self.retrieve(query, k=k, mode="dense")

    def retrieve_lexical(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """纯 BM25 检索，供消融实验使用。"""
        return self.retrieve(query, k=k, mode="lexical")

    def __repr__(self) -> str:
        return (
            f"Retriever(num_chunks={len(self.chunks)}, "
            f"chunk_size={self.manifest.get('chunk_size')}, "
            f"overlap={self.manifest.get('overlap')})"
        )
