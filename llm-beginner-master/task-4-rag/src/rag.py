"""端到端 RAG：召回 -> rerank -> 生成。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .generator import Generator
    from .reranker import Reranker
    from .retriever import Retriever
except ImportError:  # pragma: no cover - 直接脚本运行时的回退
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.generator import Generator
    from src.reranker import Reranker
    from src.retriever import Retriever

_PIPELINE_CACHE: Dict[str, Any] = {}


def _get_pipeline() -> Dict[str, Any]:
    if _PIPELINE_CACHE:
        return _PIPELINE_CACHE
    pipeline: Dict[str, Any] = {"retriever": Retriever()}
    enable_reranker = os.environ.get("RAG_ENABLE_RERANKER", "1").lower() not in {"0", "false", "no"}
    try:
        pipeline["reranker"] = Reranker() if enable_reranker else None
    except Exception as exc:  # reranker 模型缺失时退化为纯向量检索
        pipeline["reranker"] = None
        pipeline["reranker_error"] = str(exc)
    pipeline["generator"] = Generator()
    _PIPELINE_CACHE.update(pipeline)
    return _PIPELINE_CACHE


def _to_sources(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for document in documents:
        source = dict(document.get("source") or {})
        source["chunk_id"] = document.get("chunk_id", source.get("chunk_id"))
        source["score"] = round(float(document.get("score", 0.0)), 6)
        if "rerank_score" in document:
            source["rerank_score"] = round(float(document["rerank_score"]), 6)
        source["preview"] = str(document.get("text", ""))[:160]
        sources.append(source)
    return sources


def answer(
    query: str,
    top_k: int = 5,
    recall_k: int = 20,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    """端到端回答，返回 ``{"answer": str, "sources": list[dict]}``。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")

    pipeline = _get_pipeline()
    retriever: Retriever = pipeline["retriever"]
    candidates = retriever.retrieve(query, k=max(int(recall_k), int(top_k)))

    reranker: Optional[Reranker] = pipeline.get("reranker")
    if use_reranker and reranker is not None and candidates:
        try:
            selected = reranker.rerank(query, candidates, top_k=top_k)
        except Exception:
            selected = candidates[:top_k]
    else:
        selected = candidates[:top_k]

    answer_text = pipeline["generator"].generate(query, selected)
    return {
        "answer": answer_text,
        "sources": _to_sources(selected),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="任务四 RAG 端到端问答")
    parser.add_argument("query", nargs="?", default="什么是前馈神经网络？")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()
    result = answer(
        args.query,
        top_k=args.top_k,
        recall_k=args.recall_k,
        use_reranker=not args.no_reranker,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
