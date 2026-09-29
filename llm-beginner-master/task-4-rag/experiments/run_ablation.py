"""任务四加分项消融实验。

用法（在 task-4-rag 目录下）：
    python experiments/run_ablation.py chunk-size
    python experiments/run_ablation.py collect-candidates
    python experiments/run_ablation.py rerank

三个阶段分开跑，避免在同一进程里同时驻留 embedding 模型和 reranker，
在 4GB 显卡 / 小页面文件的 Windows 机器上更稳。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import faiss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_text_with_offsets, extract_pdf_pages  # noqa: E402
from src.indexer import BGEEmbedder, DEFAULT_MODEL_PATH, DEFAULT_PDF  # noqa: E402
from src.lexical import BM25Index, ranks_from_scores  # noqa: E402
from src.reranker import Reranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
GOLD_PATH = DATA_DIR / "gold_qa.jsonl"


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text))


def load_gold() -> List[Dict[str, Any]]:
    items = []
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def hit_rank(texts: Sequence[str], anchors: Sequence[str]) -> Optional[int]:
    for rank, text in enumerate(texts, 1):
        normalized = normalize_text(text)
        if any(anchor in normalized for anchor in anchors):
            return rank
    return None


def summarize(ranks: Sequence[Optional[int]]) -> Dict[str, float]:
    total = max(1, len(ranks))
    result: Dict[str, float] = {}
    for k in (1, 3, 5, 10):
        result[f"recall_at_{k}"] = round(
            sum(1 for rank in ranks if rank is not None and rank <= k) / total, 3
        )
    result["mrr"] = round(
        sum(1.0 / rank for rank in ranks if rank is not None) / total, 3
    )
    return result


def rrf_order(
    dense_scores: Sequence[float],
    lexical_scores: Sequence[float],
    rrf_k: float = 10.0,
) -> List[int]:
    dense_ranks = ranks_from_scores(dense_scores)
    lexical_ranks = ranks_from_scores(lexical_scores)
    fused = [
        1.0 / (rrf_k + dense_ranks[index])
        + 1.0 / (rrf_k + lexical_ranks[index])
        for index in range(len(dense_scores))
    ]
    return sorted(range(len(fused)), key=lambda index: fused[index], reverse=True)


def build_concatenated_text() -> str:
    pages = extract_pdf_pages(DEFAULT_PDF)
    parts: List[str] = []
    for page in pages:
        if parts:
            parts.append("\n\n")
        parts.append(page["text"])
    return "".join(parts)


def command_chunk_size(args: argparse.Namespace) -> None:
    items = load_gold()
    anchors = [
        [normalize_text(anchor) for anchor in item["gold_anchors"]]
        for item in items
    ]
    text = build_concatenated_text()
    embedder = BGEEmbedder(
        model_path=DEFAULT_MODEL_PATH,
        device=args.device,
        batch_size=args.batch_size,
    )
    results: Dict[str, Any] = {}
    for size in args.sizes:
        overlap = max(8, size // 8)
        chunks = [
            chunk["text"]
            for chunk in chunk_text_with_offsets(text, chunk_size=size, overlap=overlap)
        ]
        embeddings = embedder.encode(
            chunks,
            is_query=False,
            batch_size=args.batch_size,
            normalize=True,
            show_progress=True,
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        bm25 = BM25Index(chunks)

        dense_ranks: List[Optional[int]] = []
        hybrid_ranks: List[Optional[int]] = []
        for item, item_anchors in zip(items, anchors):
            query_embedding = embedder.encode(
                [item["question"]], is_query=True, normalize=True
            )
            scores, indices = index.search(query_embedding, len(chunks))
            dense_scores = [0.0] * len(chunks)
            for score, index_id in zip(scores[0], indices[0]):
                if index_id >= 0:
                    dense_scores[int(index_id)] = float(score)
            lexical_scores = bm25.score(item["question"])

            dense_order = sorted(
                range(len(chunks)), key=lambda index_id: dense_scores[index_id], reverse=True
            )[:10]
            dense_ranks.append(hit_rank([chunks[index_id] for index_id in dense_order], item_anchors))

            hybrid_order = rrf_order(dense_scores, lexical_scores)[:10]
            hybrid_ranks.append(hit_rank([chunks[index_id] for index_id in hybrid_order], item_anchors))

        entry = {
            "chunk_size": size,
            "overlap": overlap,
            "num_chunks": len(chunks),
            "avg_len": round(sum(len(chunk) for chunk in chunks) / len(chunks), 1),
            "dense": summarize(dense_ranks),
            "hybrid": summarize(hybrid_ranks),
        }
        results[str(size)] = entry
        print(f"[chunk_size={size}] {json.dumps(entry, ensure_ascii=False)}", flush=True)

    (EVAL_DIR / "ablation_chunk_size.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"写入 {EVAL_DIR / 'ablation_chunk_size.json'}")


def command_collect_candidates(args: argparse.Namespace) -> None:
    items = load_gold()
    anchors = [
        [normalize_text(anchor) for anchor in item["gold_anchors"]]
        for item in items
    ]
    retriever = Retriever()
    records: List[Dict[str, Any]] = []
    dense_ranks: List[Optional[int]] = []
    hybrid_ranks: List[Optional[int]] = []
    for item, item_anchors in zip(items, anchors):
        dense = retriever.retrieve(item["question"], k=args.recall_k, mode="dense")
        hybrid = retriever.retrieve(item["question"], k=args.recall_k, mode="hybrid")
        dense_ranks.append(hit_rank([doc["text"] for doc in dense[:10]], item_anchors))
        hybrid_ranks.append(hit_rank([doc["text"] for doc in hybrid[:10]], item_anchors))
        records.append({
            "id": item["id"],
            "question": item["question"],
            "gold_anchors": item["gold_anchors"],
            "dense": dense,
            "hybrid": hybrid,
        })
    payload = {
        "recall_k": args.recall_k,
        "dense_top10": summarize(dense_ranks),
        "hybrid_top10": summarize(hybrid_ranks),
        "records": records,
    }
    (EVAL_DIR / "candidates_hybrid_top20.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, ensure_ascii=False, indent=2))
    print(f"写入 {EVAL_DIR / 'candidates_hybrid_top20.json'}")


def command_rerank(args: argparse.Namespace) -> None:
    payload = json.loads(
        (EVAL_DIR / "candidates_hybrid_top20.json").read_text(encoding="utf-8")
    )
    reranker = Reranker(
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    reranked_ranks: List[Optional[int]] = []
    baseline_ranks: List[Optional[int]] = []
    details = []
    started = time.time()
    for record in payload["records"]:
        anchors = [normalize_text(anchor) for anchor in record["gold_anchors"]]
        candidates = record["hybrid"]
        baseline_ranks.append(hit_rank([doc["text"] for doc in candidates[:10]], anchors))
        reranked = reranker.rerank(record["question"], candidates, top_k=10)
        rank = hit_rank([doc["text"] for doc in reranked], anchors)
        reranked_ranks.append(rank)
        details.append({"id": record["id"], "rank": rank})
        print(f"{record['id']}: rank={rank}", flush=True)
    result = {
        "reranker_model": "bge-reranker-base",
        "device": reranker.device,
        "recall_k": payload["recall_k"],
        "baseline_hybrid_top10": summarize(baseline_ranks),
        "hybrid_top20_rerank_top10": summarize(reranked_ranks),
        "seconds": round(time.time() - started, 1),
        "details": details,
    }
    (EVAL_DIR / "ablation_reranker.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="任务四消融实验")
    sub = parser.add_subparsers(dest="command", required=True)

    chunk = sub.add_parser("chunk-size", help="扫描 chunk_size 对召回的影响")
    chunk.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    chunk.add_argument("--batch-size", type=int, default=64)
    chunk.add_argument("--device", default=None)
    chunk.set_defaults(func=command_chunk_size)

    collect = sub.add_parser("collect-candidates", help="保存默认 512 索引的候选")
    collect.add_argument("--recall-k", type=int, default=20)
    collect.set_defaults(func=command_collect_candidates)

    rerank = sub.add_parser("rerank", help="对候选做 reranker 消融")
    rerank.add_argument("--batch-size", type=int, default=16)
    rerank.add_argument("--max-length", type=int, default=256)
    rerank.add_argument("--device", default="cpu")
    rerank.set_defaults(func=command_rerank)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
