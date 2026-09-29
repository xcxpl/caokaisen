"""BGE embedding + FAISS 索引构建。

默认从 ``data/kb.pdf`` 抽取文本（不是 LaTeX 源），切分后用
``bge-small-zh-v1.5`` 编码，建立内积索引。向量已 L2 归一化，
因此内积等价于 cosine 相似度。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:  # 支持 python -m src.indexer 与 python src/indexer.py 两种运行方式
    from .chunker import build_corpus
except ImportError:  # pragma: no cover - 直接脚本运行时的回退
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.chunker import build_corpus

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = TASK_ROOT / "data" / "kb.pdf"
DEFAULT_MODEL_PATH = TASK_ROOT / "models" / "bge-small-zh-v1.5"
DEFAULT_INDEX_DIR = TASK_ROOT / "data" / "index"

# BGE 官方检索前缀：只加在 query 侧，document 侧不加。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BGEEmbedder:
    """直接基于 transformers 的 BGE encoder，使用 CLS pooling。"""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到 embedding 模型：{self.model_path}\n"
                f"先运行：python data/download.py"
            )
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModel.from_pretrained(str(self.model_path))
        self.model.to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)

    def encode(
        self,
        texts: str | Sequence[str],
        is_query: bool = False,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(text) for text in texts]
        if is_query:
            texts = [
                text if text.startswith(QUERY_PREFIX) else QUERY_PREFIX + text
                for text in texts
            ]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        batch_size = int(batch_size or self.batch_size)
        iterator: Iterable[int] = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="embedding", total=(len(texts) + batch_size - 1) // batch_size)
            except ImportError:
                pass

        all_embeddings: List[np.ndarray] = []
        with torch.inference_mode():
            for start in iterator:
                batch = texts[start:start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                # BGE-small-zh-v1.5 / v1.5 系列使用 CLS pooling。
                cls = outputs.last_hidden_state[:, 0]
                if normalize:
                    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                all_embeddings.append(cls.detach().float().cpu().numpy())
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        return embeddings


def _chunk_to_record(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "source": chunk["source"],
        "section": chunk.get("section", ""),
        "start": chunk.get("start"),
        "end": chunk.get("end"),
    }


def load_index(index_dir: str | Path = DEFAULT_INDEX_DIR):
    """读取 FAISS 索引、chunk 元数据和 manifest。"""
    index_dir = Path(index_dir)
    manifest_path = index_dir / "manifest.json"
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.jsonl"
    if not manifest_path.exists() or not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"索引文件不完整：{index_dir}\n"
            f"先运行：python -m src.indexer"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return index, chunks, manifest


def build_index(
    pdf_path: str | Path = DEFAULT_PDF,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    chunk_size: int = 512,
    overlap: int = 64,
    batch_size: int = 64,
    device: Optional[str] = None,
    force: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """从 PDF 构建并保存 FAISS 索引，返回 manifest。"""
    pdf_path = Path(pdf_path)
    index_dir = Path(index_dir)
    model_path = Path(model_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"找不到知识库 PDF：{pdf_path}\n"
            f"先运行：python data/download.py --skip-models"
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.jsonl"

    if not force and manifest_path.exists() and index_path.exists() and chunks_path.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_manifest = {}
        if (
            old_manifest.get("chunk_size") == int(chunk_size)
            and old_manifest.get("overlap") == int(overlap)
            and old_manifest.get("model_path") == str(model_path)
            and old_manifest.get("pdf_sha256") == _sha256(pdf_path)
        ):
            return old_manifest

    started = time.time()
    corpus = build_corpus(pdf_path, chunk_size=chunk_size, overlap=overlap)
    chunks = corpus["chunks"]
    if not chunks:
        raise RuntimeError("切分后没有 chunk，请检查 PDF 文本抽取是否成功。")

    embedder = BGEEmbedder(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
    )
    # 把章节标题拼进 embedding 输入，给 chunk 补充上下文；
    # 但保存给检索结果/reranker 的 text 仍是原始 chunk 文本。
    embed_texts = []
    for chunk in chunks:
        section = (chunk.get("section") or "").strip()
        embed_texts.append(f"{section}\n{chunk['text']}" if section else chunk["text"])

    embeddings = embedder.encode(
        embed_texts,
        is_query=False,
        batch_size=batch_size,
        normalize=True,
        show_progress=show_progress,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(_chunk_to_record(chunk), ensure_ascii=False) + "\n")

    manifest = {
        "pdf_path": str(pdf_path),
        "pdf_sha256": _sha256(pdf_path),
        "pdf_size": pdf_path.stat().st_size,
        "model_path": str(model_path),
        "embedding_model": model_path.name,
        "chunk_size": int(chunk_size),
        "overlap": int(overlap),
        "dim": int(embeddings.shape[1]),
        "num_chunks": int(embeddings.shape[0]),
        "normalize": True,
        "query_prefix": QUERY_PREFIX,
        "index_type": "IndexFlatIP",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - started, 2),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="从 NNDL PDF 构建 BGE + FAISS 索引")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="知识库 PDF 路径")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="索引输出目录")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="BGE embedding 模型路径")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动")
    parser.add_argument("--force", action="store_true", help="忽略已有索引，强制重建")
    parser.add_argument("--quiet", action="store_true", help="不显示进度条")
    args = parser.parse_args()
    manifest = build_index(
        pdf_path=args.pdf,
        index_dir=args.index_dir,
        model_path=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force,
        show_progress=not args.quiet,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
