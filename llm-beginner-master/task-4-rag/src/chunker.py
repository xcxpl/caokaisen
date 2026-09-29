"""PDF 文本抽取、清洗与字符级切分。

任务四自检只要求 ``chunk_text(text, chunk_size, overlap) -> List[str]``，
且 chunk_size / overlap 均以字符计。本模块额外提供带 offset 的切分和
PDF 语料构建函数，供 indexer 使用。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# 句子边界。换行也视为边界，避免跨段硬切。
_SENTENCE_ENDS = "。！？；!?;…\n"

# CJK 字符与常见中文标点，用于去掉 PDF 抽取时插入的多余空格。
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff"
_CJK_PUNCT = r"，。；：！？、（）【】《》“”‘’·—…"

_HEADER_PATTERNS = (
    re.compile(r"^\s*\d{1,4}\s*$"),                                  # 页码
    re.compile(r"^\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*$"),  # 日期页眉
    re.compile(r"^\s*https?://\S+\s*$", re.I),                        # URL 页脚
    re.compile(r"^\s*www\.\S+\s*$", re.I),
)

_HEADING_PATTERNS = (
    re.compile(r"^第\s*[0-9一二三四五六七八九十百零]+\s*章"),
    re.compile(r"^(?:\d+\.)+\d+(?:\s|$)"),
    re.compile(r"^[一二三四五六七八九十]+、"),
    re.compile(r"^附录\s*[A-Za-z0-9一二三四五六七八九十]*"),
)


def _normalize_line(line: str) -> str:
    """去掉 PDF 抽取时在汉字/中文标点之间插入的空格，并压缩连续空白。"""
    line = line.replace("\u00a0", " ").replace("\u3000", " ")
    line = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", line)
    line = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK_PUNCT}])", "", line)
    line = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK}])", "", line)
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def _iter_raw_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def _is_heading(line: str) -> bool:
    normalized = _normalize_line(line)
    if not normalized or len(normalized) > 80:
        return False
    return any(pattern.match(normalized) for pattern in _HEADING_PATTERNS)


def _find_repeated_lines(pages: Sequence[str]) -> Set[str]:
    """找出跨页重复的短行（页眉/页脚/水印）。

    这些行只在第一次出现时保留，后续出现会删除；这样既清理了页眉，
    又不会丢掉章节标题第一次出现的位置。
    """
    counts: Counter[str] = Counter()
    for page in pages:
        for line in set(_iter_raw_lines(page)):
            counts[line] += 1
    if not pages:
        return set()
    threshold = max(3, int(len(pages) * 0.05))
    repeated = set()
    for line, count in counts.items():
        if count < threshold:
            continue
        normalized = _normalize_line(line)
        if not normalized or len(normalized) > 60:
            continue
        # 纯数字、日期、URL 已经由 _HEADER_PATTERNS 处理。
        if any(pattern.match(normalized) for pattern in _HEADER_PATTERNS):
            continue
        repeated.add(normalized)
    return repeated


def clean_page_text(
    raw_text: str,
    repeated_lines: Optional[Set[str]] = None,
    seen_repeated: Optional[Dict[str, int]] = None,
) -> str:
    """清洗单页 PDF 文本，返回按行组织的文本。"""
    repeated_lines = repeated_lines or set()
    seen_repeated = seen_repeated if seen_repeated is not None else {}
    output: List[str] = []
    for line in _iter_raw_lines(raw_text):
        normalized = _normalize_line(line)
        if not normalized:
            continue
        if any(pattern.match(normalized) for pattern in _HEADER_PATTERNS):
            continue
        if normalized in repeated_lines:
            if seen_repeated.get(normalized, 0) > 0:
                continue
            seen_repeated[normalized] = seen_repeated.get(normalized, 0) + 1
        output.append(normalized)
    return "\n".join(output)


def extract_pdf_pages(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """从 PDF 抽取并清洗文本，返回 ``[{"page": 1, "text": "..."}, ...]``。"""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    raw_pages: List[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        raw_pages.append((page_no, text))

    repeated_lines = _find_repeated_lines([text for _, text in raw_pages])
    seen_repeated: Dict[str, int] = {}
    pages: List[Dict[str, Any]] = []
    for page_no, raw_text in raw_pages:
        cleaned = clean_page_text(raw_text, repeated_lines, seen_repeated)
        if len(cleaned) >= 20:
            pages.append({"page": page_no, "text": cleaned})
    return pages


def chunk_text_with_offsets(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Dict[str, Any]]:
    """字符级滑动窗口切分，并尽量在句子边界收尾。

    Returns:
        每个元素形如 ``{"text": ..., "start": ..., "end": ...}``，
        start/end 是相对输入 text 的字符偏移（左闭右开）。
    """
    if chunk_size is None or int(chunk_size) <= 0:
        raise ValueError("chunk_size 必须是正整数")
    text = "" if text is None else str(text)
    if not text.strip():
        return []

    chunk_size = int(chunk_size)
    # overlap 最多取一半，保证窗口仍然前进。
    overlap = max(0, min(int(overlap or 0), chunk_size // 2))
    min_content = max(1, int(chunk_size * 0.6))
    n = len(text)
    chunks: List[Dict[str, Any]] = []
    start = 0

    while start < n:
        target_end = min(start + chunk_size, n)
        end = target_end

        # 不落在文本末尾时，向前找最近的句子边界，避免把句子切断。
        if target_end < n:
            search_start = start + min_content
            boundary = -1
            for ch in _SENTENCE_ENDS:
                pos = text.rfind(ch, search_start, target_end)
                if pos > boundary:
                    boundary = pos
            if boundary >= search_start:
                end = boundary + 1

        raw = text[start:end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        real_start, real_end = start + left_trim, end - right_trim
        if real_end > real_start:
            chunks.append({
                "text": text[real_start:real_end],
                "start": real_start,
                "end": real_end,
            })

        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)
        start = next_start

    return chunks


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """任务四 M1 要求的公开接口：按字符切分文本。"""
    return [item["text"] for item in chunk_text_with_offsets(text, chunk_size, overlap)]


def _relative_pdf_name(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except Exception:
        return path.as_posix()


def build_corpus(
    pdf_path: str | Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> Dict[str, Any]:
    """从 PDF 构建带页码/章节元数据的 chunk 语料。"""
    pages = extract_pdf_pages(pdf_path)
    if not pages:
        raise RuntimeError(f"未能从 PDF 抽取到文本：{pdf_path}")

    parts: List[str] = []
    page_spans: List[Dict[str, Any]] = []
    cursor = 0
    for page in pages:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(page["text"])
        cursor += len(page["text"])
        page_spans.append({"page": page["page"], "start": start, "end": cursor})
    text = "".join(parts)

    chunks = chunk_text_with_offsets(text, chunk_size=chunk_size, overlap=overlap)

    headings: List[Dict[str, Any]] = []
    for match in re.finditer(r"^.+$", text, flags=re.MULTILINE):
        line = match.group().strip()
        if _is_heading(line):
            headings.append({"offset": match.start(), "title": line})

    pdf_name = _relative_pdf_name(pdf_path)
    for chunk_id, chunk in enumerate(chunks):
        pages_hit = [
            span["page"] for span in page_spans
            if span["end"] > chunk["start"] and span["start"] < chunk["end"]
        ]
        section = ""
        for heading in headings:
            if heading["offset"] <= chunk["start"]:
                section = heading["title"]
            else:
                break
        chunk["chunk_id"] = chunk_id
        chunk["section"] = section
        chunk["source"] = {
            "file": pdf_name,
            "page": pages_hit[0] if pages_hit else None,
            "page_end": pages_hit[-1] if pages_hit else None,
            "pages": pages_hit,
            "section": section,
            "chunk_id": chunk_id,
        }

    return {
        "text": text,
        "pages": pages,
        "page_spans": page_spans,
        "headings": headings,
        "chunks": chunks,
    }
