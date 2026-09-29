"""file_search：本地目录的文件名检索 + 内容检索 + 文件读取。

自检固定参数键：run({"pattern": "README.md", "dir": <项目根>}) -> 结果须包含 "README.md"

安全边界：所有路径先 resolve，再校验落在允许根目录（项目目录 / 当前工作目录 /
环境变量 FILE_SEARCH_ROOTS）之内，`../../` 这类越界访问会被拒绝。
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "在本地目录里检索文件：可以按文件名/扩展名匹配（如 '*.md'、'todo_note'），"
            "也可以按文件内容里的关键词匹配（如 'TODO'），"
            "pattern 直接给某个文件路径时（如 'data/agent-fixtures/README.md'）会返回该文件的内容。"
            "返回结果是文件路径列表、命中行，或文件内容片段。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "文件名/通配符（*.md、todo_note.md）、内容关键词（TODO）、或待读取的文件路径（data/agent-fixtures/README.md）",
                    "example": "*.md",
                },
                "dir": {
                    "type": "string",
                    "description": "检索的起始目录，相对路径按项目目录解析；默认项目目录",
                    "example": "data/agent-fixtures",
                },
                "mode": {
                    "type": "string",
                    "description": "auto（默认）/ name（只比文件名）/ content（只搜内容）/ read（读文件内容）",
                    "example": "auto",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回多少条命中，默认 20",
                    "example": 20,
                },
            },
            "required": ["pattern"],
        },
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea",
             ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints", "cache"}
TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".jsonl", ".yaml", ".yml", ".toml",
                 ".cfg", ".ini", ".csv", ".tsv", ".log", ".html", ".htm", ".rst",
                 ".tex", ".js", ".ts", ".sh", ".ps1", ".bat", ".env", ""}
MAX_FILE_BYTES = 1_000_000
MAX_OUTPUT_CHARS = 4000
CONTENT_LINE_CHARS = 160


class FileSearchError(RuntimeError):
    """参数非法或路径越界；消息会作为 Observation 喂回模型。"""


def _allowed_roots() -> list[Path]:
    roots = [PROJECT_ROOT]
    try:
        cwd = Path.cwd().resolve()
        roots.append(cwd)
    except OSError:
        pass
    for item in os.environ.get("FILE_SEARCH_ROOTS", "").split(os.pathsep):
        item = item.strip()
        if item:
            roots.append(Path(item))
    unique: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in unique and resolved.exists():
            unique.append(resolved)
    return unique


def _check_inside(path: Path) -> Path:
    resolved = path.resolve()
    for root in _allowed_roots():
        if resolved == root or root in resolved.parents:
            return resolved
    raise FileSearchError(
        f"路径越界被拒绝：{resolved}；只允许访问 {[str(r) for r in _allowed_roots()]} 内的文件")


def _resolve_dir(dir_arg: str | None) -> Path:
    if not dir_arg or not str(dir_arg).strip() or str(dir_arg).strip() in (".", "./"):
        return _check_inside(PROJECT_ROOT)
    raw = Path(str(dir_arg).strip().strip('"').strip("'"))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(PROJECT_ROOT / raw)
        try:
            candidates.append(Path.cwd() / raw)
        except OSError:
            pass
        candidates.append(raw)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists():
            return _check_inside(resolved if resolved.is_dir() else resolved.parent)
    raise FileSearchError(
        f"目录不存在：{dir_arg}（相对路径按项目目录 {PROJECT_ROOT} 解析）")


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
        with path.open("rb") as fh:
            chunk = fh.read(2048)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    return True


def _iter_files(root: Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        base = Path(current)
        for name in files:
            path = base / name
            if _is_text_file(path):
                yield path


def _looks_like_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def _match_names(pattern: str, root: Path, limit: int) -> list[Path]:
    hits: list[Path] = []
    needle = pattern.lower()
    # 小模型经常把通配符写成正则（`.*\.md$` 而不是 `*.md`），三种口径都试一遍，
    # 命中任意一种就算匹配——收得更宽，比 "0 个命中" 让模型瞎猜要好。
    regex = None
    try:
        regex = re.compile(needle)
    except re.error:
        regex = None
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        name = path.name.lower()
        matched = (fnmatch.fnmatch(name, needle) or fnmatch.fnmatch(rel.lower(), needle)
                   or needle in name or needle in rel.lower())
        if not matched and regex is not None:
            matched = bool(regex.search(name)) or bool(regex.search(rel.lower()))
        if matched:
            hits.append(path)
            if len(hits) >= limit * 4:      # 多收一些，外层截断
                break
    return hits


def _match_content(pattern: str, root: Path, limit: int) -> list[tuple[Path, int, str]]:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
        use_regex = True
    except re.error:
        use_regex = False
        needle = pattern.lower()
    hits: list[tuple[Path, int, str]] = []
    for path in _iter_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            matched = bool(regex.search(stripped)) if use_regex else needle in stripped.lower()
            if matched:
                hits.append((path, lineno, stripped[:CONTENT_LINE_CHARS]))
                break   # 每个文件只报第一次命中
        if len(hits) >= limit * 4:
            break
    return hits


def _find_namesakes(path: Path, limit: int = 3) -> list[Path]:
    """项目里同名的其它文件（小模型常常只给文件名、给错目录，这里显式提示歧义）。"""
    found: list[Path] = []
    try:
        resolved = path.resolve()
    except OSError:
        return found
    for root in _allowed_roots():
        for other in _iter_files(root):
            try:
                if other.resolve() == resolved or other.name != path.name:
                    continue
            except OSError:
                continue
            found.append(other)
            if len(found) >= limit:
                return found
    return found


def _first_block(path: Path, limit: int = 200) -> str:
    """取文件的第一段（首个连续非空文本块），同名文件歧义时供模型比对。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            return block[:limit]
    return ""


def _read_file(path: Path, max_chars: int = 1200, ambiguous: bool = False) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    lines: list[str] = []
    namesakes = _find_namesakes(path) if ambiguous else []
    if namesakes:
        lines.append("错误：只给了文件名，项目里存在同名文件，很可能读到的是错误的那个。")
    lines += [
        f"文件：{path}",
        f"字符数：{len(text)}，行数：{len(text.splitlines())}",
    ]
    if namesakes:
        lines.append(f"同名文件还有：{'、'.join(str(p) for p in namesakes)}")
        # 把每个同名文件的第一段也列出来：小模型往往不会「重新调用」，直接抄一眼就能选对
        for other in namesakes[:3]:
            head = _first_block(other)
            if head:
                lines.append(f"  · {other} 的第一段：{head}")
        lines.append("请把你要读的那个文件的完整路径（含目录）直接作为 pattern 重新调用，"
                     "例如 {\"pattern\": \"data/agent-fixtures/README.md\"}。")
    if blocks:
        lines.append(f"第一段：{blocks[0][:600]}")
        lines.append(f"段落数：{len(blocks)}")
    body = text[:max_chars]
    lines.append("内容片段：")
    lines.append(body + ("…（已截断）" if len(text) > max_chars else ""))
    return "\n".join(lines)


def run(args) -> str:
    if isinstance(args, str):
        args = {"pattern": args}
    if not isinstance(args, dict):
        raise FileSearchError(f"file_search 参数必须是对象，收到 {type(args).__name__}")

    pattern = None
    for key in ("pattern", "query", "keyword", "name", "filename", "file", "path", "q", "text"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            pattern = value.strip()
            break
    if pattern is None:
        raise FileSearchError(
            f"缺少 pattern 参数，收到的键为 {sorted(args)}；"
            '正确格式：{"pattern": "*.md", "dir": "data/agent-fixtures"}')

    dir_arg = None
    for key in ("dir", "directory", "folder", "root", "base_dir"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            dir_arg = value.strip()
            break

    mode = str(args.get("mode") or "auto").strip().lower()
    try:
        limit = int(args.get("max_results", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 100))

    # pattern 直接指向一个文件 -> 读取内容（第 10 题这类"文件里写了什么"）
    if mode in ("auto", "read"):
        bare_name = not any(sep in pattern for sep in ("/", "\\")) and not dir_arg
        for candidate in _file_candidates(pattern, dir_arg):
            if candidate.is_file():
                return _read_file(_check_inside(candidate), ambiguous=bare_name)
        if mode == "read":
            raise FileSearchError(f"文件不存在：{pattern}（相对路径按项目目录 {PROJECT_ROOT} 解析）")

    root = _resolve_dir(dir_arg)
    lines = [f"检索目录：{root}", f"pattern：{pattern}", f"模式：{mode}"]

    name_hits = _match_names(pattern, root, limit) if mode in ("auto", "name") else []
    content_hits = _match_content(pattern, root, limit) if mode in ("auto", "content") else []

    if mode in ("auto", "name"):
        lines.append(f"文件名匹配：{len(name_hits)} 个")
        if name_hits:
            rel = [p.relative_to(root).as_posix() for p in name_hits[:limit]]
            lines.extend(f"  {i}. {r}" for i, r in enumerate(rel, 1))
            if len(name_hits) > limit:
                lines.append(f"  …共 {len(name_hits)} 个，已截断显示前 {limit} 个")
        else:
            lines.append("  （无）")

    if mode in ("auto", "content") and (mode == "content" or not name_hits or len(name_hits) < 3):
        lines.append(f"内容匹配：{len(content_hits)} 个文件")
        if content_hits:
            for i, (path, lineno, snippet) in enumerate(content_hits[:limit], 1):
                rel = path.relative_to(root).as_posix()
                lines.append(f"  {i}. {rel}:{lineno}: {snippet}")
        else:
            lines.append("  （无）")

    text = "\n".join(lines)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n…（结果过长已截断）"
    return text


def _file_candidates(pattern: str, dir_arg: str | None = None) -> list[Path]:
    raw = Path(pattern.strip().strip('"').strip("'"))
    if raw.is_absolute():
        return [raw]
    candidates: list[Path] = []
    if dir_arg and not raw.is_absolute():
        try:
            candidates.append(_resolve_dir(dir_arg) / raw)
        except FileSearchError:
            pass
    candidates.append(PROJECT_ROOT / raw)
    try:
        candidates.append(Path.cwd() / raw)
    except OSError:
        pass
    return candidates
