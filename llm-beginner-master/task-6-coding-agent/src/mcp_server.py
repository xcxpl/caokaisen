"""任务六 · 底层：手写 MCP server（零依赖，stdio JSON-RPC 2.0）。

设计要点
--------
* 顶层导出 ``list_tools()``：返回 7 个工具 schema（name / description / inputSchema），
  自检 ``mcp_server_lists_tools`` 直接 import 调用即可枚举，不需要真拉起进程。
* 同时可独立启动：``python src/mcp_server.py``，按 MCP stdio 传输（换行分隔的
  JSON-RPC 2.0 报文）应答 ``initialize`` / ``tools/list`` / ``tools/call``。
* 安全约束（README 明确要求）：
  - 文件类工具先 ``Path.resolve()`` 再校验落在 repo 根目录内，拒绝 ``..`` 越界与
    仓库外绝对路径；
  - 所有子进程都用 ``subprocess.run(list, shell=False, cwd=repo)``，命令字符串只经
    ``shlex.split`` 拆分、绝不拼 shell 串（杜绝注入）；
  - 只读 / 可写，不做 ``git reset --hard``、``git clean -fd``、``git checkout --``
    这类会丢改动或删文件的危险操作。
* 工具异常一律转成结构化错误（``ok=False``）返回，绝不让 server 崩掉。
"""
from __future__ import annotations

import copy
import difflib
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "mini-coding-agent", "version": "0.1.0"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = PROJECT_ROOT / "data" / "toy-repo"

MAX_READ_BYTES = 24_000
MAX_OUTPUT_CHARS = 6_000
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv",
                "node_modules", "site-packages", ".mypy_cache"}

# 测试文件保护：issue 明确"不许改测试文件"。放在工具层做硬约束，
# 而不是只在 prompt 里写一句——实测 0.5B 会无视 prompt 直接去改 test_*.py。
_TEST_FILE_RE = re.compile(r"^(?:tests?/.*\.py|test_.*\.py|.*_test\.py)$")


class ToolError(Exception):
    """工具执行失败。附带人类可读原因，由调用方转成结构化错误。"""


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n...（已截断 {len(text) - limit} 字符）...\n{text[-half:]}"


# --------------------------------------------------------------------------- #
# 工具 schema：MCP tools/list 与自检 list_tools() 共用同一份定义
# --------------------------------------------------------------------------- #
_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "read_file",
        "description": "读取仓库内某个文本文件的内容（只读，路径必须在仓库内）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对仓库根目录的文件路径，例如 calculator.py"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "用给定内容整体覆盖写入仓库内某个文件（文件不存在则创建，自动建父目录）。测试文件被禁止写入。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对仓库根目录的文件路径"},
                "content": {"type": "string", "description": "要写入的完整文本内容"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "在文件里做精确字符串替换（把 old 换成 new），比整体重写更安全。找不到 old 会报错而不会瞎改；测试文件被禁止修改。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对仓库根目录的文件路径"},
                "old": {"type": "string", "description": "要被替换的原文片段（必须与文件里完全一致）"},
                "new": {"type": "string", "description": "替换后的新片段"},
                "count": {"type": "integer", "description": "最多替换几处，<=0 表示全部替换，默认 1"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "list_dir",
        "description": "列出仓库内某个目录下的文件（可递归），返回相对路径列表。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对仓库根目录的目录路径，默认仓库根目录"},
                "recursive": {"type": "boolean", "description": "是否递归列出子目录，默认 false"},
            },
            "required": [],
        },
    },
    {
        "name": "run_tests",
        "description": "在仓库根目录执行测试命令并返回输出，默认 `python -m pytest -q`。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "自定义命令，如 `python -m pytest test_calculator.py -q`；留空用默认"},
                "timeout": {"type": "integer", "description": "超时秒数，默认 60"},
            },
            "required": [],
        },
    },
    {
        "name": "git_diff",
        "description": "查看工作区相对基线（默认 HEAD，也可指定某个 commit）的 unified diff；仓库没有 git 历史时退回与 *.orig 基准快照对比。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "对比基线，默认 HEAD；传空字符串表示只看未暂存改动"},
            },
            "required": [],
        },
    },
    {
        "name": "git_apply",
        "description": "把一段 unified diff 应用到工作区（只做 apply，不会 reset / clean / 丢改动）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "unified diff 文本"},
            },
            "required": ["patch"],
        },
    },
]


def list_tools() -> list[dict]:
    """返回工具 schema 列表（深拷贝，调用方随便改）。每个元素都含 ``name``。"""
    return copy.deepcopy(_TOOL_SCHEMAS)


def describe_tools() -> str:
    """把工具清单压成一段紧凑文本，塞进 agent 的 system prompt。"""
    lines = []
    for tool in _TOOL_SCHEMAS:
        props = tool["inputSchema"].get("properties", {})
        params = ", ".join(
            f"{k}{'' if k in tool['inputSchema'].get('required', []) else '?'}"
            for k in props
        )
        lines.append(f"- {tool['name']}({params}): {tool['description']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 工具实现：绑定到某个仓库根目录
# --------------------------------------------------------------------------- #
class RepoTools:
    """绑定到某个仓库根目录的工具集。每个任务一个实例，无跨任务状态。"""

    def __init__(self, repo_root, protect_test_files: bool = True):
        self.repo = Path(repo_root).resolve()
        if not self.repo.exists():
            raise ToolError(f"仓库目录不存在：{self.repo}")
        self.protect_test_files = protect_test_files

    # ---------- 路径安全 ----------
    def _safe_path(self, rel) -> Path:
        raw = str(rel if rel is not None else "").strip().strip('"').strip("'")
        if not raw:
            raise ToolError("path 不能为空")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.repo / candidate
        candidate = candidate.resolve()
        if candidate != self.repo and self.repo not in candidate.parents:
            raise ToolError(f"拒绝仓库外的路径（越界或绝对路径）：{rel}")
        return candidate

    def _guard_writable(self, path: Path) -> None:
        """测试文件只读：不让"改测试让它通过"这条路存在。"""
        if not self.protect_test_files:
            return
        rel = self.rel(path)
        if _TEST_FILE_RE.match(rel):
            raise ToolError(
                f"拒绝修改测试文件 {rel}：测试是用例契约，改测试来让它通过等于作弊。"
                "请修改被测源码（例如 calculator.py）。")

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo).as_posix()
        except ValueError:  # pragma: no cover - _safe_path 已经挡住了
            return str(path)

    # ---------- 子进程 ----------
    def _run(self, args: list[str], timeout: float = 60, stdin_text: str | None = None):
        try:
            return subprocess.run(
                args,
                cwd=str(self.repo),
                input=stdin_text,
                text=True,
                capture_output=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                shell=False,          # 关键：list 形式 + 不用 shell
            )
        except FileNotFoundError as e:
            raise ToolError(f"命令不存在：{args[0]}（{e}）") from e
        except subprocess.TimeoutExpired as e:
            raise ToolError(f"命令超时（>{timeout}s）：{' '.join(args)}") from e

    # ---------- 文件工具 ----------
    def read_text(self, path) -> str:
        """内部使用：读原文（不截断），供 agent 做修复分析。"""
        target = self._safe_path(path)
        if not target.is_file():
            raise ToolError(f"文件不存在：{self.rel(target)}")
        return target.read_text(encoding="utf-8", errors="replace")

    def read_file(self, path, max_bytes: int = MAX_READ_BYTES) -> dict:
        target = self._safe_path(path)
        if not target.is_file():
            raise ToolError(f"文件不存在：{self.rel(target)}")
        raw = target.read_bytes()
        truncated = len(raw) > int(max_bytes)
        text = raw[: int(max_bytes)].decode("utf-8", errors="replace")
        if truncated:
            text += "\n...（文件过大，已截断）"
        return {"text": text or "（空文件）",
                "data": {"path": self.rel(target), "chars": len(text), "truncated": truncated}}

    def write_file(self, path, content) -> dict:
        target = self._safe_path(path)
        self._guard_writable(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = str(content)
        target.write_text(text, encoding="utf-8")
        return {"text": f"已写入 {self.rel(target)}（{len(text)} 字符）",
                "data": {"path": self.rel(target), "chars": len(text)}}

    def edit_file(self, path, old, new, count: int = 1) -> dict:
        target = self._safe_path(path)
        self._guard_writable(target)
        if not target.is_file():
            raise ToolError(f"文件不存在：{self.rel(target)}")
        text = target.read_text(encoding="utf-8", errors="replace")
        old, new = str(old), str(new)
        occurrences = text.count(old)
        if occurrences == 0:
            raise ToolError(
                f"在 {self.rel(target)} 里找不到待替换片段（old 必须与文件内容逐字一致）："
                f"{old[:80]!r}")
        limit = -1 if int(count) <= 0 else int(count)
        replaced = text.replace(old, new, limit) if limit > 0 else text.replace(old, new)
        actual = occurrences if limit <= 0 else min(limit, occurrences)
        target.write_text(replaced, encoding="utf-8")
        return {"text": f"已在 {self.rel(target)} 替换 {actual} 处",
                "data": {"path": self.rel(target), "replaced": actual}}

    def list_dir(self, path: str = ".", recursive: bool = False) -> dict:
        target = self._safe_path(path or ".")
        if not target.is_dir():
            raise ToolError(f"不是目录：{self.rel(target)}")
        entries: list[str] = []
        it = target.rglob("*") if recursive else target.iterdir()
        for p in sorted(it):
            rel_parts = p.relative_to(target).parts
            if any(part in IGNORED_DIRS for part in rel_parts):
                continue
            rel = p.relative_to(self.repo).as_posix()
            entries.append(rel + "/" if p.is_dir() else rel)
            if len(entries) >= 400:
                entries.append("...（条目过多，已截断）")
                break
        return {"text": "\n".join(entries) or "（空目录）",
                "data": {"path": self.rel(target), "count": len(entries)}}

    def python_files(self) -> list[str]:
        """仓库内所有 .py 相对路径（跳过 .git / __pycache__ 等）。"""
        out = []
        for p in sorted(self.repo.rglob("*.py")):
            if any(part in IGNORED_DIRS for part in p.relative_to(self.repo).parts):
                continue
            out.append(p.relative_to(self.repo).as_posix())
        return out

    # ---------- 测试 ----------
    def run_tests(self, command=None, timeout: int = 60) -> dict:
        if command:
            args = list(command) if isinstance(command, (list, tuple)) else shlex.split(str(command))
        else:
            args = [sys.executable, "-m", "pytest", "-q"]
        if not args:
            raise ToolError("测试命令为空")
        proc = self._run(args, timeout=int(timeout))
        output = _clip((proc.stdout or "") + (proc.stderr or ""), 4_000)
        passed = proc.returncode == 0
        head = "测试通过" if passed else f"测试失败（exit={proc.returncode}）"
        return {"text": f"{head}\n{output}",
                "data": {"passed": passed, "returncode": proc.returncode, "output": output,
                         "command": " ".join(args)}}

    # ---------- git ----------
    def git_diff(self, base: str = "HEAD") -> dict:
        text = ""
        if (self.repo / ".git").exists():
            args = ["git", "diff", "--no-color"]
            if base:
                args.append(base)
            try:
                text = self._run(args, timeout=30).stdout or ""
            except ToolError:
                text = ""
        if not text.strip():
            text = self._snapshot_diff()
        return {"text": _clip(text) or "（没有检测到差异）", "data": {"patch": text}}

    def _snapshot_diff(self) -> str:
        """没有 git 历史时的兜底：拿 *.orig 基准快照与当前文件比 unified diff。"""
        chunks = []
        for orig in sorted(self.repo.rglob("*.orig")):
            if any(part in IGNORED_DIRS for part in orig.relative_to(self.repo).parts):
                continue
            current = orig.with_suffix("")     # calculator.py.orig -> calculator.py
            if not current.exists():
                continue
            old = orig.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            new = current.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            diff = difflib.unified_diff(
                old, new,
                fromfile=f"a/{current.relative_to(self.repo).as_posix()}",
                tofile=f"b/{current.relative_to(self.repo).as_posix()}",
            )
            chunks.append("".join(diff))
        return "".join(chunks)

    def git_apply(self, patch) -> dict:
        text = str(patch or "")
        if not text.strip():
            raise ToolError("patch 为空")
        proc = self._run(["git", "apply", "--whitespace=nowarn", "-"], timeout=30, stdin_text=text)
        if proc.returncode != 0:
            raise ToolError(f"git apply 失败：{(proc.stderr or proc.stdout or '').strip()[:300]}")
        return {"text": "已应用 patch", "data": {"applied": True}}

    # ---------- 统一调用入口 ----------
    def _handlers(self) -> dict:
        return {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "list_dir": self.list_dir,
            "run_tests": self.run_tests,
            "git_diff": self.git_diff,
            "git_apply": self.git_apply,
        }

    def call(self, name: str, args: dict | None = None) -> dict:
        """执行一个工具，永不抛异常；返回 {tool, ok, text, data, dropped_args}。"""
        args = dict(args or {})
        handler = self._handlers().get(str(name))
        if handler is None:
            return {"tool": str(name), "ok": False, "text": f"错误：未知工具 {name}", "data": {}}
        # 弱模型经常多塞参数：按签名过滤掉多余的键，而不是直接报错
        params = inspect.signature(handler).parameters
        valid = {k: v for k, v in args.items() if k in params}
        dropped = sorted(set(args) - set(valid))
        try:
            out = handler(**valid)
        except ToolError as e:
            return {"tool": str(name), "ok": False, "text": f"错误：{e}", "data": {},
                    "dropped_args": dropped}
        except Exception as e:  # noqa: BLE001 —— 工具异常不能冒泡 crash 整个循环
            return {"tool": str(name), "ok": False, "text": f"错误：{type(e).__name__}: {e}",
                    "data": {}, "dropped_args": dropped}
        out = out if isinstance(out, dict) else {"text": str(out), "data": {}}
        text = out.get("text", "")
        if dropped:
            text += f"\n（注意：已忽略多余的参数 {dropped}）"
        return {"tool": str(name), "ok": True, "text": text,
                "data": out.get("data", {}) or {}, "dropped_args": dropped}


_DEFAULT_SESSION: RepoTools | None = None


def default_session() -> RepoTools:
    global _DEFAULT_SESSION
    if _DEFAULT_SESSION is None or not _DEFAULT_SESSION.repo.exists():
        _DEFAULT_SESSION = RepoTools(os.environ.get("AGENT_REPO") or DEFAULT_REPO)
    return _DEFAULT_SESSION


def call_tool(name: str, args: dict | None = None, repo_root=None) -> dict:
    """模块级调用入口：给定 repo_root 就新建会话，否则用默认 toy-repo 会话。"""
    session = RepoTools(repo_root) if repo_root else default_session()
    return session.call(name, args)


# --------------------------------------------------------------------------- #
# stdio JSON-RPC（MCP 传输层）
# --------------------------------------------------------------------------- #
def _result(mid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def handle_message(msg: dict, session: RepoTools | None = None) -> dict | None:
    """处理一条 JSON-RPC 报文；通知类（无 id）返回 None 表示不应答。"""
    session = session or default_session()
    if not isinstance(msg, dict) or "method" not in msg:
        return None
    mid = msg.get("id")
    method = msg.get("method")

    if method == "initialize":
        return _result(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })
    if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return _result(mid, {})
    if method == "tools/list":
        return _result(mid, {"tools": list_tools()})
    if method == "tools/call":
        params = msg.get("params") or {}
        out = session.call(params.get("name"), params.get("arguments") or {})
        return _result(mid, {
            "content": [{"type": "text", "text": out["text"]}],
            "structuredContent": out["data"],
            "isError": not out["ok"],
        })
    if mid is None:
        return None
    return _error(mid, -32601, f"未实现的方法：{method}")


def serve_stdio(repo_root=None) -> None:
    """换行分隔 JSON-RPC 主循环，直到 stdin 关闭。"""
    session = RepoTools(repo_root or os.environ.get("AGENT_REPO") or DEFAULT_REPO)
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line.lower().startswith(("content-length:", "content-type:")):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_message(msg, session)
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> None:
    serve_stdio()


if __name__ == "__main__":
    main()
