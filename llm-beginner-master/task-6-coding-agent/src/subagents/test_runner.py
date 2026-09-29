"""Subagent：测试执行与失败诊断。

pytest 的输出又长又杂，直接塞回主 agent 会挤爆 context。这个 subagent 自己跑测试、
自己解析 traceback，只回一份 3~5 行的摘要（失败用例 / 断言差异 / 嫌疑源码文件）。
"""
from __future__ import annotations

import re

_FAILED_RE = re.compile(r"FAILED\s+(\S+?)(?:::(\S+))?")
_LOC_RE = re.compile(r"([\w./\\-]+\.py):(\d+):\s*in\s+(\w+)")
_ASSERT_RE = re.compile(r"^E\s+(assert .*)$", re.M)
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)


class TestRunnerSubagent:
    name = "test-runner"
    tool_subset = ("run_tests", "read_file", "list_dir")

    def __init__(self, max_steps: int = 3, backend=None, timeout: int = 60):
        self.max_steps = max_steps
        self.backend = backend
        self.timeout = timeout

    # ---------- 解析 ----------
    @staticmethod
    def _parse_failure(output: str) -> dict:
        info: dict = {"failed_tests": [], "location": None, "assertion": None,
                      "error_type": None}
        info["failed_tests"] = [m.group(2) or m.group(1)
                                for m in _FAILED_RE.finditer(output)][:8]
        loc = _LOC_RE.search(output)
        if loc:
            info["location"] = {"file": loc.group(1).replace("\\", "/"),
                                "line": int(loc.group(2)), "func": loc.group(3)}
        asrt = _ASSERT_RE.search(output)
        if asrt:
            info["assertion"] = asrt.group(1).strip()[:200]
        err = re.search(r"^E\s+(\w*Error|\w*Exception):\s*(.*)$", output, re.M)
        if err:
            info["error_type"] = f"{err.group(1)}: {err.group(2)[:160]}"
        return info

    @staticmethod
    def _suspect_sources(session, info: dict) -> list[str]:
        """从失败测试文件的 import 反推被测源码文件。"""
        loc = info.get("location") or {}
        test_file = loc.get("file")
        candidates: list[str] = []
        if test_file and not test_file.startswith(("..", "/")):
            try:
                text = session.read_text(test_file)
                for m in _IMPORT_RE.finditer(text):
                    module = (m.group(1) or m.group(2) or "").split(".")[0]
                    guess = f"{module}.py"
                    if guess in session.python_files():
                        candidates.append(guess)
            except Exception:  # noqa: BLE001
                pass
        if not candidates:
            # 兜底：找定义了 add 的源码文件
            for rel in session.python_files():
                if rel.startswith("test_"):
                    continue
                try:
                    if re.search(r"^\s*def\s+add\s*\(", session.read_text(rel), re.M):
                        candidates.append(rel)
                        break
                except Exception:  # noqa: BLE001
                    continue
        return candidates

    # ---------- 主入口 ----------
    def run(self, session, command=None) -> dict:
        messages = [
            {"role": "system",
             "content": "你是测试执行子 agent。跑测试、读失败输出、指出嫌疑文件与函数，"
                        "只回摘要。工具子集：run_tests / read_file / list_dir。"},
            {"role": "user", "content": f"跑这个仓库的测试并诊断失败原因。command={command!r}"},
        ]
        steps: list[dict] = []
        remaining = self.max_steps

        args = {"command": command} if command else {}
        result = session.call("run_tests", args)
        remaining -= 1
        passed = bool((result.get("data") or {}).get("passed"))
        output = (result.get("data") or {}).get("output", "")
        steps.append({"tool": "run_tests", "ok": result["ok"], "passed": passed,
                      "summary": result["text"][:400]})
        messages.append({"role": "assistant", "content": "已执行 run_tests。"})
        messages.append({"role": "user", "content": f"测试输出：{output[:1500]}"})

        info = self._parse_failure(output) if not passed else {"failed_tests": [],
                                                              "location": None}
        sources: list[str] = []
        if not passed and remaining > 0:
            sources = self._suspect_sources(session, info)
            if sources:
                read = session.call("read_file", {"path": sources[0]})
                remaining -= 1
                steps.append({"tool": "read_file", "args": {"path": sources[0]},
                              "ok": read["ok"], "summary": read["text"][:300]})
                messages.append({"role": "assistant", "content": f"读嫌疑源码 {sources[0]}"})
                messages.append({"role": "user", "content": read["text"][:800]})

        if passed:
            summary = "pytest 全部通过（exit=0）。"
        else:
            bits = ["pytest 失败"]
            if info["failed_tests"]:
                bits.append("失败用例：" + ", ".join(info["failed_tests"][:4]))
            if info["assertion"]:
                bits.append("断言：" + info["assertion"])
            if info["error_type"]:
                bits.append("异常：" + info["error_type"])
            if sources:
                bits.append("嫌疑源码：" + ", ".join(sources[:3]))
            summary = "；".join(bits) + "。"

        return {"subagent": self.name, "steps": steps, "messages": messages,
                "passed": passed, "output": output, "diagnosis": info,
                "sources": sources, "summary": summary}
