"""Subagent：仓库代码搜索。

主 agent 常问「这个功能实现在哪个文件里」。把"列目录 + 打分 + 读片段"这套琐碎动作
交给独立 context 的子 agent，只回一份摘要——主 agent 的 context 里不会塞进整仓文件。
"""
from __future__ import annotations

import re

from ..skill_loader import tokenize

_MAX_SNIPPET_CHARS = 320


class CodeSearchSubagent:
    name = "code-search"
    tool_subset = ("list_dir", "read_file")

    def __init__(self, max_steps: int = 2, backend=None):
        self.max_steps = max_steps
        self.backend = backend          # 保留接口：需要时可用模型总结

    # ---------- 内部：给文件打分（纯规则，零依赖、可复现） ----------
    def _rank(self, session, query: str) -> list[str]:
        tokens = tokenize(query)
        scores: list[tuple[float, str]] = []
        for rel in session.python_files():
            name = rel.lower()
            try:
                text = session.read_text(rel)
            except Exception:  # noqa: BLE001
                continue
            lower = text.lower()
            score = 0.0
            for tok in tokens:
                if tok in name:
                    score += 3.0            # 文件名命中权重最高
                hits = lower.count(tok)
                if hits:
                    score += min(hits, 5) * 0.5
            if re.search(r"^\s*def\s+add\s*\(", text, re.M) and "add" in tokens:
                score += 2.0
            if score > 0:
                scores.append((score, rel))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [rel for _, rel in scores]

    def run(self, session, query: str) -> dict:
        """returns {"subagent", "steps", "matches", "summary"}"""
        # 独立 message 列表（这里只记录子 agent 自己的对话，不回灌主 agent）
        messages = [
            {"role": "system",
             "content": "你是仓库检索子 agent。只负责找出与查询最相关的源码文件，"
                        "并给出简短摘要。工具子集：list_dir / read_file。"},
            {"role": "user", "content": f"在仓库里找与这个需求最相关的实现：{query}"},
        ]
        steps: list[dict] = []
        remaining = self.max_steps

        # 第 1 步：列目录（建立仓库全貌）
        listing = session.call("list_dir", {"recursive": True})
        remaining -= 1
        steps.append({"tool": "list_dir", "ok": listing["ok"],
                      "summary": listing["text"][:300]})
        messages.append({"role": "assistant", "content": "先列出仓库文件。"})
        messages.append({"role": "user", "content": listing["text"][:1200]})

        # 第 2 步：打分排序，读最相关文件的开头片段
        matches = self._rank(session, query)
        snippet = ""
        if matches and remaining > 0:
            top = matches[0]
            read = session.call("read_file", {"path": top})
            remaining -= 1
            snippet = (read.get("data") or {}).get("path", top)
            steps.append({"tool": "read_file", "args": {"path": top}, "ok": read["ok"],
                          "summary": read["text"][:_MAX_SNIPPET_CHARS]})
            messages.append({"role": "assistant", "content": f"读 {top}"})
            messages.append({"role": "user", "content": read["text"][:_MAX_SNIPPET_CHARS]})

        if matches:
            summary = (f"最相关文件：{', '.join(matches[:5])}；"
                       f"建议先读 {snippet or matches[0]}。")
        else:
            summary = "没有找到与需求匹配度高的源码文件（可能关键字对不上）。"

        return {"subagent": self.name, "steps": steps, "matches": matches[:5],
                "messages": messages, "summary": summary}
