"""任务六 · 顶层：CodingAgent —— 把 Tools / Skills / Subagents 三层串成一条 agentic loop。

设计取舍（本机只有 4GB 显存，用 0.5B 小模型）
-----------------------------------------------
小模型的"规划能力"很不可靠：经常输出不了合法 JSON、选错工具、改错地方、或者在测试
还没过时就宣布完成。所以这条 loop 采用 **模型优先 + 规则 Planner 兜底** 的双轨制：

1. 每一轮先请模型给出下一步动作（JSON），解析成功且不是重复动作就执行；
2. 模型输出不可用 / 动作重复 / 过早收工 → 规则 Planner 接管（它的状态机基于真实
   测试输出：跑测试 → 找罪魁文件 → 读源码 → 最小改动 → 再跑测试），保证 loop 一定
   向前推进；
3. 每一步都记进 ``Trace.steps``，并标注该步来自 ``model`` 还是 ``script``。

这样即使模型全程"不听话"，toy-repo 也能被真实地读文件、改代码、跑 pytest 修通；
而模型一旦给出可用规划，它就走模型那条路。哪一步是谁做的，trace 里一目了然。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .mcp_server import RepoTools, describe_tools
from .skill_loader import SkillLoader
from .subagents import CodeSearchSubagent, TestRunnerSubagent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

_SYSTEM_TEMPLATE = """你是一个 mini coding agent 的规划器。目标：读懂 issue，用工具修改这个仓库的代码，直到 `python -m pytest` 全绿。

## 可用工具
{tools}

## 输出格式
每次只输出 **一个 JSON 对象**，不要输出解释、不要用 markdown 代码块：
{{"thought": "一句话想法", "tool": "工具名", "args": {{"参数名": "值"}}}}
任务确实完成时输出：
{{"thought": "一句话想法", "final": "完成说明"}}

## 规则
1. 只改源码，**不要改测试文件**（测试是需求契约）。
2. 一次只调用一个工具。
3. 改完代码必须再用 run_tests 确认，看到"测试通过"才算完成。
4. 不要重复调用完全相同的工具+参数。

## 示例
用户：修复 add 让测试通过（当前测试失败）
助手：{{"thought": "先跑一次测试看失败详情", "tool": "run_tests", "args": {{}}}}
助手：{{"thought": "读源码定位 add 的实现", "tool": "read_file", "args": {{"path": "calculator.py"}}}}
助手：{{"thought": "add 写成了减法，改回加法", "tool": "edit_file", "args": {{"path": "calculator.py", "old": "return a - b", "new": "return a + b"}}}}
助手：{{"thought": "重新跑测试确认", "tool": "run_tests", "args": {{}}}}"""

_JSON_RE = re.compile(r"\{.*\}", re.S)
_TOOL_ALIASES = {
    "read": "read_file", "readfile": "read_file", "cat": "read_file",
    "write": "write_file", "writefile": "write_file",
    "edit": "edit_file", "replace": "edit_file", "patch_file": "edit_file",
    "ls": "list_dir", "list": "list_dir", "list_files": "list_dir",
    "test": "run_tests", "pytest": "run_tests", "run_test": "run_tests",
    "diff": "git_diff", "apply": "git_apply",
}


class Trace(dict):
    """一次 run 的完整轨迹。dict 子类，自检直接 trace.get("steps"/"patch"/"tests_passed")。"""

    def summary(self) -> str:
        return (f"steps={len(self.get('steps', []))} "
                f"tests_passed={self.get('tests_passed')} "
                f"patch={'有' if self.get('patch') else '无'}")


def _sig(tool: str, args: dict) -> str:
    try:
        payload = json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        payload = str(args)
    return f"{tool}|{payload}"


def _parse_action(text: str) -> dict | None:
    """从模型输出里抠出一个动作 JSON；抠不出来就返回 None（交给规则 Planner）。"""
    if not text:
        return None
    match = _JSON_RE.search(text)
    raw = match.group(0) if match else text.strip()
    data = None
    for candidate in (raw, raw.replace("'", '"'), raw.replace("，", ",").replace("：", ":")):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        return None
    thought = str(data.get("thought") or data.get("reasoning") or "")
    tool = data.get("tool") or data.get("action") or data.get("tool_name")
    if isinstance(tool, str) and tool.strip():
        args = data.get("args") if data.get("args") is not None else data.get("arguments")
        if not isinstance(args, dict):
            args = {}
        # 有些模型会把参数平铺在同一层：{"tool": "read_file", "path": "x"}
        for key in ("path", "pattern", "old", "new", "content", "command", "recursive"):
            if key in data and key not in args:
                args[key] = data[key]
        return {"type": "action", "tool": tool.strip(), "args": args, "thought": thought}
    for key in ("final", "final_answer", "answer", "done"):
        if key in data:
            return {"type": "final", "final": str(data[key]), "thought": thought}
    return None


def _add_return_expr(source: str) -> str | None:
    """返回 `def add(...)` 里的 return 表达式（找不到函数/return 返回 None）。"""
    lines = source.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*def\s+add\s*\(", line):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip() and not lines[j][:1].isspace():   # 顶格的新定义 → 函数结束
            end = j
            break
    for j in range(start, end):
        body = lines[j].strip()
        if body.startswith("return "):
            return body[len("return "):]
    return None


def _fix_add_source(source: str) -> tuple[str, str] | None:
    """最小修复规则：`def add(...)` 的 return 里首个减号换成加号。

    返回 (old, new) 两个逐字对应的片段，直接喂给 edit_file。
    规则刻意收得很窄——只处理"加法写成了减法"这一类，不猜别的。
    """
    expr = _add_return_expr(source)
    if expr is None or "-" not in expr or "+" in expr:
        return None
    return "return " + expr, "return " + expr.replace("-", "+", 1)


def _add_is_fixed(source: str) -> bool:
    """源码里 add 是否已经是"两数相加"的正确实现（用于判断修复是否真的落地）。"""
    expr = _add_return_expr(source)
    return expr is not None and "+" in expr and "-" not in expr


def _guess_source_file(session: RepoTools, hint: str = "") -> str | None:
    """定位定义了 add 的源码文件（排除测试文件）。"""
    hinted = re.findall(r"([\w./\\-]+\.py)", hint or "")
    for rel in hinted:
        rel = rel.replace("\\", "/").lstrip("./")
        if rel.startswith("test_") or "/test_" in rel:
            continue
        try:
            if re.search(r"^\s*def\s+add\s*\(", session.read_text(rel), re.M):
                return rel
        except Exception:  # noqa: BLE001
            continue
    for rel in session.python_files():
        if Path(rel).name.startswith("test_"):
            continue
        try:
            if re.search(r"^\s*def\s+add\s*\(", session.read_text(rel), re.M):
                return rel
        except Exception:  # noqa: BLE001
            continue
    return None


class CodingAgent:
    """Tools(MCP) + Skills + Subagents 的编排层。"""

    def __init__(self, backend=None, max_steps: int | None = None, skills_dir=None,
                 verbose: bool = True, use_model: bool | None = None,
                 use_subagents: bool = True, use_skills: bool = True,
                 use_planner: bool = True, temperature: float = 0.0):
        self.max_steps = int(max_steps or os.environ.get("AGENT_MAX_STEPS", 4))
        self.skills_dir = str(skills_dir or DEFAULT_SKILLS_DIR)
        self.verbose = verbose
        self.use_model = (os.environ.get("AGENT_NO_MODEL") != "1") if use_model is None else use_model
        self.use_subagents = use_subagents
        self.use_skills = use_skills
        self.use_planner = use_planner
        self.temperature = temperature
        self._backend = backend
        self._backend_error: str | None = None

    # ---------- 惰性加载模型：失败也不影响 loop 跑通 ----------
    def _get_backend(self):
        if self._backend is not None or self._backend_error is not None:
            return self._backend
        if not self.use_model:
            self._backend_error = "use_model=False"
            return None
        try:
            from .llm import get_backend
            self._backend = get_backend(verbose=self.verbose)
        except Exception as e:  # noqa: BLE001 —— 模型起不来就纯规则 Planner 兜底
            self._backend_error = str(e)
            self._backend = None
            if self.verbose:
                print(f"[agent] 模型后端不可用，改用规则 Planner：{e}")
        return self._backend

    # ---------- 规则 Planner：模型不可用时的前进引擎 ----------
    def _scripted_action(self, state: dict, session: RepoTools):
        """基于真实测试输出的确定性状态机，保证 loop 一定会向前推进。"""
        def mk(tool: str, args: dict, thought: str, dedupe: bool = True):
            if dedupe and _sig(tool, args) in state["seen"]:
                return None
            return {"type": "action", "tool": tool, "args": args,
                    "thought": thought, "source": "script"}

        # (a) 一次测试都没跑过 → 先建立事实
        if state["runs"] == 0:
            return mk("run_tests", {}, "跑 pytest 确认当前状态", dedupe=False)
        if state["passed"]:
            return None
        # (b) 已应用修复、改动还没验证 → 立刻验证
        if state["applied_fix"] and state["dirty"]:
            return mk("run_tests", {}, "改完代码，重新跑 pytest 验证", dedupe=False)
        # (c) 定位可疑源码文件
        if not state["suspect_file"]:
            guess = _guess_source_file(session, state.get("fail_output", ""))
            if not guess:
                return None
            state["suspect_file"] = guess
        suspect = state["suspect_file"]
        # (d) 还没读过源码 → 读
        if suspect not in state["read"]:
            action = mk("read_file", {"path": suspect}, f"读取 {suspect} 定位 bug")
            if action:
                return action
        # (e) 已知修复还没落地 → 落地（必要时先从 .orig 基准快照恢复，挡住模型的乱改）
        if not state["applied_fix"]:
            try:
                source = session.read_text(suspect)
            except ToolError:
                return None
            if _add_is_fixed(source):
                # 源码里已经是正确实现（可能是模型改对了但我们没记上）→ 只差验证
                state["applied_fix"] = True
                return mk("run_tests", {}, f"{suspect} 已是正确实现，跑测试验证", dedupe=False)
            fix = _fix_add_source(source)
            if fix is None:
                try:
                    original = session.read_text(suspect + ".orig")
                except ToolError:
                    return None
                action = mk("write_file", {"path": suspect, "content": original},
                            f"{suspect} 当前已不是可修复形态，先恢复到基准快照")
                if action:
                    return action
                return None
            old, new = fix
            action = mk("edit_file", {"path": suspect, "old": old, "new": new},
                        f"最小改动：把 {suspect} 里的 `{old}` 改成 `{new}`")
            if action:
                return action
        return None

    # ---------- prompt 组装 ----------
    def _system_prompt(self, skill_ctx: str) -> str:
        prompt = _SYSTEM_TEMPLATE.format(tools=describe_tools())
        if skill_ctx:
            prompt += "\n\n## 已按需加载的 Skill（progressive disclosure）\n" + skill_ctx
        return prompt

    def _build_messages(self, skill_ctx: str, issue: str, history: list[str],
                        used: set[str] | None = None) -> list[dict]:
        tail = "\n".join(history) if history else "（还没有执行任何动作）"
        used_txt = "；".join(sorted(used or [])[:12]) or "（无）"
        user = (f"## issue\n{issue}\n\n"
                f"## 已经执行过的动作（**不要重复选**）\n{used_txt}\n\n"
                f"## 到目前为止的轨迹\n{tail}\n\n"
                f"请输出**下一个**动作的 JSON；它必须与上面已经执行过的动作不同。")
        return [{"role": "system", "content": self._system_prompt(skill_ctx)},
                {"role": "user", "content": user}]

    # ---------- 主入口 ----------
    def run(self, repo_path: str, issue: str) -> Trace:
        started = time.time()
        repo = Path(repo_path).resolve()
        session = RepoTools(repo)
        trace = Trace(steps=[], patch="", tests_passed=False, repo=str(repo),
                      issue=issue, skills=[], subagents=[], notes=[],
                      max_steps=self.max_steps, temperature=self.temperature,
                      config={"use_model": self.use_model, "use_skills": self.use_skills,
                              "use_subagents": self.use_subagents,
                              "use_planner": self.use_planner})

        # ---- 1) Skills：先用 description 做匹配，命中后才读正文（progressive disclosure）----
        loader = SkillLoader(self.skills_dir)
        if self.use_skills:
            matched = loader.match(issue, top_k=2)
            if not matched:   # 关键词没对上时，退回最贴合"修 bug"的一个
                matched = [s["name"] for s in loader.list_skills() if s["name"] == "bug-fix"]
        else:
            matched = []      # 消融：纯 prompt，不加载任何 Skill 正文
        trace["skill_catalog"] = [{"name": s["name"], "description": s["description"]}
                                  for s in loader.list_skills()]
        trace["skills"] = [{"name": name, "description": (loader.get(name) or {}).get("description", "")}
                           for name in matched]
        skill_ctx = "\n\n".join(f"### {name}\n{loader.load(name)}" for name in matched)
        trace["notes"].append(
            f"匹配到 Skill：{matched}（正文已按需载入 context）" if matched
            else "use_skills=False：未加载任何 Skill（纯 prompt 消融组）")

        # ---- 1.5) 派 CodeSearchSubagent 定位相关文件（主 agent 只拿摘要）----
        if self.use_subagents:
            found = CodeSearchSubagent().run(session, issue)
            trace["subagents"].append({"name": "code-search", "phase": "locate",
                                       "summary": found["summary"],
                                       "matches": found["matches"],
                                       "steps": len(found["steps"])})
            trace["notes"].append(f"code-search 子 agent 命中：{found['matches'][:3]}")

        # ---- 2) 基线：派 TestRunnerSubagent 跑一次测试，主 agent 只拿摘要 ----
        state = {"runs": 0, "passed": False, "dirty": False, "applied_fix": False,
                 "read": set(), "seen": set(), "suspect_file": None,
                 "fail_output": "", "model_final": 0}
        if self.use_subagents:
            base = TestRunnerSubagent().run(session)
            base_passed, base_output, base_summary = base["passed"], base["output"], base["summary"]
            trace["subagents"].append({"name": "test-runner", "phase": "baseline",
                                       "summary": base["summary"], "steps": len(base["steps"])})
            source, via = "subagent", "TestRunnerSubagent"
        else:   # 消融：不派 subagent，直接在 loop 里跑测试
            raw = session.call("run_tests", {})
            base_passed = bool((raw.get("data") or {}).get("passed"))
            base_output = (raw.get("data") or {}).get("output", "")
            base_summary = raw["text"][:400]
            source, via = "script", "in-loop"
        state["runs"] += 1
        state["passed"] = base_passed
        state["fail_output"] = base_output
        if self.use_planner:
            # 规则 Planner 需要知道基线跑过了；纯模型模式不预占这个签名，
            # 否则模型第一步提的 run_tests 会被误判成"重复"。
            state["seen"].add(_sig("run_tests", {}))
        trace["steps"].append({
            "step": 0, "phase": "baseline", "source": source,
            "thought": "先跑一次基线测试，确认当前失败情况",
            "tool_call": {"tool": "run_tests", "args": {}},
            "observation": base_summary, "error": False, "via": via,
        })

        # ---- 3) agentic loop ----
        backend = self._get_backend()
        trace["backend"] = backend.describe() if backend else f"不可用（{self._backend_error}）"
        history = [f"[0] run_tests() -> {base_summary}"]
        step = 1
        final_answer = ""

        while step <= self.max_steps:
            if state["passed"]:
                break
            # 3.1 优先问模型
            action, source = None, "model"
            if backend is not None and state["model_final"] < 1:
                try:
                    raw = backend.chat(
                        self._build_messages(skill_ctx, issue, history, state["seen"]),
                        max_tokens=160, temperature=self.temperature, stop=["\n\n"])
                    action = _parse_action(raw)
                    trace.setdefault("model_outputs", []).append(
                        {"step": step, "parsed": bool(action), "raw": (raw or "").strip()[:400]})
                except Exception as e:  # noqa: BLE001
                    trace["notes"].append(f"第 {step} 步模型调用失败：{e}")
                    action = None
                if action is None:
                    trace["notes"].append(f"第 {step} 步模型输出不可解析，规则 Planner 接管")

            # 3.2 模型动作不合法/重复/过早收工 → 规则 Planner
            def _stale(act):
                return act and act.get("type") == "action" and \
                    _sig(act.get("tool", ""), act.get("args", {})) in state["seen"]

            if action is None or _stale(action):
                if not self.use_planner:     # 纯模型对照：不兜底，让 trace 如实反映模型能力
                    trace["notes"].append(
                        f"第 {step} 步模型动作{'重复' if action else '不可用'}，"
                        "use_planner=False 不接管，loop 结束")
                    break
                if action is not None:
                    trace["notes"].append(f"第 {step} 步模型动作重复，规则 Planner 接管")
                action = self._scripted_action(state, session)
                source = "script"
            if action is None:
                break

            # 3.3 收工判定
            if action["type"] == "final":
                record = {"step": step, "phase": "final", "source": source,
                          "thought": action.get("thought", ""),
                          "tool_call": {"tool": "final_answer", "args": {}},
                          "observation": action.get("final", ""), "error": False}
                trace["steps"].append(record)
                if state["passed"]:
                    final_answer = action.get("final", "")
                    break
                state["model_final"] += 1
                trace["notes"].append(f"第 {step} 步模型宣布完成但 pytest 未通过，拒绝收工")
                history.append(f"[{step}] final_answer -> {action.get('final', '')}（被拒绝）")
                step += 1
                continue

            # 3.4 执行工具
            tool = _TOOL_ALIASES.get(action["tool"].lower(), action["tool"])
            args = action.get("args", {}) or {}
            result = session.call(tool, args)
            observation = result["text"]
            state["seen"].add(_sig(tool, args))
            if result["ok"]:
                self._update_state(state, tool, args, result)
            else:
                # 工具失败（越界 / 改测试文件 / old 对不上…）不算"已修复"
                trace["notes"].append(
                    f"第 {step} 步 {tool} 执行失败，不更新状态：{observation[:120]}")

            trace["steps"].append({
                "step": step, "phase": "act", "source": source,
                "thought": action.get("thought", ""),
                "tool_call": {"tool": tool, "args": args},
                "observation": observation, "error": not result["ok"],
            })
            history.append(f"[{step}] {tool}({json.dumps(args, ensure_ascii=False)}) -> "
                           f"{observation[:300]}")
            if self.verbose:
                print(f"[step {step}][{source}] {tool} {json.dumps(args, ensure_ascii=False)[:120]}"
                      f" -> {'OK' if result['ok'] else 'ERR'}", flush=True)
            step += 1

        # ---- 3.5 兜底收敛：模型没在步数上限内修通，就让规则 Planner 走完 ----
        if not state["passed"] and self.use_planner:
            trace["notes"].append("模型未在步数上限内修通，进入规则 Planner 兜底阶段")
            guard = 0
            while not state["passed"] and guard < 6:
                guard += 1
                action = self._scripted_action(state, session)
                if action is None or action["type"] != "action":
                    break
                tool = _TOOL_ALIASES.get(action["tool"].lower(), action["tool"])
                args = action.get("args", {}) or {}
                result = session.call(tool, args)
                state["seen"].add(_sig(tool, args))
                if result["ok"]:
                    self._update_state(state, tool, args, result)
                step += 1
                trace["steps"].append({
                    "step": step, "phase": "repair", "source": "script",
                    "thought": action.get("thought", ""),
                    "tool_call": {"tool": tool, "args": args},
                    "observation": result["text"], "error": not result["ok"],
                })
                history.append(f"[{step}] {tool}({json.dumps(args, ensure_ascii=False)}) -> "
                               f"{result['text'][:300]}")
                if self.verbose:
                    print(f"[step {step}][script/repair] {tool} -> "
                          f"{'OK' if result['ok'] else 'ERR'}", flush=True)

        # ---- 4) 收尾：生成 patch + 跑最终验收 ----
        state["dirty"] = False
        diff = session.call("git_diff", {})
        trace["patch"] = (diff.get("data") or {}).get("patch", "")
        final = session.call("run_tests", {})
        trace["tests_passed"] = bool((final.get("data") or {}).get("passed"))
        trace["steps"].append({
            "step": step, "phase": "verify", "source": "script",
            "thought": "收尾：跑一次完整 pytest 并生成 patch",
            "tool_call": {"tool": "run_tests", "args": {}},
            "observation": final["text"], "error": False,
        })
        if not state["passed"] and trace["tests_passed"]:
            state["passed"] = True
        trace["final_answer"] = final_answer or (
            "已修复并让 pytest 全绿。" if trace["tests_passed"]
            else "未能让 pytest 全绿，详见 steps。")
        trace["duration_s"] = round(time.time() - started, 2)
        trace["stopped_by"] = "tests_passed" if trace["tests_passed"] else "step_budget"
        return trace

    @staticmethod
    def _update_state(state: dict, tool: str, args: dict, result: dict) -> None:
        if tool == "run_tests":
            state["runs"] += 1
            data = result.get("data") or {}
            state["passed"] = bool(data.get("passed"))
            state["fail_output"] = data.get("output", "") or state.get("fail_output", "")
            state["dirty"] = False
        elif tool == "read_file":
            path = str(args.get("path", ""))
            state["read"].add(path.replace("\\", "/"))
        elif tool in ("edit_file", "write_file"):
            blob = str(args.get("new", "")) + str(args.get("content", ""))
            if re.search(r"return\s+\w+\s*\+\s*\w+", blob):
                state["applied_fix"] = True     # 已落地的就是已知正确修复
            state["dirty"] = True
