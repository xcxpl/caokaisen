"""任务五自检：工具单元测试 + 多工具任务成功率 + 错误恢复。"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # from src.* —— 学生实现
sys.path.insert(0, str(ROOT.parent))    # from _eval_harness —— 共用运行壳

from _eval_harness import run_tests

SUCCESS_RATE_PASS = 0.6   # 多工具任务成功率通过线
ERROR_RECOVERY_PASS = 0.4  # 错误恢复通过线
TRACE_DIR = ROOT / "eval" / "traces"


def normalize_answer(text):
    text = str(text).lower()
    text = text.replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", text)


def answer_matches(answer, expected_keywords):
    norm_answer = normalize_answer(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(normalize_answer(keyword) in norm_answer
                       for keyword in expected):
                return False
        elif normalize_answer(expected) not in norm_answer:
            return False
    return True


def extract_used_tools(trace):
    used = []
    for step in trace.get("steps", []) if isinstance(trace, dict) else []:
        if not isinstance(step, dict):
            continue
        name = step.get("tool") or step.get("tool_name") or step.get("action")
        if isinstance(name, str) and name:
            used.append(name)
    return used


def test_tools_individual():
    """每个工具单元测试。"""
    results = {}
    try:
        from src.tools import calculator, python_sandbox, file_search, wiki
    except ImportError as e:
        return {"test": "tools_individual", "pass": False,
                "error": f"工具导入失败：{e}"}

    # (工具名, 模块, 调用参数, 预期子串, 是否依赖网络)：
    # expected=None 表示只检查响应长度 > 50（如 wiki 网络回包）；
    # network=True 的工具若抛异常按“跳过”处理（多半是离线/被墙），不拖累其余工具判定。
    checks = [
        ("calculator",     calculator,     {"expression": "2 + 3 * 4"},                  "14",        False),
        ("python_sandbox", python_sandbox, {"code": "print(sum(range(10)))"},            "45",        False),
        ("file_search",    file_search,    {"pattern": "README.md", "dir": str(ROOT)},   "README.md", False),
        ("wiki",           wiki,           {"query": "Alan Turing"},                     None,        True),
    ]
    network_skipped = []
    for name, mod, args, expected, network in checks:
        try:
            out = str(mod.run(args))
            results[name] = (expected in out) if expected is not None else (len(out) > 50)
        except Exception as e:
            if network:
                results[name] = f"skip(网络不可用？): {e}"
                network_skipped.append(name)
            else:
                results[name] = f"error: {e}"

    gated = [v for k, v in results.items() if k not in network_skipped]
    all_pass = bool(gated) and all(v is True for v in gated)
    return {"test": "tools_individual", "pass": all_pass, "results": results,
            "network_skipped": network_skipped or None}


def test_multi_tool_success_rate():
    from src.agent import ReActAgent
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        return {"test": "multi_tool_success_rate", "pass": None,
                "skip": "data/tasks.json 不存在；跑 data/download.py 生成"}
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not tasks:
        return {"test": "multi_tool_success_rate", "pass": None,
                "skip": "data/tasks.json 为空；重跑 data/download.py 生成"}
    agent = ReActAgent()
    success = 0
    details = []
    for t in tasks:
        try:
            trace = agent.run(t["task"])
            dump_trace(t, trace)
            final_answer = trace.get("final_answer", "") if isinstance(trace, dict) else ""
            expected = t.get("expected_answer_contains", [])
            ok = answer_matches(final_answer, expected)
            success += int(ok)
            used_tools = extract_used_tools(trace)
            expected_tools = t.get("expected_tools", [])
            details.append({
                "id": t["id"],
                "success": ok,
                "final_answer_preview": str(final_answer)[:120],
                "expected_answer_contains": expected,
                "expected_tools": expected_tools,
                "used_tools": used_tools,
                "used_expected_tools": all(
                    any(expected_tool in used for used in used_tools)
                    for expected_tool in expected_tools
                ) if used_tools else None,
                "n_steps": trace.get("n_steps") if isinstance(trace, dict) else None,
                "terminated_by": trace.get("terminated_by") if isinstance(trace, dict) else None,
                "elapsed_sec": trace.get("elapsed_sec") if isinstance(trace, dict) else None,
                "trace_preview": trace_preview(trace),
            })
        except Exception as e:
            details.append({"id": t["id"], "success": False, "error": str(e)})
    rate = success / len(tasks)
    return {"test": "multi_tool_success_rate", "pass": rate > SUCCESS_RATE_PASS,
            "rate": round(rate, 3), "n": len(tasks),
            "backend": getattr(agent.backend, "describe", lambda: "?")(),
            "traces": str(TRACE_DIR.relative_to(ROOT)), "details": details}


def test_error_recovery():
    """S4：每道题注入 1 次工具故障，看 agent 能否自我纠正后仍答对（> 40%）。

    只取不依赖 wiki 的题目：本机/CI 可能无法访问维基百科，
    那类失败属于网络问题而不是错误恢复能力。
    设 SKIP_ERROR_RECOVERY=1 可跳过这一项。
    """
    if os.environ.get("SKIP_ERROR_RECOVERY"):
        return {"test": "error_recovery", "pass": None, "skip": "SKIP_ERROR_RECOVERY=1"}

    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        return {"test": "error_recovery", "pass": None,
                "skip": "data/tasks.json 不存在；跑 data/download.py 生成"}
    try:
        from src.agent import ReActAgent
    except ImportError as e:
        return {"test": "error_recovery", "pass": None, "skip": f"未实现 src/agent.py：{e}"}

    tasks = [t for t in json.loads(tasks_path.read_text(encoding="utf-8"))
             if "wiki" not in t.get("expected_tools", [])]
    agent = ReActAgent(inject_error={"occurrences": 1})
    success = 0
    details = []
    for t in tasks:
        try:
            trace = agent.run(t["task"])
            dump_trace(t, trace, suffix="-inject-error")
            final_answer = trace.get("final_answer", "")
            ok = answer_matches(final_answer, t.get("expected_answer_contains", []))
            success += int(ok)
            details.append({
                "id": t["id"],
                "success": ok,
                "injected": any(s.get("error") for s in trace.get("steps", [])),
                "recovered": ok,
                "n_steps": trace.get("n_steps"),
                "final_answer_preview": str(final_answer)[:120],
            })
        except Exception as e:
            details.append({"id": t["id"], "success": False, "error": str(e)})
    rate = success / len(tasks) if tasks else 0.0
    return {"test": "error_recovery", "pass": rate > ERROR_RECOVERY_PASS,
            "rate": round(rate, 3), "n": len(tasks),
            "note": "每题第 1 次工具调用被 inject_error 钩子打断，agent 须自我纠错后答对",
            "details": details}


def trace_preview(trace, limit=4):
    """trace 摘要：每一步的动作与观察前 100 字，避免 result.json 过大。"""
    if not isinstance(trace, dict):
        return []
    preview = []
    for step in trace.get("steps", [])[:limit]:
        preview.append({
            "step": step.get("step"),
            "thought": (step.get("thought") or "")[:100],
            "action": step.get("action"),
            "action_input": (step.get("action_input") or "")[:160],
            "observation": str(step.get("observation") or "")[:160],
            "error": bool(step.get("error")),
        })
    return preview


def dump_trace(task, trace, suffix=""):
    """完整 trace 落盘到 eval/traces/，方便写报告时直接贴 Thought/Action/Observation。"""
    if not isinstance(trace, dict):
        return
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"task-{task['id']}{suffix}.json"
        path.write_text(json.dumps({"task": task, "trace": trace},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    run_tests([test_tools_individual, test_multi_tool_success_rate,
               test_error_recovery], ROOT)
