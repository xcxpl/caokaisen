"""任务六 · 消融实验：单 agent / +Skill / +Subagent 的对比（对应 S2、S3）。

在 toy-repo 上跑四种配置，每种都先把 `calculator.py` 从 `calculator.py.orig` 基准快照
恢复成 buggy 版本，再让 agent 自主修复，比较：

* tests_passed —— 最终 pytest 是否全绿（toy repo 上这是硬指标）；
* steps —— trace 里的步数（越少越省）；
* model_steps / script_steps —— 动作分别来自模型还是规则 Planner；
* subagent_steps、prompt_chars —— 三层栈的"省 context"效果；
* duration_s。

用法：
    python experiments/run_ablation.py
结果写到 `experiments/ablation_results.json`。

说明：模型用本地 Qwen2.5-0.5B-Instruct（4GB 显存跑不动 7B），temperature=0 时解码
是确定的，因此每个条件只跑 1 次（多次重复会得到完全相同的结果）。S1（Q4_K_M vs FP16）
本机没有 GGUF / Ollama 环境，未做，见 EXPERIMENT.md。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import CodingAgent          # noqa: E402
from src.llm import get_backend            # noqa: E402

TOY_REPO = ROOT / "data" / "toy-repo"

CONDITIONS = [
    # —— 带规则 Planner 兜底（工程上"能跑通"的配置）——
    ("single_no_skill",  "单 agent（无 Skill / 无 Subagent）", dict(use_skills=False, use_subagents=False)),
    ("skill_only",       "+Skill（无 Subagent）",              dict(use_skills=True,  use_subagents=False)),
    ("subagent_only",    "+Subagent（无 Skill）",              dict(use_skills=False, use_subagents=True)),
    ("full_stack",       "完整三层栈（Skill + Subagent）",      dict(use_skills=True,  use_subagents=True)),
    # —— 纯模型、关掉规则兜底：直接量 0.5B 自己的规划能力 ——
    ("model_only_bare",  "纯模型（无 Skill / 无 Subagent / 无兜底）",
     dict(use_skills=False, use_subagents=False, use_planner=False)),
    ("model_only_skill", "纯模型 + Skill（无 Subagent / 无兜底）",
     dict(use_skills=True, use_subagents=False, use_planner=False)),
]


def run_condition(name: str, label: str, kwargs: dict, backend) -> dict:
    # 每轮从基准快照恢复 buggy 版本，避免上一轮修好了这轮空跑
    shutil.copy(TOY_REPO / "calculator.py.orig", TOY_REPO / "calculator.py")
    issue = (TOY_REPO / "ISSUE.md").read_text(encoding="utf-8")

    agent = CodingAgent(backend=backend, verbose=False, max_steps=3, **kwargs)
    started = time.time()
    try:
        trace = agent.run(str(TOY_REPO), issue)
        error = None
    except Exception as e:  # noqa: BLE001
        trace, error = {}, f"{type(e).__name__}: {e}"

    steps = trace.get("steps", [])
    act_steps = [s for s in steps if s.get("phase") == "act"]
    return {
        "condition": name,
        "label": label,
        "config": {k: v for k, v in kwargs.items()},
        "tests_passed": bool(trace.get("tests_passed")),
        "steps": len(steps),
        "model_steps": sum(1 for s in act_steps if s.get("source") == "model"),
        "script_steps": sum(1 for s in act_steps if s.get("source") == "script"),
        "subagent_calls": len(trace.get("subagents", [])),
        "subagent_steps": sum(s.get("steps", 0) for s in trace.get("subagents", [])),
        "skills_loaded": [s.get("name") for s in trace.get("skills", [])],
        "patch_chars": len(trace.get("patch", "") or ""),
        "duration_s": round(time.time() - started, 1),
        "stopped_by": trace.get("stopped_by"),
        "error": error,
    }


def main() -> None:
    print("[ablation] 加载模型后端 …", flush=True)
    backend = get_backend(verbose=True)
    print(f"[ablation] backend = {backend.describe()}", flush=True)

    results = []
    for name, label, kwargs in CONDITIONS:
        print(f"[ablation] 运行 {name} …", flush=True)
        row = run_condition(name, label, kwargs, backend)
        results.append(row)
        print(f"[ablation]   -> tests_passed={row['tests_passed']} steps={row['steps']} "
              f"model={row['model_steps']} script={row['script_steps']} "
              f"{row['duration_s']}s", flush=True)

    # 额外落一份完整 trace，方便提交时附上
    shutil.copy(TOY_REPO / "calculator.py.orig", TOY_REPO / "calculator.py")
    issue = (TOY_REPO / "ISSUE.md").read_text(encoding="utf-8")
    trace = CodingAgent(backend=backend, max_steps=3).run(str(TOY_REPO), issue)
    trace_path = ROOT / "eval" / "trace_toy_repo.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    out = {
        "backend": backend.describe(),
        "note": "toy-repo 上四种配置各跑 1 次（temperature=0，确定性解码）",
        "conditions": results,
        "trace_file": str(trace_path.relative_to(ROOT)),
    }
    out_path = ROOT / "experiments" / "ablation_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ablation] 结果写入 {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
