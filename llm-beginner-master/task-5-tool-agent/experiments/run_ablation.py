"""消融实验：模型尺寸（S2）与 few-shot 条数（S3）。

用法示例：
    # 全部 10 题，比较两个模型 + zero-shot / 默认 few-shot
    python experiments/run_ablation.py

    # 只跑其中几题，快速试
    python experiments/run_ablation.py --tasks 2,6,8 --models 0.5b --few-shot 3,0

    # 4GB 显存跑 1.5B 时限制显存占用（超出部分自动放到 CPU）
    set AGENT_MAX_GPU_MEM=3000MiB
    python experiments/run_ablation.py

结果写入 experiments/ablation_results.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from src.agent import ReActAgent, few_shot_subset   # noqa: E402
from src import llm as llm_module                   # noqa: E402

import run as eval_run                              # noqa: E402  eval/run.py

# 候选模型：key -> 权重目录（相对仓库根）
MODEL_CHOICES = {
    "0.5b": "task-4-rag/models/Qwen2.5-0.5B-Instruct",
    "1.5b": "task-5-tool-agent/models/Qwen2.5-1.5B-Instruct",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="0.5b,1.5b",
                        help="逗号分隔，取值见 MODEL_CHOICES，或直接给本地目录")
    parser.add_argument("--few-shot", default="all,0",
                        help="逗号分隔的示例组数，all 表示全部示例，0 表示 zero-shot")
    parser.add_argument("--tasks", default="all", help="逗号分隔的题目 id，all 表示全部")
    parser.add_argument("--max-steps", type=int, default=6)
    return parser.parse_args()


def load_tasks(spec: str):
    tasks = json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))
    if spec.strip().lower() == "all":
        return tasks
    wanted = {int(x) for x in spec.split(",") if x.strip()}
    return [t for t in tasks if t["id"] in wanted]


def resolve_model(name: str) -> Path | None:
    rel = MODEL_CHOICES.get(name.strip().lower(), name.strip())
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT.parent / rel if rel.startswith("task-") else ROOT / rel
    return path if (path / "config.json").exists() else None


def parse_few_shot(spec: str):
    values = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        values.append((item, None if item == "all" else int(item)))
    return values


def main():
    args = parse_args()
    tasks = load_tasks(args.tasks)
    results = []

    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        model_path = resolve_model(model_name)
        if model_path is None:
            print(f"[跳过] 找不到模型 {model_name} 的本地权重（见 MODEL_CHOICES）")
            continue
        try:
            backend = llm_module.LocalHFBackend(model_path)
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] 加载 {model_name} 失败：{e}")
            continue
        print(f"\n=== 模型 {model_name}（device={backend.device}）===", flush=True)

        for label, n_shot in parse_few_shot(args.few_shot):
            agent = ReActAgent(backend=backend, max_steps=args.max_steps,
                               few_shot=few_shot_subset(n_shot))
            passed, per_task, seconds, steps = 0, [], 0.0, 0
            for task in tasks:
                trace = agent.run(task["task"])
                ok = eval_run.answer_matches(trace["final_answer"],
                                             task["expected_answer_contains"])
                passed += int(ok)
                seconds += trace["elapsed_sec"]
                steps += trace["n_steps"]
                per_task.append({
                    "id": task["id"],
                    "success": ok,
                    "n_steps": trace["n_steps"],
                    "used_tools": trace["used_tools"],
                    "final_answer": trace["final_answer"][:200],
                })
                print(f"  [{label:>3} shot] 题 {task['id']:>2} "
                      f"{'通过' if ok else '失败'}  {trace['n_steps']} 步 "
                      f"{trace['elapsed_sec']:>5.1f}s", flush=True)
            rate = passed / len(tasks) if tasks else 0.0
            results.append({
                "model": model_name,
                "model_path": str(model_path),
                "few_shot": label,
                "few_shot_groups": None if n_shot is None else n_shot,
                "n": len(tasks),
                "passed": passed,
                "rate": round(rate, 3),
                "avg_steps": round(steps / len(tasks), 2) if tasks else None,
                "total_sec": round(seconds, 1),
                "details": per_task,
            })
            print(f"  -> {label} shot：{passed}/{len(tasks)} = {rate:.2f}", flush=True)

    out = ROOT / "experiments" / "ablation_results.json"
    merged: list[dict] = []
    if out.exists():
        try:
            merged = json.loads(out.read_text(encoding="utf-8")).get("results", [])
        except ValueError:
            merged = []
    # 同一 (模型, few-shot, 题集) 的旧结果被本次覆盖，其余保留，方便分多次跑
    fresh_keys = {(r["model"], r["few_shot"], r["n"]) for r in results}
    merged = [r for r in merged if (r["model"], r["few_shot"], r["n"]) not in fresh_keys]
    merged.extend(results)
    out.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "results": merged}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n写入 {out.relative_to(ROOT)}")
    print("\n| 模型 | few-shot | 通过/总数 | 成功率 | 平均步数 | 总耗时(s) |")
    print("|---|---|---|---|---|---|")
    for r in merged:
        print(f"| {r['model']} | {r['few_shot']} | {r['passed']}/{r['n']} | "
              f"{r['rate']:.2f} | {r['avg_steps']} | {r['total_sec']} |")


if __name__ == "__main__":
    main()
