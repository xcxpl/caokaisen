"""4 个工具的快速冒烟测试（开发时用；正式自检仍是 eval/run.py）。

用法：python eval/tools_smoke.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools import calculator, file_search, python_sandbox, wiki  # noqa: E402


def show(name, module, args):
    print(f"\n=== {name} ===")
    try:
        print(module.run(args))
    except Exception as e:  # noqa: BLE001
        print(f"[{type(e).__name__}] {e}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    show("calculator: (123+456)*789", calculator, {"expression": "(123 + 456) * 789"})
    show("calculator: sqrt(2026)", calculator, {"expression": "sqrt(2026)"})
    show("calculator: 除零（错误路径）", calculator, {"expression": "1/0"})
    show("python_sandbox: 质数求和", python_sandbox,
         {"code": "print(sum(n for n in range(2, 100) if all(n % d for d in range(2, int(n**0.5)+1))))"})
    show("python_sandbox: 最后一行表达式自动打印", python_sandbox,
         {"code": "def f(s):\n    return s == s[::-1]\nprint(f('level'), f('world'))"})
    show("python_sandbox: 死循环（熔断路径）", python_sandbox, {"code": "while True:\n    pass"})
    show("file_search: 目录下 *.md", file_search, {"pattern": "*.md", "dir": "data/agent-fixtures"})
    show("file_search: 内容 TODO", file_search, {"pattern": "TODO", "dir": "data/agent-fixtures"})
    show("file_search: 读文件", file_search, {"pattern": "data/agent-fixtures/README.md"})
    show("file_search: 越界（应被拒绝）", file_search, {"pattern": "*.md", "dir": "../../"})
    show("wiki: Alan Turing", wiki, {"query": "Alan Turing"})
    show("wiki: 图灵机", wiki, {"query": "图灵机"})


if __name__ == "__main__":
    main()
