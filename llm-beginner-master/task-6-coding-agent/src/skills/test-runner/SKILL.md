---
name: test-runner
description: 跑测试、读懂失败输出、定位可疑源码文件。当任务要求"让 python -m pytest 通过"、需要确认当前是否全绿、或测试报错要判断是哪个文件哪一行出问题时加载。
---

# Skill: 测试运行与失败诊断

## 何时用

- issue 里出现 "pytest"、"测试通过"、"全绿"、"测试失败" 等字样；
- 改完代码需要确认没有引入回归；
- 拿到一段 pytest 输出，想知道"到底哪个函数坏了"。

## Workflow

1. **先跑一次基线**：`run_tests`（默认 `python -m pytest -q`）。不要凭猜测下结论，
   永远以真实输出为准。
2. **读失败摘要**，按下面顺序抽取信息：
   - `FAILED test_x.py::test_name` → 哪个测试用例挂了；
   - `test_x.py:NN: in test_name` → 失败发生在测试文件的哪一行；
   - `assert <左> == <右>` → 期望值 vs 实际值（左为期望，右为实际）；
   - `ImportError` / `AttributeError` → 往往是接口签名或名字被写错。
3. **从测试文件反推被测源码**：看测试文件的 `from xxx import ...`，被导入的模块就是
   嫌疑源码文件（例如 `from calculator import add` → 看 `calculator.py`）。
4. **只改源码、不改测试**：测试是需求契约。改测试来"让它通过"等于骗自己。
5. **收尾再跑一次**：`run_tests` 必须真正看到 `passed=true` 才算完成，不要因为
   上一次输出里没有红色就假定通过。

## 输出建议

给主 agent 的摘要保持 3~5 行：失败用例名 / 断言差异 / 嫌疑文件 / 建议动作。
不要把整段 pytest 输出原样丢回去。
