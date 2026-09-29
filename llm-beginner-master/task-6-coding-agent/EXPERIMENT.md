# 任务六实验观察（EXPERIMENT.md）

## 实验设置

- 模型：本地 `Qwen2.5-0.5B-Instruct`（4GB 显存能跑的最大档；7B 装不下，见 SOLUTION.md）
- 任务：`data/toy-repo`（`calculator.add` 写成 `a - b`，要改回 `a + b` 并让 3 个 pytest 全绿）
- 每组实验前都从 `calculator.py.orig` 基准快照恢复 buggy 版本，避免上一轮修好了这轮空跑
- `temperature=0`，确定性解码，所以每个条件跑 1 次即可复现
- 完整数据：`experiments/ablation_results.json`；完整 trace：`eval/trace_toy_repo.json`

```bash
python experiments/run_ablation.py
```

## 结果

| 配置 | 通过 | 步数 | 模型步 | 规则步 | 派 subagent | 加载 Skill |
|---|---|---|---|---|---|---|
| 单 agent（无 Skill / 无 Subagent） | ✅ | 5 | 1 | 2 | 0 | — |
| +Skill | ✅ | 7 | 2 | 1 | 0 | test-runner, bug-fix |
| +Subagent | ✅ | 6 | 1 | 2 | 2 | — |
| 完整三层栈 | ✅ | 5 | 1 | 2 | 2 | test-runner, bug-fix |
| **纯模型（关掉规则兜底）** | ✅ | 5 | 3 | 0 | 0 | — |
| **纯模型 + Skill（关掉兜底）** | ❌ | 4 | 2 | 0 | 0 | test-runner, bug-fix |

## 观察（约 480 字）

**1. 0.5B 其实能修对这个 bug，但只在对的上下文长度下。**
关掉规则兜底的"纯模型"组，模型自己两步就走对了：先 `run_tests` 摸现状，再
`edit_file(calculator.py, "return a - b" → "return a + b")`，测试直接转绿。这个
结论比我预期的乐观——toy-repo 这类"单点运算符写错"的任务，0.5B 靠 prompt 就能解决。

**2. 但一加载 Skill 正文，模型反而跑偏了。**
同样是纯模型条件，把命中 Skill 的正文（`test-runner` + `bug-fix`，约 1.5k 字）塞进
system prompt 后，模型第一步照旧 `run_tests`，第二步却去改 **测试文件**
（`edit_file(test_add_positive_numbers.py, "assert -1 == 5" → ...)`）——这个文件名
本身还是它编出来的。原因大概是：Skill 正文里"失败用例 / 断言 / E assert"这些词密度
很高，0.5B 的注意力被拉到了"测试"上，而系统提示里那句"不要改测试文件"它根本没守住。
**弱模型上，更长的"方法论上下文"未必是加分项，可能是干扰项。**

**3. 于是我把约束从 prompt 挪到了工具层。**
针对上面的现象，`write_file` / `edit_file` 现在对 `test_*.py`、`tests/**` 一律返回
`错误：拒绝修改测试文件…`。这条硬约束下，重跑消融：四个带规则兜底的配置全部通过；
"纯模型 + Skill"仍然失败——模型挨了错误后没有改回正确目标，而是在测试文件上又试了
一次，把 3 步预算耗尽。**结论：prompt 层的"不要做 X"对弱模型基本无效，必须由工具层
兜住；但工具层只能拦住危险动作，救不回"选错了目标"的规划。**

**4. 规则 Planner 的价值是"把成功率从不确定变成确定"。**
带兜底的四个配置成功率 4/4，且 trace 显示模型在每一步都真实参与了（通常贡献"改代码"
那一击），规则 Planner 只补它掉链子的环节（重复动作、解析失败、过早收工）。
代价是多 1-2 步——在"必须修通"的场景里这笔交易很划算。

## 未做的部分（如实说明）

- **S1（Q4_K_M vs FP16）**：本机没有 Ollama / llama.cpp / GGUF 环境，也没有
  Qwen2.5-Coder 的量化权重，未做。
- **S4（SWE-bench Lite）**：需要 `--with-swebench` 下载元数据 + 自行 clone 仓库，
  网络当前不通，未做（自检里该项为"跳过"）。
