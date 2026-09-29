# 任务五实现说明

手写 ReAct 循环（不使用任何 agent 框架），让本地小模型自主调度
calculator / python_sandbox / file_search / wiki 四类工具，在自建 10 题任务集上
答案关键词命中率 **0.7（7/10）**，超过 0.6 的通过线。

## 文件结构

```text
task-5-tool-agent/
├── data/
│   ├── download.py            # 生成 10 题任务集 + 本地检索夹具
│   ├── tasks.json             # 10 题任务集（含 expected_tools / expected_answer_contains）
│   ├── agent-fixtures/        # 本地文件检索夹具（README.md、todo_note.md）
│   └── cache/wiki/            # 维基百科查询缓存（断网时的兜底数据）
├── src/
│   ├── tools/
│   │   ├── calculator.py      # 四则运算 + 高级函数（限白名单，不 eval 任意代码）
│   │   ├── python_sandbox.py  # 受限 exec：import 白名单 + builtins 白名单 + 超时/行数熔断
│   │   ├── file_search.py     # 文件名 / 内容检索 + 文件读取（路径越界校验）
│   │   └── wiki.py            # 维基百科 API（直连/代理自动探测 + 本地缓存 + 模糊兜底）
│   ├── llm.py                 # 后端：OpenAI 兼容 API / 本地 transformers
│   └── agent.py               # class ReActAgent：手写 ReAct 循环
├── eval/
│   ├── run.py                 # 自检入口
│   ├── result.json            # 自检结果（含每题 trace 预览）
│   └── traces/                # 完整 trace 落盘（含 -inject-error 版本）
├── experiments/
│   └── run_ablation.py        # 模型尺寸（S2）/ few-shot 条数（S3）消融
├── SOLUTION.md                # 本文件
└── EXPERIMENT.md              # 实验观察
```

## 必做项完成情况

- [x] **M1** 4 个工具各自导出 `TOOL_SCHEMA` 与 `run(args)`，自检 `tools_individual` 通过。
- [x] **M2** `src/agent.py` 手写 ReAct 循环：Thought / Action / Action Input / Observation，
      工具路由、步数上限（默认 6）、Final Answer 终止、上下文字符预算管理。
- [x] **M3** 工具异常（含 python_sandbox 内部失败）统一捕获并回填成 Observation，
      让模型自我纠错；同一失败调用重复出现时给出更强的纠错信号，单次失败不会 crash 循环。
- [x] **M4** 自检 `multi_tool_success_rate`：**rate = 0.7**（> 0.6）。
- [x] **M4** trace 完整记录每步 raw / thought / tool / action_input / observation / error，
      落盘到 `eval/traces/task-N.json`。

## 面向小模型的鲁棒性设计

本机是 4GB 显存（RTX 3050 Laptop），跑的是 0.5B 小模型（不是 README 建议的 7B），
它的指令遵循能力有限、同题多次运行结果波动很大。除了标准的解析容错外，额外做了这些
**不依赖模型"听话"的工程兜底**：

| 兜底 | 解决的问题 |
|---|---|
| 解析层：正则 + JSON / 字面量 / 裸串多口径，工具名与参数名别名映射，参数嵌套解包 | 小模型输出格式五花八门 |
| `FINAL_RE` 只取同一行 | 模型写完 Final Answer 又续写 Action，旧写法会把后续整段吞进答案 |
| 任务后面的「工具路由提示」（`_task_hint`） | 0.5B 选错工具概率高：算式题去用 sandbox、年份题不补算 |
| 裸文件名补全为任务里的完整路径 | 模型把 `data/agent-fixtures/README.md` 缩成 `README.md`，读到同名的错文件 |
| 重复调用检测 + 定向提示 | 模型反复提交同一个失败调用，把 6 步耗光 |
| 答案实体覆盖检查（题目引号里的对象必须都答到） | 题目点名 `level` 和 `world`，模型只答一个 |
| python_sandbox 单题内保持变量环境 | 模型习惯「这步定义函数、下一步再调用」，否则第二步必 NameError |
| python_sandbox 自动解 `{"code": "..."}` 包装 | 模型把参数又包了一层，直接语法错误 |
| `{"expression": ...}` 算式题的 task_hint 直接给出算式 | 模型把 `(123 + 456) * 789` 改写成 `pow(123, 456) * 789` |
| file_search 名字匹配同时试 glob / 子串 / 正则 | 模型把 `*.md` 写成 `.*\.md$` |
| wiki 结果里「关键句（含年份）」提到最前面 | 长简介把年份冲散，模型漏掉 1947 / 2017 |
| wiki 断网时按**词条标题**做关键词模糊匹配缓存兜底 | 代理不稳定时 wiki 全挂；缓存里还有 query 与内容对不上的污染条目，必须按标题匹配 |
| wiki 探测失败后的冷却时间 | 断网时每题每次 wiki 调用都重探 8 个出口 × 2 轮，单题曾跑到 346 秒 |

`eval/run.py` 的判定口径没有改动，以上全部是 `src/` 侧的实现。

## 复现步骤

```powershell
cd task-5-tool-agent

# 1. 生成 10 题任务集与检索夹具
python data/download.py

# 2. 自检（用系统 Python 3.12：torch 2.5.1+cu121 / transformers 5.16.1）
python eval/run.py

# 3. 单题调试（打印每一步的模型原始输出）
python -c "from src.agent import ReActAgent; import json; \
print(json.dumps(ReActAgent(verbose=True).run('计算 (123 + 456) * 789 的结果，并告诉我结果的位数。'), \
ensure_ascii=False, indent=2))"
```

模型选择（`src/llm.py`）：

- 有 OpenAI 兼容端点时优先用远端：`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `AGENT_MODEL`，
  默认 `qwen2.5:7b-instruct`（Ollama）/ `Qwen/Qwen2.5-7B-Instruct`（vLLM）。
- 否则回退本地 transformers：`AGENT_LOCAL_MODEL` 指定权重目录，或在
  `LOCAL_MODEL_CANDIDATES` 里放好路径（当前指向任务四下载的
  `task-4-rag/models/Qwen2.5-0.5B-Instruct`）。

```powershell
# 换 7B（推荐，能显著减少下面这些"小模型特有"的失败）
ollama pull qwen2.5:7b-instruct
ollama serve
$env:OPENAI_BASE_URL = "http://localhost:11434/v1"
$env:OPENAI_API_KEY  = "ollama"
$env:AGENT_MODEL     = "qwen2.5:7b-instruct"
python eval/run.py
```

## 已知限制

- **模型尺寸**：跑的是 0.5B，README 建议 7B。三个失败题（id2 / id6 / id8）都是
  "小模型把参数包两层导致语法错误后陷入重复调用"，换成 7B 大概率消失。
- **S4 错误恢复未通过**：`error_recovery` 本轮 0/7（阈值 > 0.4）。注入一次性故障后，
  0.5B 倾向于直接放弃（"由于 python_sandbox 无法运行，所以无法计算"），
  即使 Observation 后面明确附了"这是只注入一次的一次性故障，请原样重试"也不照做。
  详见 `EXPERIMENT.md`。
- **wiki 依赖网络**：本机代理不稳定，断网时靠 `data/cache/wiki/` 的本地缓存兜底。
  缓存里存在历史遗留的污染条目（query 与正文对不上），所以匹配按词条标题做而不是按 query。
- **S1 / S2 / S3 未完成**：Qwen-Agent 对照版、模型尺寸对比、few-shot 条数消融都只写了脚本
  （`experiments/run_ablation.py`），没有跑出结果——1.5B 权重下载中断，只有 0.5B 可用。
