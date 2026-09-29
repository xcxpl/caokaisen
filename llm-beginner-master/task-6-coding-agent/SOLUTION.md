# 任务六实现说明（SOLUTION.md）

## 一句话

用本地 Qwen2.5-0.5B-Instruct 搭了一个 mini Claude Code：手写 MCP server（7 个工具）
+ 50 行 Skill 加载器（3 个 Skill）+ 2 个 Subagent + 一条 agentic loop，在
`data/toy-repo` 上自主改代码并让 `python -m pytest` 全绿。

## 目录结构

```
task-6-coding-agent/
├── src/
│   ├── mcp_server.py          # 底层 Tools：手写 MCP server（零依赖 stdio JSON-RPC）
│   ├── skill_loader.py        # 中层 Skills：SkillLoader（约 50 行核心逻辑）
│   ├── skills/
│   │   ├── bug-fix/SKILL.md        # 定位并最小化修复 bug
│   │   ├── test-runner/SKILL.md    # 跑测试 + 读失败输出 + 定位嫌疑文件
│   │   └── code-review/SKILL.md    # 改动自查 + 写 PR 描述
│   ├── subagents/
│   │   ├── code_search.py     # 顶层 Subagent：仓库代码搜索
│   │   └── test_runner.py     # 顶层 Subagent：测试执行与失败诊断
│   ├── llm.py                 # 模型后端（OpenAI 兼容 / 本地 transformers 双级）
│   └── agent.py               # CodingAgent：三层栈的编排 + agentic loop
├── experiments/run_ablation.py
└── eval/                      # 自检结果 + trace 产物
```

## 三层能力栈各自解决什么

| 层 | 实现 | 解决什么 |
|---|---|---|
| **Tools / MCP** | `src/mcp_server.py` 暴露 `read_file` / `write_file` / `edit_file` / `list_dir` / `run_tests` / `git_diff` / `git_apply` | 把"动文件、跑进程"这类有副作用的动作收成无状态原子操作，跨 agent 复用；同时是安全边界所在 |
| **Skills** | `src/skill_loader.py` + 3 个 `SKILL.md` | 把"怎么修 bug / 怎么读测试输出"这类方法论按需注入，避免一开始就占满 context |
| **Subagents** | `src/subagents/` 两个独立 context 的子 agent | 把"翻仓库、跑测试、解 traceback"这种产出很啰嗦的工作隔离掉，主 agent 只看摘要 |

## Tools：MCP server 的两条通路

1. **自检通路**：模块顶层 `list_tools()` 返回 7 个工具的 schema（name / description /
   inputSchema），`eval/run.py` 直接 `import` 枚举。
2. **真协议通路**：`python src/mcp_server.py` 起一个 stdio JSON-RPC 2.0 服务，实现
   `initialize` / `tools/list` / `tools/call`，换行分隔报文（MCP stdio 传输）。

没有依赖官方 `mcp` SDK：一是本机没装、网络受限，二是"手写 MCP server"本来就是任务
要求。协议要点（`protocolVersion` / `capabilities.tools` / `content[].text` /
`isError`）都按规范实现，已用管道喂报文实测握手与调用。

### 安全约束（逐条对应 README 要求）

- **路径越界**：所有文件类工具先把参数 `Path.resolve()`，再用
  `candidate != repo and repo not in candidate.parents` 判定；`../..`、`C:\Windows`
  这类一律拒绝。实测 `read_file("../../etc/passwd")` 返回
  `错误：拒绝仓库外的路径`。
- **命令注入**：所有子进程都是 `subprocess.run([...], shell=False, cwd=repo)`；
  自定义测试命令先用 `shlex.split` 拆成参数列表，绝不拼进 shell 串。
- **不丢数据**：只提供 `git_apply`，不提供 `git reset --hard` / `git clean -fd` /
  `git checkout --`。
- **测试文件只读**：`write_file` / `edit_file` 命中 `test_*.py`、`*_test.py`、
  `tests/**` 时直接报错。这条不是为了"防坏人"，而是实验逼出来的——实测 0.5B 会无视
  system prompt 里"不要改测试文件"，转头去改 `assert` 语句。prompt 层的约束对弱模型
  不可靠，得由工具层兜住（详见 EXPERIMENT.md 观察 2/3）。
- **不让 server 崩**：`RepoTools.call()` 把 `ToolError` 和一切异常都转成
  `{"ok": False, "text": "错误：…"}`；弱模型多塞参数时按函数签名过滤（并回显忽略项），
  而不是直接抛 `TypeError`。

`git_diff` 还做了兜底：仓库有 git 历史时用 `git diff HEAD`，没有时用 `difflib` 把
`*.orig` 基准快照与当前文件对比生成 unified diff——所以新机器没配 git 身份也能拿到 patch。

## Skills：真的做了渐进式披露

```
list_skills()  → 只解析 YAML front-matter，拿 name + description（不读正文）
match(issue)   → 用 description/name 与 issue 做词重叠打分，选出命中的 skill
load(name)     → 命中之后才把 SKILL.md 正文读进 context
```

`agent.run()` 里先 `match(issue)`：本任务的 issue 命中 `['test-runner', 'bug-fix']`，
只有这两个的正文进 system prompt，`code-review` 的正文完全不进——这是"渐进式披露"
的直接体现，而不是把 3 份 SKILL.md 全塞进去。

SKILL.md 用 Anthropic Skills 的 front-matter 约定（`---` 包住的 `name` /
`description`），零依赖手写解析（不引入 pyyaml）。

## Subagents：隔离是怎么落地的

两个子 agent 都满足"独立 message 列表 + 独立步数上限 + 工具子集"：

| | code-search | test-runner |
|---|---|---|
| 工具子集 | `list_dir` / `read_file` | `run_tests` / `read_file` / `list_dir` |
| 步数上限 | 2 | 3 |
| 独立 context | 是（自己的 `messages`，不并回主 agent） | 是 |
| 返回给主 agent | `matches` + 一行 summary | `passed` + 3~5 行失败摘要 |

`TestRunnerSubagent` 会自己解析 traceback：从 `FAILED` 行拿失败用例、从
`test_x.py:NN: in fn` 拿位置、从 `E assert …` 拿断言差异，再从测试文件的
`from calculator import add` 反推出嫌疑源码文件。主 agent 收到的是
"pytest 失败；失败用例：test_add_positive_numbers…；断言：assert add(2, 3) == 5；
嫌疑源码：calculator.py"，而不是一整段 pytest 输出。

## Agent loop：怎么停机

```
① 匹配 Skill（命中才读正文）
② 派 code-search 子 agent 定位 → 派 test-runner 子 agent 跑基线测试
③ while step <= max_steps:
     优先请模型输出 {"thought","tool","args"} JSON
     解析失败 / 动作重复 / 过早收工 → 规则 Planner 接管
     执行工具（走 MCP 层）→ 记录 step → 更新状态
     测试已通过 → break
④ 兜底收敛：模型没修通就让规则 Planner 走完（最多 6 步）
⑤ 收尾：git_diff 生成 patch + 跑最终验收
```

**停机信号有两个，且都可观测**：

1. `run_tests` 真实返回 `passed=true` —— 硬信号，主信号；
2. 模型显式输出 `{"final": …}` —— 软信号，但**必须已经 `passed=true` 才认**；
   否则记为"拒绝收工"并交给规则 Planner。

所以既不会"步数到了硬停"，也不会听信模型"我觉得改好了"。

### 为什么要"模型优先 + 规则 Planner 兜底"

本机 4GB 显存只能跑 0.5B，它的规划能力很不可靠：实测会输出重复动作（照抄 few-shot
里最后一条 `run_tests`）、偶尔吐不出合法 JSON、也会在测试没过时宣布完成。纯靠 prompt
调教 0.5B 的期望收益很低，所以 loop 里放了一个**确定性状态机**做兜底：

```
未跑测试 → 跑测试 → 从测试输出定位嫌疑文件 → 读源码
         → 应用最小修复（必要时先从 .orig 恢复） → 再跑测试 → 通过
```

每个 step 都标注 `source: model | script`，trace 里一眼能看出哪一步是模型做的。
实测（`eval/trace_toy_repo.json`）：基线→读文件是规则 Planner，**改代码那一步是模型
给出的**，最后验证又回到规则。也就是说模型在它擅长的那一步真的参与了，而不是全程
被规则架着走。

`trace.steps` 每步都含 `thought` / `tool_call{tool,args}` / `observation` / `error` /
`source`，另有 `patch`、`tests_passed`、`subagents`、`skills`、`model_outputs`、
`backend`、`notes`、`stopped_by`、`duration_s`。

## 关于模型：为什么不是 Qwen2.5-Coder-7B

README 要求本地 Qwen2.5-Coder-7B-Instruct。本机 4GB 显存 + 网络受限，7B 装不下也拉
不动，因此：

- **后端留了标准接入点**：`src/llm.py` 的 `OpenAICompatBackend` 直接读
  `OPENAI_BASE_URL` / `AGENT_MODEL`，只要本地起一个 Ollama/vLLM 并
  `ollama pull qwen2.5-coder:7b`，无需改代码就能切到 7B；
- **实际跑 0.5B**：`LocalHFBackend` 从 `task-4-rag/models/Qwen2.5-0.5B-Instruct`
  加载（task-5 下的 1.5B 只有 `.incomplete` 残缺权重，`_has_weights()` 会主动跳过）；
- **降级链**：CUDA → OOM 自动退回 CPU；模型完全起不来时 loop 仍能靠规则 Planner
  跑通（`AGENT_BACKEND=none` 可强制纯规则模式）。

## 复现

```bash
cd task-6-coding-agent
python data/download.py                 # 生成 toy-repo（幂等）
python eval/run.py                      # 自检，结果写 eval/result.json
python experiments/run_ablation.py      # 消融实验，结果写 experiments/ablation_results.json
```

常用环境变量：`AGENT_BACKEND`（auto/openai/local/none）、`AGENT_MODEL`、
`OPENAI_BASE_URL`、`AGENT_LOCAL_MODEL`、`AGENT_MAX_STEPS`、`AGENT_NO_MODEL=1`。
