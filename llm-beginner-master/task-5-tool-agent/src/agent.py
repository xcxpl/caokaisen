"""手写 ReAct 循环（约 200 行，不依赖任何 agent 框架）。

流程：Thought / Action / Action Input -> 系统执行工具 -> Observation -> … -> Final Answer

小模型（尤其是本地 0.5B~7B 量化版）经常不严格守格式，所以这里做了三层鲁棒性：
  1. 解析层：正则 + JSON/字面量/裸串多种兜底，工具名与参数名都有别名映射；
  2. 纠错层：解析失败或工具报错都把提示塞回 Observation，让模型自己修；
  3. 兜底层：步数用尽时再问一次「只根据 Observation 给结论」，避免整题空手而归。

自检接口（eval/run.py 依赖）：
    class ReActAgent: run(task: str) -> dict   # 含 steps / final_answer / success
    inject_error 钩子：run 时按配置注入 1 次工具故障，用于 error_recovery 实验
"""
from __future__ import annotations

import ast
import json
import re
import time

from . import llm as llm_module
from .tools import TOOLS, describe_tools, normalize_name, reset_sessions, run_tool

AgentTrace = dict

DEFAULT_MAX_STEPS = 6
OBS_HEAD_CHARS = 1400
OBS_TAIL_CHARS = 500
# 上下文预算（字符数）：ReAct 的历史会越滚越长，超过预算就丢最早的那几步 Observation，
# 否则 4GB 显存的小机器很容易 OOM，也更容易把模型绕晕。
PROMPT_BUDGET_CHARS = 7000
STOP_STRINGS = ["\nObservation:", "Observation:", "\nUser:", "\nuser:", "\n\n## "]

SYSTEM_TEMPLATE = """你是一个会用工具的 ReAct 智能体，靠「思考 → 行动 → 观察」的循环解决用户问题。

## 你的回复只能是下面两种格式之一

格式 A（需要调用工具时）：
Thought: <一句话说明你打算做什么>
Action: <工具名>
Action Input: <一行合法 JSON>

格式 B（信息足够，给出结论时）：
Thought: <一句话说明你已经拿到什么信息>
Final Answer: <最终答案>

## 必须遵守的规则
1. Action 只能是这些工具之一：{tool_names}
2. Action Input 必须是**一行**合法 JSON 对象，键名要与工具参数一致。
3. 一次只调一个工具；写完 Action 就停下，**不要自己编造 Observation**，系统会返回。
4. 只有 Observation 里出现过的数据才能写进 Final Answer；严禁凭记忆编造数字、年份、人名。
   反过来也一样：Observation 里已经给出了相关的年份/数字/人名，就必须从里面取出来算或写，
   **不许回答说“查不到”“无法获取”**。
5. Observation 以「错误：」开头说明工具调用失败：读懂原因后修正参数或换工具重试，同一错误不要犯两次。
6. 用户问题里出现的目录或文件路径（例如 data/agent-fixtures）必须原样填进 dir 参数；
   要读取某个具体文件的内容时，把它的完整路径直接作为 file_search 的 pattern。
7. python_sandbox 的代码必须用 print() 输出结果；只定义函数不调用的话什么都得不到。
8. Final Answer 写一句完整的中文，把关键结果连同单位/量词一起写进去（例如「共 6 位数字」
   「出生于 1947 年」「文件是 todo_note.md」）。
9. 如果题目要求测试 / 对比多个对象，Final Answer 里要把每个对象的结果都写出来
   （例如「level 是回文（True），world 不是回文（False）」）。
10. 写 Final Answer 之前对照题目检查一遍：题目里的每一问是否都答了？
    例如「查 X 的年份**并计算**到 2026 年过了多少年」是两问，必须先 wiki 再 calculator，
    只答了年份就写 Final Answer 算没做完。
11. 一个任务通常 1~3 次工具调用就够；能答就答，不要反复试探同一个工具。
12. 纯四则运算（例如 (123 + 456) * 789）请把算式**原样**填进 calculator 的 expression，
    不要改写算式，也不要绕道 python_sandbox。
13. 上面的示例对话里出现过的数字（230、2870、1867、159 等）只是用来演示格式的，
    **绝对不许**把示例里的数字当成答案搬过来。
14. 看到 Observation 是「（重复调用，结果同上）」就说明你手里已经有信息了，
    立刻写 Final Answer，不要再重复调用工具。

## 可用工具
{tool_list}"""

FEW_SHOT = [
    # 例 A：单步算术
    {"role": "user", "content": "计算 (12 + 34) * 5 的结果"},
    {"role": "assistant", "content":
        "Thought: 这是纯算术，交给 calculator。\n"
        "Action: calculator\n"
        'Action Input: {"expression": "(12 + 34) * 5"}'},
    {"role": "user", "content":
        "Observation: 表达式：(12 + 34) * 5\n结果：230\n说明：整数 230，共 3 位数字"},
    {"role": "assistant", "content":
        "Thought: 已经拿到结果 230。\n"
        "Final Answer: (12 + 34) * 5 = 230，结果共 3 位数字。"},
    # 例 B：python_sandbox 写循环 + print
    {"role": "user", "content": "用 Python 计算 1 到 20 的平方和"},
    {"role": "assistant", "content":
        "Thought: 需要循环求和，用 python_sandbox，代码里要用 print 输出。\n"
        "Action: python_sandbox\n"
        'Action Input: {"code": "print(sum(i * i for i in range(1, 21)))"}'},
    {"role": "user", "content": "Observation: 执行成功（0.00s）\nstdout:\n2870"},
    {"role": "assistant", "content":
        "Thought: 得到 2870。\n"
        "Final Answer: 1 到 20 的平方和是 2870。"},
    # 例 C：file_search 带 dir 参数做统计
    {"role": "user", "content": "src/tools 目录下有多少个 .py 文件？"},
    {"role": "assistant", "content":
        "Thought: 题目给了目录，file_search 的 dir 要原样写上。\n"
        "Action: file_search\n"
        'Action Input: {"pattern": "*.py", "dir": "src/tools"}'},
    {"role": "user", "content":
        "Observation: 检索目录：…\\src\\tools\npattern：*.py\n模式：auto\n文件名匹配：5 个\n"
        "  1. __init__.py\n  2. calculator.py\n  3. file_search.py\n  4. python_sandbox.py\n  5. wiki.py"},
    {"role": "assistant", "content":
        "Thought: 数一下是 5 个。\n"
        "Final Answer: src/tools 目录下有 5 个 .py 文件。"},
    # 例 D：file_search 读取某个文件的内容
    {"role": "user", "content": "读一下 reports/week1.md 的第一段写了什么"},
    {"role": "assistant", "content":
        "Thought: 要读文件内容，把完整路径直接作为 pattern。\n"
        "Action: file_search\n"
        'Action Input: {"pattern": "reports/week1.md"}'},
    {"role": "user", "content":
        "Observation: 文件：…\\reports\\week1.md\n字符数：420，行数：12\n"
        "第一段：本周完成了数据清洗与特征工程，共处理 3 张表。\n段落数：4"},
    {"role": "assistant", "content":
        "Thought: 第一段内容拿到了。\n"
        "Final Answer: reports/week1.md 第一段写的是：本周完成了数据清洗与特征工程，共处理 3 张表。"},
    # 例 E：wiki + calculator 两步
    {"role": "user", "content": "查维基百科 Marie Curie 的出生年份，并算她到 2026 年是多少岁"},
    {"role": "assistant", "content":
        "Thought: 先查出生年份。\n"
        "Action: wiki\n"
        'Action Input: {"query": "Marie Curie"}'},
    {"role": "user", "content":
        "Observation: 词条：Marie Curie（en 维基百科）\nMaria Salomea Skłodowska-Curie "
        "(born 7 November 1867) was a Polish and naturalised-French physicist and chemist…\n"
        "关键句（含年份）：Marie Skłodowska Curie (born 7 November 1867) was a Polish and "
        "naturalised-French physicist and chemist."},
    {"role": "assistant", "content":
        "Thought: 出生年份是 1867，接着算 2026-1867。\n"
        "Action: calculator\n"
        'Action Input: {"expression": "2026 - 1867"}'},
    {"role": "user", "content":
        "Observation: 表达式：2026 - 1867\n结果：159\n说明：整数 159，共 3 位数字"},
    {"role": "assistant", "content":
        "Thought: 年龄是 159。\n"
        "Final Answer: Marie Curie 出生于 1867 年，到 2026 年是 159 岁。"},
    # 例 F：参数写错，看到错误后自我纠错
    {"role": "user", "content": "在 data 目录里找文件名包含 download 的文件"},
    {"role": "assistant", "content":
        "Thought: 按文件名找文件，用 file_search。\n"
        "Action: file_search\n"
        'Action Input: {"keyword": "download", "dir": "data"}'},
    {"role": "user", "content":
        "Observation: 错误：缺少 pattern 参数，收到的键为 ['keyword', 'dir']；"
        '正确格式：{"pattern": "*.md", "dir": "data"}'},
    {"role": "assistant", "content":
        "Thought: 参数名写错了，应该用 pattern。\n"
        "Action: file_search\n"
        'Action Input: {"pattern": "download", "dir": "data"}'},
    {"role": "user", "content":
        "Observation: 检索目录：…\\data\npattern：download\n模式：auto\n文件名匹配：1 个\n"
        "  1. download.py"},
    {"role": "assistant", "content":
        "Thought: 找到了文件。\n"
        "Final Answer: data 目录下文件名包含 download 的是 download.py。"},
]

# 只在同一行内取值：小模型写完 Final Answer 后常顺手再续一段 Thought/Action，
# 旧写法带 re.S 会把后续整段都吞进答案（见 task-10 的 trace）。
FINAL_RE = re.compile(r"(?:Final\s*Answer|最终答案|最后答案)\s*[:：]\s*([^\n\r]+)", re.I)
ACTION_RE = re.compile(r"^\s*Action\s*[:：]\s*(.+)$", re.M | re.I)
ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*[:：]?\s*(.*)", re.S | re.I)
THOUGHT_RE = re.compile(r"^\s*Thought\s*[:：]\s*(.+)$", re.M | re.I)
# 只去掉包住关键词的 markdown 加粗，不能全局删 "**"：
# Python 里的幂运算符就是 "**"，全局删会把 int(i ** 0.5) 变成 int(i  0.5)。
KEYWORD_BOLD_RE = re.compile(
    r"\*\*\s*(Thought|Action\s*Input|Action|Final\s*Answer|最终答案)\s*[:：]?\s*\*\*\s*[:：]?", re.I)

ARG_ALIASES = {
    "calculator": {"expression": "expression", "expr": "expression", "formula": "expression",
                   "input": "expression", "math": "expression", "query": "expression",
                   "question": "expression", "code": "expression"},
    "python_sandbox": {"code": "code", "source": "code", "script": "code", "python": "code",
                       "program": "code", "snippet": "code", "input": "code",
                       "expression": "code"},
    "file_search": {"pattern": "pattern", "query": "pattern", "keyword": "pattern",
                    "name": "pattern", "filename": "pattern", "file": "pattern",
                    "path": "pattern", "q": "pattern", "text": "pattern",
                    "dir": "dir", "directory": "dir", "folder": "dir", "root": "dir",
                    "base_dir": "dir", "path_dir": "dir"},
    "wiki": {"query": "query", "q": "query", "term": "query", "title": "query",
             "entity": "query", "name": "query", "question": "query",
             "keyword": "query", "search": "query", "lang": "lang", "language": "lang"},
}


def build_system_prompt() -> str:
    return SYSTEM_TEMPLATE.format(
        tool_names=", ".join(TOOLS), tool_list=describe_tools())


def few_shot_subset(n: int | None) -> list[dict]:
    """取前 n 组示例（1 组 = 一次完整的多轮示范）；n<=0 即 zero-shot。

    用于 S3 消融：few-shot 条数对工具调用准确率的影响。
    """
    if n is None:
        return list(FEW_SHOT)
    if n <= 0:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = []
    for message in FEW_SHOT:
        if message["role"] == "user" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return [message for group in groups[:n] for message in group]


def _clean(text: str) -> str:
    text = text.strip()
    text = KEYWORD_BOLD_RE.sub(lambda m: re.sub(r"\s+", " ", m.group(1)).strip() + ":", text)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _parse_json_like(raw: str):
    """尽力把模型给的一段文本解析成 dict / 标量；失败返回 None。"""
    text = raw.strip().strip("`").strip()
    if not text:
        return None
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        for attempt in (candidate,):
            try:
                return json.loads(attempt)
            except ValueError:
                pass
        # 单引号 / Python 字面量
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            pass
        try:
            return json.loads(attempt.replace("'", '"'))
        except ValueError:
            pass
    return None


def _normalize_args(tool: str, parsed) -> dict:
    aliases = ARG_ALIASES.get(tool, {})
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        key = {"calculator": "expression", "python_sandbox": "code",
               "file_search": "pattern", "wiki": "query"}[tool]
        return {key: parsed}

    args: dict = {}
    for key, value in parsed.items():
        canonical = aliases.get(str(key).strip().lower(), str(key).strip())
        args.setdefault(canonical, value)
    return args


def _unwrap_args(tool: str, value, depth: int = 0):
    """小模型有时把参数对象又包了一层字符串，例如：

        {"code": "{\\"code\\": \\"print(1)\\"}"}

    这里逐层拆掉外层包装（只在键名确实都是该工具的参数名时才拆）。
    """
    if depth > 3:
        return value
    aliases = ARG_ALIASES.get(tool, {})

    def _is_arg_dict(obj) -> bool:
        return (isinstance(obj, dict) and bool(obj)
                and all(str(key).strip().lower() in aliases for key in obj))

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            inner = _parse_json_like(stripped)
            if _is_arg_dict(inner):
                return _unwrap_args(tool, inner, depth + 1)
        return value
    if isinstance(value, dict):
        # 单键包裹：{"code": {"code": "print(1)"}} 或 {"code": "{\"code\": \"print(1)\"}"}
        if len(value) == 1:
            only_key, only_value = next(iter(value.items()))
            if str(only_key).strip().lower() in aliases:
                inner = _unwrap_args(tool, only_value, depth + 1)
                if _is_arg_dict(inner):
                    return inner
        return {key: _unwrap_args(tool, item, depth + 1) for key, item in value.items()}
    return value


def parse_response(text: str) -> dict:
    """把模型输出解析成 {"type": "final"/"action"/"invalid", ...}。"""
    cleaned = _clean(text)
    thought = ""
    thought_match = THOUGHT_RE.search(cleaned)
    if thought_match:
        thought = thought_match.group(1).strip()

    final_match = FINAL_RE.search(cleaned)
    if final_match:
        return {"type": "final", "final_answer": final_match.group(1).strip(),
                "thought": thought}

    action_match = ACTION_RE.search(cleaned)
    tool = None
    payload = ""
    if action_match:
        line = action_match.group(1).strip().strip("`").strip()
        call_match = re.match(r"([A-Za-z_][\w\- ]*?)\s*\((.*)\)\s*$", line, re.S)
        if call_match:
            tool = normalize_name(call_match.group(1))
            payload = call_match.group(2).strip()
        else:
            token = re.split(r"[\s:：,;，（(\[{'\"`]+", line)[0]
            tool = normalize_name(token) or normalize_name(
                line.split()[0] if line.split() else "")

    if tool is None:
        # 兜底：有些模型直接吐 function-call 风格的 JSON
        blob = _parse_json_like(cleaned)
        if isinstance(blob, dict):
            name = blob.get("name") or blob.get("tool") or blob.get("function")
            tool = normalize_name(str(name)) if name else None
            if tool:
                payload = blob.get("arguments") or blob.get("action_input") or blob.get("input") or {}

    if tool is None:
        return {"type": "invalid", "thought": thought, "raw": cleaned,
                "reason": "没有解析出 Action / Final Answer"}

    if not payload:
        input_match = ACTION_INPUT_RE.search(cleaned)
        if input_match:
            payload = input_match.group(1)
    if isinstance(payload, dict):
        parsed_args = payload
    else:
        payload_text = str(payload)
        for marker in ("\nObservation:", "\nThought:", "\nUser:", "\nFinal Answer"):
            if marker in payload_text:
                payload_text = payload_text.split(marker)[0]
        parsed_args = _parse_json_like(payload_text)
        if parsed_args is None:
            stripped = payload_text.strip().strip("`").strip()
            parsed_args = stripped or None
    args = _unwrap_args(tool, _normalize_args(tool, parsed_args))
    return {"type": "action", "tool": tool, "args": args,
            "thought": thought, "raw": cleaned}


def _truncate(text: str, head: int = OBS_HEAD_CHARS, tail: int = OBS_TAIL_CHARS) -> str:
    text = str(text)
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n…（中间省略 {len(text) - head - tail} 字符）…\n{text[-tail:]}"


class ReActAgent:
    """手写 ReAct 循环。"""

    def __init__(self, backend=None, max_steps: int = DEFAULT_MAX_STEPS, verbose: bool = False,
                 inject_error: dict | None = None, max_tokens: int = 400,
                 temperature: float = 0.0, system_prompt: str | None = None,
                 few_shot: list[dict] | None = None):
        self.verbose = verbose
        self.backend = backend if backend is not None else llm_module.get_backend(verbose=verbose)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or build_system_prompt()
        # few_shot=None 用默认示例；few_shot=[] 即 zero-shot（S3 消融用）
        self.few_shot = FEW_SHOT if few_shot is None else list(few_shot)
        # inject_error 钩子（S4 错误恢复实验）：{"tool": 可选, "occurrences": 1, "message": 可选}
        self.inject_error = inject_error
        self._injected = 0
        self._task = ""     # 当前任务原文，供 _autofill_file_path / _coach 参考

    # ---------- 对外接口 ----------

    def run(self, task: str) -> AgentTrace:
        started = time.time()
        self._injected = 0
        self._task = task
        reset_sessions()        # 清空 python_sandbox 的跨题变量环境
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.few_shot)
        messages.append({"role": "user", "content": task + self._task_hint()})

        steps: list[dict] = []
        final_answer = ""
        terminated_by = "max_steps"
        error: str | None = None
        seen_calls: dict[str, tuple[bool, str]] = {}
        entity_retry = False     # 「答案漏掉题目点名对象」只补问一次，避免死循环

        for step in range(1, self.max_steps + 1):
            try:
                raw = self.backend.chat(self._trim(messages), max_tokens=self.max_tokens,
                                        temperature=self.temperature, stop=STOP_STRINGS)
            except Exception as e:  # noqa: BLE001 —— 单次生成失败不该炸掉整轮评测
                error = f"模型调用失败（step {step}）：{type(e).__name__}: {e}"
                terminated_by = "backend_error"
                break

            parsed = parse_response(raw)
            record = {"step": step, "raw": raw, "thought": parsed.get("thought", ""),
                      "tool": None, "action": None, "action_input": None,
                      "observation": None, "error": False}

            if self.verbose:
                print(f"\n--- step {step} ---\n{raw}", flush=True)

            if parsed["type"] == "final":
                answer = parsed["final_answer"]
                missing = [] if entity_retry else self._missing_entities(answer)
                if missing:
                    # 题目点名了多个对象（如 'level' 和 'world'），小模型常常只答一个
                    entity_retry = True
                    record["final_answer"] = answer
                    record["action"] = "final_answer"
                    record["observation"] = "答案未覆盖题目点名的对象：" + "、".join(missing)
                    record["error"] = True
                    steps.append(record)
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": (
                        f"你的 Final Answer 漏掉了题目点名的对象：{'、'.join(missing)}。"
                        "请把它们的结论都写进同一条 Final Answer"
                        "（例如「level 是回文（True），world 不是回文（False）」），"
                        "用一行 'Final Answer: …' 重新输出，不要再调用工具。")})
                    continue
                record["final_answer"] = answer
                record["action"] = "final_answer"
                steps.append(record)
                final_answer = answer
                terminated_by = "final_answer"
                break

            if parsed["type"] == "action":
                tool = parsed["tool"]
                # 小模型常把题目里的完整路径缩成裸文件名，这里先补全再执行
                args = self._autofill_file_path(tool, parsed["args"])
                record["tool"] = tool
                record["action"] = tool
                record["action_input"] = json.dumps(args, ensure_ascii=False)
                signature = f"{tool}|{record['action_input']}"
                previous = seen_calls.get(signature)
                injected = False
                if previous is None:
                    observation, failed, injected = self._invoke(tool, args)
                    seen_calls[signature] = (failed, observation)
                elif previous[0]:
                    # 同一个失败调用重复出现：不再执行，直接给更强的纠错信号
                    observation = (f"错误：完全相同的 {tool} 调用刚刚已经失败过一次，"
                                   "重复同样的参数不会有新结果。请修改参数、换用别的工具，"
                                   "或者基于已有 Observation 直接给出 Final Answer。")
                    failed = True
                else:
                    observation = f"（重复调用，结果同上）\n{previous[1]}"
                    failed = False
                record["observation"] = observation
                record["error"] = failed
                hint = self._coach(tool, observation, failed, injected,
                                   repeated=previous is not None,
                                   tool_calls=[s["tool"] for s in steps if s.get("tool")])
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": f"Observation: {_truncate(observation)}{hint}"})
                steps.append(record)
                if self.verbose:
                    print(f"Observation: {_truncate(observation, 400, 200)}", flush=True)
                continue

            # 解析失败：把格式要求再喂回去，让模型重试
            observation = ("格式错误：我没有解析出 Action / Final Answer。"
                           "请严格按格式重来，只输出下面之一：\n"
                           "Thought: ...\\nAction: <工具名>\\nAction Input: <一行 JSON>\n"
                           "或者 Thought: ...\\nFinal Answer: ...")
            record["observation"] = observation
            record["error"] = True
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            steps.append(record)

        if not final_answer:
            final_answer, extra = self._finalize(messages, steps)
            if final_answer:
                terminated_by = ("finalized_after_max_steps" if terminated_by == "max_steps"
                                 else terminated_by)
            if extra:
                steps.append(extra)

        used_tools = [s["tool"] for s in steps if s.get("tool")]
        return {
            "task": task,
            "backend": self.backend.describe(),
            "n_steps": len(steps),
            "max_steps": self.max_steps,
            "steps": steps,
            "used_tools": used_tools,
            "final_answer": final_answer,
            # 注意：自检不信任这个字段，只按 expected_answer_contains 校验 final_answer
            "success": bool(final_answer) and error is None,
            "terminated_by": terminated_by,
            "error": error,
            "elapsed_sec": round(time.time() - started, 2),
        }

    # ---------- 内部实现 ----------

    # 任务解析 / 工具路由：在任务后面直接给一句「这类题用哪个工具」。
    # 只在 system prompt 里写规则对 0.5B 效果很差（它选错工具的频率很高），
    # 把建议放到最靠近生成位置的 user turn，命中率明显更高。
    _ROUTE_EXPR_RE = re.compile(r"[（(]?\d[\d\s]*[-+*/×x][^，。；,;、]*\d")

    def _task_hint(self) -> str:
        task = self._task or ""
        tips: list[str] = []
        expr_match = self._ROUTE_EXPR_RE.search(task)
        if expr_match and re.search(r"计算|结果|位数", task):
            expr = expr_match.group(0).strip()
            tips.append("这是算式计算题：请用 calculator，Action Input 填 "
                        f"{{\"expression\": \"{expr}\"}}（式子照抄，不要用 python_sandbox）。")
        elif re.search(r"Python|python", task):
            tips.append("需要用循环/函数算的题用 python_sandbox，代码里务必 print() 出结果；"
                        "同一题内上一步定义的函数下一步还能用。")
        if re.search(r"sqrt|根号|平方根|math|小数|round", task):
            tips.append("数学函数（sqrt/log 等）在 python_sandbox 里必须先 `import math`，"
                        "再写成 math.sqrt(...)。")
        if re.search(r"目录|文件夹|读取|文件内容|文件路径|查找.{0,12}文件|找出.{0,12}文件", task):
            tips.append("涉及文件的问题用 file_search；题目里出现的目录/路径原样填进 dir/pattern。")
        if re.search(r"维基|百科|出生|发表|发明者|是谁|年份", task):
            tips.append("涉及人物/年份/百科事实先用 wiki 查；查到年份后若题目还要求计算，"
                        "就再用 calculator 算一步，别只答年份。")
        if not tips:
            return ""
        return "\n\n（任务提示：" + " ".join(tips) + "）"

    _QUOTED_RE = re.compile(r"[\"'“”‘’「」『』]([^\"'“”‘’「」『』]{1,40})[\"'“”‘’「」『』]")

    def _missing_entities(self, answer: str) -> list[str]:
        """题目里被引号点名的对象，答案里必须都出现（小模型常只答其中一个）。"""
        if not self._task:
            return []
        wanted: list[str] = []
        for match in self._QUOTED_RE.finditer(self._task):
            token = match.group(1).strip()
            if token and re.search(r"[A-Za-z0-9]", token) and token not in wanted:
                wanted.append(token)
        norm = re.sub(r"\s+", "", str(answer)).lower()
        return [t for t in wanted if re.sub(r"\s+", "", t).lower() not in norm]

    _PATH_IN_TASK_RE = re.compile(r"[\w.\-]+(?:[/\\][\w.\-]+)+")

    def _autofill_file_path(self, tool: str, args: dict) -> dict:
        """把裸文件名补全成任务原文里出现过的完整路径。

        小模型经常把题目里的 `data/agent-fixtures/README.md` 缩成 `README.md`，
        项目里同名文件一多就会读到错的那个（task-10 的失败原因）。
        这里只在「没给 dir、pattern 里没有路径分隔符」时才动手，且必须在任务原文里
        找到同名文件的路径才会替换，不会凭空发明路径。
        """
        if tool != "file_search" or not isinstance(args, dict) or args.get("dir"):
            return args
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return args
        bare = pattern.strip().strip('"').strip("'")
        if not bare or any(sep in bare for sep in ("/", "\\")):
            return args
        for match in self._PATH_IN_TASK_RE.finditer(self._task or ""):
            candidate = match.group(0).replace("\\", "/")
            if candidate.lower().endswith("/" + bare.lower()):
                filled = dict(args)
                filled["pattern"] = candidate
                return filled
        return args

    # 多跳题：题目要「查年份 + 算到某年」，小模型查到年份就以为完事了，这里补一句推它一步
    _CALC_INTENT_RE = re.compile(r"并计算|并算|再算|多少岁|多大|多少年|几年|相隔多少|相差|间隔")
    _YEAR_RE = re.compile(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)")

    _NETWORK_FAIL_RE = re.compile(r"联网失败|不可达|请求失败|Connection|timed?\s*out", re.I)

    def _coach(self, tool: str, observation: str, failed: bool, injected: bool,
               repeated: bool, tool_calls: list[str]) -> str:
        """针对小模型的典型卡点，在 Observation 后追加一行短提示（只提示方向，不给答案）。"""
        tips: list[str] = []
        if injected:
            # inject_error 只注入一次性故障，重试就会成功——推模型重试而不是放弃
            tips.append("这是只注入一次的一次性故障（重试就会成功）：请原样重试同一个调用，不要修改参数。")
        elif failed:
            if self._NETWORK_FAIL_RE.search(str(observation)):
                tips.append("这是网络/服务不可达，不是参数错误：不要凭记忆编造答案；"
                            "可以换一个更常见的词条名重试一次，若仍失败就说明拿不到该信息。")
            else:
                tips.append("请根据错误信息修正参数或换用其他工具；"
                            "如果这个信息确实拿不到，就基于已有 Observation 直接给 Final Answer。")
        elif repeated:
            # 重复成功调用 = 模型在打转，这里必须把它推向收尾（task-4 就是这么卡满 6 步的）
            tips.append("这个调用你已经做过、结果就在上面，**不要再重复调用工具**；"
                        "请直接从 Observation 里找出关键信息，用「Final Answer: …」作答。")
        else:
            year = self._missing_calc_year(observation, tool_calls)
            if year:
                tips.append(f"Observation 里已经有年份 {year}，而题目还要求计算；"
                            f"请立刻用 calculator 算一步（例如 2026 - {year}）再收尾。")
        if not tips:
            return ""
        return "\n（提示：" + " ".join(tips) + "）"

    def _missing_calc_year(self, observation: str, tool_calls: list[str]) -> str | None:
        if "calculator" in tool_calls or not self._CALC_INTENT_RE.search(self._task or ""):
            return None
        years = self._YEAR_RE.findall(str(observation))
        if not years:
            return None
        return min(years, key=int)     # 出生/首次发表一般是最早的那个年份

    def _invoke(self, tool: str, args: dict) -> tuple[str, bool, bool]:
        """执行工具：注入故障 / 捕获异常，都变成 Observation 文本。

        返回 (observation, failed, injected)。
        """
        injected = self._maybe_inject_error(tool)
        if injected:
            return f"错误：{injected}", True, True
        try:
            output = str(run_tool(tool, args))
        except Exception as e:  # noqa: BLE001 —— M3：工具异常不能冒泡 crash 整个循环
            return f"错误：{type(e).__name__}: {e}", True, False
        # 有些工具（如 python_sandbox）把失败写成「错误：…」而不是抛异常，
        # 这里统一识别，让 agent 走同一套纠错分支。
        failed = output.lstrip().startswith("错误：") or "\n错误：" in output
        return output, failed, False

    def _maybe_inject_error(self, tool: str) -> str | None:
        cfg = self.inject_error
        if not cfg:
            return None
        if self._injected >= int(cfg.get("occurrences", 1)):
            return None
        target = cfg.get("tool")
        if target and normalize_name(str(target)) != tool:
            return None
        self._injected += 1
        return cfg.get("message") or f"模拟工具故障：{tool} 暂时不可用（inject_error 钩子注入）"

    def _trim(self, messages: list[dict], budget: int = PROMPT_BUDGET_CHARS) -> list[dict]:
        """上下文预算管理：system + few-shot + 原始任务固定保留，历史 Observation 从最早的开始丢。"""
        head_len = 2 + len(self.few_shot)     # system + few-shot 示例 + 任务本身
        head, rest = messages[:head_len], messages[head_len:]
        total = sum(len(m.get("content") or "") for m in head)
        kept: list[dict] = []
        for message in reversed(rest):
            size = len(message.get("content") or "")
            if kept and total + size > budget:
                break
            total += size
            kept.append(message)
        return head + list(reversed(kept))

    def _finalize(self, messages: list[dict], steps: list[dict]):
        """步数用尽还没有 Final Answer 时，再问一次只做总结。"""
        messages = list(messages) + [{
            "role": "user",
            "content": ("已经用完工具调用步数，请不要再调用任何工具。"
                        "请从上面成功的 Observation 里找出与问题最相关的那一句"
                        "（含人名 / 年份 / 数字 / 文件名的优先），"
                        "把它的关键信息写成一行 'Final Answer: <答案>'；"
                        "信息确实不足时，才说明哪一步失败了。"),
        }]
        try:
            raw = self.backend.chat(self._trim(messages), max_tokens=200, temperature=0.0,
                                    stop=["\nObservation:"])
        except Exception as e:  # noqa: BLE001
            return "", {"step": len(steps) + 1, "tool": None, "action": "finalize_failed",
                        "error": True, "observation": f"总结调用失败：{e}", "raw": ""}

        parsed = parse_response(raw)
        if parsed["type"] == "final":
            answer = parsed["final_answer"]
        else:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            answer = lines[-1] if lines else ""
            answer = re.sub(r"^(Thought|Action|Action Input)\s*[:：]\s*", "", answer)
        return answer, {"step": len(steps) + 1, "tool": None, "action": "finalize",
                        "thought": parsed.get("thought", ""), "raw": raw,
                        "observation": None, "error": False}
