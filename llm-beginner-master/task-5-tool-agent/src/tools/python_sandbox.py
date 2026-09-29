"""python_sandbox：受限的 Python 执行环境（教学级防护）。

自检固定参数键：run({"code": "print(sum(range(10)))"}) -> 结果须包含 "45"

⚠️ 防护边界（务必如实理解，别当沙箱用）：
  - 这里做的是「白名单 import + 白名单 builtins + 行数/超时熔断」，
    属于**教学级**限制，不是真正的安全隔离。
  - 仍可通过 `().__class__.__bases__`、`__globals__`、`__subclasses__` 等路径逃逸，
    也挡不住内存耗尽（没有 rlimit / cgroup）。
  - 只对可信或自产代码使用；真要跑不可信输入，应上子进程 + resource 限制、
    RestrictedPython 或容器。
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import re
import sys
import threading
import time
import traceback

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "在受限的本地 Python 解释器里执行一小段代码并返回 stdout。"
            "适合：循环/条件/函数定义、数列与质数统计、字符串处理、精确小数运算、"
            "以及任何计算器表达不出来的计算。"
            "代码里必须用 print() 输出结果；如果最后一行是表达式，系统会自动打印它。"
            "不可用：文件读写、网络、系统命令、安装包（import 只放开标准库白名单）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 源码，多行用 \\n 连接，务必包含 print 或把结果放在最后一行表达式",
                    "example": "print(sum(n for n in range(2, 100) if all(n % d for d in range(2, int(n ** 0.5) + 1))))",
                },
                "timeout": {
                    "type": "number",
                    "description": "墙钟超时秒数，默认 5，上限 15",
                    "example": 5,
                },
            },
            "required": ["code"],
        },
    },
}

ALLOWED_MODULES = {
    "math", "cmath", "statistics", "random", "itertools", "functools", "operator",
    "collections", "heapq", "bisect", "re", "json", "string", "textwrap",
    "decimal", "fractions", "datetime", "time", "zoneinfo", "calendar", "unicodedata",
    "typing", "dataclasses", "enum", "copy", "pprint", "difflib", "hashlib",
    "uuid", "base64", "binascii", "struct", "array", "numbers", "abc", "types",
    "contextlib", "warnings", "traceback", "sys", "platform",
}

# 即使 import 放开，这几个模块的部分能力也受限（sys/platform 只用于读取常量）
_SAFE_BUILTIN_NAMES = (
    "abs all any ascii bin bool bytearray bytes callable chr complex dict divmod "
    "enumerate filter float format frozenset getattr hasattr hash hex int isinstance "
    "issubclass iter len list map max min next oct ord pow print range repr reversed "
    "round set setattr slice sorted str sum tuple type zip"
).split()

_SAFE_EXCEPTIONS = (
    "Exception BaseException ArithmeticError ZeroDivisionError OverflowError "
    "ValueError TypeError KeyError IndexError AttributeError NameError "
    "RuntimeError StopIteration StopAsyncIteration NotImplementedError "
    "AssertionError LookupError UnboundLocalError ImportError "
    "FileNotFoundError OSError EOFError RecursionError MemoryError"
).split()

MAX_LINES = 400_000          # 熔断：执行的源码行数上限（挡死循环）
MAX_OUTPUT_CHARS = 4000      # stdout 截断
DEFAULT_TIMEOUT = 5.0
MAX_TIMEOUT = 15.0

# NameError 里如果是这些名字，多半是忘了 import math（小模型的高频错误）
_MATH_NAMES = {"sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "log2", "log10",
               "exp", "pi", "e", "pow", "factorial", "floor", "ceil", "fabs", "hypot",
               "degrees", "radians", "gcd", "comb", "perm", "isqrt", "fsum"}


class SandboxError(RuntimeError):
    """参数非法或执行不可恢复时抛出（消息会作为 Observation 喂回模型）。"""


def _build_builtins() -> dict:
    safe = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}
    for name in _SAFE_EXCEPTIONS:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    safe["__import__"] = _guarded_import
    safe["True"], safe["False"], safe["None"] = True, False, None
    return safe


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("sandbox 不支持相对导入")
    root = str(name).split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(
            f"sandbox 拒绝 import {name!r}；允许的模块白名单：{', '.join(sorted(ALLOWED_MODULES))}")
    return builtins.__import__(name, globals, locals, fromlist, level)


class _Abort(Exception):
    """熔断信号（行数超限 / 超时）。"""


def _make_tracer(deadline: float, counter: list):
    def tracer(frame, event, arg):
        if event == "line":
            counter[0] += 1
            if counter[0] > MAX_LINES:
                raise _Abort(f"执行行数超过 {MAX_LINES} 行，已中止（疑似死循环）")
            if counter[0] % 512 == 0 and time.monotonic() > deadline:
                raise _Abort("执行超时，已中止")
        return tracer

    return tracer


# 单题内保持变量：小模型习惯「这一步定义函数、下一步再调用」，
# 如果每次调用都是全新的空环境，第二步就会 NameError（task-6 的回文题正是这么挂的）。
# 由 agent 在每道题开始前调用 reset_state() 清空，避免题目之间串变量。
_SESSION_ENV: dict | None = None


def reset_state() -> None:
    """清空跨调用的变量环境（每道题开始前调用）。"""
    global _SESSION_ENV
    _SESSION_ENV = None


def _session_env() -> dict:
    global _SESSION_ENV
    if _SESSION_ENV is None:
        _SESSION_ENV = {"__name__": "__sandbox__", "__builtins__": _build_builtins()}
    return _SESSION_ENV


def _exec_in_thread(code: str, timeout: float, stdout: io.StringIO, box: dict):
    code_lines = code.splitlines()

    def worker():
        env = _session_env()
        env.pop("__sandbox_result__", None)     # 清掉上一步的自动打印残留
        counter = [0]
        sys.settrace(_make_tracer(time.monotonic() + timeout, counter))
        try:
            tree = ast.parse(code, mode="exec")
            # 最后一行若是表达式，自动打印它的值（小模型常常忘记 print）
            auto = False
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = tree.body.pop()
                assign = ast.Assign(targets=[ast.Name(id="__sandbox_result__", ctx=ast.Store())],
                                    value=last.value)
                ast.copy_location(assign, last)
                ast.fix_missing_locations(assign)
                tree.body.append(assign)
                auto = True
            compiled = compile(tree, "<python_sandbox>", "exec")
            exec(compiled, env)  # noqa: S102 —— 受限环境见模块 docstring
            if auto and env.get("__sandbox_result__") is not None:
                print(f"[自动打印最后一行] {env['__sandbox_result__']!r}")
            box["ok"] = True
        except _Abort as e:
            box["aborted"] = str(e)
        except SyntaxError as e:
            box["error"] = f"语法错误：{e.msg}（第 {e.lineno} 行）"
        except BaseException as e:  # noqa: BLE001 —— 任何异常都要变成 Observation
            line = ""
            frames = traceback.extract_tb(e.__traceback__)
            for frame in reversed(frames):
                if frame.filename == "<python_sandbox>":
                    source = ""
                    if 0 < frame.lineno <= len(code_lines):
                        source = code_lines[frame.lineno - 1].strip()
                    line = f"（第 {frame.lineno} 行：{source or (frame.line or '').strip()}）"
                    break
            message = f"{type(e).__name__}: {e}{line}"
            if isinstance(e, NameError):
                missing = re.search(r"name '([\w.]+)' is not defined", str(e))
                if missing and missing.group(1).split(".")[0] in _MATH_NAMES:
                    name = missing.group(1)
                    message += (f"\n提示：{name} 属于 math 模块。请先 `import math`，"
                                f"再写成 math.{name}(...)；如果只是简单算术，也可以改用 calculator 工具。")
            box["error"] = message
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=worker, daemon=True)
    with contextlib.redirect_stdout(stdout):
        thread.start()
        thread.join(timeout + 1.0)
    if thread.is_alive():
        box["aborted"] = f"执行超过 {timeout:g} 秒仍未结束，已放弃等待（线程可能仍在后台跑）"


def run(args) -> str:
    """执行代码，返回 stdout / 异常信息文本。"""
    if isinstance(args, str):
        args = {"code": args}
    if not isinstance(args, dict):
        raise SandboxError(f"python_sandbox 参数必须是对象，收到 {type(args).__name__}")

    code = None
    for key in ("code", "source", "script", "python", "program", "snippet", "input"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            code = value
            break
    if code is None:
        raise SandboxError(
            f"缺少 code 参数，收到的键为 {sorted(args)}；"
            '正确格式：{"code": "print(sum(range(10)))"}')

    try:
        timeout = float(args.get("timeout", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    timeout = max(0.2, min(timeout, MAX_TIMEOUT))

    code = _strip_code_fence(code)
    stdout = io.StringIO()
    box: dict = {}
    started = time.monotonic()
    _exec_in_thread(code, timeout, stdout, box)
    elapsed = time.monotonic() - started

    output = stdout.getvalue()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n...(stdout 超长，已截断到 {MAX_OUTPUT_CHARS} 字符)"
    output = output.strip()

    header = f"执行{'成功' if box.get('ok') else '未成功'}（{elapsed:.2f}s）"
    body = f"stdout:\n{output}" if output else "stdout:（空）"
    if box.get("aborted"):
        return f"{header}\n{body}\n错误：{box['aborted']}"
    if box.get("error"):
        return f"{header}\n{body}\n错误：{box['error']}"
    if not output:
        # 用「错误：」开头，上层 agent 会当成失败观测并提示模型自我纠错
        return (f"{header}\n{body}\n"
                "错误：代码执行完了但没有任何输出。请修改代码，把要看的计算结果用 print(...) "
                "打印出来（定义了函数的话，记得调用它），然后重新调用 python_sandbox。")
    return f"{header}\n{body}"


def _strip_code_fence(code: str) -> str:
    text = code.strip()
    # 小模型常把整个参数对象又塞进 code：{"code": "print(1)"}（多一层包装就直接语法错误）
    if text.startswith("{") and text.rstrip().endswith("}"):
        inner = _unwrap_code_object(text)
        if inner:
            text = inner.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _unwrap_code_object(text: str) -> str | None:
    """从 {"code": "..."} 这类包装里取出真正的代码（标准 JSON 与畸形 JSON 都尽力处理）。"""
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        for key in ("code", "source", "script", "python", "program", "snippet"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None
    # 畸形 JSON 兜底：直接抠 "code": "...." 里的内容
    match = re.search(r'"(?:code|source|script|python)"\s*:\s*"(.*?)"\s*\}?\s*$', text, re.S)
    if not match:
        return None
    inner = match.group(1)
    try:
        return json.loads(f'"{inner}"')
    except ValueError:
        return inner.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
