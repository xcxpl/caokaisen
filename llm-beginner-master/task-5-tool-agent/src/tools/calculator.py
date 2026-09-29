"""calculator：安全的数学表达式求值器（不用 eval）。

自检固定参数键：run({"expression": "2 + 3 * 4"}) -> 结果须包含 "14"
"""
from __future__ import annotations

import ast
import math
import operator

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "对数学表达式求值。支持 + - * / // % ** 与括号，"
            "支持 sqrt/log/log10/log2/exp/sin/cos/tan/factorial/gcd/floor/ceil/round/abs/min/max/sum/pow/hypot 等函数，"
            "常量 pi、e。只接受表达式，不要传代码、赋值语句或中文说明。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "单个数学表达式，例如 \"(123 + 456) * 789\" 或 \"sqrt(2026)\" 或 \"sum([1,2,3])\"。注意：它只能算表达式，不能写循环、函数定义或赋值语句（那类需求请用 python_sandbox）",
                    "example": "(123 + 456) * 789",
                },
                "precision": {
                    "type": "integer",
                    "description": "小数保留位数，默认 6",
                    "example": 6,
                },
            },
            "required": ["expression"],
        },
    },
}

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}

_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "cbrt": getattr(math, "cbrt", lambda x: x ** (1 / 3)),
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "fabs": math.fabs,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "comb": math.comb,
    "perm": math.perm,
    "isqrt": math.isqrt,
    "pi_value": lambda: math.pi,
}

_FULLWIDTH = str.maketrans({
    "（": "(", "）": ")", "，": ",", "。": ".", "×": "*", "÷": "/",
    "－": "-", "＋": "+", "＝": "=", "％": "%", "＾": "^", "：": ":",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})


class CalculatorError(ValueError):
    """表达式非法或含不允许的语法；消息会作为 Observation 喂回模型。"""


def _sanitize(expression: str) -> str:
    expr = str(expression).strip()
    expr = expr.translate(_FULLWIDTH)
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = expr.replace("math.", "").replace("Math.", "")
    # 去掉行首等号，如 "= 1 + 2"
    expr = expr.lstrip("=").strip()
    # 去掉数字里的千分位逗号（123,456 -> 123456），但不破坏 max(1, 2) 这类参数分隔
    import re

    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr)
    # 允许中文问句里夹带的"结果是/等于"
    for word in ("结果是", "等于", "答案", "计算", "求"):
        expr = expr.replace(word, "")
    return expr.strip()


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"不支持的常量：{node.value!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        if node.id in _FUNCS:
            return _FUNCS[node.id]
        raise CalculatorError(f"未知的名字：{node.id}（只支持常量 pi/e 与白名单函数）")
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"不支持的运算符：{type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("只允许调用白名单函数，不支持属性/方法调用")
        if node.func.id not in _FUNCS:
            raise CalculatorError(f"不允许调用：{node.func.id}")
        if node.keywords:
            raise CalculatorError("不支持关键字参数")
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [ _eval_node(e) for e in node.elts ]
    raise CalculatorError(f"不允许的语法：{type(node).__name__}")


def evaluate(expression: str, precision: int = 6):
    expr = _sanitize(expression)
    if not expr:
        raise CalculatorError("表达式为空")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"无法解析表达式 {expr!r}：{e.msg}") from e
    try:
        return _eval_node(tree)
    except ZeroDivisionError as e:
        raise CalculatorError("除零错误") from e
    except (OverflowError, ValueError) as e:
        raise CalculatorError(f"计算失败：{e}") from e


def _describe(value, precision: int) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "非有限值"
        if value.is_integer() and abs(value) < 1e15:
            try:
                whole = int(value)
            except (OverflowError, ValueError):
                return f"保留 {precision} 位小数 = {value:.{precision}f}"
            return (f"整数 {whole}，共 {len(str(abs(whole)))} 位数字；"
                    f"保留 {precision} 位小数 = {value:.{precision}f}")
        return f"保留 {precision} 位小数 = {value:.{precision}f}"
    if isinstance(value, int):
        if abs(value) >= 10:
            return f"整数 {value}，共 {len(str(abs(value)))} 位数字"
        return f"整数 {value}"
    return f"{value}"


def run(args) -> str:
    """求值入口。args 可以是 {"expression": ...} 或直接一段表达式字符串。"""
    if isinstance(args, str):
        args = {"expression": args}
    if not isinstance(args, dict):
        raise CalculatorError(f"calculator 参数必须是对象，收到 {type(args).__name__}")

    expression = None
    for key in ("expression", "expr", "formula", "input", "query", "question", "math"):
        if isinstance(args.get(key), str) and args[key].strip():
            expression = args[key]
            break
    if expression is None:
        raise CalculatorError(
            f"缺少 expression 参数，收到的键为 {sorted(args)}；"
            '正确格式：{"expression": "2 + 3 * 4"}')

    try:
        precision = int(args.get("precision", 6))
    except (TypeError, ValueError):
        precision = 6
    precision = max(0, min(precision, 12))

    value = evaluate(expression, precision)
    lines = [
        f"表达式：{_sanitize(expression)}",
        f"结果：{value}",
        f"说明：{_describe(value, precision)}",
    ]
    return "\n".join(lines)
