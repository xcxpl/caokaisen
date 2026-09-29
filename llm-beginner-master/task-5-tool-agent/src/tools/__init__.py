"""4 个工具的注册表。

每个工具模块导出 `TOOL_SCHEMA`（OpenAI function calling 格式）与 `run(args) -> str`。
自检脚本 `eval/run.py` 直接 `from src.tools import calculator, python_sandbox, file_search, wiki`
并按固定参数键调用 `run`，所以这里的模块名与参数键都不要再改。
"""
from __future__ import annotations

from . import calculator, file_search, python_sandbox, wiki

MODULES = (calculator, python_sandbox, file_search, wiki)

TOOLS = {m.TOOL_SCHEMA["function"]["name"]: m for m in MODULES}

# 小模型经常把工具名写成同义词 / 中文，这里做一层容错别名（解析端用，不影响 schema）。
ALIASES = {
    "calculator": "calculator",
    "calc": "calculator",
    "calculate": "calculator",
    "math": "calculator",
    "计算器": "calculator",
    "计算": "calculator",
    "python_sandbox": "python_sandbox",
    "python": "python_sandbox",
    "sandbox": "python_sandbox",
    "python_sandbox_tool": "python_sandbox",
    "python_interpreter": "python_sandbox",
    "python_exec": "python_sandbox",
    "code_interpreter": "python_sandbox",
    "file_search": "file_search",
    "filesearch": "file_search",
    "file search": "file_search",
    "search_file": "file_search",
    "search_files": "file_search",
    "grep": "file_search",
    "local_search": "file_search",
    "文件检索": "file_search",
    "文件搜索": "file_search",
    "wiki": "wiki",
    "wikipedia": "wiki",
    "wiki_search": "wiki",
    "wikipedia_search": "wiki",
    "维基百科": "wiki",
    "维基": "wiki",
}

__all__ = ["calculator", "python_sandbox", "file_search", "wiki",
           "TOOLS", "MODULES", "ALIASES", "normalize_name",
           "get_schemas", "describe_tools", "run_tool", "reset_sessions"]


def reset_sessions() -> None:
    """重置有状态工具（目前只有 python_sandbox 的变量环境）。每道题开始前调一次。"""
    for module in MODULES:
        reset = getattr(module, "reset_state", None)
        if callable(reset):
            reset()


def normalize_name(name: str) -> str | None:
    """把模型写出的工具名映射到 4 个规范名之一，无法识别时返回 None。"""
    if not isinstance(name, str):
        return None
    key = name.strip().strip("`\"'").lower()
    key = key.replace("-", "_").replace(" ", "_")
    if key in TOOLS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    for alias, canonical in ALIASES.items():
        if alias.replace(" ", "_") in key:
            return canonical
    return None


def get_schemas() -> list[dict]:
    """OpenAI function calling 格式的工具列表。"""
    return [m.TOOL_SCHEMA for m in MODULES]


def describe_tools() -> str:
    """拼进 system prompt 的工具清单（名称 + 描述 + 参数示例）。"""
    lines = []
    for i, module in enumerate(MODULES, 1):
        schema = module.TOOL_SCHEMA["function"]
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        example = {key: props[key].get("example", "...") for key in required if key in props}
        lines.append(f"{i}. {schema['name']}：{schema['description']}")
        lines.append(f"   参数 JSON 示例：{_json(example)}")
        for key, spec in props.items():
            optional = "" if key in required else "（可选）"
            lines.append(f"   - {key}{optional}: {spec.get('description', '')}")
    return "\n".join(lines)


def run_tool(name: str, args: dict) -> str:
    """按规范名调用工具，返回字符串结果。"""
    module = TOOLS[name]
    return str(module.run(args))


def _json(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
