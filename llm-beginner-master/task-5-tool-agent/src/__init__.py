"""任务五：手写 ReAct 工具调用 Agent。

模块划分：
    src.tools            4 个工具（calculator / python_sandbox / file_search / wiki）
    src.llm              模型后端（OpenAI 兼容 API / 本地 transformers）
    src.agent            手写 ReAct 循环
"""

__all__ = ["agent", "llm", "tools"]
