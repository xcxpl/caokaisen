"""任务六 · 顶层之一：Subagent（独立 context 的子 agent）。

每个 subagent 都满足三条硬要求：
1. **独立 message 列表** —— 不共用主 agent 的对话历史；
2. **独立步数上限** —— 自己数自己的步，不会被主 agent 的预算带偏；
3. **工具子集** —— 只拿完成本职责所需的工具，避免乱调。

主 agent 只拿回一份 **摘要**，不吞 subagent 的全部 trace。
"""
from .code_search import CodeSearchSubagent
from .test_runner import TestRunnerSubagent

__all__ = ["CodeSearchSubagent", "TestRunnerSubagent"]
