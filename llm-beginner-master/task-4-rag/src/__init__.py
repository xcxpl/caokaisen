"""任务四：RAG 文档问答。

模块划分：
    chunker.py     PDF 文本抽取、清洗与字符级切分
    indexer.py     BGE embedding + FAISS 索引构建
    retriever.py   向量召回
    reranker.py    bge-reranker-base 精排
    generator.py   拼接上下文并调用 Qwen 生成
    rag.py         端到端 answer(query)
"""

import os

# transformers 5.x 默认用线程池异步 materialize 权重；在 Windows 上加载
# bge-reranker-base 这类较大模型时会触发 access violation。关闭异步加载，
# 改用单线程装载，稳定且对推理速度无影响。
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

__all__ = ["chunker", "indexer", "retriever", "reranker", "generator", "rag"]
