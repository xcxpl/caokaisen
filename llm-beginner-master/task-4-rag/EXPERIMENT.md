# 任务四实验观察

## 主结果

默认配置：`data/kb.pdf` 抽取文本 → 512 字符切分 / 64 overlap →
`bge-small-zh-v1.5`（CLS pooling，query 加检索前缀）→ FAISS 内积 →
BM25 + RRF 混合召回。自检 `eval/result.json`：

| 指标 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| dense-only | 0.467 | 0.700 | 0.800 | 0.867 | 0.601 |
| **hybrid（默认）** | **0.600** | **0.767** | **0.800** | **0.900** | **0.700** |

通过线为 Recall@10 > 0.6；默认配置 0.90，通过。

## 消融 1：chunk_size 扫描

固定 `overlap = chunk_size / 8`，其余流程相同（不额外拼接章节标题）：

| chunk_size | overlap | chunk 数 | 平均长度 | dense R@10 | hybrid R@10 | hybrid MRR |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 5929 | 110.7 | 0.767 | 0.800 | 0.482 |
| 256 | 32 | 2761 | 236.4 | 0.733 | 0.767 | 0.545 |
| **512** | **64** | **1319** | **493.1** | **0.800** | **0.867** | **0.679** |
| 1024 | 128 | 646 | 1005.2 | 0.800 | 0.867 | 0.616 |

512 与 1024 的 Recall@10 持平，但 512 的 MRR 更高，且 chunk 数适中，
因此默认选 512。默认索引还把章节标题拼进 embedding 输入，实际
Recall@10 进一步提升到 0.90（见主结果表）。

原始数据：`eval/ablation_chunk_size.json`。

## 消融 2：dense vs BM25 + RRF 混合

- 纯 BGE 向量：Recall@10 0.867、MRR 0.601。
- BM25 + RRF（dense/lexical 等权，`rrf_k=10`）：Recall@10 0.900、
  MRR 0.700。
- 混合召回主要救回了 `nndl-inductive-bias` 一题；该题问题里同时出现
  “卷积神经网络 / 循环神经网络 / Transformer / 归纳偏置”，BM25 能把
  绪论中含这些术语的 chunk 提到前十。

原始数据：`eval/candidates_hybrid_top20.json`。

## 消融 3：reranker

`bge-reranker-base` 已实现于 `src/reranker.py`，输入是 `[query, doc]`
文本对，输出 sigmoid 分数。单次验证：

| 片段 | rerank 分数 |
|---|---:|
| “前馈神经网络的信息从输入到输出单向传播……” | 0.9868 |
| “今天天气很好。” | 0.000037 |

排序区分度很好。受 4GB 显存和 Windows 页面文件限制，30 题全量
reranker 消融未跑完：GPU 路径会触发 CUDA access violation，CPU
fp16/bf16 推理极慢（batch16 约 27 秒），CPU fp32 虽只需约 0.2 秒/批，
但加载 1.1GB 权重时多次遇到页面文件不足。代码已做优雅降级：
reranker 加载失败时自动退回混合召回，不影响 M4 自检。

## 生成忠实性

本地 Qwen2.5-0.5B-Instruct 对第一题的生成结果：

> 前馈神经网络的基本特点是信息从输入到输出单向传播，没有反馈连接。
> 它能够集中体现神经元、层的堆叠、非线性变换、反向传播和自动微分等
> 核心概念。因此，前馈神经网络是理解深度学习的基础，同时也是后续
> 卷积神经网络、循环神经网络以及 Transformer 等复杂模型的重要基础。

答案与检索到的第 102 页原文一致，没有引入上下文外断言。代码同时支持
Qwen2.5-7B-Instruct 的 OpenAI 兼容 API（Ollama / vLLM / llama.cpp）。

## 实验观察（约 450 字）

在 30 条 NNDL gold QA 上，512 字符切分 + BGE-small 向量召回 +
BM25/RRF 混合召回得到 Recall@1/3/5/10 =
0.60/0.77/0.80/0.90、MRR = 0.70，明显高于纯向量的
0.47/0.70/0.80/0.87、MRR 0.60。chunk_size 扫描显示 128 和 256 太碎，
平均长度虽小但召回下降；1024 的 Recall@10 与 512 持平，但 MRR 更低，
说明过大的 chunk 会稀释关键句，最终选 512/64。加入章节标题作为
embedding 前缀后，默认索引的 Recall@10 从消融版的 0.867 提升到 0.90，
说明给 chunk 补充章节上下文对中文教材检索有效。剩余 3 个失败案例
（Transformer 长距离依赖、FlashAttention/GQA、GraphSAGE/GAT）的相关
片段在向量检索中排到 100 名之外，属于“问题表述与正文另一章更相似”的
源歧义问题；混合检索救回了归纳偏置一题，但单靠 rerank top-20 无法覆盖。
reranker 在 CPU 上以 fp32 对相关/无关片段的打分分别为 0.987/0.00004，
排序区分度很好；受 4GB 显存和 Windows 页面文件限制，未完成 30 题全量
rerank 消融。生成侧用 Qwen2.5-0.5B-Instruct 做了本地验证，答案能严格
依据检索上下文，未出现明显幻觉；代码同时保留 Qwen2.5-7B-Instruct 的
OpenAI 兼容 API 接入口。

## 复现消融

```powershell
cd task-4-rag

# chunk_size 扫描（128/256/512/1024）
python experiments/run_ablation.py chunk-size

# dense vs hybrid：先收集候选，再单独跑 reranker（避免同进程占内存）
python experiments/run_ablation.py collect-candidates
python experiments/run_ablation.py rerank
```
