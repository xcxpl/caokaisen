# 任务四实现说明

本目录已完成 README 中的 M1–M4 必做项，并补做了 chunk_size 扫描与
dense / hybrid 检索消融。

## 文件结构

```text
task-4-rag/
├── data/
│   ├── download.py            # PDF / 模型下载与 gold QA 校验
│   ├── gold_qa.jsonl          # 30 条 NNDL gold QA
│   ├── kb.pdf                 # NNDL v2 PDF（已下载）
│   └── index/                 # FAISS 索引 + chunks.jsonl + manifest.json
├── models/
│   ├── bge-small-zh-v1.5/     # embedding
│   ├── bge-reranker-base/     # reranker
│   └── Qwen2.5-0.5B-Instruct/ # 本地生成模型（演示用，可选）
├── src/
│   ├── chunker.py             # PDF 抽取、清洗、字符级切分
│   ├── indexer.py             # BGE embedding + FAISS 索引
│   ├── lexical.py             # BM25 词法检索
│   ├── retriever.py           # dense / BM25 / RRF 混合召回
│   ├── reranker.py            # bge-reranker-base 精排
│   ├── generator.py           # 上下文拼接 + Qwen 生成
│   └── rag.py                 # 端到端 answer(query)
├── experiments/
│   └── run_ablation.py        # chunk_size / dense vs hybrid / reranker 消融
├── eval/
│   ├── run.py                 # 自检入口
│   ├── result.json            # 自检结果
│   ├── ablation_chunk_size.json
│   └── candidates_hybrid_top20.json
├── SOLUTION.md                # 本文件
└── EXPERIMENT.md              # 实验观察
```

## 必做项完成情况

- [x] **M1** `src/chunker.py::chunk_text`：字符级滑动窗口，优先在句末切分；
  自检 chunk 数 17，平均长度 241.9，落在 `(128, 307.2)` 内。
- [x] **M2** 从 `data/kb.pdf`（不是 LaTeX 源）抽取文本，用
  `bge-small-zh-v1.5` 编码并建立 FAISS `IndexFlatIP`；向量已 L2 归一化，
  query 侧加 BGE 检索前缀。
- [x] **M3** 在 30 条 gold QA 上评测：Recall@1/3/5/10 =
  0.60 / 0.77 / 0.80 / **0.90**，MRR = **0.70**（通过线为 Recall@10 > 0.6）。
- [x] **M4** `src/rag.py::answer` 串起召回 → rerank → 生成，返回
  `{"answer": str, "sources": list[dict]}`，自检通过。

## 复现步骤

```powershell
cd task-4-rag

# 1. 下载 PDF 并校验 gold QA（模型可用 ModelScope / HuggingFace 镜像下载）
python data/download.py --skip-models

# 2. 建索引（默认 512 字符 / 64 overlap）
python -m src.indexer --chunk-size 512 --overlap 64

# 3. 自检
python eval/run.py

# 4. 命令行问答
python -m src.rag "前馈神经网络的基本特点是什么？"
```

## Qwen2.5-7B-Instruct 接法

生成器优先调用 OpenAI 兼容接口；没有本地服务时才用
`models/Qwen2.5-0.5B-Instruct` 或 extractive 兜底。

```powershell
# Ollama
ollama pull qwen2.5:7b-instruct
ollama serve
$env:RAG_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:RAG_LLM_MODEL = "qwen2.5:7b-instruct"

# vLLM
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
$env:RAG_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:RAG_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

python -m src.rag "什么是 RAG？"
```

常用环境变量：

| 变量 | 作用 | 默认 |
|---|---|---|
| `RAG_LLM_BACKEND` | `auto` / `api` / `local` / `extractive` | `auto` |
| `RAG_LLM_BASE_URL` | OpenAI 兼容服务地址 | `http://127.0.0.1:11434/v1` |
| `RAG_LLM_MODEL` | API 模型名 | `qwen2.5:7b-instruct` |
| `RAG_LOCAL_MODEL` | 本地 transformers 模型目录 | `models/Qwen2.5-0.5B-Instruct` |
| `RAG_ENABLE_RERANKER` | 设为 `0` 可跳过 reranker，省内存 | `1` |
| `RERANKER_DEVICE` | reranker 设备，默认 CPU | `cpu` |

## 环境说明

- 本机为 Windows + Python 3.12 + torch 2.5.1+cu121 +
  transformers 5.16.1，GPU 为 RTX 3050 Laptop 4GB。
- transformers 5.x 在 Windows 上异步加载大权重会触发 access violation，
  `src/__init__.py` 已设置 `HF_DEACTIVATE_ASYNC_LOAD=1`；
  reranker 用 `safetensors` + NumPy 逐张量加载，绕开 PyTorch 对内存映射
  张量切片的崩溃路径。
- reranker 默认 CPU fp32：PyTorch CPU 的 fp16/bf16 是模拟实现，batch16
  要几十秒；fp32 只需约 0.2 秒。CUDA 稳定时可用
  `$env:RERANKER_DEVICE="cuda"`。
- 4GB 显存 + 较小 Windows 页面文件下，同一进程同时加载 reranker 和
  本地 Qwen 可能触发内存不足；`answer` 会按 reranker → Qwen → extractive
  的顺序优雅降级，不影响自检通过。7B 模型建议走 Ollama / vLLM API。

## 已知限制

- 30 题中有 3 题未命中（`nndl-transformer-attention`、
  `nndl-flashattention-gqa`、`nndl-graphsage-gat`）：相关 gold anchor
  在 dense 排序中位于 100 名之后，属于“问题表述与正文另一章更相似”的
  源歧义问题，top-20 rerank 无法覆盖。
- 受 CPU / 内存限制，reranker 的 30 题全量消融未跑完；已单独验证
  `bge-reranker-base` 对相关/无关片段打分为 0.987 / 0.00004。
- 本地生成演示使用 Qwen2.5-0.5B-Instruct；提交完整实验时建议按上面的
  API 接法换成 Qwen2.5-7B-Instruct。
