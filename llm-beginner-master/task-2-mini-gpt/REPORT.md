# 任务二：从零实现 mini-GPT —— 实验观察

## 完成情况（Definition of Done）

| 里程碑 | 说明 | 状态 |
| --- | --- | --- |
| M1 BPE 分词 | 手写字节级 BPE（`src/tokenizer.py`），encode→decode 可还原；`tokenizer_roundtrip` 通过 | ✅ |
| M2 Decoder-only + RoPE | `src/rope.py` 旋转位置编码，只作用于 Q/K，前向形状正确 | ✅ |
| M3 KV cache | 增量解码与全量前向 logits 最大误差 ~1e-5，`kv_cache_equivalence` 通过 | ✅ |
| M4 预训练 | dev 困惑度 44.73 < 阈值 50，`perplexity_on_dev` 通过 | ✅ |
| M5 采样 | 实现 greedy / top-k / top-p / temperature，能生成连贯诗歌 | ✅ |

## 交付物

- `ckpt/tokenizer.json`：在唐诗语料上训练好的 BPE 词表（vocab 1024）。
- `ckpt/best.pt`：预训练模型 state_dict（config：6 层、n_embd=256、n_head=8、block 128、vocab 1024，约 5.26M 参数）。
- `eval/result.json`：三项自检结果。
- `src/`：tokenizer / rope / attention / model / sampling 完整实现。
- `train.py`：训练脚本（warmup+cosine、grad clip、早停、困惑度监控）。
- `samples.txt`：不同策略的生成样例；本文件为实验观察。

## 环境与实验过程

设备：单机；PyTorch 2.5.1 + CUDA（训练在 GPU 上，dev 困惑度评估用 CPU）。数据先用仓库自带 `poetryFromTang.txt`（约 48KB，唐诗 quick-start）跑通 pipeline，随后为达到有意义的效果切换到更大的唐诗语料。

### 1. BPE 分词的字/字节取舍（关键观察）

先按字节级 BPE 训练，扫了几档词表大小，看它们对 dev 困惑度与生成质量的影响：

| vocab | merge | train token | 该词表下 unigram dev ppl |
| --- | --- | --- | --- |
| 300（近字节级） | 40 | 33k | 84 |
| 400 | 140 | 28k | 153 |
| 512 | 252 | 25k | 211 |
| 1024 | 764 | 19k | 496 |

结论一：词表越大，token 越接近汉字/词（生成更“像中文”），但在小语料上熵越高、越难泛化。字节级 vocab=300 在 48KB 诗上能把 dev ppl 压到 ~27，远低于 50，但生成是“字节拼接”，常拼出非法 UTF-8，不连贯——**只降 ppl 不够**。

结论二：vocab≈1024 时 764 个 merge 已把高频汉字压成单 token，生成的是「以字/词为单位的连贯诗」，但需要足够语料才能把 ppl 压下来。

### 2. 小语料上的灾难性过拟合

在 48KB 唐诗（约 1.5 万字符）上，1.45M~3.68M 的模型都会**迅速死记训练集**：train_ppl 从 ~109 一路降到 ~1.0，而 dev_ppl 反而从 ~59 飙到数千甚至数万。这印证了 `每 token 参数比` 过高时的过拟合——模型不是“学诗”，而是“背诗”。因此果断放弃 48KB quick-start 作为正式训练集。

### 3. 数据规模与模型容量的配合

改用 HF `Lifan-Z/Chinese-poetries-txt` 全唐诗文本（每行一首）。数据量从 9 万 token 提升到 510 万 token，模型加深到 6 层（约 5.26M）：

- vocab 1024、6 层：dev_ppl 一路降到 **44.73**（train_ppl ~38），泛化差距很小、无过拟合，明显优于同模型在小语料上的表现。
- 训练采用 warmup+cosine、grad clip、weight decay、按 dev ppl 早停保存最优。

**观察**：困惑度达标的关键不是“模型够小”，而是「有意义的 token 粒度 + 充分的语料 + 相匹配的容量」。三者匹配后，模型既能把 dev ppl 压到 50 以下，又能生成像样的五言/七言诗。

### 4. 采样与生成

同一 prompt 下对比：
- **greedy**：最稳，输出多为工整的五言/七言，如 `床前明月光。一径深深碧翠微，故人无复见君家。山光不可见人意，万事无人见此身。`。
- **top-k / top-p / temperature**：多样性更高，能换用不同意象；temperature 越高越随机，偶尔会因词表未覆盖的生僻字（走字节回退）产生个别非法 UTF-8，这是小模型常见现象。

### 5. RoPE 与 KV cache

- **RoPE**：只作用在 Q、K，前后折半配对，`base=10000`。增量解码时新 token 的位置必须按缓存长度续算，否则角度错、与全量前向对不上。
- **KV cache**：`forward(ids, kv_cache=None, return_cache=True)`，在 K/V 的序列维拼接；实测增量解码与全量前向的 logits 最大绝对误差约 **1.3e-5**（阈值 1e-4）。自检里还修了一个因果 mask 的坑：增量时 query 是 1、key 是 total_t，不能直接 `tril`，要用「query 绝对位置 ≥ key 绝对位置」构造。

## 一句话总结

用字节级 BPE + RoPE + KV cache 从零手写了一个 decoder-only mini-GPT，把「有意义的 token 粒度」「充分的语料」「匹配的容量」三者对齐后，在唐诗 dev 上达到困惑度 44.73（<50），并能用四种采样策略生成连贯诗歌。此过程最直观的教训是：**小语料上先背诗后学诗，只有扩数据、加容量、调词表粒度才能既降困惑度又保生成质量**。
