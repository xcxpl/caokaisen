# 任务一实验报告

## 一句话总结

从零手写了缩放点积注意力、多头注意力、Transformer encoder block 与分类器，
在 ChnSentiCorp 中文情感二分类上 **dev 正确率 0.8708**（要求 ≥0.80，参考基线 0.85），
并通过了 `eval/run.py` 的全部自检项。

## 实现要点

- `src/attention.py`：手写 `scaled_dot_product_attention`（缩放 + mask + softmax + 加权求和），
  mask 用「填充大负数」而不是「乘 0」，避免 softmax 概率泄漏。
- `src/block.py`：`TransformerBlock` = 多头注意力 + FFN + 两处残差 + 两处 LayerNorm，采用 Pre-LN。
- `src/model.py`：`TransformerClassifier` 堆 4 层 block，用 CLS 池化 + 分类头。
- `train.py`：AdamW + cosine LR（含 warmup）+ 梯度裁剪 + padding mask + 早停。
- 分词用字符级词表（含 `<pad>/<unk>/<cls>`，训练集去重约 4253 个字符），未加载任何预训练模型。

## 实验观察（约 300 字）

模型约 1.35M 参数，10 个 epoch 内 dev 准确率从 0.77 升到 **0.8708**，训练损失从 0.58 降到 0.20，
可见手写编码器在这个小任务上足以收敛，甚至超过官方自检给出的 0.85 参考基线。

在注意力热图里我观察到一个有意思的现象：**CLS 词元（输入的第一行）会把注意力集中到情感关键词上**。
例如正面句「……速度真是快，很是不错……」，第 3 层 head=0 的 CLS 行明显高亮在「不」字附近；
负面句「XP的驱动不好找！……」的 CLS 行则集中在句尾的「！」和「不好」处。也就是说，
模型在做分类时并不是均匀地看整句话，而是学会了把「不错」「不好」「！」这类带情绪的 token 作为决策依据，
这正说明 attention 学到的是可解释的分词/语义线索，而不是单纯的位置噪声。

另外我把同一个注意力函数加上 causal mask 跑了一个迷你因果语言模型（唐诗 next-token prediction），
300 步损失从 5.94 降到 2.61，并且 `causal_mask` 自检的「未来词元泄漏」为 0，
验证了未来信息没有被偷看，这是任务二（mini-GPT）的预热。

可以进一步做的消融：head 数/层数对比、去掉 residual 或 LayerNorm 看是否还能收敛、把绝对位置编码换 RoPE。

## Definition of Done 勾选

- [x] M1 手写 `scaled_dot_product_attention`（`attention_correctness` 通过，误差 9.5e-7）
- [x] M2 手写 `MultiHeadAttention` + `TransformerBlock`（前向形状正确，无报错）
- [x] M3 ChnSentiCorp 分类器 dev 准确率 ≥0.80（实测 0.8708）
- [x] M4 causal mask 跑 toy 语言模型（`causal_mask` 自检通过，泄漏 0；生成示例见 toy_lm.py）
- [x] M5 ≥3 张注意力热图（`figures/`: 正面 / 负面 / 长句）

## 加分项（未系统做，仅留观察）

- S1/S2/S3/S4 未系统做；当前结果已接近 S3（>0.88）的边界（0.8708）。
