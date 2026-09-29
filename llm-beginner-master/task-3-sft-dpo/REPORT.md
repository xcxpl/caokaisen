# 任务三实验观察

本实验在 Qwen2.5-0.5B 上用自写 LoRA 完成了 SFT 与 DPO。LoRA 只注入 `q_proj`、`v_proj`，rank=8、alpha=16，A 用 Kaiming 初始化、B 置零，原始权重全部冻结；注入后可训练参数为 540,672，占 494,573,440 总参数的 0.109%。chat template 使用 Qwen 的 `<|im_start|>`/`<|im_end|>` 格式，loss 只对 assistant 内容计算，mock 多轮对话的 mask 比例为 0.556，user/system 与模板控制符均为 -100。

SFT 从 MOSS-003-sft 中采样 800 条多轮对话，截断到 256 token，训练 400 步，loss 从 5.59 降到 2.29，耗时约 229 秒，显存约 1.05 GB。DPO 在 SFT 权重上使用 DPO-En-Zh-20k 的 100 组短偏好对，β=0.1，训练 120 步，loss 从 0.693 降到 0.663，reward margin 最终为 0.061，过程中多次出现正 margin，说明 chosen 的偏好被逐步拉开；reference 采用冻结的 SFT LoRA 副本，避免再加载一份 0.5B 模型。

对比样例显示：base 容易续写代码片段或无关文本；SFT 已能按中文指令组织回答；DPO 在礼貌改写等样本上更倾向于安抚和结构化表达，但在安全拒答上仍有限，主要原因是本次受 4GB 显存限制只使用了小子集和较短训练。三项自检 `lora_param_count`、`loss_masking`、`sft_vs_base` 均已通过。
