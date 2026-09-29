#!/usr/bin/env bash
# CPU / Windows(Git Bash) 版本的启动脚本。
# 与原 train.sh 的区别：
#   1) --device 改为 cpu；
#   2) 指定 HF 镜像，便于在国内下载 bert-base-chinese；
#   3) 关闭 datasets 库（已改用轻量自读 txt 的 Dataset），并让 torch 跑满 CPU 线程。
# 用法：
#   PYTHON=/path/to/venv/python bash train_cpu.sh
#   （不指定 PYTHON 时使用 PATH 里的 python）

# ---- 国内模型下载镜像 ----
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# ---- 让 torch 用满 CPU 线程（默认只取一半） ----
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}

# ---- 模型已缓存到本地，离线加载，避免联网校验 ----
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

PYTHON=${PYTHON:-python}

"$PYTHON" -u train.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "BERT" \
    --num_labels 8 \
    --batch_size 32 \
    --max_seq_len 96 \
    --valid_steps 25 \
    --logging_steps 10 \
    --num_train_epochs 20 \
    --device "cpu"
