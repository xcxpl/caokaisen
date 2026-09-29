# -*- coding: utf-8 -*-
"""
重建训练曲线图。

背景:
    train.py 原版在 evaluate_model 里没有调用 metric.reset()，导致每次评估都在
    累加历史预测，训练日志中的 F1/acc 会被污染（已修复）。
    为了得到真实曲线，这里用每个已保存的 checkpoint（model_25 ~ model_250）
    在 dev 集上重新评测一次，loss 曲线则直接取自训练日志（loss 未被污染）。
"""
import os
import re
import sys

ROOT = r'F:\zhuomian\transformers_tasks-main\text_classification'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          default_data_collator)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

from utils import TextClassificationDataset
from iTrainingLogger import iSummaryWriter

LOG = r'F:\zhuomian\transformers_tasks-main\.workbuddy\full_train.log'
DEV = 'data/comment_classify/dev.txt'
CKPT_DIR = 'checkpoints/comment_classify'
OUT_DIR = 'logs/comment_classify'
OUT_NAME = 'BERT'
MAX_SEQ_LEN = 96
BATCH = 32
DEVICE = 'cpu'

# ---- 1. 从日志解析 loss 点（global step -> loss） ----
raw = open(LOG, encoding='utf-8', errors='ignore').read()
raw = re.sub(r'\x1b\[[0-9;]*m', '', raw)
loss_pts = [(int(m.group(1)), float(m.group(2)))
            for m in re.finditer(r'global step (\d+), epoch: \d+, loss: ([\d.]+)', raw)]
print(f'[+] 解析到 {len(loss_pts)} 个 loss 点')

# ---- 2. 逐个 checkpoint 在 dev 集上评测 ----
tokenizer = AutoTokenizer.from_pretrained(os.path.join(CKPT_DIR, 'model_best'))
dataset = TextClassificationDataset(DEV, tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, collate_fn=default_data_collator, batch_size=BATCH)

writer = iSummaryWriter(log_path=OUT_DIR, log_name=OUT_NAME)
for step, loss in loss_pts:
    writer.add_scalar('train/train_loss', loss, step)

steps = sorted(int(d.split('_')[1]) for d in os.listdir(CKPT_DIR)
               if d.startswith('model_') and d.split('_')[1].isdigit())
print(f'[+] 待评测 checkpoint: {steps}')

for step in steps:
    ckpt = os.path.join(CKPT_DIR, f'model_{step}')
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEVICE).eval()
    golds, preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                input_ids=batch['input_ids'].to(DEVICE),
                token_type_ids=batch['token_type_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
            ).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            golds.extend(batch['labels'].numpy().tolist())
    writer.add_scalar('eval/accuracy', round(accuracy_score(golds, preds), 4), step)
    writer.add_scalar('eval/precision', round(precision_score(golds, preds, average='weighted', zero_division=0), 4), step)
    writer.add_scalar('eval/recall', round(recall_score(golds, preds, average='weighted', zero_division=0), 4), step)
    writer.add_scalar('eval/f1', round(f1_score(golds, preds, average='weighted', zero_division=0), 4), step)
    print(f'    step {step}: acc={accuracy_score(golds, preds):.4f} '
          f'f1={f1_score(golds, preds, average="weighted", zero_division=0):.4f}')

writer.record()
print(f'[+] 已重建曲线: {os.path.join(OUT_DIR, OUT_NAME)}.png')
