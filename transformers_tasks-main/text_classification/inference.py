# !/usr/bin/env python3
r"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

测试训练好的模型。

Author: pankeyu
Date: 2023/01/11
"""
import os
from typing import List

import torch
from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def inference(
    model, 
    tokenizer, 
    sentences: List[str],
    device: str,
    batch_size=16,
    max_seq_len=96
    ) -> List[int]:
    """
    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        sentences (List[str]): _description_
        batch_size (int, optional): _description_. Defaults to 16.
        max_seq_len (int, optional): _description_. Defaults to 128.

    Returns:
        List[int]: [laebl1, label2, label3, ...]
    """
    res = []
    for i in range(0, len(sentences), batch_size):
        batch_sentence = sentences[i:i+batch_size]
        ipnuts = tokenizer(
            batch_sentence,
            truncation=True,
            max_length=max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        output = model(
            input_ids=ipnuts['input_ids'].to(device),
            token_type_ids=ipnuts['token_type_ids'].to(device),
            attention_mask=ipnuts['attention_mask'].to(device),
        ).logits
        output = torch.argmax(output, dim=-1).cpu().tolist()
        res.extend(output)
    return res


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'           # 自动选择设备
    saved_model_path = 'checkpoints/comment_classify/model_best'       # 训练模型存放地址
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path) 
    model.to(device).eval()

    label_mapping = {}                                                 # 读取 label id -> 类别名 的映射
    label_mapping_path = 'data/comment_classify/label_mapping.txt'
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label_id, label_name = line.split(',')
                label_mapping[int(label_id)] = label_name

    sentences = [
        '外表跟图片不太像，而且号码偏大一些，穿着宽松而且裤腿不是很长，除了穿着暖和外还凑合吧，可能我个人试着不太合适',
        '不好看，不值这个价钱',
        '房间超级小，根本就不值688元的价格，特别是在广州这样一个酒店业十分发达的城市，酒店服务差，入住登记时强调要安静的房间。',
    ]
    res = inference(model, tokenizer, sentences, device)
    print('res: ', res)
    # * rich.print 会把 [xxx] 当成标记语言解析，这里用 markup=False 的 Console 原样输出
    console = Console(markup=False)
    for sentence, label_id in zip(sentences, res):
        console.print(f'[label={label_id} -> {label_mapping.get(label_id, "未知")}] {sentence}')