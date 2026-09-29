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

工具类。

Author: pankeyu02
Date: 2022/09/13
"""
import json
import traceback

import torch
import numpy as np


def convert_example(examples: dict, tokenizer, max_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '1	这是一条正样本',
                                                            '0	这是一条负样本',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]], 
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    tokenized_output = {
            'input_ids': [], 
            'token_type_ids': [],
            'attention_mask': [],
            'labels': []
        }

    for example in examples['text']:
        try:
            label, content = example.split('\t')
            encoded_inputs = tokenizer(
                text=content,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['labels'].append(int(label))
    
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


class TextClassificationDataset(torch.utils.data.Dataset):
    """
    轻量级文本分类数据集。

    说明:
        原实现依赖 huggingface `datasets` 库（load_dataset + map），该库会在
        ~/.cache/huggingface/datasets 下做缓存、写 lock 文件、清理临时文件。
        在受限环境下这些删除动作会被安全守卫拦截导致训练中断。
        这里改为直接读取 txt 文件并在初始化阶段一次性完成分词，行为等价且更快。

    数据格式:
        每行 -> "label\\tcontent"，例如: "1\\t这个产品很好用"

    Args:
        data_path (str): 数据文件路径。
        tokenizer: transformers tokenizer。
        max_seq_len (int): 最大序列长度。
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int):
        self.data_path = data_path
        self.max_seq_len = max_seq_len

        labels, contents = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                if '\t' not in line:
                    print(f'[warn] {data_path} 第 {line_no} 行缺少分隔符 \\t，已跳过: {line[:60]}')
                    continue
                label, content = line.split('\t', 1)
                try:
                    label = int(label)
                except ValueError:
                    print(f'[warn] {data_path} 第 {line_no} 行标签非整数，已跳过: {line[:60]}')
                    continue
                labels.append(label)
                contents.append(content)

        if not labels:
            raise ValueError(f'数据集为空或格式不正确: {data_path}')

        # 一次性批量分词，比逐条处理快得多
        encoded = tokenizer(
            text=contents,
            truncation=True,
            max_length=max_seq_len,
            padding='max_length',
        )

        self.input_ids = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']
        # 部分 tokenizer（如 RoBERTa）不输出 token_type_ids，用 0 兜底
        self.token_type_ids = encoded.get('token_type_ids', [[0] * max_seq_len] * len(labels))
        self.labels = labels

        print(f'[+] 已加载 {data_path}: {len(self.labels)} 条样本 '
              f'(标签 {min(labels)} ~ {max(labels)})')

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            'input_ids': self.input_ids[index],
            'token_type_ids': self.token_type_ids[index],
            'attention_mask': self.attention_mask[index],
            'labels': self.labels[index],
        }