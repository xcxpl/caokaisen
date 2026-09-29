# !/usr/bin/env python3
r"""
对训练好的文本分类模型做完整评测：整体指标 + 各类别指标 + 混淆矩阵图。

用法:
    python evaluate.py \
        --model_path "checkpoints/comment_classify/model_best" \
        --dev_path "data/comment_classify/dev.txt" \
        --label_mapping_path "data/comment_classify/label_mapping.txt" \
        --num_labels 8

Author: WorkBuddy
Date: 2026/09/29
"""
import os
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')                                                       # 无界面环境下必须使用非交互后端
import matplotlib.pyplot as plt
# 中文字体配置：优先使用 Windows 自带的中文字体，避免中文标签显示为方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from rich.console import Console
from rich.table import Table

from utils import TextClassificationDataset


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='checkpoints/comment_classify/model_best', type=str, help="模型存放目录。")
parser.add_argument("--dev_path", default='data/comment_classify/dev.txt', type=str, help="验证集路径。")
parser.add_argument("--label_mapping_path", default='data/comment_classify/label_mapping.txt', type=str, help="label id -> 类别名 映射文件。")
parser.add_argument("--num_labels", default=8, type=int, help="类别数。")
parser.add_argument("--max_seq_len", default=96, type=int, help="最大序列长度（需与训练时保持一致）。")
parser.add_argument("--batch_size", default=32, type=int, help="推理 batch size。")
parser.add_argument('--device', default='cpu', type=str, help="推理设备。")
parser.add_argument("--output_dir", default='logs/comment_classify', type=str, help="评测产物输出目录。")
args = parser.parse_args()

console = Console()


def load_label_mapping(path: str) -> dict:
    """
    读取 label id -> 类别名 的映射。
    """
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_id, label_name = line.split(',', 1)
            mapping[int(label_id)] = label_name
    return mapping


def plot_confusion_matrix(cm: np.ndarray, labels: list, save_path: str) -> None:
    """
    绘制混淆矩阵热力图。注意 cm[i][j] 表示「真实为 i、预测为 j」。
    """
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Confusion Matrix (dev set)', fontsize=12)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    label_mapping = load_label_mapping(args.label_mapping_path)

    console.print(f'[bold]加载模型[/]: {args.model_path}', highlight=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(args.device).eval()

    dataset = TextClassificationDataset(args.dev_path, tokenizer, args.max_seq_len)
    dataloader = DataLoader(dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    golds, preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                input_ids=batch['input_ids'].to(args.device),
                token_type_ids=batch['token_type_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device),
            ).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            golds.extend(batch['labels'].numpy().tolist())

    classes = list(range(args.num_labels))
    cm = confusion_matrix(golds, preds, labels=classes)

    overall = {
        'num_samples': len(golds),
        'accuracy': round(accuracy_score(golds, preds), 4),
        'precision_weighted': round(precision_score(golds, preds, average='weighted', zero_division=0), 4),
        'recall_weighted': round(recall_score(golds, preds, average='weighted', zero_division=0), 4),
        'f1_weighted': round(f1_score(golds, preds, average='weighted', zero_division=0), 4),
        'f1_macro': round(f1_score(golds, preds, average='macro', zero_division=0), 4),
    }

    # ---- 各类别指标（基于混淆矩阵计算，支持类别名） ----
    per_class = {}
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        support = int(cm[i, :].sum())
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        per_class[str(cls)] = {
            'name': label_mapping.get(cls, f'class_{cls}'),
            'support': support,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }

    # ---- 打印整体指标 ----
    console.rule('[bold]Overall Metrics (dev set)')
    t = Table(show_header=True, header_style='bold cyan')
    t.add_column('metric'); t.add_column('value', justify='right')
    for k, v in overall.items():
        t.add_row(k, str(v))
    console.print(t)

    # ---- 打印各类别指标 ----
    console.rule('[bold]Per-class Metrics')
    t2 = Table(show_header=True, header_style='bold cyan')
    for col in ['id', 'label', 'support', 'precision', 'recall', 'f1']:
        t2.add_column(col, justify='right' if col not in ('label',) else 'left')
    for cls in classes:
        m = per_class[str(cls)]
        t2.add_row(str(cls), m['name'], str(m['support']),
                   f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}")
    console.print(t2)

    # ---- 打印混淆矩阵（行=真实, 列=预测） ----
    console.rule('[bold]Confusion Matrix (rows=true, cols=pred)')
    header = '     ' + ''.join(f'{c:>6}' for c in classes)
    console.print(header, highlight=False, markup=False)
    for i, cls in enumerate(classes):
        row = f'{cls:>4} ' + ''.join(f'{int(v):>6}' for v in cm[i])
        console.print(row, highlight=False, markup=False)

    # ---- 落盘 ----
    cm_png = os.path.join(args.output_dir, 'confusion_matrix.png')
    axis_labels = [label_mapping.get(c, f'class_{c}') for c in classes]
    plot_confusion_matrix(cm, axis_labels, cm_png)

    metrics_json = os.path.join(args.output_dir, 'eval_metrics.json')
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump({'overall': overall, 'per_class': per_class,
                   'confusion_matrix': cm.tolist(), 'classes': classes},
                  f, ensure_ascii=False, indent=2)

    console.print(f'\n[green]已保存[/]: {cm_png}')
    console.print(f'[green]已保存[/]: {metrics_json}')


if __name__ == '__main__':
    main()
