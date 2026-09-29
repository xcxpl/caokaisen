# 任务二、图上的链路预测（Link Prediction）

基于 **PyTorch Geometric (PyG)** 实现链路预测，对比 4 个主流 GNN 编码器在
**全图训练** 与 **LinkNeighborLoader 子图采样训练** 下的性能与运行时间。

---

## 1. 目录结构

```
任务二/
├── data/
│   ├── Cora/  Citeseer/  Computers/
├── code/
│   ├── common.py         # 数据集、GNN 编码器、点积解码器、边划分、训练/评估
│   ├── run.py            # 单组实验入口
│   └── run_all.py        # 批量实验编排（4 个阶段）
└── README.md
```

## 2. 任务设定

* **边划分**：PyG `RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
  split_labels=True, add_negative_train_samples=True, neg_sampling_ratio=1.0)`
    * 训练 85% / 验证 5% / 测试 10%，训练集自带等量负样本
    * `val_data.edge_index` / `test_data.edge_index` **只包含训练边**，因此评估无信息泄漏
* **模型**：GNN 编码器（GCN / GAT / GraphSAGE / GIN）+ **点积解码器**
  `score(u,v) = <h_u, h_v>`，损失函数 `BCEWithLogitsLoss`
* **指标**：ROC-AUC、Average Precision（AP）
* 隐藏维度 64、2 层、Dropout 0.5、Adam lr 0.01、weight_decay 5e-4、100 epoch
* 两种训练模式都在**全图**上做验证/测试推理，保证只有「训练方式」这一个变量

## 3. 训练与测试脚本

```bash
cd code

# 全图训练
python run.py --dataset Cora   --model gcn  --mode full   --verbose
# 子图采样训练
python run.py --dataset Cora   --model gcn  --mode sample --verbose
# Amazon 10 组划分（边划分随机种子随 split_id 变化）
python run.py --dataset Amazon --model gat  --mode full --all-splits

# 超参
python run.py --dataset Citeseer --model sage --mode sample \
    --layers 3 --lr 0.005 --batch-size 512 --num-neighbors 15
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--dataset` | `Cora` / `Citeseer` / `Amazon` | Cora |
| `--model` | `gcn` / `gat` / `sage` / `gin` | gcn |
| `--mode` | `full` / `sample` | full |
| `--epochs` `--lr` `--layers` `--hidden` `--dropout` | 训练超参 | 100 / 0.01 / 2 / 64 / 0.5 |
| `--batch-size` `--num-neighbors` | 采样器参数 | 1024 / 10 |
| `--all-splits` | 跑数据集所有划分 | 关 |

### 批量实验

```bash
python run_all.py              # 全部阶段（断点续跑）
python run_all.py --stages 1   # 只跑主对比
```

| 阶段 | 内容 | 输出 |
|---|---|---|
| stage1 | 3 数据集 × 4 模型 × {全图, 采样} | `../results/task2_main.csv` |
| stage2 | Cora × 4 模型 × lr ∈ {0.1,0.01,0.001} | `../results/task2_lr.csv` |
| stage3 | Cora × 4 模型 × layers ∈ {1,2,3} | `../results/task2_layers.csv` |
| stage4 | Cora × GCN × batch ∈ {256,1024,4096} × neighbors ∈ {5,10,20} | `../results/task2_sampler.csv` |

## 4. 关于 Flickr 的说明

同任务一：PyG 的 Flickr 下载源（Google Drive）在本机网络下不可达，
改用 **Amazon-Computers** 作为大规模图。`common.py` 保留 `Flickr` 分支。

## 5. 环境

```bash
pip install -r ../requirements.txt
pip install pyg-lib torch-sparse torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

## 6. 实现要点与踩坑记录

* **PyG 2.8 的 `RandomLinkSplit` 行为变化**：必须显式传入 `split_labels=False`，
  否则会报 `'GlobalStorage' object has no attribute 'edge_label'`。本任务用
  `split_labels=False`，正负样本以 `edge_label` / `edge_label_index` 的形式一并返回。
* **`LinkNeighborLoader` 同样应在 CPU 上采样**，图不要提前搬到 GPU。
* **Amazon 的采样训练要用大 batch**：训练边共 417 964 条，用
  `batch_size=16384` + `num_neighbors=5` 时每个 epoch 约 1.2 s
  （`batch_size=1024` 时需要约 9 s），显存峰值仅 ~400 MB，
  3 个划分的 AUC 与全图训练基本持平。
* **评估统一在全图上做**：val/test 的 `edge_index` 只含训练边，
  因此指标不存在信息泄漏。
