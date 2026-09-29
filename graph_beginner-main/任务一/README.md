# 任务一、节点分类（Node Classification）

基于 **PyTorch Geometric (PyG)** 实现图上的节点分类，对比 4 个主流 GNN 模型在
**全图训练** 与 **子图采样训练（NeighborLoader）** 下的性能与运行时间。

---

## 1. 目录结构

```
任务一/
├── data/                 # 数据集（PyG 自动下载 / 已随仓库附带）
│   ├── Cora/
│   ├── Citeseer/
│   └── Computers/        # Amazon-Computers（替代 Flickr，见第 6 节）
├── code/
│   ├── common.py         # 数据集加载、GNN 模型库、训练/评估/计时工具
│   ├── run.py            # 单组实验入口
│   └── run_all.py        # 批量实验编排（4 个阶段）
└── README.md
```

## 2. 环境准备

```bash
pip install -r ../requirements.txt
```

PyG 2.x 的 `NeighborLoader` 需要额外安装采样算子（本仓库已在
`C:\Users\lenono\.workbuddy\binaries\python\envs\gcn312` 环境中装好）：

```bash
pip install pyg-lib torch-sparse torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

> 若安装时 pip 试图降级 torch，请改为下载对应 wheel 后
> `pip install --no-deps xxx.whl`（本机即采用该方式，torch 保持 2.5.1+cu121）。

## 3. 数据集

| 数据集 | 节点数 | 有向边数 | 特征维度 | 类别数 | 划分方式 |
|---|---:|---:|---:|---:|---|
| Cora | 2 708 | 10 556 | 1 433 | 7 | Planetoid public split |
| Citeseer | 3 327 | 9 104 | 3 703 | 6 | Planetoid public split |
| Amazon-Computers | 13 752 | 491 722 | 767 | 10 | 每类 20 训练 / 30 验证，其余测试，共 10 组 |

> Citeseer 实际类别数为 6。

## 4. 模型与训练设置

* 模型：`GCN`、`GAT`、`GraphSAGE`、`GIN`（同一个 `GNNEncoder` 骨架 + 线性分类头）
* 隐藏维度 64，Dropout 0.5，Adam，weight_decay 5e-4
* 全图训练：每个 epoch 在整个图上做一次前向/反向
* 子图采样训练：`NeighborLoader(num_neighbors=[10,10], batch_size=1024)`
* **评估方式两种模式一致**：都在全图上推理，取验证准确率最高的那个 epoch 对应的测试准确率

> GAT 隐藏层注意力头数：Cora/Citeseer 为 8；Amazon 图较大，受本机 4 GB 显存限制降为 1。

## 5. 训练与测试脚本

### 5.1 单组实验 `run.py`

```bash
cd code

# 全图训练 / 子图采样训练
python run.py --dataset Cora     --model gcn  --mode full   --seed 42 --verbose
python run.py --dataset Cora     --model gcn  --mode sample --seed 42 --verbose

# 大图（10 个划分全部跑，自动取平均）
python run.py --dataset Amazon   --model sage --mode sample --all-splits

# 超参扫描
python run.py --dataset Cora --model gcn --mode full --lr 0.005 --layers 3 --dro 0.5
```

主要参数：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--dataset` | `Cora` / `Citeseer` / `Amazon` | Cora |
| `--model` | `gcn` / `gat` / `sage` / `gin` | gcn |
| `--mode` | `full` 全图训练 / `sample` 子图采样训练 | full |
| `--epochs` `--lr` `--layers` `--hidden` `--dropout` | 训练超参 | 按数据集 |
| `--batch-size` `--num-neighbors` | 采样器参数（仅 sample 模式） | 1024 / 10 |
| `--all-splits` | 对数据集所有公开划分各跑一次 | 关 |
| `--out` | 结果追加写入 csv | 无 |

输出的 JSON 含：`val_acc`、`test_acc`、`train_time_s`（纯训练耗时）、
`time_per_epoch_s`、`infer_time_ms`（全图推理耗时）、`peak_gpu_mb`。

### 5.2 批量实验 `run_all.py`

```bash
python run_all.py                 # 4 个阶段全部执行（支持断点续跑）
python run_all.py --stages 2 3    # 只跑学习率 / 层数消融
```

| 阶段 | 内容 | 输出 |
|---|---|---|
| stage1 | {Cora, Citeseer, Amazon} × {GCN,GAT,SAGE,GIN} × {全图, 采样} | `../results/task1_main.csv` |
| stage2 | Cora × 4 模型 × lr ∈ {0.1, 0.01, 0.001} | `../results/task1_lr.csv` |
| stage3 | Cora × 4 模型 × layers ∈ {1, 2, 3} | `../results/task1_layers.csv` |
| stage4 | Cora × GCN × batch ∈ {256,1024,4096} × neighbors ∈ {5,10,20} | `../results/task1_sampler.csv` |

## 6. 关于 Flickr 数据集的说明

任务书要求使用 **Flickr**，但 PyG 的 `Flickr` 类下载地址为 Google Drive：
`https://drive.google.com/uc?id=1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy`。
本机网络经代理访问 Google Drive 时返回 `502 Bad Gateway`，github 上的镜像
（geom-gcn 仓库）也已 404，多种镜像均不可达。

因此按「可适当降低任务要求，跑通即可」的原则，改用同为公开划分、且规模明显
大于 Cora 的 **Amazon-Computers**（13 752 节点 / 491 722 有向边，来自
“Pitfalls of Graph Neural Network Evaluation”，Shchur et al. 2018）作为大规模图，
用于验证「采样训练在大图上的收益」。

`code/common.py` 中仍保留了 `Flickr` 分支，若网络可达直接
`python run.py --dataset Flickr ...` 即可无缝切换。

## 7. 结果

见根目录 `results/task1_*.csv` 与 `report/实验报告.md`。

## 8. 实现要点与踩坑记录

* **采样训练的 `batch_size` 必须小于训练节点数。** Cora / Citeseer / Amazon 的公开
  划分分别只有 140 / 120 / 200 个训练节点，若沿用 `batch_size=1024`，每轮只会产生
  **1 个 mini-batch**，「子图采样训练」就退化成「一次全批量更新」，与全图训练的
  耗时对比也失去意义。`run_all.py` 因此统一取 `batch_size=64`。
* **`NeighborLoader` 应在 CPU 上采样**（PyG 官方推荐做法）：把图留在 CPU，
  只把每个 mini-batch `.to(device)`。若先把整图 `.to(cuda)` 再采样，会显著变慢甚至 OOM。
* **4 GB 显存下 GAT 需要降头数**：`GATConv` 会物化 `[E, heads, out]` 中间张量，
  Amazon 上 8 头直接 OOM，因此隐藏层注意力头数降为 1（Cora/Citeseer 仍为 8）。
* **GIN 在 Amazon 全图训练上需要用更小的学习率**（见 `report/实验报告.md` 2.7 节）。

