# 图神经网络 Beginner —— 四个任务的完整实现

本仓库按任务书要求，基于 **PyTorch Geometric (PyG)** 实现节点分类、链路预测、
图分类三个任务，并**自行实现**知识图谱补全框架（TransE / RotatE / ConvE）。

> 每个任务都可独立运行：`任务X/data` + `任务X/code` + `任务X/README.md`。

---

## 0. 项目结构

```
graph_beginner-main/
├── README.md                 # 本文件
├── requirements.txt          # 运行环境
├── 任务一/                    # 节点分类
│   ├── data/  code/  README.md
├── 任务二/                    # 链路预测
│   ├── data/  code/  README.md
├── 任务三/                    # 图分类
│   ├── data/  code/  README.md
├── 任务四/                    # 知识图谱补全
│   ├── data/  code/  README.md
├── results/                  # 全部实验结果的 csv / 日志 / 图表脚本
│   ├── task1_main.csv  task1_lr.csv  task1_layers.csv  task1_sampler.csv
│   ├── task1_gin_amazon_lr.csv     # 补充：GIN@Amazon 全图训练的学习率敏感性
│   ├── task2_main.csv  task2_lr.csv  task2_layers.csv  task2_sampler.csv
│   ├── task3_pooling.csv  task3_layers.csv  task3_lr.csv
│   ├── task4_kge.csv  task4_dim.csv
│   ├── task4_conve_long.csv        # 补充：ConvE@WN18RR 训练 200 epoch
│   ├── summary_tables.md           # 全部结果均值±标准差汇总
│   └── make_figures.py
└── report/                   # 实验报告与插图
    ├── 实验报告.md
    └── figures/*.png
```

## 1. 环境

```bash
pip install -r requirements.txt
# PyG 的采样算子（NeighborLoader / LinkNeighborLoader 必需）
pip install pyg-lib torch-sparse torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

本机实测环境：Windows 11 / Python 3.12.10 / torch 2.5.1+cu121 /
PyG 2.8.0 / RTX 3050 Laptop **4 GB 显存** / 16 核 CPU。

## 2. 一键复现

```bash
# 任务一 节点分类
cd 任务一/code && python run_all.py

# 任务二 链路预测
cd 任务二/code && python run_all.py

# 任务三 图分类
cd 任务三/code && python run_all.py

# 任务四 知识图谱
cd 任务四/code && python download_data.py && python run_all.py

# 生成图表
python results/make_figures.py
```

`run_all.py` 都带 **断点续跑**：结果已存在的实验会自动跳过，可中断后继续。

## 3. 任务一览

| 任务 | 数据集 | 模型 | 对比维度 | 结果文件 |
|---|---|---|---|---|
| 一 节点分类 | Cora, Citeseer, Amazon-Computers | GCN, GAT, GraphSAGE, GIN | 全图训练 vs `NeighborLoader` 子图采样；lr / 层数 / 采样参数 | `task1_*.csv` |
| 二 链路预测 | Cora, Citeseer, Amazon-Computers | 同上 + 点积解码器 | 全图训练 vs `LinkNeighborLoader` 子图采样；lr / 层数 / 采样参数 | `task2_*.csv` |
| 三 图分类 | MUTAG, PROTEINS, IMDB-BINARY, ogbg-molhiv | 同上 | **Avg / Max / Min 池化**；lr / 层数 | `task3_*.csv` |
| 四 知识图谱 | countries_S1, WN18RR | TransE, RotatE, ConvE | 模型对比；嵌入维度 | `task4_*.csv` |

## 4. 数据集替换说明（重要）

任务书中的两个数据集官方下载源在本机网络下不可达，按「可适当降低任务要求，
跑通即可」的原则做了等价替换，并在各任务 README 中详细说明：

| 任务书要求 | 实际问题 | 本仓库采用 |
|---|---|---|
| Flickr | PyG 的 Flickr 从 Google Drive 下载，代理返回 502；github 镜像 404 | **Amazon-Computers**（13 752 节点 / 491 722 有向边，10 组公开划分协议） |
| ZINC | PyG 的 ZINC 从 Dropbox 下载，代理返回 502；TU 站无回归数据集 | **OGB ogbg-molhiv**（41 127 个分子图，官方 scaffold split） |

代码中仍保留 `Flickr` / `ZINC` 的分支与调用方式，网络可达时可直接切换。

## 5. 硬件限制下的取舍

* 显卡仅 **4 GB 显存**，PyG 的 `GATConv` 在稠密大图上会物化 `[E, heads, out]`
  中间张量，因此 Amazon 数据集上 GAT 的隐藏层注意力头数由 8 降为 **1**（其余设置不变）。
* **子图采样在 CPU 上做**（PyG 官方推荐）：图留在 CPU，只把每个 mini-batch 搬到 GPU；
  先把整图搬上 GPU 再采样会显著变慢甚至 OOM。
* **采样训练的 `batch_size` 必须小于训练节点数**（Cora/Citeseer/Amazon 只有
  140/120/200 个训练节点），否则每轮只有 1 个 batch，采样训练会退化成一次全批量更新。
  任务一因此统一使用 `batch_size=64`。
* 任务二在 Amazon 上的采样训练使用 `batch_size=16384` + `num_neighbors=5`
  （每 epoch 约 1.2 s，显存峰值约 400 MB），并取 **3 组随机划分**；
  全图训练取全部 10 组划分。
* ogbg-molhiv 训练使用 8000 / 2000 / 2000 的官方划分子集。
* Amazon 的 10 组划分、WN18RR 的 120 epoch 等都属于「跑得动同时结论稳定」的折中。
* 详细的取舍理由、踩坑记录见各任务 README 的「实现要点与踩坑记录」一节。

## 6. 实验报告

结论、图表与参数影响分析见 [`report/实验报告.md`](report/实验报告.md)。

---

## 附：原始任务书

<details>
<summary>点击展开</summary>

### 任务一、节点分类
1. 基于现有模型框架(DGL, Pytorch_geometric)实现图上的节点分类任务
2. 使用 Cora, Citeseer, Flickr 数据集
3. 测试 GCN, GAT, GraphSAGE, GIN 模型
4. 利用框架自带的 Sampler 采样子图进行训练，并与全图训练进行性能和运行时间的对比

### 任务二、图上的链路预测
1. 基于现有模型框架(DGL, Pytorch_geometric)实现图上的链路预测
2. 使用 Cora, Citeseer, Flickr 数据集
3. 测试 GCN, GAT, GraphSAGE, GIN 模型
4. 利用框架自带的 Sampler 采样子图进行训练，并与全图训练进行性能和运行时间的对比

### 任务三、图分类
1. 基于现有模型框架(DGL, Pytorch_geometric)实现图分类任务
2. 使用 TUDataset, ZINC 数据集
3. 分析不同的池化方法对图分类性能的影响(AvgPooling, MaxPooling, MinPooling)
4. 测试 GCN, GAT, GraphSAGE, GIN 模型

### 任务四、知识图谱
1. 参考 KGE 框架与 TransE / RotatE / ConvE 论文
2. 基于知识图谱补全框架，支持 TransE, RotatE, ConvE
3. 需要了解：常见知识图谱补全模型的原理；数据集 train/valid/test 的划分

### 任务要求
1. 代码文件夹统一格式：`任务X/{data, code, README.md}` + `requirements.txt`
2. README.md 中写明训练和测试的脚本
3. 报告要求：
   1. 分析不同参数（学习率、网络层数）和不同的神经网络对性能的影响
   2. 测试全图训练和分批次训练对模型性能和运行时间的影响（任务四不需要）

</details>
