# 任务三、图分类（Graph Classification）

基于 **PyTorch Geometric (PyG)** 实现图级分类，重点分析
**AvgPooling / MaxPooling / MinPooling** 三种 READOUT 池化方法对性能的影响。

---

## 1. 目录结构

```
任务三/
├── data/
│   ├── TU/               # TUDataset: MUTAG / PROTEINS / IMDB-BINARY
│   └── OGB/              # ogbg-molhiv
├── code/
│   ├── common.py         # 数据集、GNN 编码器 + 池化、训练/评估
│   ├── run.py            # 单组实验入口
│   └── run_all.py        # 批量实验编排（3 个阶段）
└── README.md
```

## 2. 数据集

| 数据集 | 图数 | 节点特征 | 类别 | 任务 | 训练/验证/测试 |
|---|---:|---|---:|---|---|
| MUTAG | 188 | 7 | 2 | 分类 | 150 / 18 / 20 |
| PROTEINS | 1113 | 3 | 2 | 分类 | 890 / 111 / 112 |
| IMDB-BINARY | 1000 | 无 → 度 one-hot(101) | 2 | 分类 | 800 / 100 / 100 |
| ogbg-molhiv | 41127 | 9 | 2 | 二分类(ROC-AUC) | 8000 / 2000 / 2000（子集） |

说明：
* TU 数据集采用固定随机种子下的 **80% / 10% / 10%** 划分，保证不同模型、不同池化
  方法之间完全可比。
* **IMDB-BINARY 不含节点特征**，按通行做法补上「节点度 one-hot」（max_degree=100）。
* ogbg-molhiv 使用 **OGB 官方 scaffold split**，为控制本机耗时抽取了 8000/2000/2000 的子集。

## 3. 模型与池化

```
输入图 → GNNEncoder(L 层 GCN/GAT/SAGE/GIN) → 节点表示 z
        → READOUT 池化 (Avg / Max / Min)   → 图表示 g
        → MLP 分类头 → logits
```

* `AvgPooling` = `global_mean_pool`，`MaxPooling` = `global_max_pool`
* `MinPooling`：PyG 未提供，用 `scatter(reduce="min")` 自行实现
  （`common.global_min_pool`）
* 默认 3 层、隐藏维度 64（ogbg-molhiv 为 128）、Dropout 0.5、Adam

## 4. 训练与测试脚本

```bash
cd code

# 单组：数据集 × 模型 × 池化
python run.py --dataset MUTAG    --model gin --pool avg
python run.py --dataset MUTAG    --model gin --pool min --epochs 100
python run.py --dataset PROTEINS --model gcn --pool max --verbose
python run.py --dataset ogbg-molhiv --model sage --pool avg

# 层数 / 学习率
python run.py --dataset MUTAG --model gcn --pool avg --layers 2 --lr 0.001
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--dataset` | `MUTAG` / `PROTEINS` / `IMDB-BINARY` / `ogbg-molhiv` | MUTAG |
| `--model` | `gcn` / `gat` / `sage` / `gin` | gin |
| `--pool` | `avg` / `max` / `min` | avg |
| `--epochs` `--lr` `--layers` `--hidden` `--batch-size` | 训练超参 | 按数据集 |

### 批量实验

```bash
python run_all.py            # 3 个阶段
python run_all.py --stages 1
```

| 阶段 | 内容 | 输出 |
|---|---|---|
| stage1 | 4 数据集 × 4 模型 × 3 池化 | `../results/task3_pooling.csv` |
| stage2 | MUTAG × 4 模型 × layers ∈ {2,3,4}（AvgPool） | `../results/task3_layers.csv` |
| stage3 | MUTAG × 4 模型 × lr ∈ {0.1,0.01,0.001}（AvgPool） | `../results/task3_lr.csv` |

## 5. 关于 ZINC 数据集的说明

任务书要求 `TUDataset + ZINC`。PyG 的 `ZINC` 类下载地址为 Dropbox
（`https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1`），
本机网络经代理访问 Dropbox 返回 `502 Bad Gateway`，TU 官方站也没有可用的
ZINC/回归数据集（AQSOL、ESOL 等均 404），因此按「可适当降低任务要求，
跑通即可」的原则，改用 **OGB `ogbg-molhiv`** 作为第二个（分子级）图数据集 —
它同样带有官方划分，且规模远大于 TU 数据集，更能体现池化方法的差异。

## 6. 环境

```bash
pip install -r ../requirements.txt
pip install pyg-lib torch-sparse torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

## 7. 实现要点与踩坑记录

* **TUDataset 的 `__getitem__` 每次返回的是切片副本**，直接修改遍历得到的
  `data.x` 不会写回数据集，后续再取仍然是 `x=None`。IMDB-BINARY 的
  「节点度 one-hot」特征必须通过 `TUDataset(..., transform=AddDegreeFeatures())`
  在取数据时注入。
* **OGB 的节点特征是 `torch.int64`**（原子序数等整数编码），直接送进 `nn.Linear`
  会报 `expected mat1 and mat2 to have the same dtype, but got: __int64 != float`，
  需要先做 `x = x.float()`。
* **PyG 没有提供 `global_min_pool`**，用 `scatter(x, batch, dim=0, reduce="min")`
  自行实现 MinPooling。
* **同一 csv 内同时存在 Accuracy 与 AUC 时容易错位**（molhiv 是二分类、用 AUC），
  因此每行额外写入 `metric_name` 列，标明该行的 `test_acc` / `test_auc`
  哪一列才是有效指标。
