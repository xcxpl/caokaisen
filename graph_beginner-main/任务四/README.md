# 任务四、知识图谱补全（Knowledge Graph Completion）

参考 `kge_framework` 的组织方式，**自行实现** 三个经典知识图谱补全模型：
**TransE**、**RotatE**、**ConvE**，并在标准数据集上完成 filtered 排序评估。

---

## 1. 目录结构

```
任务四/
├── data/
│   ├── countries_S1/     # 小规模数据集（快速验证）
│   └── wn18rr/           # 主数据集
├── code/
│   ├── download_data.py  # 数据集下载（curl 优先，urllib 兜底）
│   ├── common.py         # KG 数据、TransE / RotatE / ConvE、训练与 filtered 评估
│   ├── run.py            # 单组实验入口
│   └── run_all.py        # 批量实验编排（2 个阶段）
└── README.md
```

## 2. 数据集格式与划分

RotatE 官方仓库格式，每行一个三元组（`\t` 分隔）：

```
data/<name>/train.txt      # h \t r \t t
data/<name>/valid.txt
data/<name>/test.txt
data/<name>/entities.dict  # <id> \t <实体名>
data/<name>/relations.dict
```

| 数据集 | 实体 | 关系 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|---:|---:|
| countries_S1 | 271 | 2 | 1068 | 122 | 122 |
| wn18rr | 40 943 | 11 | 86 835 | 3 034 | 3 134 |

> 训练/验证/测试集是官方给定的固定划分（不是随机切分）；评估时会把
> train + valid + test 的全部三元组作为「已知真值」用于 **filtered** 排序。

## 3. 三个模型的原理与实现要点

| 模型 | 核心思想 | 打分函数 | 训练目标 |
|---|---|---|---|
| **TransE** | 平移假设：`h + r ≈ t` | `-‖h + r − t‖₂` | margin ranking：`max(0, γ + d_pos − d_neg)`，随机替换头/尾做负采样，实体嵌入每步 L2 归一化 |
| **RotatE** | 复数空间旋转：`h ∘ r ≈ t`，关系建模为相位 `r = e^{iθ}`，可表达对称/反对称/逆/组合四种关系模式 | `-‖h∘r − t‖₂` | 同 TransE（本实现采用均匀负采样，未用论文中的自对抗采样） |
| **ConvE** | 把头实体和关系嵌入 reshape 成 2D 图像，拼接后做 2D 卷积，再与全部实体嵌入做内积 | `f(conv([ĥ;r̂])) · e_t` | **1-N 打分**：一次前向对全部实体做 BCE 二分类（正样本 + 随机替换尾实体的负样本） |

实现细节：
* `score_all_tails(h, r)` 对所有模型统一返回 `[B, num_entities]` 的打分矩阵，
  用范数展开（`‖x−t‖² = ‖x‖² + ‖t‖² − 2x·t`）避免 `E×d×B` 的显存爆炸。
* TransE / RotatE 提供逐三元组的精确距离 `score_triples` 用于 margin 损失。
* 评估为 **filtered 尾实体排序**：把该 `(h, r)` 下其它已知真尾实体的分数置为 `-inf`
  后再计算排名，最后统计 MRR、Hits@1/3/10。

## 4. 训练与测试脚本

### 4.1 下载数据

```bash
cd code
python download_data.py                       # countries_S1 + wn18rr
python download_data.py --datasets wn18rr
```

### 4.2 单组实验

```bash
python run.py --dataset countries_S1 --model transe --verbose
python run.py --dataset countries_S1 --model rotate  --epochs 300
python run.py --dataset wn18rr       --model conve   --epochs 60
python run.py --dataset wn18rr       --model transe  --dim 100 --lr 1e-3
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--dataset` | `countries_S1` / `wn18rr` | wn18rr |
| `--model` | `transe` / `rotate` / `conve` | transe |
| `--epochs` `--batch-size` `--lr` `--dim` | 训练超参 | 见 `MODEL_CFG` |
| `--neg-num` | 每个正样本的负样本数（TransE/RotatE） | 1 |

输出指标：验证/测试的 `MRR`、`Hits@1`、`Hits@3`、`Hits@10`，以及每轮耗时、显存占用。

### 4.3 批量实验

```bash
python run_all.py              # 2 个阶段
python run_all.py --stages 2
```

| 阶段 | 内容 | 输出 |
|---|---|---|
| stage1 | {countries_S1, wn18rr} × {TransE, RotatE, ConvE} | `../results/task4_kge.csv` |
| stage2 | countries_S1 × {TransE, RotatE} × dim ∈ {50,100,200} | `../results/task4_dim.csv` |

## 5. 环境

```bash
pip install -r ../requirements.txt
```
（本任务只依赖 torch，不需要 PyG 的采样算子。）

## 6. 实现要点与踩坑记录

* **ConvE 的 1-N 损失必须对「全部实体」做 BCE。** 只取「1 个正样本 + 1 个负样本」
  两个 logit 会让训练信号极弱（WN18RR 上 MRR 只有 0.06）。正确做法是对
  `[B, E]` 的完整打分矩阵计算 BCE，目标向量在真实尾实体处置 `1-ε`、
  其余实体置 `ε/(E-1)`（label smoothing，ε=0.1）。
* **过滤评估要用矩阵乘法形式**：`-‖h+r-t‖²` 展开为 `|x|² + |T|² − 2xTᵀ`，
  一次 matmul 就能算完全部尾实体的分数；若直接物化 `[B, E, d]` 张量，
  WN18RR 的 40 943 个实体在 4 GB 显存下会直接 OOM。
* **ConvE 需要更长的训练预算**：WN18RR 上 80 epoch 的 MRR 为 0.0996，
  200 epoch 提升到 0.2004（仍在上升），说明瓶颈是 epoch 预算而不是模型本身。
* 负样本采样中的随机下标要显式指定 `dtype=np.int64`，否则会与
  `torch.from_numpy` 的 dtype 校验冲突。
