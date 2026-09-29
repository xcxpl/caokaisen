# -*- coding: utf-8 -*-
"""
任务三 公共模块：图分类
========================
- 数据集：TUDataset（MUTAG / PROTEINS / IMDB-BINARY）+ OGB ogbg-molhiv
- 模型：GCN / GAT / GraphSAGE / GIN  +  Avg / Max / Min 三种池化
- 训练/评估工具

说明：
  * ZINC 的官方下载源（Dropbox）在本机网络下不可达（502），因此用同为图级任务的
    OGB `ogbg-molhiv` 替代，见 tasks README。
  * IMDB-BINARY 不含节点特征，按通行做法补上「节点度 one-hot」特征。
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def peak_gpu_mem_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def reset_gpu_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class Timer:
    def __init__(self):
        self.t = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t += time.perf_counter() - self._t0


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------

DATASET_INFO = {
    "MUTAG": "TUDataset/MUTAG — 188 个图, 2 类, 平均 17.9 节点",
    "PROTEINS": "TUDataset/PROTEINS — 1113 个图, 2 类, 平均 39.1 节点",
    "IMDB-BINARY": "TUDataset/IMDB-BINARY — 1000 个图, 2 类, 无节点特征(补度特征)",
    "ogbg-molhiv": "OGB/ogbg-molhiv — 41127 个图, 二分类(ROC-AUC), 9 维特征",
}

# 训练超参：max_epochs / batch_size / 任务类型
DATASET_CFG = {
    "MUTAG": dict(epochs=100, batch_size=32, lr=0.01, task="cls", hidden=64),
    "PROTEINS": dict(epochs=100, batch_size=64, lr=0.01, task="cls", hidden=64),
    "IMDB-BINARY": dict(epochs=100, batch_size=64, lr=0.01, task="cls", hidden=64),
    "ogbg-molhiv": dict(epochs=60, batch_size=128, lr=0.001, task="binclf", hidden=128),
}

# ogbg-molhiv 抽样子集规模（本地 4GB 显卡，全量 41k 图训练过慢）
OGB_SUBSET = dict(train=8000, val=2000, test=2000)


def _degree_features(data, max_degree=100):
    """给单个图添加节点度 one-hot 特征（返回新的 Data）。"""
    from torch_geometric.utils import degree

    _, col = data.edge_index
    deg = degree(col, data.num_nodes, dtype=torch.long)
    deg = torch.clamp(deg, 0, max_degree)
    data.x = F.one_hot(deg, num_classes=max_degree + 1).float()
    return data


class AddDegreeFeatures:
    """
    TUDataset 的 transform：为「没有节点特征」的数据集（如 IMDB-BINARY）
    补齐节点度 one-hot 特征。

    必须用 transform 的方式在 ``__getitem__`` 时注入，因为 PyG 的
    ``InMemoryDataset.__getitem__`` 每次返回的是切片副本，直接改返回值
    （下面的 ``_add_degree_features`` 旧写法）不会写回数据集，后续再取
    仍然是 ``x=None``。
    """

    def __init__(self, max_degree=100):
        self.max_degree = max_degree

    def __call__(self, data):
        return _degree_features(data, self.max_degree)


def load_graph_dataset(name, root, seed=42):
    """
    返回 (train_list, val_list, test_list, in_dim, out_dim, task)

    root 为任务三的 data 根目录：TUDataset 放在 <root>/TU，OGB 放在 <root>/OGB。
    """
    from torch_geometric.datasets import TUDataset

    if name in ("MUTAG", "PROTEINS", "IMDB-BINARY"):
        tu_root = os.path.join(root, "TU")
        ds = TUDataset(root=tu_root, name=name)
        if ds.num_node_features == 0 or ds.num_features == 0:
            # 无特征数据集（IMDB-BINARY）：用 transform 注入度特征
            ds = TUDataset(root=tu_root, name=name,
                           transform=AddDegreeFeatures(max_degree=100))
            in_dim = 101
        else:
            in_dim = ds.num_features
        out_dim = max(int(ds.num_classes), 2)

        # 固定 80% / 10% / 10% 随机划分，保证不同模型、不同池化方法可比
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(ds))
        n_train = int(0.8 * len(ds))
        n_val = int(0.1 * len(ds))
        tr = [ds[i] for i in idx[:n_train]]
        va = [ds[i] for i in idx[n_train:n_train + n_val]]
        te = [ds[i] for i in idx[n_train + n_val:]]
        return tr, va, te, in_dim, out_dim, "cls"

    if name == "ogbg-molhiv":
        from ogb.graphproppred import PygGraphPropPredDataset

        ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
        split = ds.get_idx_split()
        rng = np.random.RandomState(seed)
        tr_idx = rng.permutation(split["train"].numpy())[:OGB_SUBSET["train"]]
        va_idx = rng.permutation(split["valid"].numpy())[:OGB_SUBSET["val"]]
        te_idx = rng.permutation(split["test"].numpy())[:OGB_SUBSET["test"]]

        def _prep(i):
            # OGB 的节点特征 x 是 int64（原子序数等整数特征），
            # 直接送进 nn.Linear 会报 dtype 不匹配，这里统一转 float。
            g = ds[int(i)]
            g.x = g.x.float()
            return g

        tr = [_prep(i) for i in tr_idx]
        va = [_prep(i) for i in va_idx]
        te = [_prep(i) for i in te_idx]
        in_dim = ds.num_node_features      # 9
        out_dim = 1                        # 二分类
        return tr, va, te, in_dim, out_dim, "binclf"

    raise ValueError(f"不支持的数据集: {name}")


# --------------------------------------------------------------------------
# 模型：GNN + 池化
# --------------------------------------------------------------------------


def make_conv(kind, in_dim, out_dim, is_last=False, gat_heads=8):
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

    kind = kind.lower()
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "gat":
        heads = 1 if is_last else gat_heads
        return GATConv(in_dim, out_dim, heads=heads, concat=False,
                       add_self_loops=True)
    if kind == "sage":
        return SAGEConv(in_dim, out_dim, aggr="mean")
    if kind == "gin":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        return GINConv(mlp, train_eps=True)
    raise ValueError(f"不支持的模型: {kind}")


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, kind="gcn",
                 dropout=0.5, gat_heads=8, use_bn=True):
        super().__init__()
        self.kind = kind.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(make_conv(self.kind, dims[i], dims[i + 1],
                                        is_last=(i == num_layers - 1),
                                        gat_heads=gat_heads))
            if use_bn and i < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))
            else:
                self.norms.append(nn.Identity())

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


POOLINGS = ["avg", "max", "min"]


def global_min_pool(x, batch, size=None):
    """PyG 的 torch_geometric.nn 未提供 Min 池化，这里用 scatter(reduce='min') 实现。"""
    from torch_geometric.utils import scatter

    size = int(batch.max().item()) + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="min")


def pool_graph(z, batch, pool="avg"):
    from torch_geometric.nn import global_mean_pool, global_max_pool

    if pool == "avg":
        return global_mean_pool(z, batch)
    if pool == "max":
        return global_max_pool(z, batch)
    if pool == "min":
        return global_min_pool(z, batch)
    raise ValueError(f"不支持的池化方式: {pool}")


class GraphClassifier(nn.Module):
    """GNN 编码器 + READOUT 池化 + 线性分类头"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, kind="gcn",
                 pool="avg", dropout=0.5, gat_heads=8):
        super().__init__()
        self.pool = pool
        self.encoder = GNNEncoder(in_dim, hidden_dim, hidden_dim,
                                  num_layers=num_layers, kind=kind,
                                  dropout=dropout, gat_heads=gat_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index)
        g = pool_graph(z, batch, self.pool)
        return self.head(g)


# --------------------------------------------------------------------------
# 训练 / 评估
# --------------------------------------------------------------------------


def _step_loss(task, out, y):
    if task == "cls":
        return F.cross_entropy(out, y.view(-1))
    # 二分类（多标签形式，可能含 NaN 标签，需屏蔽）
    out = out.view(-1)
    y = y.view(-1).float()
    mask = ~torch.isnan(y)
    if mask.sum() == 0:
        return None
    return F.binary_cross_entropy_with_logits(out[mask], y[mask])


@torch.no_grad()
def evaluate_clf(model, loader, device, task):
    from sklearn.metrics import roc_auc_score

    model.eval()
    correct = total = 0
    all_logit, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        if task == "cls":
            pred = out.argmax(dim=-1)
            correct += int((pred == batch.y.view(-1)).sum())
            total += int(batch.y.numel())
        else:
            all_logit.append(out.view(-1).cpu())
            all_y.append(batch.y.view(-1).cpu())
    if task == "cls":
        return correct / max(total, 1)
    logits = torch.cat(all_logit).numpy()
    y = torch.cat(all_y).numpy()
    mask = ~np.isnan(y)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return 0.5
    return float(roc_auc_score(y[mask], logits[mask]))


def train_graph_clf(model, train_loader, val_loader, test_loader, device,
                    task="cls", epochs=100, lr=0.01, weight_decay=5e-4,
                    verbose=True):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    best_val, best_test = -1.0, 0.0
    epoch_times = []
    timer = Timer()
    with timer:
        for epoch in range(1, epochs + 1):
            et = Timer()
            total_loss, nb = 0.0, 0
            with et:
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = _step_loss(task, out, batch.y)
                    if loss is None:
                        continue
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    nb += 1
            epoch_times.append(et.t)
            val = evaluate_clf(model, val_loader, device, task)
            if val > best_val:
                best_val = val
                best_test = evaluate_clf(model, test_loader, device, task)
            if verbose and epoch % 20 == 0:
                print(f"    epoch {epoch:3d} | loss {total_loss / max(nb,1):.4f} "
                      f"| val {val:.4f} | best_test {best_test:.4f}", flush=True)
    return best_val, best_test, timer.t, epoch_times
