# -*- coding: utf-8 -*-
"""
任务一 / 任务二 公共模块
------------------------
- 数据集加载（Planetoid: Cora/Citeseer；Flickr）
- GNN 模型库（GCN / GAT / GraphSAGE / GIN）
- 统一的训练、评估、计时工具

约定：所有模型都拆成 "编码器 + 任务头"，方便任务一（节点分类）与
任务二（链路预测）复用同一套编码器做公平对比。
"""
import os
import time
import json
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
    """上下文计时器，单位秒。"""

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

# 数据集 -> 说明
DATASET_INFO = {
    "Cora": "Planetoid/Cora — 2708 节点, 10556 有向边, 1433 维特征, 7 类",
    "Citeseer": "Planetoid/Citeseer — 3327 节点, 9104 有向边, 3703 维特征, 6 类",
    "Amazon": "Amazon-Computers — 13752 节点, 491722 有向边, 767 维特征, 10 类 (10 个公开划分)",
    "Flickr": "Flickr — 89250 节点, 899756 边, 500 维特征, 7 类（官方源 Google Drive 不可达，见 README）",
}

# 每个数据集的推荐训练超参
#   gat_heads：GAT 隐藏层注意力头数。本地显卡仅 4GB 显存，Amazon 图较大
#   （49 万条有向边），8 头会导致 OOM，因此降为 2 头。
DATASET_CFG = {
    "Cora": dict(epochs=200, lr=0.01, hidden=64, gat_heads=8),
    "Citeseer": dict(epochs=200, lr=0.01, hidden=64, gat_heads=8),
    "Amazon": dict(epochs=100, lr=0.01, hidden=64, gat_heads=1),
}


def make_stratified_splits(data, num_splits=10, train_per_class=20,
                           val_per_class=30, seed=42):
    """
    Planetoid 风格的分层随机划分：
      每类 20 个节点作训练、30 个作验证、其余作测试，重复 num_splits 组。
    用于本身不带官方划分的数据集（如 Amazon-Computers）。
    """
    y = data.y.view(-1).cpu().numpy()
    n = len(y)
    rng = np.random.RandomState(seed)
    tr = np.zeros((n, num_splits), dtype=bool)
    va = np.zeros((n, num_splits), dtype=bool)
    te = np.zeros((n, num_splits), dtype=bool)
    for s in range(num_splits):
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            k_tr = min(train_per_class, len(idx))
            k_va = min(val_per_class, len(idx) - k_tr)
            tr[idx[:k_tr], s] = True
            va[idx[k_tr:k_tr + k_va], s] = True
            te[idx[k_tr + k_va:], s] = True
    data.train_mask = torch.from_numpy(tr)
    data.val_mask = torch.from_numpy(va)
    data.test_mask = torch.from_numpy(te)
    return data


_LOAD_CACHE = {}


def load_dataset(name: str, root: str):
    """
    返回 dict(data=..., num_features=..., num_classes=...)。root 为统一的 data 根目录，
    PyG 数据集会自动在 root 下建立 <数据集名>/raw、<数据集名>/processed。

    同一进程内重复调用会命中缓存（大图 data.pt 读取较慢）。
    注意：缓存里的 Data 始终保持在 CPU 上，调用方若要训练请先 .clone()。

    - Cora / Citeseer : Planetoid 内置公开划分 (public split)
    - Amazon          : Amazon-Computers，按经典协议生成 10 组分层划分
    - Flickr          : 数据集自带 train/val/test mask（本机网络不可达，见 README）
    """
    key = (name, os.path.abspath(root))
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]

    from torch_geometric.datasets import Planetoid, Flickr, Amazon
    from torch_geometric.transforms import NormalizeFeatures

    key = name.lower()
    if key in ("cora", "citeseer"):
        display = {"cora": "Cora", "citeseer": "Citeseer"}[key]
        ds = Planetoid(root=root, name=display, transform=NormalizeFeatures())
        data = ds[0]
    elif key == "amazon":
        ds = Amazon(root=root, name="Computers", transform=NormalizeFeatures())
        data = ds[0]
        # PyG 的 Amazon 不含官方划分，这里按经典协议生成 10 组分层划分
        if not hasattr(data, "train_mask"):
            make_stratified_splits(data, num_splits=10)
    elif key == "flickr":
        ds = Flickr(root=root)
        data = ds[0]
        data.x = F.normalize(data.x, p=2, dim=1)
    else:
        raise ValueError(f"不支持的数据集: {name}")

    info = {
        "data": data,
        "num_features": ds.num_features,
        "num_classes": ds.num_classes,
        "name": name,
    }
    _LOAD_CACHE[key] = info
    return info


def get_masks(data, split_id=0):
    """
    统一取出 1 维 bool 的 train/val/test mask。

    - Planetoid / Flickr：mask 本身就是 1 维
    - Amazon / Coauthor 等多划分数据集：mask 形状为 [N, K]，取第 split_id 个划分
    返回 (train, val, test, num_splits)
    """
    def _flat(m):
        m = m.to(torch.bool)
        if m.dim() > 1:
            n_splits = m.size(1)
            return m[:, split_id], n_splits
        return m.view(-1), 1

    tr, k = _flat(data.train_mask)
    va, _ = _flat(data.val_mask)
    te, _ = _flat(data.test_mask)
    return tr, va, te, k


# --------------------------------------------------------------------------
# 模型库：GCN / GAT / GraphSAGE / GIN
# --------------------------------------------------------------------------


def make_conv(kind: str, in_dim: int, out_dim: int, is_last: bool = False,
              gat_heads: int = 8):
    """构造单层卷积，4 个模型超参尽量对齐（隐藏维度统一由外部控制）。"""
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

    kind = kind.lower()
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "gat":
        # 隐藏层多头注意力（多头平均，输出维度不变），输出层单头
        heads = 1 if is_last else gat_heads
        return GATConv(in_dim, out_dim, heads=heads, concat=False,
                       add_self_loops=True)
    if kind == "sage":
        return SAGEConv(in_dim, out_dim, aggr="mean")
    if kind == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        return GINConv(mlp, train_eps=True)
    raise ValueError(f"不支持的模型: {kind}")


class GNNEncoder(nn.Module):
    """
    L 层消息传递编码器。
    num_layers=2 表示 2 层卷积（对应 2-hop 感受野）。
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2,
                 kind="gcn", dropout=0.5, use_bn=True, gat_heads=8):
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


class NodeClassifier(nn.Module):
    """节点分类：编码器 + 线性分类头。"""

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2,
                 kind="gcn", dropout=0.5, gat_heads=8):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, hidden_dim,
                                  num_layers=num_layers, kind=kind,
                                  dropout=dropout, gat_heads=gat_heads)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.head(z)


# --------------------------------------------------------------------------
# 训练 / 评估
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate_node_clf(model, data, mask, device):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    return correct / int(mask.sum())


def train_full_graph(model, data, train_mask, val_mask, test_mask, device,
                     epochs=200, lr=0.01, weight_decay=5e-4, verbose=True):
    """全图训练。返回 (最佳验证准确率, 对应测试准确率, 训练耗时, 每轮耗时列表)"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    best_val, best_test = 0.0, 0.0
    epoch_times = []
    timer = Timer()
    with timer:
        for epoch in range(1, epochs + 1):
            et = Timer()
            with et:
                model.train()
                optimizer.zero_grad()
                logits = model(data.x, data.edge_index)
                loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
                loss.backward()
                optimizer.step()
            epoch_times.append(et.t)

            val_acc = evaluate_node_clf(model, data, val_mask, device)
            if val_acc > best_val:
                best_val = val_acc
                best_test = evaluate_node_clf(model, data, test_mask, device)

            if verbose and epoch % 20 == 0:
                print(f"    epoch {epoch:3d} | loss {loss.item():.4f} "
                      f"| val {val_acc:.4f} | best_test {best_test:.4f}",
                      flush=True)
    return best_val, best_test, timer.t, epoch_times


def train_sampled_graph(model, data, train_mask, val_mask, test_mask, device,
                        epochs=200, lr=0.01, weight_decay=5e-4,
                        batch_size=1024, num_neighbors=10, verbose=True):
    """
    基于框架自带 Sampler（NeighborLoader）的子图采样训练。

    返回 (最佳验证准确率, 对应测试准确率, 训练耗时, 每轮耗时列表)。

    说明：模型质量统一在 **全图** 上做推理评估（这也是 PyG 官方推荐的
    做法），因此两种训练模式的差异只体现在「训练方式 + 训练耗时」上。
    """
    from torch_geometric.loader import NeighborLoader

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    # 采样在 CPU 上完成（PyG 官方推荐）：把整图放到 GPU 再采样会非常慢，
    # 且大图上极易显存溢出。评估时才把整图搬到 GPU。
    eval_data = data.clone().to(device)
    val_mask_d = val_mask.to(device)
    test_mask_d = test_mask.to(device)

    n_layers = model.encoder.num_layers
    loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbors] * n_layers,
        batch_size=batch_size,
        input_nodes=train_mask,
        shuffle=True,
    )

    best_val, best_test = 0.0, 0.0
    epoch_times = []
    timer = Timer()
    with timer:
        for epoch in range(1, epochs + 1):
            et = Timer()
            total_loss, nb = 0.0, 0
            with et:
                model.train()
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index)
                    # NeighborLoader 把种子节点放在最前面
                    bs = batch.batch_size
                    loss = F.cross_entropy(out[:bs], batch.y[:bs])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    nb += 1
            epoch_times.append(et.t)

            val_acc = evaluate_node_clf(model, eval_data, val_mask_d, device)
            if val_acc > best_val:
                best_val = val_acc
                best_test = evaluate_node_clf(model, eval_data, test_mask_d, device)

            if verbose and epoch % 20 == 0:
                print(f"    epoch {epoch:3d} | loss {total_loss / max(nb,1):.4f} "
                      f"| val {val_acc:.4f} | best_test {best_test:.4f}",
                      flush=True)
    return best_val, best_test, timer.t, epoch_times


@torch.no_grad()
def measure_inference_time(model, data, device, repeats=5):
    """全图推理耗时（毫秒，取平均）。"""
    model.eval()
    d = data.to(device)
    for _ in range(2):  # 预热
        _ = model(d.x, d.edge_index)
    t = Timer()
    with t:
        for _ in range(repeats):
            _ = model(d.x, d.edge_index)
    return t.t / repeats * 1000.0

