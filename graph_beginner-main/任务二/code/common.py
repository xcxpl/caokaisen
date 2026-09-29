# -*- coding: utf-8 -*-
"""
任务二 公共模块
------------------------
- 数据集加载（Planetoid: Cora/Citeseer；Amazon-Computers）
- GNN 模型库（GCN / GAT / GraphSAGE / GIN）
- 链路预测：边划分、点积解码器、全图训练 / LinkNeighborLoader 子图采样训练
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

# 每个数据集的推荐训练超参（链路预测统一 100 epoch）
DATASET_CFG = {
    "Cora": dict(epochs=100, lr=0.01, hidden=64, gat_heads=8),
    "Citeseer": dict(epochs=100, lr=0.01, hidden=64, gat_heads=8),
    "Amazon": dict(epochs=100, lr=0.01, hidden=64, gat_heads=1),
}


def make_stratified_splits(data, num_splits=10, train_per_class=20,
                           val_per_class=30, seed=42):
    """
    Planetoid 风格的分层随机划分（每类 20 训练 / 30 验证 / 其余测试），
    生成 num_splits 组。用于本身不带官方划分的数据集（如 Amazon-Computers）。
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
    返回 dict(data=..., num_features=..., num_classes=...)。root 为统一的 data 根目录。
    同一进程内重复调用会命中缓存；缓存中的 Data 保持在 CPU，调用方自行 .clone()。

    - Cora / Citeseer : Planetoid 内置公开划分 (public split)
    - Amazon          : Amazon-Computers，按经典协议生成 10 组分层划分
    - Flickr          : 数据集自带 train/val/test mask（本机网络不可达，见 README）
    """
    from torch_geometric.datasets import Planetoid, Flickr, Amazon
    from torch_geometric.transforms import NormalizeFeatures

    ckey = (name, os.path.abspath(root))
    if ckey in _LOAD_CACHE:
        return _LOAD_CACHE[ckey]

    key = name.lower()
    if key in ("cora", "citeseer"):
        display = {"cora": "Cora", "citeseer": "Citeseer"}[key]
        ds = Planetoid(root=root, name=display, transform=NormalizeFeatures())
        data = ds[0]
    elif key == "amazon":
        ds = Amazon(root=root, name="Computers", transform=NormalizeFeatures())
        data = ds[0]
        if not hasattr(data, "train_mask"):
            make_stratified_splits(data, num_splits=10)
    elif key == "flickr":
        ds = Flickr(root=root)
        data = ds[0]
        data.x = F.normalize(data.x, p=2, dim=1)
    else:
        raise ValueError(f"不支持的数据集: {name}")

    _LOAD_CACHE[ckey] = {
        "data": data,
        "num_features": ds.num_features,
        "num_classes": ds.num_classes,
        "name": name,
    }
    return _LOAD_CACHE[ckey]


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
                 kind="gcn", dropout=0.5):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, hidden_dim,
                                  num_layers=num_layers, kind=kind,
                                  dropout=dropout)
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
    data = data.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

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

            val_acc = evaluate_node_clf(model, data, val_mask, device)
            if val_acc > best_val:
                best_val = val_acc
                best_test = evaluate_node_clf(model, data, test_mask, device)

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


# ==========================================================================
#                        任务二：链路预测专用部分
# ==========================================================================


class LinkPredictor(nn.Module):
    """
    链路预测模型 = GNN 编码器 + 点积解码器。

    score(u, v) = <h_u, h_v>，训练时用 BCEWithLogitsLoss，负样本随机采样。
    """

    def __init__(self, in_dim, hidden_dim, num_layers=2, kind="gcn",
                 dropout=0.5, gat_heads=8):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, hidden_dim,
                                  num_layers=num_layers, kind=kind,
                                  dropout=dropout, gat_heads=gat_heads)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        return self.decode(z, edge_label_index)


def make_link_split(data, num_val=0.05, num_test=0.10, seed=42, disjoint_train_ratio=0.0):
    """
    按 85% / 5% / 10% 划分边，得到 (train_data, val_data, test_data)。

    - is_undirected=True：无向图只保留一个方向的边做监督信号
    - add_negative_train_samples=True：训练集自带等量负样本
    - split_labels=False：正负样本合并保存在 edge_label / edge_label_index
    - val/test 的 edge_index 只含训练边，因此评估不存在信息泄漏
    """
    from torch_geometric.transforms import RandomLinkSplit

    set_seed(seed)
    transform = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        split_labels=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        disjoint_train_ratio=disjoint_train_ratio,
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


@torch.no_grad()
def evaluate_link(model, data, device):
    """在全图的边集合上评估：返回 (ROC-AUC, Average Precision)。"""
    from sklearn.metrics import roc_auc_score, average_precision_score

    model.eval()
    d = data.to(device)
    z = model.encoder(d.x, d.edge_index)
    out = model.decode(z, d.edge_label_index).sigmoid()
    y = d.edge_label
    auc = roc_auc_score(y.cpu().numpy(), out.cpu().numpy())
    ap = average_precision_score(y.cpu().numpy(), out.cpu().numpy())
    return float(auc), float(ap)


def train_link_full(model, train_data, val_data, test_data, device,
                    epochs=100, lr=0.01, weight_decay=5e-4, verbose=True):
    """全图训练链路预测。返回 (best_val_auc, test_auc, test_ap, 训练耗时, 每轮耗时)"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    train_data = train_data.to(device)
    y_train = train_data.edge_label.float()

    best_val, best_test_auc, best_test_ap = -1.0, 0.0, 0.0
    epoch_times = []
    timer = Timer()
    with timer:
        for epoch in range(1, epochs + 1):
            et = Timer()
            with et:
                model.train()
                optimizer.zero_grad()
                z = model.encoder(train_data.x, train_data.edge_index)
                out = model.decode(z, train_data.edge_label_index)
                loss = F.binary_cross_entropy_with_logits(out, y_train)
                loss.backward()
                optimizer.step()
            epoch_times.append(et.t)

            val_auc, _ = evaluate_link(model, val_data, device)
            if val_auc > best_val:
                best_val = val_auc
                best_test_auc, best_test_ap = evaluate_link(model, test_data, device)

            if verbose and epoch % 20 == 0:
                print(f"    epoch {epoch:3d} | loss {loss.item():.4f} "
                      f"| val_auc {val_auc:.4f} | best_test_auc {best_test_auc:.4f}",
                      flush=True)
    return best_val, best_test_auc, best_test_ap, timer.t, epoch_times


def train_link_sampled(model, train_data, val_data, test_data, device,
                       epochs=100, lr=0.01, weight_decay=5e-4,
                       batch_size=1024, num_neighbors=10, verbose=True):
    """
    用 LinkNeighborLoader 采样-边邻域子图训练；评估仍在全图上进行。
    """
    from torch_geometric.loader import LinkNeighborLoader

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    # 训练边保留在 CPU，由 LinkNeighborLoader 在 CPU 上采样子图（PyG 官方做法），
    # 每个 batch 再搬到 GPU；把整图先放到 GPU 上采样会极大拖慢速度。
    n_layers = model.encoder.num_layers
    loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[num_neighbors] * n_layers,
        batch_size=batch_size,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        neg_sampling_ratio=0.0,       # 训练负样本已在 train_data 中
        shuffle=True,
    )

    best_val, best_test_auc, best_test_ap = -1.0, 0.0, 0.0
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
                    z = model.encoder(batch.x, batch.edge_index)
                    out = model.decode(z, batch.edge_label_index)
                    loss = F.binary_cross_entropy_with_logits(
                        out, batch.edge_label.float())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    nb += 1
            epoch_times.append(et.t)

            val_auc, _ = evaluate_link(model, val_data, device)
            if val_auc > best_val:
                best_val = val_auc
                best_test_auc, best_test_ap = evaluate_link(model, test_data, device)

            if verbose and epoch % 20 == 0:
                print(f"    epoch {epoch:3d} | loss {total_loss / max(nb,1):.4f} "
                      f"| val_auc {val_auc:.4f} | best_test_auc {best_test_auc:.4f}",
                      flush=True)
    return best_val, best_test_auc, best_test_ap, timer.t, epoch_times

