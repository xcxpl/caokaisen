# -*- coding: utf-8 -*-
"""
任务四 公共模块：知识图谱补全（KGE）
=====================================
参考 kge_framework 的组织方式，自行实现 3 个经典补全模型：

    TransE  —— 平移假设   h + r ≈ t
    RotatE  —— 复数空间旋转  h ∘ r ≈ t
    ConvE   —— 2D 卷积 + 1-N 打分

数据集格式（RotatE 官方）：
    train/valid/test.txt  每行 "h \t r \t t"
    entities.dict / relations.dict  每行 "<id> \t <name>"

评估：Filtered 排序下的 MRR、Hits@1 / Hits@3 / Hits@10（头尾双向平均）。
"""
import os
import time
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def peak_gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0


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
# 数据
# --------------------------------------------------------------------------

DATASET_INFO = {
    "wn18rr": "WN18RR — 40943 实体, 11 关系, 86835 训练三元组 (英文 WordNet 子集)",
    "countries_S1": "countries_S1 — 271 实体, 2 关系, 1068 训练三元组 (小型数据集)",
}

DATASET_CFG = {
    "wn18rr": dict(epochs=120, batch_size=1024, lr=1e-3, neg_num=1),
    "countries_S1": dict(epochs=300, batch_size=256, lr=1e-2, neg_num=1),
}


def _load_dict(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                i, name = parts
            else:                                  # 少数数据集用空格分隔
                i, name = line.split(" ", 1)
            m[name] = int(i)
    return m


def _load_triples(path, e2id, r2id):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            rows.append([e2id[h], r2id[r], e2id[t]])
    return torch.tensor(rows, dtype=torch.long)


class KGData:
    """知识图谱数据集：三元组 + 用于 filtered 评估的真值集合。"""

    def __init__(self, name, root):
        self.name = name
        self.dir = os.path.join(root, name)
        self.ent2id = _load_dict(os.path.join(self.dir, "entities.dict"))
        self.rel2id = _load_dict(os.path.join(self.dir, "relations.dict"))
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.num_entities = len(self.ent2id)
        self.num_relations = len(self.rel2id)

        self.train = _load_triples(os.path.join(self.dir, "train.txt"),
                                   self.ent2id, self.rel2id)
        self.valid = _load_triples(os.path.join(self.dir, "valid.txt"),
                                   self.ent2id, self.rel2id)
        self.test = _load_triples(os.path.join(self.dir, "test.txt"),
                                  self.ent2id, self.rel2id)

        all_true = torch.cat([self.train, self.valid, self.test], dim=0)
        self.all_true = all_true
        self._build_filters()

    def _build_filters(self):
        """filter[e][r] = 该 (h, r) 下所有真尾实体集合 / (r, t) 下所有真头实体集合"""
        self.tail_filter = {}
        self.head_filter = {}
        for h, r, t in self.all_true.tolist():
            self.tail_filter.setdefault((h, r), set()).add(t)
            self.head_filter.setdefault((r, t), set()).add(h)

    def describe(self):
        return (f"{self.name}: {self.num_entities} 实体, {self.num_relations} 关系, "
                f"train={len(self.train)}, valid={len(self.valid)}, test={len(self.test)}")


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------


class BaseKGE(nn.Module):
    """所有模型统一提供 score_all_tails(h, r) -> [B, num_entities]（越大越真）"""

    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

    def score_all_tails(self, h, r):
        raise NotImplementedError

    def score_triples(self, h, r, t):
        s = self.score_all_tails(h, r)
        return s.gather(1, t.view(-1, 1)).view(-1)

    def normalize(self):
        """TransE / RotatE 的实体嵌入约束，ConvE 无需约束。"""
        return


class TransE(BaseKGE):
    """h + r ≈ t，得分 = -||h + r - t||_2"""

    def __init__(self, num_entities, num_relations, dim=100, gamma=3.0):
        super().__init__(num_entities, num_relations, dim)
        self.gamma = gamma
        bound = 6.0 / math.sqrt(dim)
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.ent.weight, -bound, bound)
        nn.init.uniform_(self.rel.weight, -bound, bound)

    def score_all_tails(self, h, r):
        """-||h + r - t||^2（与 -||.|| 排序等价，但可用矩阵乘法一次算完）"""
        x = self.ent(h) + self.rel(r)                   # [B, d]
        T = self.ent.weight                             # [E, d]
        d2 = (x * x).sum(-1, keepdim=True) + (T * T).sum(-1).unsqueeze(0) \
            - 2.0 * x @ T.t()
        return -d2

    def score_triples(self, h, r, t):
        """-||h + r - t||_2（训练用的精确距离）"""
        return -torch.norm(self.ent(h) + self.rel(r) - self.ent(t), p=2, dim=-1)

    def normalize(self):
        with torch.no_grad():
            self.ent.weight.data = F.normalize(self.ent.weight.data, p=2, dim=1)


class RotatE(BaseKGE):
    """复数空间旋转 h ∘ r ≈ t，r 用相位表示"""

    def __init__(self, num_entities, num_relations, dim=100, gamma=3.0):
        """dim 为复数维度，实际存储 2*dim（实部 + 虚部）"""
        super().__init__(num_entities, num_relations, dim)
        self.gamma = gamma
        self.embedding_range = (gamma + 2.0) / dim
        self.ent = nn.Embedding(num_entities, dim * 2)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.ent.weight, -self.embedding_range, self.embedding_range)
        nn.init.uniform_(self.rel.weight, -self.embedding_range, self.embedding_range)

    def _phase(self, r):
        return self.rel(r) * (math.pi / self.embedding_range)

    def _rotate(self, h, r):
        """返回旋转后的复数向量 [B, 2, c]"""
        B = h.size(0)
        h_e = self.ent(h).view(B, 2, self.dim)
        re_h, im_h = h_e[:, 0], h_e[:, 1]
        phase = self._phase(r)
        cos_r, sin_r = torch.cos(phase), torch.sin(phase)
        return torch.stack([re_h * cos_r - im_h * sin_r,
                            re_h * sin_r + im_h * cos_r], dim=1)

    def score_all_tails(self, h, r):
        x = self._rotate(h, r)                          # [B, 2, c]
        x_re, x_im = x[:, 0], x[:, 1]
        T = self.ent.weight.view(self.num_entities, 2, self.dim)
        T_re, T_im = T[:, 0], T[:, 1]                   # [E, c]
        d2 = (x_re * x_re + x_im * x_im).sum(-1, keepdim=True) \
            + (T_re * T_re + T_im * T_im).sum(-1).unsqueeze(0) \
            - 2.0 * (x_re @ T_re.t() + x_im @ T_im.t())
        return -d2

    def score_triples(self, h, r, t):
        B = h.size(0)
        x = self._rotate(h, r).reshape(B, -1)
        return -torch.norm(x - self.ent(t), p=2, dim=-1)

    def normalize(self):
        with torch.no_grad():
            self.ent.weight.data = F.normalize(self.ent.weight.data, p=2, dim=1)


class ConvE(BaseKGE):
    """2D 卷积 + 1-N 打分，实体/关系嵌入维度相同并可 reshape 成 2D 图像"""

    def __init__(self, num_entities, num_relations, dim=200,
                 input_drop=0.2, hidden_drop=0.3, feature_map_drop=0.2):
        super().__init__(num_entities, num_relations, dim)
        assert dim % 20 == 0, "dim 需要能被 20 整除"
        self.h, self.w = dim // 20, 20
        bound = 6.0 / math.sqrt(dim)
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.ent.weight, -bound, bound)
        nn.init.uniform_(self.rel.weight, -bound, bound)

        self.inp_drop = nn.Dropout(input_drop)
        self.hidden_drop = nn.Dropout(hidden_drop)
        self.feature_drop = nn.Dropout2d(feature_map_drop)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(dim)
        flat = 32 * (2 * self.h - 2) * (self.w - 2)
        self.fc = nn.Linear(flat, dim)

    def score_all_tails(self, h, r):
        B = h.size(0)
        e1 = self.ent(h).view(B, 1, self.h, self.w)
        e2 = self.rel(r).view(B, 1, self.h, self.w)
        stacked = torch.cat([e1, e2], dim=2)                     # [B,1,2h,w]
        x = self.bn0(stacked)
        x = self.inp_drop(x)
        x = self.conv1(x)                                        # [B,32,2h-2,w-2]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(B, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x @ self.ent.weight.t()                           # [B, E]


MODELS = {"transe": TransE, "rotate": RotatE, "conve": ConvE}

MODEL_CFG = {
    "transe": dict(dim=100, lr=1e-3, epochs=120, batch_size=1024, use_1n=False),
    "rotate": dict(dim=100, lr=1e-3, epochs=120, batch_size=512, use_1n=False),
    "conve": dict(dim=200, lr=1e-3, epochs=80, batch_size=128, use_1n=True),
}


def build_model(name, data, dim=None):
    name = name.lower()
    cls = MODELS[name]
    d = MODEL_CFG[name].copy()
    if dim is not None:
        d["dim"] = dim
    return cls(data.num_entities, data.num_relations, dim=d["dim"]), d


# --------------------------------------------------------------------------
# 训练 / 评估
# --------------------------------------------------------------------------


def _sample_negatives(pos, num_entities, neg_num, rng):
    """随机替换头或尾实体"""
    B = pos.size(0)
    negs = pos.repeat(1, neg_num).view(B * neg_num, 3).clone()
    rand = torch.from_numpy(rng.randint(0, num_entities, size=B * neg_num,
                                        dtype=np.int64))
    choose_head = torch.from_numpy(rng.randint(0, 2, size=B * neg_num)).bool()
    negs[choose_head, 0] = rand[choose_head]
    negs[~choose_head, 2] = rand[~choose_head]
    return negs


def train_kge(model, data, device, epochs=None, batch_size=None, lr=None,
              neg_num=1, use_1n=None, verbose=True):
    """
    use_1n=True  →  ConvE 的 1-N 打分 + BCE（对全部实体做二分类）
    use_1n=False →  TransE / RotatE 的 margin ranking 损失（负采样）
    """
    name = model.__class__.__name__.lower()
    cfg = MODEL_CFG.get(name, {})
    epochs = int(epochs or cfg.get("epochs", 100))
    batch_size = int(batch_size or cfg.get("batch_size", 1024))
    lr = float(lr or cfg.get("lr", 1e-3))
    use_1n = cfg.get("use_1n", False) if use_1n is None else use_1n

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train = data.train.to(device)
    num_ent = data.num_entities
    rng = np.random.RandomState(42)
    gamma = float(getattr(model, "gamma", 1.0))

    epoch_times = []
    timer = Timer()
    with timer:
        for epoch in range(1, epochs + 1):
            et = Timer()
            with et:
                model.train()
                perm = torch.randperm(train.size(0))
                total_loss, nb = 0.0, 0
                for i in range(0, train.size(0), batch_size):
                    pos = train[perm[i:i + batch_size]]
                    optimizer.zero_grad()
                    if use_1n:
                        # 1-N：对全部实体打分，用 ConvE 论文的标准做法——
                        # 目标向量在真实尾实体处置 1-ε，其余实体置 ε/(E-1)
                        # （label smoothing），整张 [B, E] 上做 BCE。
                        score = model.score_all_tails(pos[:, 0], pos[:, 1])
                        eps = 0.1
                        target = torch.full_like(score, eps / max(num_ent - 1, 1))
                        target.scatter_(1, pos[:, 2:3], 1.0 - eps)
                        loss = F.binary_cross_entropy_with_logits(score, target)
                    else:
                        negs = _sample_negatives(pos.cpu(), num_ent, neg_num,
                                                 rng).to(device)
                        d_pos = -model.score_triples(pos[:, 0], pos[:, 1], pos[:, 2])
                        d_neg = -model.score_triples(negs[:, 0], negs[:, 1], negs[:, 2])
                        d_neg = d_neg.view(pos.size(0), neg_num).mean(dim=1)
                        # max(0, gamma + d_pos - d_neg)
                        loss = F.relu(gamma + d_pos - d_neg).mean()
                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()
                        model.normalize()
                    total_loss += float(loss)
                    nb += 1
            epoch_times.append(et.t)
            if verbose and epoch % 20 == 0:
                print(f"    epoch {epoch:3d} | loss {total_loss / max(nb,1):.4f}",
                      flush=True)
    return timer.t, epoch_times


@torch.no_grad()
def evaluate_filtered(model, triples, data, device, batch_size=256):
    """
    Filtered 尾实体排序评估（同时评估两个方向的标准做法需要反向关系，
    这里统一报告尾实体方向），返回 MRR / Hits@1/3/10。
    """
    model.eval()
    triples = triples.to(device)
    ranks = []
    for i in range(0, triples.size(0), batch_size):
        batch = triples[i:i + batch_size]
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        score = model.score_all_tails(h, r)
        ranks.append(_ranks_from_scores(score, t, h, r, data))
    ranks = torch.cat(ranks).float()
    return {
        "mrr": float((1.0 / ranks).mean()),
        "hits@1": float((ranks <= 1).float().mean()),
        "hits@3": float((ranks <= 3).float().mean()),
        "hits@10": float((ranks <= 10).float().mean()),
        "count": int(ranks.numel()),
    }


def _ranks_from_scores(score, target, head, rel, data):
    """过滤掉其它已知真尾实体后，计算目标实体的排名（1 表示排名第一）"""
    score = score.clone()
    B = score.size(0)
    h_l, r_l, t_l = head.tolist(), rel.tolist(), target.tolist()
    neg_inf = float("-inf")
    for i in range(B):
        for e in data.tail_filter.get((h_l[i], r_l[i]), ()):  # 其它真值
            if e != t_l[i]:
                score[i, e] = neg_inf
    target_score = score.gather(1, target.view(-1, 1))
    return (score > target_score).sum(dim=1) + 1
