"""bge-reranker-base 精排。

Reranker 吃的是 ``[query, document]`` 文本对，不是第二次 embedding。
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:
    from .indexer import resolve_device
except ImportError:  # pragma: no cover - 直接脚本运行时的回退
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.indexer import resolve_device

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RERANKER_PATH = TASK_ROOT / "models" / "bge-reranker-base"
_MODEL_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _copy_from_safetensors(handle, name: str, target: torch.Tensor) -> None:
    """逐张量/分块拷贝 NumPy 权重，避免一次性分配大矩阵。"""
    if target.numel() < 5_000_000 or len(target.shape) == 0:
        array = np.array(handle.get_tensor(name), copy=True)
        target.data.copy_(torch.from_numpy(array))
        del array
        return
    slice_ = handle.get_slice(name)
    rows = target.shape[0]
    chunk_rows = 4096
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        array = np.array(slice_[start:end], copy=True)
        target.data[start:end].copy_(torch.from_numpy(array))
        del array


def _load_reranker_model(
    model_path: str | Path,
    device: Optional[str],
) -> Dict[str, Any]:
    model_path = Path(model_path)
    # 4GB 显卡上把 1.1GB 的 reranker 放到 CUDA 容易触发驱动级崩溃；
    # 默认 CPU，需要时可用 RERANKER_DEVICE=cuda 覆盖。
    resolved_device = device or os.environ.get("RERANKER_DEVICE", "cpu")
    # PyTorch CPU 的 fp16 算子是模拟实现，batch16 要几十秒；fp32 只要
    # 约 0.2 秒。因此 CPU 用 fp32，CUDA 才用 fp16。
    if resolved_device.startswith("cuda"):
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        except Exception:
            pass
    key = (str(model_path.resolve()), resolved_device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到 reranker 模型：{model_path}\n"
            f"先运行：python data/download.py"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    # 不用 from_pretrained 加载权重：transformers 5.x 的 _materialize_copy
    # 在 Windows 上处理该 XLM-R 权重时会触发 access violation。这里用
    # safe_open 逐张量读取并直接写入 fp16 模型，既绕开该路径，也把
    # 峰值内存从约 2.2GB 降到约 1GB（15GB 机器 + 小页面文件下更稳）。
    config = AutoConfig.from_pretrained(str(model_path))
    try:
        model = AutoModelForSequenceClassification.from_config(
            config, dtype=model_dtype
        )
    except TypeError:  # 兼容旧版 transformers 参数名
        model = AutoModelForSequenceClassification.from_config(
            config, torch_dtype=model_dtype
        )
    pytorch_path = model_path / "pytorch_model.bin"
    safetensors_path = model_path / "model.safetensors"
    # 优先用完整的 pytorch_model.bin；如果只有 safetensors，则用
    # safetensors.numpy 读取为普通 NumPy 数组再拷贝。PyTorch 2.5 在
    # Windows 上直接对 safetensors 的 torch 张量做切片会触发 access violation。
    if pytorch_path.exists() and pytorch_path.stat().st_size >= 1_000_000_000:
        state_dict = torch.load(str(pytorch_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
    elif safetensors_path.exists():
        # 逐张量读取为 NumPy 数组再拷入 fp16 模型：既避开 PyTorch 对
        # safetensors 内存映射张量的切片崩溃，也避免一次性分配完整 state_dict。
        with safe_open(str(safetensors_path), framework="np", device="cpu") as handle:
            keys = set(handle.keys())
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in keys:
                        _copy_from_safetensors(handle, name, parameter)
                for name, buffer in model.named_buffers():
                    if name in keys:
                        _copy_from_safetensors(handle, name, buffer)
        gc.collect()
    else:
        raise FileNotFoundError(f"模型目录下没有权重文件：{model_path}")
    model.to(resolved_device).eval()
    cached = {"tokenizer": tokenizer, "model": model, "device": resolved_device}
    _MODEL_CACHE[key] = cached
    return cached


class Reranker:
    """BGE cross-encoder reranker。"""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_RERANKER_PATH,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        loaded = _load_reranker_model(self.model_path, device)
        self.tokenizer = loaded["tokenizer"]
        self.model = loaded["model"]
        self.device = loaded["device"]

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        pairs = [(query, str(document)) for document in documents]
        scores: List[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs).logits
                if logits.dim() > 1 and logits.shape[-1] > 1:
                    logits = logits[:, -1]
                else:
                    logits = logits.reshape(-1)
                scores.extend(torch.sigmoid(logits).detach().float().cpu().tolist())
        return scores

    def rerank(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """对候选 chunk 精排；返回新 list，不修改输入。"""
        if not documents:
            return []
        texts = [str(document.get("text", "")) for document in documents]
        scores = self.score(query, texts)
        reranked: List[Dict[str, Any]] = []
        for document, score in zip(documents, scores):
            item = dict(document)
            item["rerank_score"] = float(score)
            reranked.append(item)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        if top_k is not None:
            reranked = reranked[:max(0, int(top_k))]
        return reranked
