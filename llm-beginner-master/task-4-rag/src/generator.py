"""上下文拼接与 Qwen 生成。

优先调用 OpenAI 兼容的本地 Qwen 服务（Ollama / vLLM / llama.cpp 均可）：

    # Ollama
    ollama pull qwen2.5:7b-instruct
    ollama serve
    $env:RAG_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
    $env:RAG_LLM_MODEL = "qwen2.5:7b-instruct"

    # vLLM
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
    $env:RAG_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
    $env:RAG_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

若本地服务不可用，会尝试 transformers 本地模型（RAG_LOCAL_MODEL）；
再不可用则退化为 extractive 答案，保证离线也能跑通流水线。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL = TASK_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = (
    "你是一个严谨的中文文档问答助手。你只能依据用户提供的资料回答问题。\n"
    "要求：\n"
    "1. 如果资料足以回答，请给出简洁、准确的答案，并尽量引用资料中的关键表述。\n"
    "2. 如果资料不足以回答，请明确说“根据提供的资料无法回答”，不要编造。\n"
    "3. 不要使用资料之外的知识，也不要臆测。"
)


def _source_label(source: Dict[str, Any], index: int) -> str:
    page = source.get("page")
    page_end = source.get("page_end")
    if page and page_end and page_end != page:
        page_text = f"第{page}-{page_end}页"
    elif page:
        page_text = f"第{page}页"
    else:
        page_text = "页码未知"
    section = source.get("section") or ""
    section_text = f" | {section}" if section else ""
    return f"[资料{index} | {page_text}{section_text}]"


def build_context(
    documents: Sequence[Dict[str, Any]],
    max_chars: int = 6000,
) -> str:
    """按分数排序、去重并截断上下文。"""
    if not documents:
        return ""
    ordered = sorted(
        documents,
        key=lambda item: item.get("rerank_score", item.get("score", 0.0)),
        reverse=True,
    )
    seen: List[str] = []
    blocks: List[str] = []
    used = 0
    for document in ordered:
        text = re.sub(r"\s+", " ", str(document.get("text", ""))).strip()
        if not text:
            continue
        # 去掉完全重复或已被更完整 chunk 覆盖的片段。
        if any(text == old or text in old for old in seen):
            continue
        label = _source_label(document.get("source", {}), len(blocks) + 1)
        block = f"{label}\n{text}"
        if used + len(block) > max_chars and blocks:
            remaining = max_chars - used
            if remaining > 200:
                blocks.append(block[:remaining])
            break
        blocks.append(block)
        seen.append(text)
        used += len(block)
    return "\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    return (
        f"请只依据下面的资料回答问题。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{query}\n\n"
        f"答案："
    )


def _split_sentences(text: str) -> List[str]:
    pieces = re.split(r"(?<=[。！？；!?;])\s*", str(text))
    return [piece.strip() for piece in pieces if len(piece.strip()) >= 8]


def _char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(text))
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def extractive_answer(query: str, documents: Sequence[Dict[str, Any]]) -> str:
    """离线兜底：从检索片段中抽取与问题最相关的句子。"""
    query_bigrams = _char_bigrams(query)
    candidates: List[tuple[float, str]] = []
    for document in documents:
        for sentence in _split_sentences(document.get("text", "")):
            sentence_bigrams = _char_bigrams(sentence)
            if not sentence_bigrams:
                continue
            overlap = len(query_bigrams & sentence_bigrams)
            score = overlap / max(1, len(query_bigrams))
            candidates.append((score, sentence))
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    selected: List[str] = []
    seen = set()
    for _, sentence in candidates:
        key = sentence[:40]
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= 3:
            break
    if selected:
        return "根据检索到的资料：" + "".join(selected)
    if documents:
        return "根据检索到的资料：" + str(documents[0].get("text", ""))[:300]
    return "根据提供的资料无法回答。"


class Generator:
    """Qwen 生成器：API 优先，本地 transformers / extractive 兜底。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_model: Optional[str] = None,
        local_model: Optional[str | Path] = None,
        timeout: float = 120.0,
        max_new_tokens: int = 256,
        max_context_chars: int = 6000,
    ) -> None:
        self.base_url = (base_url or os.environ.get("RAG_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_model = api_model or os.environ.get("RAG_LLM_MODEL") or DEFAULT_API_MODEL
        self.api_key = os.environ.get("RAG_LLM_API_KEY", "EMPTY")
        self.local_model_path = Path(local_model or os.environ.get("RAG_LOCAL_MODEL") or DEFAULT_LOCAL_MODEL)
        self.timeout = float(timeout)
        self.max_new_tokens = int(max_new_tokens)
        self.max_context_chars = int(max_context_chars)
        self._local_bundle: Optional[Dict[str, Any]] = None
        self._api_checked = False
        self._api_ok = False

    def _api_available(self) -> bool:
        if self._api_checked:
            return self._api_ok
        self._api_checked = True
        try:
            response = requests.get(f"{self.base_url}/models", timeout=3)
            self._api_ok = response.status_code < 500
        except Exception:
            self._api_ok = False
        return self._api_ok

    def _generate_api(self, prompt: str) -> str:
        payload = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": self.max_new_tokens,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def _load_local_model(self) -> Optional[Dict[str, Any]]:
        if self._local_bundle is not None:
            return self._local_bundle
        if not self.local_model_path.exists():
            return None
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = os.environ.get("RAG_LOCAL_DEVICE", "cpu")
        if not device.startswith("cuda"):
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            except Exception:
                pass
        tokenizer = AutoTokenizer.from_pretrained(str(self.local_model_path), trust_remote_code=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.local_model_path),
                dtype=dtype,
                trust_remote_code=True,
            )
        except TypeError:  # 兼容旧版 transformers
            model = AutoModelForCausalLM.from_pretrained(
                str(self.local_model_path),
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        model.to(device).eval()
        self._local_bundle = {"tokenizer": tokenizer, "model": model, "device": device}
        return self._local_bundle

    def _generate_local(self, prompt: str) -> str:
        bundle = self._load_local_model()
        if bundle is None:
            raise RuntimeError("本地生成模型不存在")
        import torch

        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        device = bundle["device"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{SYSTEM_PROMPT}\n\n{prompt}"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
    ) -> str:
        context = build_context(documents, max_chars=self.max_context_chars)
        if not context:
            return "根据提供的资料无法回答。"
        prompt = build_prompt(query, context)
        backend = os.environ.get("RAG_LLM_BACKEND", "auto").lower()

        if backend in {"auto", "api"} and self._api_available():
            try:
                answer = self._generate_api(prompt)
                if answer:
                    return answer
            except Exception:
                if backend == "api":
                    raise

        if backend in {"auto", "local"}:
            try:
                answer = self._generate_local(prompt)
                if answer:
                    return answer
            except Exception:
                if backend == "local":
                    raise

        return extractive_answer(query, documents)
