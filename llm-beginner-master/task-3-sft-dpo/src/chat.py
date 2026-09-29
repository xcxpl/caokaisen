"""Qwen chat formatting and assistant-only loss masking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List

import torch

_TOKENIZER: Any | None = None
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B"


def _role_of(message: dict) -> str:
    role = message.get("role", message.get("from", "user"))
    role = str(role).lower()
    aliases = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
    }
    return aliases.get(role, role)


def _content_of(message: dict) -> str:
    content = message.get("content", message.get("value", ""))
    return "" if content is None else str(content)


def format_messages(messages: List[dict]) -> str:
    """Format a message list with Qwen2.5's official chat markers.

    Example::

        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        你好！<|im_end|>
    """
    parts: list[str] = []
    for message in messages:
        role = _role_of(message)
        content = _content_of(message)
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def format_generation_prompt(messages: List[dict]) -> str:
    """Format history and append the assistant header for generation."""
    text = format_messages(messages)
    if messages and _role_of(messages[-1]) == "assistant":
        return text
    return text + "<|im_start|>assistant\n"


def _format_with_spans(messages: List[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Return the full text and character spans of assistant content."""
    parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    cursor = 0

    for message in messages:
        role = _role_of(message)
        prefix = f"<|im_start|>{role}\n"
        parts.append(prefix)
        cursor += len(prefix)

        content = _content_of(message)
        start = cursor
        parts.append(content)
        cursor += len(content)
        end = cursor
        if role == "assistant":
            assistant_spans.append((start, end))

        suffix = "<|im_end|>\n"
        parts.append(suffix)
        cursor += len(suffix)

    return "".join(parts), assistant_spans


def set_tokenizer(tokenizer: Any | None) -> None:
    """Set a process-wide tokenizer used by :func:`build_labels`."""
    global _TOKENIZER
    _TOKENIZER = tokenizer


def get_tokenizer(tokenizer: Any | None = None) -> Any:
    """Return the supplied tokenizer or lazily load the local Qwen tokenizer."""
    global _TOKENIZER
    if tokenizer is not None:
        return tokenizer
    if _TOKENIZER is not None:
        return _TOKENIZER

    from transformers import AutoTokenizer

    model_path = os.environ.get("QWEN_MODEL_PATH", str(_DEFAULT_MODEL_PATH))
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    return _TOKENIZER


def _labels_from_offsets(
    input_ids: torch.Tensor,
    assistant_spans: list[tuple[int, int]],
    offsets: Iterable[tuple[int, int] | None],
    ignore_index: int = -100,
) -> torch.Tensor | None:
    labels = torch.full_like(input_ids, ignore_index)
    if len(offsets) != input_ids.numel():
        return None

    for index, offset in enumerate(offsets):
        if offset is None:
            continue
        start, end = offset
        if start == end:
            continue
        if any(end > span_start and start < span_end for span_start, span_end in assistant_spans):
            labels[index] = input_ids[index]
    return labels


def _labels_from_segments(
    messages: List[dict],
    tokenizer: Any,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Tokenization fallback for tokenizers without offset mapping."""
    labels: list[int] = []
    for message in messages:
        role = _role_of(message)
        prefix_ids = tokenizer(f"<|im_start|>{role}\n", add_special_tokens=False)["input_ids"]
        content_ids = tokenizer(_content_of(message), add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]

        labels.extend([ignore_index] * len(prefix_ids))
        if role == "assistant":
            labels.extend(content_ids)
        else:
            labels.extend([ignore_index] * len(content_ids))
        labels.extend([ignore_index] * len(suffix_ids))
    return torch.tensor(labels, dtype=torch.long)


def build_labels(
    input_ids,
    messages: List[dict],
    tokenizer: Any | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build labels with only assistant content tokens supervised.

    User/system content and every template control marker are set to
    ``ignore_index``.  All assistant turns participate in the loss.
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(-1)
    tokenizer = get_tokenizer(tokenizer)
    text, assistant_spans = _format_with_spans(messages)

    # The evaluator tokenizes with ``add_special_tokens=True`` (the default).
    # Some tokenizers use different defaults, so try both before falling back.
    for add_special_tokens in (True, False):
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
            )
        except (TypeError, NotImplementedError, ValueError):
            break
        encoded_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        if encoded_ids.numel() != ids.numel():
            continue
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            continue
        labels = _labels_from_offsets(ids, assistant_spans, offsets, ignore_index)
        if labels is not None:
            return labels

    fallback = _labels_from_segments(messages, tokenizer, ignore_index)
    if fallback.numel() < ids.numel():
        fallback = torch.cat(
            [
                fallback,
                torch.full((ids.numel() - fallback.numel(),), ignore_index, dtype=torch.long),
            ]
        )
    return fallback[: ids.numel()]


def _build_labels_for_spans(
    input_ids,
    text: str,
    spans: list[tuple[int, int]],
    tokenizer: Any,
    ignore_index: int = -100,
) -> torch.Tensor:
    ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(-1)
    for add_special_tokens in (True, False):
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
            )
        except (TypeError, NotImplementedError, ValueError):
            break
        encoded_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        if encoded_ids.numel() != ids.numel():
            continue
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            continue
        labels = _labels_from_offsets(ids, spans, offsets, ignore_index)
        if labels is not None:
            return labels

    labels = torch.full_like(ids, ignore_index)
    # Fallback: find the response tokenization at the end of the sequence.
    response_ids = tokenizer(text[spans[-1][0] : spans[-1][1]], add_special_tokens=False)["input_ids"]
    if response_ids:
        labels[-len(response_ids) :] = torch.tensor(response_ids, dtype=torch.long)
    return labels


def build_response_labels(
    input_ids,
    prompt_messages: List[dict],
    response: str,
    tokenizer: Any | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mask prompt/history and supervise only the final assistant response."""
    tokenizer = get_tokenizer(tokenizer)
    messages = list(prompt_messages) + [{"role": "assistant", "content": response}]
    text, assistant_spans = _format_with_spans(messages)
    if not assistant_spans:
        raise ValueError("No assistant span found")
    return _build_labels_for_spans(input_ids, text, assistant_spans[-1:], tokenizer, ignore_index)


def tokenize_supervised(messages: List[dict], tokenizer: Any, max_length: int | None = None):
    """Tokenize one conversation and return ``(input_ids, labels)`` tensors."""
    text = format_messages(messages)
    encoded = tokenizer(text, add_special_tokens=True)
    ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    labels = build_labels(ids, messages, tokenizer=tokenizer)
    if max_length is not None:
        ids = ids[:max_length]
        labels = labels[:max_length]
    return ids, labels
