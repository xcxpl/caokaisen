"""Dataset readers for MOSS SFT and DPO-En-Zh-20k."""

from __future__ import annotations

import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any, Iterator


def normalize_role(role: Any) -> str:
    role = str(role or "user").lower()
    aliases = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
    }
    return aliases.get(role, role)


def _normalize_message(message: dict) -> dict[str, str]:
    return {
        "role": normalize_role(message.get("role", message.get("from", "user"))),
        "content": str(message.get("content", message.get("value", ""))),
    }


def iter_jsonl(path: str | Path, max_records: int | None = None) -> Iterator[dict]:
    """Iterate JSONL records from a plain file or a zip containing JSONL."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".jsonl", ".json"))
            ]
            if not members:
                raise ValueError(f"No .jsonl/.json member found in {path}")
            with archive.open(members[0]) as raw:
                stream = io.TextIOWrapper(raw, encoding="utf-8")
                yield from _iter_lines(stream, max_records)
        return

    with path.open("r", encoding="utf-8") as stream:
        yield from _iter_lines(stream, max_records)


def _iter_lines(stream: io.TextIOBase, max_records: int | None) -> Iterator[dict]:
    count = 0
    for line_number, line in enumerate(stream, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_number}") from exc
        yield record
        count += 1
        if max_records is not None and count >= max_records:
            break


def load_json_array(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON array in {path}")
    return data


def moss_record_to_messages(
    record: dict,
    include_meta_instruction: bool = False,
) -> list[dict[str, str]]:
    """Convert one MOSS-003-sft record to chat messages."""
    if "messages" in record and isinstance(record["messages"], list):
        return [_normalize_message(message) for message in record["messages"]]

    # The official no-tools JSONL stores turns under ``chat`` as
    # {"turn_1": {"Human": ..., "MOSS": ...}, ...}.
    if isinstance(record.get("chat"), dict):
        messages = []
        if include_meta_instruction and record.get("meta_instruction"):
            messages.append({"role": "system", "content": str(record["meta_instruction"])})
        for turn_name in sorted(record["chat"], key=lambda name: int(str(name).split("_")[-1])):
            turn = record["chat"][turn_name]
            if not isinstance(turn, dict):
                continue
            human = _clean_moss_text(turn.get("Human", ""))
            assistant = _clean_moss_text(turn.get("MOSS", ""))
            if human:
                messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
        if any(message["role"] == "assistant" for message in messages):
            return messages

    conversation = record.get("conversation") or record.get("conversations")
    if not isinstance(conversation, list):
        raise ValueError("MOSS record has no conversation/messages field")

    messages: list[dict[str, str]] = []
    system = record.get("system")
    if include_meta_instruction and record.get("meta_instruction"):
        system = record["meta_instruction"]
    if system:
        messages.append({"role": "system", "content": str(system)})

    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        if "human" in turn or "user" in turn:
            user = turn.get("human", turn.get("user", ""))
            assistant = turn.get("assistant", turn.get("gpt", turn.get("model", "")))
            if str(user).strip():
                messages.append({"role": "user", "content": str(user)})
            if str(assistant).strip():
                messages.append({"role": "assistant", "content": str(assistant)})
        elif "role" in turn or "from" in turn:
            message = _normalize_message(turn)
            if message["content"].strip():
                messages.append(message)

    if not any(message["role"] == "assistant" and message["content"].strip() for message in messages):
        raise ValueError("MOSS record contains no assistant response")
    return messages


def _clean_moss_text(text: Any) -> str:
    text = str(text or "")
    prefixes = (
        "<|Human|>: ",
        "<|MOSS|>: ",
        "<|Inner Thoughts|>: ",
        "<|Commands|>: ",
        "<|Results|>: ",
    )
    suffixes = ("<eoh>\n", "<eom>\n", "<eot>\n", "<eoc>\n", "<eor>\n", "<eoh>", "<eom>", "<eot>", "<eoc>", "<eor>")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    for suffix in suffixes:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip()


def dpo_record_to_example(record: dict) -> tuple[list[dict[str, str]], str, str]:
    """Return ``(prompt_messages, chosen, rejected)`` for one DPO record."""
    conversation = record.get("conversations")
    if not isinstance(conversation, list):
        raise ValueError("DPO record has no conversations field")

    prompt_messages = [_normalize_message(turn) for turn in conversation]
    chosen = str(record["chosen"]["value"])
    rejected = str(record["rejected"]["value"])
    if not chosen.strip() or not rejected.strip():
        raise ValueError("Empty chosen/rejected response")
    return prompt_messages, chosen, rejected


def sample_records(
    path: str | Path,
    limit: int,
    shuffle: bool = True,
    seed: int = 42,
    max_scan: int | None = None,
) -> list[dict]:
    """Read a bounded sample.

    JSON arrays are loaded directly.  JSONL/zip files use reservoir sampling
    over at most ``max_scan`` records, which gives a more diverse small subset
    than simply taking the first N lines.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        records = load_json_array(path)
        if shuffle:
            random.Random(seed).shuffle(records)
        return records[:limit]

    rng = random.Random(seed)
    reservoir: list[dict] = []
    for index, record in enumerate(iter_jsonl(path)):
        if index < limit:
            reservoir.append(record)
        else:
            replace = rng.randint(0, index)
            if replace < limit:
                reservoir[replace] = record
        if max_scan is not None and index + 1 >= max_scan:
            break
    if shuffle:
        rng.shuffle(reservoir)
    return reservoir
