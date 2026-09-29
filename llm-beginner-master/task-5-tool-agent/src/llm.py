"""模型后端：OpenAI 兼容 API 优先，本地 transformers 兜底。

- OpenAICompatBackend：任何 OpenAI 兼容服务（Ollama / vLLM / llama.cpp / LM Studio）。
  通过 OPENAI_BASE_URL、OPENAI_API_KEY、AGENT_MODEL 配置。
- LocalHFBackend：直接加载本地权重目录（或 HuggingFace 仓库 id）做贪心解码，
  方便没有 Ollama 时也能跑通整条链路。

环境变量：
    AGENT_BACKEND      auto（默认）/ openai / local
    OPENAI_BASE_URL    默认 http://localhost:11434/v1（Ollama）
    OPENAI_API_KEY     默认 ollama（本地服务随便填）
    AGENT_MODEL        默认 qwen2.5:7b-instruct（OpenAI 兼容后端的模型名）
    AGENT_LOCAL_MODEL  本地权重目录或 HF 仓库 id
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

LOCAL_MODEL_CANDIDATES = (
    "task-4-rag/models/Qwen2.5-0.5B-Instruct",
    "task-3-sft-dpo/models/Qwen2.5-0.5B",
    "task-4-rag/models/Qwen2.5-0.5B",
)


class BackendError(RuntimeError):
    """后端不可用。"""


class ChatBackend:
    name = "base"
    model = "?"
    temperature_default = 0.0

    def chat(self, messages, max_tokens: int = 400, temperature: float | None = None,
             stop: list[str] | None = None) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


class OpenAICompatBackend(ChatBackend):
    """走 OpenAI 兼容 HTTP 接口（Ollama / vLLM / llama.cpp server 都支持）。"""

    name = "openai-compat"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 180.0, ping: bool = True):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise BackendError(f"未安装 openai 包：{e}") from e

        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
        self.model = model or os.environ.get("AGENT_MODEL") or DEFAULT_MODEL
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)
        if ping:
            self._ping()

    def _ping(self) -> None:
        try:
            self._client.models.list()
            return
        except Exception:  # noqa: BLE001 —— 有些服务没实现 /models，退回一次最小补全
            pass
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )
        except Exception as e:  # noqa: BLE001
            raise BackendError(f"OpenAI 兼容后端不可用（{self.base_url} / {self.model}）：{e}") from e

    def chat(self, messages, max_tokens: int = 400, temperature: float | None = None,
             stop: list[str] | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature_default if temperature is None else temperature,
            stop=stop or None,
        )
        return (resp.choices[0].message.content or "").strip()


class _StopOnStrings:
    """HF 的 StoppingCriteria：生成文本里一出现 stop 串就停。"""

    def __init__(self, tokenizer, stop: list[str], prompt_len: int):
        self.tokenizer = tokenizer
        self.stop = [s for s in stop if s]
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.stop:
            return False
        text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
        return any(s in text for s in self.stop)


class LocalHFBackend(ChatBackend):
    """本地 transformers 后端（默认 fp16 + CUDA，OOM 自动退回 CPU）。"""

    name = "local-transformers"

    def __init__(self, model_path: str | Path, max_new_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target = _resolve_local_model(model_path)
        self.model = str(target)
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self._lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(str(target))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._dispatched = False
        self.model_obj = self._load_model(target, dtype, AutoModelForCausalLM)
        self.model_obj.eval()
        self._prompt_len = 0
        self._cpu_fallback = False

    def _load_model(self, target, dtype, auto_cls):
        """按 AGENT_MAX_GPU_MEM 决定是整块上 GPU、分层 GPU+CPU，还是全 CPU。"""
        import torch

        gpu_budget = os.environ.get("AGENT_MAX_GPU_MEM")
        if self.device == "cuda" and gpu_budget:
            max_memory = {0: gpu_budget, "cpu": os.environ.get("AGENT_MAX_CPU_MEM", "16GiB")}
            try:
                model = auto_cls.from_pretrained(str(target), dtype=dtype,
                                                 device_map="auto", max_memory=max_memory)
                self._dispatched = True
                self.device = str(next(model.parameters()).device)
                return model
            except Exception as e:  # noqa: BLE001
                print(f"[llm] 分层加载失败（{e}），退回单设备加载", flush=True)
        try:
            return auto_cls.from_pretrained(str(target), dtype=dtype, device_map=self.device)
        except TypeError:   # 老版本 transformers 用的是 torch_dtype
            return auto_cls.from_pretrained(str(target), torch_dtype=dtype, device_map=self.device)
        except RuntimeError as e:   # 显存不够就退回 CPU
            if self.device != "cuda":
                raise
            print(f"[llm] CUDA 加载失败（{e}），改用 CPU", flush=True)
            torch.cuda.empty_cache()
            self.device = "cpu"
            return auto_cls.from_pretrained(str(target), dtype=torch.float32)

    def chat(self, messages, max_tokens: int | None = None, temperature: float | None = None,
             stop: list[str] | None = None) -> str:
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = bool(temperature and temperature > 0)
        kwargs = {}
        if stop:
            criteria = _StopOnStrings(self.tokenizer, stop, prompt_len)
            kwargs["stopping_criteria"] = [criteria]
        try:
            output = self._generate(inputs, max_tokens, do_sample, temperature, kwargs)
        except RuntimeError as e:
            # 4GB 显存跑长上下文很容易 OOM：整体退回 CPU 再试一次（慢但能跑完）
            if "out of memory" not in str(e).lower() or self.device != "cuda":
                raise
            print(f"[llm] CUDA OOM，本次起改用 CPU 推理（{e}）", flush=True)
            self._to_cpu()
            inputs = inputs.to("cpu")
            prompt_len = inputs["input_ids"].shape[1]
            if stop:
                kwargs["stopping_criteria"] = [_StopOnStrings(self.tokenizer, stop, prompt_len)]
            output = self._generate(inputs, max_tokens, do_sample, temperature, kwargs)
        text = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        if stop:
            for token in stop:
                if token and token in text:
                    text = text.split(token)[0]
        return text.strip()

    def _generate(self, inputs, max_tokens, do_sample, temperature, kwargs):
        with self._lock, self._torch.inference_mode():
            return self.model_obj.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=do_sample,
                temperature=(temperature or 1.0) if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

    def _to_cpu(self) -> None:
        if self._cpu_fallback:
            return
        self._cpu_fallback = True
        if self._dispatched:
            # 分层加载的模型没法整体 .to("cpu")，只能清空缓存后重试
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            self.model_obj.to("cpu")
            self._torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[llm] 移到 CPU 失败：{e}", flush=True)
        self.device = "cpu"


def _resolve_local_model(model_path: str | Path) -> Path | str:
    raw = str(model_path)
    path = Path(raw)
    if path.exists():
        return path
    candidate = REPO_ROOT / raw
    if candidate.exists():
        return candidate
    if "/" in raw and not raw.endswith((".json", ".safetensors")):
        return raw          # 当成 HuggingFace 仓库 id
    raise BackendError(f"本地模型目录不存在：{raw}")


def find_local_model() -> Path | str:
    env = os.environ.get("AGENT_LOCAL_MODEL")
    if env:
        return _resolve_local_model(env)
    for rel in LOCAL_MODEL_CANDIDATES:
        candidate = REPO_ROOT / rel
        if (candidate / "config.json").exists():
            return candidate
    raise BackendError(
        "找不到可用的本地模型目录。可用 AGENT_LOCAL_MODEL 指定，"
        f"或在任务三/四的 models/ 下准备权重（候选：{', '.join(LOCAL_MODEL_CANDIDATES)}）")


_BACKEND: ChatBackend | None = None
_BACKEND_LOCK = threading.Lock()


def get_backend(force: str | None = None, verbose: bool = False) -> ChatBackend:
    """进程内单例，避免 10 道题反复加载模型。"""
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None and force is None:
            return _BACKEND
        backend = build_backend(force=force, verbose=verbose)
        if force is None:
            _BACKEND = backend
        return backend


def build_backend(force: str | None = None, verbose: bool = False) -> ChatBackend:
    prefer = (force or os.environ.get("AGENT_BACKEND") or "auto").strip().lower()
    errors: list[str] = []

    if prefer in ("auto", "openai", "openai-compat", "api"):
        try:
            backend = OpenAICompatBackend()
            if verbose:
                print(f"[llm] 使用 OpenAI 兼容后端 {backend.base_url} / {backend.model}", flush=True)
            return backend
        except Exception as e:  # noqa: BLE001
            errors.append(f"openai-compat: {e}")
            if prefer != "auto":
                raise

    if prefer in ("auto", "local", "local-transformers", "hf", "transformers"):
        try:
            started = time.time()
            backend = LocalHFBackend(find_local_model())
            if verbose:
                print(f"[llm] 使用本地后端 {backend.model}（device={backend.device}，"
                      f"加载 {time.time() - started:.1f}s）", flush=True)
            return backend
        except Exception as e:  # noqa: BLE001
            errors.append(f"local-transformers: {e}")
            if prefer != "auto":
                raise

    raise BackendError(
        "没有可用的模型后端。请二选一：\n"
        "  A) 起本地 OpenAI 兼容服务，例如 `ollama serve` 后 "
        "`set OPENAI_BASE_URL=http://localhost:11434/v1`；\n"
        "  B) 设置 AGENT_LOCAL_MODEL 指向本地权重目录。\n"
        "尝试记录：\n  - " + "\n  - ".join(errors))
