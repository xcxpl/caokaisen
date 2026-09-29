"""wiki：维基百科查询（中文 / 英文），带直连+代理自动探测与本地缓存。

自检固定参数键：run({"query": "Alan Turing"}) -> 网络可用时须返回长度 > 50 的文本；
离线 / 被墙时抛 WikiError（自检按「跳过」处理）。

取数流程：先 list=search 找到最可能的词条名，再取 intro 摘要（explaintext，纯文本）。
结果缓存到 task-5-tool-agent/data/cache/wiki/，断网时可用缓存兜底。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "查询维基百科词条，返回词条名、简介段落（纯文本）与链接。"
            "中英文词条都支持：中文问题会自动查中文维基，英文人名/术语会自动查英文维基。"
            "适合查人物生日、事件年份、术语定义等事实；返回的是网页摘录，可能较长。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要查的词条或关键词，例如 \"图灵机\"、\"Geoffrey Hinton\"、\"Transformer 机器学习模型\"",
                    "example": "Geoffrey Hinton",
                },
                "lang": {
                    "type": "string",
                    "description": "可选，指定语言：zh 或 en；不填则按查询内容自动选择",
                    "example": "en",
                },
            },
            "required": ["query"],
        },
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "wiki"
API_URL = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "llm-beginner-task5/0.1 (educational ReAct demo)"
_REQUEST_TIMEOUT = 8.0
_MAX_EXTRACT_CHARS = 1500
_COMMON_PROXY_PORTS = ("7897", "7890", "10809", "10808", "1080", "2080", "33210")

_route_cache: dict = {"route": None, "failed_until": 0.0}
# 探测失败后的冷却时间：否则每题每次 wiki 调用都要重探 8 个出口 × 2 轮，
# 断网时单题能耗掉好几分钟（task-9 曾跑到 346 秒）。
_ROUTE_RETRY_COOLDOWN = 300.0


class WikiError(RuntimeError):
    """网络不可用 / 查不到词条；消息会作为 Observation 喂回模型。"""


def _extract_keys(query: str, lang: str) -> list[str]:
    text = query.strip()
    keys = [text]
    # 去掉括号注释："Transformer (机器学习模型)" -> "Transformer"
    stripped = re.sub(r"[（(].*?[)）]", " ", text).strip()
    if stripped and stripped != text:
        keys.append(stripped)
    # 中英混杂的查询（"Transformer 发表年份"）在中英文维基里常常命中奇怪的条目，
    # 这里额外补一个「只保留拉丁词」的候选，通常才是真正的词条名。
    latin = re.sub(r"[\u4e00-\u9fff\u3000-\u303f\s]+", " ", text).strip(" .,;:-—_·")
    if len(latin) >= 3 and latin != text and latin not in keys:
        keys.append(latin)
    return keys


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _routes() -> list[tuple[str, dict | None]]:
    """候选出口列表 [(说明, proxies 或 None), ...]，按优先级排列。"""
    routes: list[tuple[str, dict | None]] = []
    env_proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
                 or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
    routes.append(("direct", None))
    if env_proxy:
        routes.append((f"env-proxy {env_proxy}", {"http": env_proxy, "https": env_proxy}))
    for port in _COMMON_PROXY_PORTS:
        url = f"http://127.0.0.1:{port}"
        routes.append((f"local-proxy {url}", {"http": url, "https": url}))
    return routes


def _probe_route() -> dict | None:
    """找一个能连通维基百科的出口，结果缓存到进程内。

    代理抖动很常见（一次探测全挂、几十秒后又通），所以整体扫两轮再放弃。
    """
    if _route_cache["route"] is not None:
        return _route_cache["route"]
    if time.monotonic() < _route_cache.get("failed_until", 0.0):
        raise WikiError("维基百科 API 暂时不可达（本轮已探测过一遍，直接走本地缓存兜底）。")
    for attempt in range(2):
        for label, proxies in _routes():
            try:
                resp = requests.get(
                    API_URL.format(lang="zh"),
                    params={"action": "query", "format": "json", "titles": "图灵机", "prop": "info"},
                    timeout=4.0, proxies=proxies,
                    headers={"User-Agent": USER_AGENT},
                )
                if resp.status_code == 200 and "query" in resp.text:
                    _route_cache["route"] = {"label": label, "proxies": proxies}
                    return _route_cache["route"]
            except requests.RequestException:
                continue
        if attempt == 0:
            time.sleep(0.5)
    _route_cache["failed_until"] = time.monotonic() + _ROUTE_RETRY_COOLDOWN
    raise WikiError(
        "联网失败：维基百科 API 不可达。已尝试直连、环境变量代理与本地常见代理端口"
        f"（{', '.join(_COMMON_PROXY_PORTS)}）各两轮。若已开代理，请设置 HTTPS_PROXY=http://127.0.0.1:<端口>。")


def _http(params: dict, lang: str) -> dict:
    route = _probe_route()
    proxies = route.get("proxies") if isinstance(route, dict) else None
    try:
        resp = requests.get(API_URL.format(lang=lang), params=params,
                            timeout=_REQUEST_TIMEOUT, proxies=proxies,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise WikiError(f"维基百科请求失败（{lang}，出口 {route.get('label') if route else '?'}）：{e}") from e
    except ValueError as e:
        raise WikiError(f"维基百科返回了非 JSON 内容：{e}") from e


def _search_titles(query: str, lang: str, limit: int = 3) -> list[str]:
    data = _http({"action": "query", "list": "search", "srsearch": query,
                  "srlimit": limit, "format": "json", "formatversion": 2}, lang)
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def _fetch_extract(title: str, lang: str) -> tuple[str, str, str]:
    data = _http({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                  "redirects": 1, "titles": title, "format": "json", "formatversion": 2}, lang)
    pages = data.get("query", {}).get("pages", [])
    if isinstance(pages, dict):      # formatversion=1 的兼容分支
        pages = list(pages.values())
    for page in pages:
        extract = (page.get("extract") or "").strip()
        if extract:
            page_title = page.get("title", title)
            url = f"https://{lang}.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            return page_title, url, extract
    raise WikiError(f"{lang} 维基百科没有词条 {title!r} 的摘要")


_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")


def _key_sentences(text: str, limit: int = 3) -> list[str]:
    """挑出含年份的句子（小模型很容易漏掉长段落里的日期，单独提一行）。"""
    sentences = [s.strip() for s in re.split(r"(?<=[。！？；])|(?<=[.!?;])\s+", text) if s.strip()]
    picked: list[str] = []
    for sentence in sentences:
        if len(sentence) < 10 or not _YEAR_RE.search(sentence):
            continue
        if any(sentence[:60] == prev[:60] for prev in picked):
            continue
        picked.append(sentence[:240])
        if len(picked) >= limit:
            break
    return picked


def _lang_order(query: str, lang: str | None) -> list[str]:
    if lang in ("zh", "en"):
        return [lang, "en" if lang == "zh" else "zh"]
    return ["zh", "en"] if _has_cjk(query) else ["en", "zh"]


def _cache_path(query: str, lang: str) -> Path:
    digest = hashlib.sha1(f"{lang}:{query.strip().lower()}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{lang}-{digest}.json"


def _read_cache(query: str, lang: str) -> str | None:
    path = _cache_path(query, lang)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload.get("text")


_NON_KEY_WORDS = {"the", "and", "or", "of", "in", "for", "a", "an", "is", "are",
                  "to", "with", "on", "at", "by"}
# 这些是「查询意图」而不是「实体」，拿来匹配会误命中（如「李燦 发表年份」）
_GENERIC_CJK = ("发表年份", "发布年份", "出生年份", "出生日期", "出生", "年份",
                "发明者", "发明", "是谁", "介绍", "简介", "定义", "什么", "多少")


def _cache_keywords(query: str) -> list[str]:
    """从查询里抽「实体」关键词（拉丁词 + 中文片段），供离线时模糊匹配缓存。"""
    latin = [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", query.lower())
             if w not in _NON_KEY_WORDS]
    cjk = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}", query)
           if not any(g in t for g in _GENERIC_CJK)]
    return list(dict.fromkeys(latin + cjk))


_TITLE_RE = re.compile(r"词条[：:]\s*([^\n（(]+)")


def _entry_title(text: str) -> str:
    match = _TITLE_RE.search(text)
    return (match.group(1) if match else text[:40]).strip().lower()


def _relevant_title(query: str, title: str) -> bool:
    """词条标题要和查询有「实体词」重叠，否则多半是搜索接口返回的跑偏结果。"""
    keywords = _cache_keywords(query)
    if not keywords:
        return True
    low = title.lower()
    return any(kw in low for kw in keywords)


def _fuzzy_cache(query: str, langs: list[str]) -> str | None:
    """精确 key 没命中时，用实体关键词匹配**缓存里的词条标题**（离线 / 被墙时的兜底）。

    例：断网时查「图灵机 发明者」，能命中之前缓存过的「图灵机」词条。
    注意必须按词条标题匹配而不是按缓存的 query 字段——历史缓存里有 query 与内容
    对不上的污染条目（query 写着 transformer，正文却是 Google Play）。
    """
    keywords = _cache_keywords(query)
    if not keywords or not CACHE_DIR.exists():
        return None
    best_text, best_score = None, 0
    order = list(dict.fromkeys(list(langs) + ["zh", "en"]))
    for lang in order:
        for path in sorted(CACHE_DIR.glob(f"{lang}-*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            text = payload.get("text")
            if not text:
                continue
            score = sum(1 for kw in keywords if kw in _entry_title(str(text)))
            if score > best_score:
                best_text, best_score = str(text), score
    return best_text


def _write_cache(query: str, lang: str, text: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(query, lang).write_text(
            json.dumps({"query": query, "lang": lang, "text": text,
                        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError:
        pass


def run(args) -> str:
    if isinstance(args, str):
        args = {"query": args}
    if not isinstance(args, dict):
        raise WikiError(f"wiki 参数必须是对象，收到 {type(args).__name__}")

    query = None
    for key in ("query", "q", "term", "title", "entity", "name", "question", "keyword", "search"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            query = value.strip()
            break
    if query is None:
        raise WikiError(
            f"缺少 query 参数，收到的键为 {sorted(args)}；"
            '正确格式：{"query": "Geoffrey Hinton"}')

    lang_arg = args.get("lang")
    lang_arg = lang_arg.strip().lower()[:2] if isinstance(lang_arg, str) else None

    langs = _lang_order(query, lang_arg)
    errors: list[str] = []
    for lang in langs:
        try:
            titles: list[str] = []
            for candidate in _extract_keys(query, lang) + _search_titles(query, lang):
                if candidate and candidate not in titles:
                    titles.append(candidate)
            for title in titles:
                try:
                    page_title, url, extract = _fetch_extract(title, lang)
                except WikiError:
                    continue
                if not _relevant_title(query, page_title):
                    # 搜索接口偶尔返回完全不相关的条目（查 transformer 给出 Google Play），
                    # 这种结果宁可不用，继续试下一个候选。
                    continue
                lines = [f"词条：{page_title}（{lang} 维基百科）"]
                key = _key_sentences(extract)
                if key:
                    # 关键句提到最前面：长简介会把 0.5B 这类小模型的注意力冲散，年份就漏了
                    lines.append("关键句（含年份）：" + " ".join(key))
                lines.append("简介：" + extract[:_MAX_EXTRACT_CHARS])
                lines.append(f"链接：{url}")
                text = "\n".join(lines)
                _write_cache(query, lang, text)
                return text
            errors.append(f"{lang}: 未找到匹配词条")
        except WikiError as e:
            errors.append(f"{lang}: {e}")

    # 全部走不通：退回本地缓存（离线可用）
    for lang in langs:
        cached = _read_cache(query, lang)
        if cached:
            return cached + "\n（本条来自本地离线缓存，可能不是最新内容）"
    fuzzy = _fuzzy_cache(query, langs)
    if fuzzy:
        return fuzzy + "\n（本条来自本地离线缓存的关键词匹配结果，可能不是最新内容）"
    raise WikiError("；".join(errors) or "维基百科查询失败")
