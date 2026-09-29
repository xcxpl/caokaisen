"""任务六 · 中层：Skill 加载器（约 50 行核心逻辑）。

对齐 Anthropic Skills 约定：``src/skills/<name>/SKILL.md``，头部是 YAML front-matter
（至少 name / description），下面是正文 workflow。

关键点是 **progressive disclosure**（渐进式披露）：
* ``list_skills()`` 只解析 front-matter 的元数据，**不读正文**；
* ``match(query)`` 用 description 与任务做关键词匹配，命中哪些 skill；
* 只有 ``load(name)`` 才把该 SKILL.md 的正文读出来塞进 context。

于是一开始往 context 里放的只有几行 description，而不是全部 Skill 正文。
"""
from __future__ import annotations

import re
from pathlib import Path

_FRONT_MATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", re.S)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")
_STOPWORDS = {"the", "and", "or", "of", "in", "for", "with", "to", "a", "an", "is",
              "are", "be", "on", "at", "by", "it", "this", "that", "as"}


def parse_front_matter(text: str) -> tuple[dict, str]:
    """拆出 YAML front-matter（只支持扁平的 key: value）与正文。"""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[match.end():]


def tokenize(text: str) -> set[str]:
    """粗分词：拉丁词 + 中文片段，够用且零依赖。"""
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


class SkillLoader:
    """扫描 ``<skills_dir>/*/SKILL.md``，按 description 匹配、命中后再读正文。"""

    def __init__(self, skills_dir):
        self.root = Path(skills_dir).expanduser().resolve()

    def skill_files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        found = sorted(self.root.glob("*/SKILL.md"))
        if (self.root / "SKILL.md").is_file():
            found.insert(0, self.root / "SKILL.md")
        return found

    def list_skills(self) -> list[dict]:
        """只读 front-matter，返回 [{name, description, path}]（正文留到 load）。"""
        skills = []
        for path in self.skill_files():
            try:
                meta, _ = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            name = meta.get("name") or path.parent.name
            skills.append({
                "name": name,
                "description": meta.get("description", ""),
                "path": str(path),
                "dir": str(path.parent),
                "front_matter": meta,
            })
        return skills

    def get(self, name: str) -> dict | None:
        for skill in self.list_skills():
            if skill["name"] == name or Path(skill["path"]).parent.name == name:
                return skill
        return None

    def load(self, name: str) -> str:
        """命中后才读正文（progressive disclosure 的第二段）。"""
        skill = self.get(name)
        if skill is None:
            raise KeyError(f"没有名为 {name} 的 Skill")
        text = Path(skill["path"]).read_text(encoding="utf-8", errors="replace")
        _, body = parse_front_matter(text)
        return body.strip()

    def match(self, query: str, top_k: int = 2, min_score: float = 0.0) -> list[str]:
        """按 description / name 与 query 的词重叠度排序，返回命中的 skill 名。"""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored: list[tuple[float, str]] = []
        for skill in self.list_skills():
            tokens = tokenize(skill["description"] + " " + skill["name"])
            if not tokens:
                continue
            overlap = len(query_tokens & tokens)
            score = overlap / len(tokens) ** 0.5
            if overlap and score > min_score:
                scored.append((score, skill["name"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored[:top_k]]

    def describe(self) -> str:
        """放进 system prompt 的紧凑清单（只有 name + description）。"""
        return "\n".join(f"- {s['name']}: {s['description']}" for s in self.list_skills())
