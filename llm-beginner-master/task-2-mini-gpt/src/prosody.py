"""格律（近体诗）约束生成所需的音韵辅助。

- 判断汉字 / 标点。
- 用 pypinyin 取韵母，做简化的押韵判断（去掉介音 i/u/ü 取“韵核”，
  近似宽押，让 guang/fang/xiang 这类同韵字都能押上）。
"""

from __future__ import annotations

import re

try:
    from pypinyin import lazy_pinyin, Style
    _HAS_PINYIN = True
except Exception:  # pragma: no cover
    _HAS_PINYIN = False


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u300a\u300b\u3001\uff0c\,\.\!\?\;\"\'\n]")


def is_cjk(text: str) -> bool:
    """是否为单个汉字（只含一个 CJK 字符）。"""
    return bool(_CJK_RE.fullmatch(text))


def is_punct(text: str) -> bool:
    """是否为（去空白后的）单个标点。"""
    text = text.strip()
    return bool(text) and bool(_PUNCT_RE.fullmatch(text))


def count_cjk(text: str) -> int:
    """统计文本里 CJK 汉字个数。"""
    return len(_CJK_RE.findall(text))


def finals_of(ch: str) -> str:
    """取一个汉字的韵母（如 '光' -> 'uang'）。失败返回空串。"""
    if not _HAS_PINYIN:
        return ""
    try:
        f = lazy_pinyin(ch, style=Style.FINALS, errors="ignore")
        return f[0] if f else ""
    except Exception:
        return ""


def _rime_core(final: str) -> str:
    """去掉前导介音(i/u/ü)得到韵核，用于宽押判断。"""
    f = final.lower()
    while len(f) > 1 and f[0] in "iu\u00fc":
        f = f[1:]
    return f


def rhymes(a: str, b: str) -> bool:
    """两个汉字是否近似押韵（韵母相同或韵核相同）。"""
    fa, fb = finals_of(a), finals_of(b)
    if not fa or not fb:
        return False
    return fa == fb or _rime_core(fa) == _rime_core(fb)


def can_rhyme(ch: str) -> bool:
    """该汉字能否作为压韵位置（必须有韵母）。"""
    return bool(finals_of(ch))
