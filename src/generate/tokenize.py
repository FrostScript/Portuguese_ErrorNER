"""把自由文本切成「词 + 单独标点」的 token 序列，并给出字符偏移。

为什么需要：合成文本里标点紧贴单词（`Braga,` `público.`），若用 `text.split()` 当 token，
实体的结束位置会落在 token 中间，GLiNER 的词级区间映射就失败（实测 413 个实体丢了 147 个）。
真实数据（UD 分词）标点是独立 token，这里按同样口径切分。

用法：
    tokens, spans = tokenize("Em Braga, a equipa...")
    # tokens = ["Em", "Braga", ",", "a", "equipa", ...]
    # spans  = [(0,2), (3,8), (8,9), ...]
"""

from __future__ import annotations

import re

# \w 含 Unicode 字母（支持葡语重音）；连字符与撇号保留在词内（如 `pós-guerra`、`d'água`）
TOKEN_RE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens, spans = [], []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def align_entities(text: str, entities: list[dict]) -> tuple[list[dict], int]:
    """把字符区间实体对齐到 token 边界。返回 (对齐后的实体, 丢弃数)。"""
    _, spans = tokenize(text)
    starts = {s: i for i, (s, _) in enumerate(spans)}
    ends = {e: i for i, (_, e) in enumerate(spans)}
    aligned, dropped = [], 0
    for ent in entities:
        si = starts.get(ent["start"])
        ei = ends.get(ent["end"])
        if si is None or ei is None or ei < si:
            dropped += 1
            continue
        aligned.append({**ent, "start_token": si, "end_token": ei})
    return aligned, dropped
