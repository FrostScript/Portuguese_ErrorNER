"""项目 JSONL（字符区间）→ GLiNER 训练格式（词级闭区间）。

GLiNER 训练样本格式：
    {"tokenized_text": [...], "ner": [[start_word, end_word, "person"], ...]}

换算规则（见 docs/specs/03-labeling.md §4）：
    1. 按 token 在 text 中顺序定位，得到每个 token 的起始字符偏移；
    2. 实体起点 = offsets[i] == start 的 token 下标；
    3. 实体终点 = offsets[i] + len(tokens[i]) == end 的 token 下标；
    4. 任一端找不到则丢弃该实体并计数。
"""

from __future__ import annotations

import json
from pathlib import Path

LABEL_MAP = {"PER": "person", "ORG": "organization", "LOC": "location"}


def token_offsets(text: str, tokens: list[str]) -> list[int] | None:
    offsets = []
    cursor = 0
    for tok in tokens:
        idx = text.find(tok, cursor)
        if idx < 0:
            return None
        offsets.append(idx)
        cursor = idx + len(tok)
    return offsets


def to_gliner(rec: dict) -> tuple[dict | None, int]:
    """返回 (训练样本, 丢弃的实体数)；token 定位失败时样本为 None。"""
    tokens = rec["tokens"]
    text = rec["text"]
    offsets = token_offsets(text, tokens)
    if offsets is None:
        return None, len(rec["entities"])

    ner = []
    dropped = 0
    for ent in rec["entities"]:
        s_c = [i for i, off in enumerate(offsets) if off == ent["start"]]
        e_c = [i for i, off in enumerate(offsets) if off + len(tokens[i]) == ent["end"]]
        if not s_c or not e_c:
            dropped += 1
            continue
        ner.append([s_c[0], e_c[0], LABEL_MAP[ent["label"]]])
    return {"tokenized_text": tokens, "ner": ner}, dropped


def load_records(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def load_examples(path: str | Path) -> tuple[list[dict], dict]:
    """读项目 JSONL，转成 GLiNER 训练样本，并返回转换统计。"""
    records = load_records(path)
    examples, dropped, failed = [], 0, 0
    for rec in records:
        ex, d = to_gliner(rec)
        dropped += d
        if ex is None:
            failed += 1
            continue
        examples.append(ex)
    stats = {
        "records": len(records),
        "examples": len(examples),
        "tokenize_failed": failed,
        "entities_dropped": dropped,
        "entities_kept": sum(len(e["ner"]) for e in examples),
    }
    return examples, stats
