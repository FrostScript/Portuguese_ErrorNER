"""生成数据的校验器：把 LLM 输出转成项目 JSONL 格式，并拦截不合格样本。

设计依据：docs/specs/04（句长/密度/模板化）、docs/specs/05（135/165 负样本比例）。

核心决策：**不信任模型给的字符下标**。模型只输出实体文本 + 类别，
本模块在句子中定位并计算区间；定位失败或多处歧义即判无效。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

LABELS = {"PER", "ORG", "LOC"}
MIN_TOKENS, MAX_TOKENS = 20, 30
MAX_ENTITIES = 3
NGRAM = 8

WORD_RE = re.compile(r"\S+")


@dataclass
class Record:
    text: str
    entities: list[dict] = field(default_factory=list)

    @property
    def tokens(self) -> list[str]:
        return WORD_RE.findall(self.text)


@dataclass
class Verdict:
    ok: bool
    record: dict | None = None
    issues: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def locate(text: str, surface: str) -> tuple[int, int, int]:
    """返回 (start, end, 出现次数)；找不到则 start=-1。"""
    n = len(re.findall(re.escape(surface), text))
    idx = text.find(surface)
    if idx < 0:
        return -1, -1, 0
    return idx, idx + len(surface), n


def validate_sentence(
    raw: dict,
    seeds_texts: list[str],
    seen_texts: set[str],
    expect_entities: bool | None = None,
    id_prefix: str = "gen",
) -> Verdict:
    """校验单条生成样本。expect_entities=True/False 强制要求有/无实体，None 不检查。"""
    issues: list[str] = []

    if not isinstance(raw, dict):
        return Verdict(False, None, ["不是 JSON 对象"])
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return Verdict(False, None, ["text 缺失或为空"])
    text = _normalize(text)

    ents_raw = raw.get("entities", [])
    if not isinstance(ents_raw, list):
        return Verdict(False, None, ["entities 不是列表"])

    tokens = WORD_RE.findall(text)
    if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
        issues.append(f"句长 {len(tokens)} 不在 {MIN_TOKENS}-{MAX_TOKENS}")

    # 完全重复
    key = text.lower()
    if key in seen_texts:
        issues.append("与已接受样本完全重复")

    # 照抄种子
    if any(text == s or (len(text) > 40 and text in s) or (len(s) > 40 and s in text) for s in seeds_texts):
        issues.append("与种子文本高度重合（疑似照抄）")

    # 8-gram 与种子重叠
    for s in seeds_texts:
        if _ngram_overlap(text, s, NGRAM):
            issues.append(f"与种子存在 {NGRAM}-gram 重叠")
            break

    entities, spans = [], []
    for e in ents_raw:
        if not isinstance(e, dict):
            issues.append("实体不是对象")
            continue
        surface = (e.get("text") or "").strip()
        label = (e.get("label") or "").strip().upper()
        if label not in LABELS:
            issues.append(f"非法标签 {label!r}")
            continue
        if not surface:
            issues.append("实体文本为空")
            continue
        start, end, count = locate(text, surface)
        if start < 0:
            issues.append(f"实体 {surface!r} 在句中找不到")
            continue
        if count > 1:
            issues.append(f"实体 {surface!r} 在句中重复出现 {count} 次，区间有歧义")
            continue
        spans.append((start, end))
        entities.append({"start": start, "end": end, "text": surface, "label": label})

    # 重叠检查
    spans.sort()
    for a, b in zip(spans, spans[1:]):
        if b[0] < a[1]:
            issues.append("实体区间重叠")
            break

    if len(entities) > MAX_ENTITIES:
        issues.append(f"实体数 {len(entities)} 超过上限 {MAX_ENTITIES}")

    if expect_entities is True and not entities:
        issues.append("要求含实体但标不出任何实体")
    if expect_entities is False and entities:
        issues.append("要求无实体但给出了实体")

    if issues:
        return Verdict(False, None, issues)

    return Verdict(
        True,
        {"id": "", "doc_id": f"{id_prefix}", "split": "synthetic", "tokens": tokens,
         "text": text, "entities": entities},
        [],
    )


def _ngram_overlap(a: str, b: str, n: int) -> bool:
    ta, tb = WORD_RE.findall(a.lower()), WORD_RE.findall(b.lower())
    if len(ta) < n or len(tb) < n:
        return False
    ga = {" ".join(ta[i:i + n]) for i in range(len(ta) - n + 1)}
    gb = {" ".join(tb[i:i + n]) for i in range(len(tb) - n + 1)}
    return bool(ga & gb)


def parse_batch(raw_text: str) -> tuple[list[dict], str | None]:
    """从模型响应里解析出 sentences 列表。返回 (列表, 错误)。"""
    s = raw_text.strip()
    # 容忍 ```json 围栏
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        return [], f"JSON 解析失败: {exc}"
    if not isinstance(data, dict) or "sentences" not in data:
        return [], "缺少 sentences 字段"
    if not isinstance(data["sentences"], list):
        return [], "sentences 不是列表"
    return data["sentences"], None


def validate_batch(
    sentences: list[dict],
    seeds_texts: list[str],
    expect_entities: bool | None = None,
) -> tuple[list[dict], list[dict], Counter]:
    """校验整批，返回 (已接受, 被拒明细, 统计)。批内去重。"""
    accepted: list[dict] = []
    rejected: list[dict] = []
    stats: Counter = Counter()

    seen: set[str] = set()
    for i, raw in enumerate(sentences):
        v = validate_sentence(raw, seeds_texts, seen, expect_entities=expect_entities)
        stats["total"] += 1
        if v.ok:
            seen.add(v.record["text"].lower())
            accepted.append(v.record)
            stats["accepted"] += 1
            stats["with_entities" if v.record["entities"] else "no_entities"] += 1
        else:
            rejected.append({"index": i, "issues": v.issues, "text": (raw or {}).get("text", "")[:80]})
            stats["rejected"] += 1
    return accepted, rejected, stats


def class_coverage(records: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for e in r["entities"]:
            c[e["label"]] += 1
    return c


# ----------------------------------------------------------------- 自检
def _selftest() -> int:
    seeds = [
        "Vera Lúcia Barbosa, 40, morreu em a última quarta presa por o cinto de segurança a o carro em que estava, que caiu em o rio Tietê."
    ]
    good_no = {
        "text": "O índice de preços ao consumidor subiu dois por cento no mês passado, segundo dados divulgados nesta terça feira pelo instituto de pesquisa.",
        "entities": [],
    }
    good_ent = {
        "text": "A estatal informou nesta segunda feira que o investimento previsto para o próximo ano será ampliado em cerca de trinta por cento.",
        "entities": [{"text": "estatal", "label": "ORG"}],
    }
    cases = [
        ("合法-无实体", good_no, False, True),
        ("合法-含实体", good_ent, True, True),
        ("非法-要求无实体却给了", good_ent, False, False),
        ("非法-实体找不到", {"text": good_ent["text"], "entities": [{"text": "Petrobras", "label": "ORG"}]}, True, False),
        ("非法-标签错", {"text": good_ent["text"], "entities": [{"text": "estatal", "label": "GPE"}]}, True, False),
        ("非法-太短", {"text": "Curto demais.", "entities": []}, False, False),
        ("非法-照抄种子", {"text": seeds[0], "entities": [{"text": "Vera Lúcia Barbosa", "label": "PER"}]}, True, False),
        ("非法-重复实体歧义", {"text": "Banco X disse que Banco X vai crescer muito no próximo ano fiscal com novas agências em várias cidades do interior.", "entities": [{"text": "Banco X", "label": "ORG"}]}, True, False),
    ]
    print(f"{'用例':<22} {'期望':<6} {'实际':<6} 问题")
    bad = 0
    seen: set[str] = set()
    for name, payload, expect, want_ok in cases:
        v = validate_sentence(payload, seeds, set(seen), expect_entities=expect)
        got = "ok" if v.ok else "reject"
        mark = "✓" if (got == "ok") == want_ok else "✗"
        if mark == "✗":
            bad += 1
        print(f"{name:<22} {'ok' if want_ok else 'reject':<6} {got:<6} {mark} {'; '.join(v.issues)[:70]}")
        if v.ok:
            seen.add(v.record["text"].lower())
    print(f"\n自检结果：{'全部通过' if bad == 0 else f'{bad} 个用例不符合预期'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
