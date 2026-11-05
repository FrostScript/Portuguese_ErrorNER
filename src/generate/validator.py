"""项目级规范检查：跑在 `prompts/templates/convert_validate.py` **之后**。

分工：
    convert_validate.py  负责「格式与偏移」——ID/文本唯一、occurrence 定位、区间不重叠；
    本模块              负责「项目规范」——句长、实体密度、抄种子、跨批重复、负样本比例。

这样职责不重叠：格式错误在第一步就被拦住，这里只判"能不能进训练集"。

用法：
    & "...\\.venv\\Scripts\\python.exe" src/generate/validator.py checked.jsonl `
        --seeds data/curated/seeds_20.jsonl --expect entity

    --expect auto    不限制有无实体
    --expect entity  要求每条至少 1 个实体（E2/E3 含实体批次）
    --expect none    要求每条 0 个实体（无实体批次）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: F401

LABELS = {"PER", "ORG", "LOC"}
# 句长按「空格切词」计（生成文本没有 UD 分词，只能用空格）。
# 换算：空格词数 / UD token ≈ 0.861（种子实测），故 UD 20–30 对应空格 17–26，
# 这里取 17–30 作为下界对齐种子最短句、上界留余量。
MIN_TOKENS, MAX_TOKENS = 17, 30
# E1 真实数据中含实体句的实体数分布为 1:70 / 2:44 / 3:26 / 4:11 / 5:7 / 6+:7，
# 15% 的句子超过 3 个实体，因此上界取 5 而不是 3，避免合成数据系统性偏离 E1。
MAX_ENTITIES = 5
NGRAM = 8
WORD_RE = re.compile(r"\S+")


def tokens_of(text: str) -> list[str]:
    return WORD_RE.findall(text)


def ngram_overlap(a: str, b: str, n: int = NGRAM) -> bool:
    ta, tb = [w.lower() for w in tokens_of(a)], [w.lower() for w in tokens_of(b)]
    if len(ta) < n or len(tb) < n:
        return False
    ga = {" ".join(ta[i:i + n]) for i in range(len(ta) - n + 1)}
    gb = {" ".join(tb[i:i + n]) for i in range(len(tb) - n + 1)}
    return bool(ga & gb)


def check_record(rec: dict, seeds_texts: list[str], seen_texts: set[str]) -> list[str]:
    """返回问题列表；空列表表示通过。假定格式已由 convert_validate.py 校验。"""
    issues: list[str] = []
    text = rec["text"]
    ents = rec["entities"]

    toks = tokens_of(text)
    if not (MIN_TOKENS <= len(toks) <= MAX_TOKENS):
        issues.append(f"句长 {len(toks)} 不在 {MIN_TOKENS}-{MAX_TOKENS}")

    if len(ents) > MAX_ENTITIES:
        issues.append(f"实体数 {len(ents)} 超过上限 {MAX_ENTITIES}")

    for e in ents:
        if e["label"] not in LABELS:
            issues.append(f"非法标签 {e['label']}")
        if text[e["start"]:e["end"]] != e["text"]:
            issues.append(f"区间与实体文本不符: {e['text']!r}")

    if text.lower() in seen_texts:
        issues.append("与其他样本完全重复")

    for s in seeds_texts:
        if text == s:
            issues.append("与种子完全相同")
            break
        if ngram_overlap(text, s):
            issues.append(f"与种子存在 {NGRAM}-gram 重叠")
            break

    return issues


def check_file(path: Path, seeds_texts: list[str], expect: str = "auto") -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    seen: set[str] = set()
    rejected: list[dict] = []
    stats: Counter = Counter()

    for i, rec in enumerate(rows):
        issues = check_record(rec, seeds_texts, seen)
        if expect == "entity" and not rec["entities"]:
            issues.append("要求含实体但 entities 为空")
        if expect == "none" and rec["entities"]:
            issues.append("要求无实体但给出了实体")
        if issues:
            rejected.append({"line": i + 1, "id": rec.get("id"), "issues": issues})
            stats["rejected"] += 1
            continue
        seen.add(rec["text"].lower())
        stats["accepted"] += 1
        stats["with_entities" if rec["entities"] else "no_entities"] += 1
        for e in rec["entities"]:
            stats["ent_" + e["label"]] += 1

    total = stats["accepted"] + stats["rejected"]
    return {
        "file": str(path),
        "total": total,
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "no_entities": stats["no_entities"],
        "with_entities": stats["with_entities"],
        "no_entity_ratio": round(stats["no_entities"] / total, 4) if total else 0.0,
        "per_class": {k.replace("ent_", ""): stats[k] for k in ("ent_PER", "ent_ORG", "ent_LOC") if stats[k]},
        "rejected_detail": rejected[:20],
    }


def _selftest() -> int:
    seed = "Vera Lúcia Barbosa, 40, morreu em a última quarta presa por o cinto de segurança a o carro em que estava, que caiu em o rio Tietê."
    ok_text = ("O índice de preços ao consumidor subiu dois por cento no mês passado, "
               "segundo dados divulgados nesta terça feira pelo instituto de pesquisa nacional.")
    ok_ent = {"id": "b1", "text": ok_text,
              "entities": [{"start": 3, "end": 8, "text": ok_text[3:8], "label": "PER"}]}
    cases = [
        ("合法-无实体", {"id": "b0", "text": ok_text, "entities": []}, [], "auto", True),
        ("合法-含实体", ok_ent, [], "auto", True),
        ("非法-太短", {"id": "b2", "text": "Curto demais.", "entities": []}, [], "auto", False),
        ("非法-抄种子", {"id": "b3", "text": seed, "entities": []}, [seed], "auto", False),
        ("非法-实体数超限", {"id": "b4", "text": ok_text, "entities": [
            {"start": 0, "end": 1, "text": ok_text[0:1], "label": "PER"},
            {"start": 2, "end": 3, "text": ok_text[2:3], "label": "PER"},
            {"start": 4, "end": 5, "text": ok_text[4:5], "label": "PER"},
            {"start": 6, "end": 7, "text": ok_text[6:7], "label": "PER"},
            {"start": 8, "end": 9, "text": ok_text[8:9], "label": "PER"},
            {"start": 10, "end": 11, "text": ok_text[10:11], "label": "PER"}]}, [], "auto", False),
        ("非法-区间不符", {"id": "b5", "text": ok_text,
                          "entities": [{"start": 0, "end": 3, "text": "ZZZ", "label": "PER"}]}, [], "auto", False),
    ]
    bad = 0
    print(f"{'用例':<18} {'期望':<7} {'实际':<7} 问题")
    for name, rec, seeds, expect, want_ok in cases:
        issues = check_record(rec, seeds, set())
        if expect == "entity" and not rec["entities"]:
            issues.append("要求含实体但 entities 为空")
        got_ok = not issues
        mark = "✓" if got_ok == want_ok else "✗"
        bad += 0 if got_ok == want_ok else 1
        print(f"{name:<18} {'ok' if want_ok else 'reject':<7} {'ok' if got_ok else 'reject':<7} {mark} {'; '.join(issues)[:60]}")
    print(f"\n自检：{'全部通过' if bad == 0 else f'{bad} 项不符合预期'}")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="convert_validate.py 的输出 JSONL；省略则跑自检")
    ap.add_argument("--seeds", default=str(WORKSPACE / "data" / "curated" / "seeds_20.jsonl"))
    ap.add_argument("--expect", choices=["auto", "entity", "none"], default="auto")
    ap.add_argument("--report", default=None, help="把结果写到该 JSON")
    args = ap.parse_args()

    if not args.input:
        return _selftest()

    seeds_texts = [json.loads(l)["text"] for l in Path(args.seeds).open(encoding="utf-8") if l.strip()]
    result = check_file(Path(args.input), seeds_texts, args.expect)
    print(json.dumps({k: v for k, v in result.items() if k != "rejected_detail"}, ensure_ascii=False, indent=2))
    if result["rejected_detail"]:
        print("\n被拒样例：")
        for r in result["rejected_detail"][:5]:
            print(f"  第{r['line']}行 {r['id']}: {'; '.join(r['issues'])}")
    if args.report:
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入 {args.report}")
    return 0 if result["rejected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
