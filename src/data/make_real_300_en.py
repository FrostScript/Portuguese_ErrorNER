"""英文对照实验：构造 E1en 训练集 —— 300 条真实英文句，45% 无实体 + 55% 含实体。

**与葡语版的关系：** `src/data/make_real_300.py` 的副本，抽样口径逐条对应：
    去重规则、随机种子 42、135/165 配比、每类 ≥30 mention、类别分层后随机补足。
按隔离原则，不 import 也不修改葡语版。

数据源：`data/en/uner_en/train.jsonl`（UNER English-EWT，CC-BY-SA-4.0）

产物：
    data/en/curated/real_300_en.jsonl
    data/en/curated/real_300_en_manifest.json
    data/en/curated/real_300_en_stats.md

用法：
    & "...\\.venv\\Scripts\\python.exe" src/data/make_real_300_en.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: F401  统一环境

DATA = WORKSPACE / "data" / "en" / "uner_en"
OUT_DIR = WORKSPACE / "data" / "en" / "curated"

SEED = 42
N_NO_ENTITY = 135          # 45%
N_ENTITY = 165             # 55%
MIN_PER_CLASS = 30         # 每类最少 mention 数
CLASSES = ["PER", "ORG", "LOC"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").open(encoding="utf-8")]


def main() -> int:
    train, dev, test = load("train"), load("dev"), load("test")

    # ---------- 去重：内部重复保留首条；与 dev/test 重复的从 train 侧剔除
    blocked = {r["text"] for r in dev} | {r["text"] for r in test}
    seen: set[str] = set()
    pool, dropped_internal, dropped_cross = [], 0, 0
    for rec in train:
        if rec["text"] in blocked:
            dropped_cross += 1
            continue
        if rec["text"] in seen:
            dropped_internal += 1
            continue
        seen.add(rec["text"])
        pool.append(rec)

    no_ent = [r for r in pool if not r["entities"]]
    with_ent = [r for r in pool if r["entities"]]
    print(f"[池] 去重后 {len(pool)} 条（剔除内部重复 {dropped_internal}、跨集重复 {dropped_cross}）")
    print(f"     无实体句 {len(no_ent)}，含实体句 {len(with_ent)}")

    if len(no_ent) < N_NO_ENTITY or len(with_ent) < N_ENTITY:
        print("FAIL: 池子不足")
        return 1

    rng = random.Random(SEED)

    # ---------- 无实体句：直接随机抽
    picked_no = rng.sample(no_ent, N_NO_ENTITY)

    # ---------- 含实体句：先保证类别覆盖，再随机补足
    shuffled = with_ent[:]
    rng.shuffle(shuffled)
    picked_ent: list[dict] = []
    taken: set[str] = set()
    counts = Counter()

    for cls in CLASSES:
        for rec in shuffled:
            if counts[cls] >= MIN_PER_CLASS:
                break
            if rec["id"] in taken:
                continue
            labels = [e["label"] for e in rec["entities"]]
            if cls in labels:
                picked_ent.append(rec)
                taken.add(rec["id"])
                counts.update(labels)

    for rec in shuffled:
        if len(picked_ent) >= N_ENTITY:
            break
        if rec["id"] in taken:
            continue
        picked_ent.append(rec)
        taken.add(rec["id"])
        counts.update(e["label"] for e in rec["entities"])

    if len(picked_ent) != N_ENTITY:
        print(f"FAIL: 含实体句只抽到 {len(picked_ent)}")
        return 1

    # ---------- 落盘
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for rec in picked_no:
        records.append({**rec, "subset": "no_entity", "source_split": "train"})
    for rec in picked_ent:
        records.append({**rec, "subset": "entity", "source_split": "train"})
    records.sort(key=lambda r: r["id"])

    out_jsonl = OUT_DIR / "real_300_en.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ent_counts = Counter(e["label"] for rec in records for e in rec["entities"])
    n_ent = sum(len(r["entities"]) for r in records)
    ent_per_sentence = [len(r["entities"]) for r in records if r["entities"]]

    manifest = {
        "created_by": "src/data/make_real_300_en.py",
        "seed": SEED,
        "source_file": str(DATA / "train.jsonl"),
        "source_sha256": sha256(DATA / "train.jsonl"),
        "output_file": str(out_jsonl),
        "output_sha256": sha256(out_jsonl),
        "dropped": {"internal_duplicates": dropped_internal, "cross_split_duplicates": dropped_cross},
        "counts": {
            "sentences": len(records),
            "no_entity": len(picked_no),
            "entity": len(picked_ent),
            "no_entity_ratio": round(len(picked_no) / len(records), 4),
            "entities": n_ent,
            "entities_per_sentence_all": round(n_ent / len(records), 4),
            "entities_per_sentence_entity_only": round(n_ent / len(picked_ent), 4),
            "per_class": dict(ent_counts),
            "entity_count_histogram": dict(sorted(Counter(ent_per_sentence).items())),
            "documents": len({r["doc_id"] for r in records}),
            "tokens": sum(len(r["tokens"]) for r in records),
        },
        "ids": [r["id"] for r in records],
    }
    (OUT_DIR / "real_300_en_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    c = manifest["counts"]
    md = [
        "# E1en 训练集 real_300_en 统计",
        "",
        f"- 生成脚本：`src/data/make_real_300_en.py`，随机种子 **{SEED}**",
        f"- 源文件：`data/en/uner_en/train.jsonl`（sha256 `{manifest['source_sha256'][:16]}…`）",
        f"- 输出：`data/en/curated/real_300_en.jsonl`（sha256 `{manifest['output_sha256'][:16]}…`）",
        "",
        "## 规模",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| 句子总数 | {c['sentences']} |",
        f"| 无实体句 | {c['no_entity']}（{c['no_entity_ratio'] * 100:.1f}%） |",
        f"| 含实体句 | {c['entity']} |",
        f"| 实体总数 | {c['entities']} |",
        f"| 实体/句（全体） | {c['entities_per_sentence_all']} |",
        f"| 实体/句（仅含实体句） | {c['entities_per_sentence_entity_only']} |",
        f"| 覆盖文档数 | {c['documents']} |",
        f"| token 总数 | {c['tokens']} |",
        "",
        "## 类别分布",
        "",
        "| 类别 | mention 数 |",
        "|---|---:|",
    ]
    for cls in CLASSES:
        md.append(f"| {cls} | {ent_counts.get(cls, 0)} |")
    md += [
        "",
        "## 含实体句的实体数分布",
        "",
        "| 每句实体数 | 条数 |",
        "|---|---:|",
    ]
    for k, v in manifest["counts"]["entity_count_histogram"].items():
        md.append(f"| {k} | {v} |")
    md += [
        "",
        "## 剔除情况",
        "",
        f"- train 内部完全重复句：{dropped_internal} 条",
        f"- 与 dev/test 完全重复句：{dropped_cross} 条",
        "",
        "> 该文件是 E1en 的唯一训练数据来源；20 条生成种子将从这 300 条中选取。",
    ]
    (OUT_DIR / "real_300_en_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[输出] {out_jsonl}")
    print(f"[输出] {OUT_DIR / 'real_300_en_manifest.json'}")
    print(f"[输出] {OUT_DIR / 'real_300_en_stats.md'}")
    print(json.dumps(c, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
