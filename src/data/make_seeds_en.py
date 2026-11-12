"""英文对照实验：从 E1en 训练集 real_300_en 中固定 20 条生成种子。

**与葡语版的关系：** `src/data/make_seeds.py` 的副本，约束逐条对应：
    含实体、句长 20–30、每类 ≥3 次、每文档 ≤2 条、随机种子 42、确定性。
按隔离原则不 import 也不修改葡语版。

**关于句长 20–30：** 葡语侧的候选池是 43/165 条，英文侧是 47/165 条（按 token 计），
池子规模相当，因此沿用同一区间不需要放宽（见 `docs/specs/11` P-EN4 的说明：
窗口对英文而言偏高，但这是已登记的已知偏差，不在准备阶段私自修改）。

产物：
    data/en/curated/seeds_20_en.jsonl          20 条种子（原记录 + seed_idx）
    data/en/curated/seeds_20_en_manifest.json  种子、ID 清单、类别计数、sha256
    data/en/curated/seeds_20_en_stats.md       人读说明
    prompts/templates_en/data/seeds_20_en.json 供 Prompt 注入的 JSON 数组（同葡语版格式）

用法：
    & "...\\.venv\\Scripts\\python.exe" src/data/make_seeds_en.py
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
from src.common import env  # noqa: F401

SRC = WORKSPACE / "data" / "en" / "curated" / "real_300_en.jsonl"
OUT_DIR = WORKSPACE / "data" / "en" / "curated"
PROMPT_DATA = WORKSPACE / "prompts" / "templates_en" / "data"

SEED = 42
N_SEEDS = 20
MIN_TOKENS, MAX_TOKENS = 20, 30
MIN_PER_CLASS = 3
MAX_PER_DOC = 2
CLASSES = ["PER", "ORG", "LOC"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8")]

    pool = [
        r
        for r in rows
        if r["entities"] and MIN_TOKENS <= len(r["tokens"]) <= MAX_TOKENS
    ]
    print(f"[池] real_300_en 共 {len(rows)} 条，含实体且句长 {MIN_TOKENS}-{MAX_TOKENS} 的候选 {len(pool)} 条")

    rng = random.Random(SEED)
    rng.shuffle(pool)

    picked: list[dict] = []
    taken_ids: set[str] = set()
    doc_count: Counter = Counter()
    class_count: Counter = Counter()

    def can_take(rec: dict) -> bool:
        return rec["id"] not in taken_ids and doc_count[rec["doc_id"]] < MAX_PER_DOC

    for cls in CLASSES:
        for rec in pool:
            if class_count[cls] >= MIN_PER_CLASS:
                break
            if not can_take(rec):
                continue
            if cls not in {e["label"] for e in rec["entities"]}:
                continue
            picked.append(rec)
            taken_ids.add(rec["id"])
            doc_count[rec["doc_id"]] += 1
            class_count.update(e["label"] for e in rec["entities"])

    for rec in pool:
        if len(picked) >= N_SEEDS:
            break
        if not can_take(rec):
            continue
        picked.append(rec)
        taken_ids.add(rec["id"])
        doc_count[rec["doc_id"]] += 1
        class_count.update(e["label"] for e in rec["entities"])

    if len(picked) != N_SEEDS:
        print(f"FAIL: 只选到 {len(picked)} 条")
        return 1
    for cls in CLASSES:
        if class_count[cls] < MIN_PER_CLASS:
            print(f"FAIL: {cls} 只有 {class_count[cls]} 次，未达下限 {MIN_PER_CLASS}")
            return 1

    picked.sort(key=lambda r: r["id"])
    out_jsonl = OUT_DIR / "seeds_20_en.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(picked, 1):
            fh.write(json.dumps({**rec, "seed_idx": i}, ensure_ascii=False) + "\n")

    # Prompt 注入用：JSON 数组，字段与葡语版一致（text + entities[{text,label}]）
    PROMPT_DATA.mkdir(parents=True, exist_ok=True)
    prompt_seeds = [
        {
            "text": r["text"],
            "entities": [{"text": e["text"], "label": e["label"]} for e in r["entities"]],
        }
        for r in picked
    ]
    (PROMPT_DATA / "seeds_20_en.json").write_text(
        json.dumps(prompt_seeds, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    n_ent = sum(len(r["entities"]) for r in picked)
    manifest = {
        "created_by": "src/data/make_seeds_en.py",
        "source": str(SRC),
        "source_sha256": sha256(SRC),
        "output": str(out_jsonl),
        "output_sha256": sha256(out_jsonl),
        "seed": SEED,
        "constraints": {
            "n_seeds": N_SEEDS,
            "token_range": [MIN_TOKENS, MAX_TOKENS],
            "min_per_class": MIN_PER_CLASS,
            "max_per_document": MAX_PER_DOC,
            "must_have_entities": True,
        },
        "counts": {
            "seeds": len(picked),
            "entities": n_ent,
            "per_class": dict(class_count),
            "documents": len(doc_count),
            "tokens": sum(len(r["tokens"]) for r in picked),
            "entities_per_seed": round(n_ent / len(picked), 3),
        },
        "ids": [r["id"] for r in picked],
        "doc_ids": sorted(doc_count),
    }
    (OUT_DIR / "seeds_20_en_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    c = manifest["counts"]
    lines = [
        "# 英文生成种子 seeds_20_en",
        "",
        "E2en / E3en 生成时使用的 20 条真实英文示例，**固定不变**。",
        "",
        f"- 来源：`data/en/curated/real_300_en.jsonl`（sha256 `{manifest['source_sha256'][:16]}…`）",
        f"- 输出：`data/en/curated/seeds_20_en.jsonl`（sha256 `{manifest['output_sha256'][:16]}…`）",
        f"- 随机种子：**{SEED}**（重跑结果一致）",
        f"- 选择约束：含实体、句长 {MIN_TOKENS}–{MAX_TOKENS} token、每类 ≥ {MIN_PER_CLASS} 次、每文档 ≤ {MAX_PER_DOC} 条",
        "",
        "## 规模",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| 种子条数 | {c['seeds']} |",
        f"| 实体总数 | {c['entities']} |",
        f"| 实体/条 | {c['entities_per_seed']} |",
        f"| 覆盖文档数 | {c['documents']} |",
        f"| token 总数 | {c['tokens']} |",
        "",
        "## 类别分布",
        "",
        "| 类别 | 出现次数 |",
        "|---|---:|",
    ]
    for cls in CLASSES:
        lines.append(f"| {cls} | {class_count.get(cls, 0)} |")
    lines += ["", "## 种子 ID", "", "```text", ", ".join(manifest["ids"]), "```", ""]
    (OUT_DIR / "seeds_20_en_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[输出] {out_jsonl}")
    print(f"[输出] {PROMPT_DATA / 'seeds_20_en.json'}")
    print(f"[计数] {json.dumps(c, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
