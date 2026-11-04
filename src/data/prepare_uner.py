"""把 UNER pt_bosque 的 .iob2 原始文件转成项目统一 JSONL，并输出统计表。

数据源（UNER 官方仓库，与 HF 上的 universalner/universal_ner 脚本使用同一批文件）：
    https://raw.githubusercontent.com/UniversalNER/UNER_Portuguese-Bosque/master/pt_bosque-ud-{train,dev,test}.iob2

原始文件格式：每 token 一行 `id \t token \t tag \t misc \t annotator`，
句子之间空行分隔，句前有 `# newdoc id` / `# sent_id` / `# text` 元数据。

产物：
    data/pt_bosque/{train,dev,test}.jsonl
    data/pt_bosque/stats.json

关于字符区间：`# text` 是官方给出的句子原文，实体字符区间通过在 text 中
按顺序定位 token 得到；若某句有个别 token 定位失败，回退为 " ".join(tokens)
并计入 fallback 计数。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
RAW_DIR = WORKSPACE / "data" / "raw_uner"
OUT_DIR = WORKSPACE / "data" / "pt_bosque"

SPLITS = {"train": "pt_bosque-ud-train.iob2", "dev": "pt_bosque-ud-dev.iob2", "test": "pt_bosque-ud-test.iob2"}
TAG_MAP = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}


def parse_iob2(path: Path):
    """产出 (meta, tokens, tags) 三元组。"""
    meta: dict[str, str] = {}
    tokens: list[str] = []
    tags: list[str] = []

    def flush():
        if tokens:
            yield dict(meta), list(tokens), list(tags)

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    yield dict(meta), list(tokens), list(tags)
                    meta.clear()
                    tokens.clear()
                    tags.clear()
                continue
            if line.startswith("#"):
                body = line.lstrip("#").strip()
                if " = " in body:
                    key, _, val = body.partition(" = ")
                    meta[key.strip()] = val
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            tokens.append(cols[1])
            tag = cols[2]
            # 与 UNER 官方脚本一致：OTH 与 B-O 归为 O
            if "OTH" in tag or tag == "B-O":
                tag = "O"
            tags.append(tag)
    if tokens:
        yield dict(meta), tokens, tags


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


def build_entities(tokens: list[str], tags: list[str], offsets: list[int]) -> list[dict]:
    entities = []
    cur = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur:
                entities.append(cur)
                cur = None
            continue
        prefix, _, raw = tag.partition("-")
        label = TAG_MAP.get(raw)
        if label is None:
            raise ValueError(f"未知标签 {tag}")
        if prefix == "B" or cur is None or cur["label"] != label:
            if cur:
                entities.append(cur)
            cur = {"start": offsets[i], "end": offsets[i] + len(tokens[i]), "label": label}
        else:
            cur["end"] = offsets[i] + len(tokens[i])
    if cur:
        entities.append(cur)
    return entities


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {"source": "UniversalNER/UNER_Portuguese-Bosque@master", "splits": {}}
    texts_by_split: dict[str, Counter] = {}

    for split, fname in SPLITS.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"  ! 缺少 {path}")
            continue

        records = []
        ent_counter: Counter = Counter()
        n_ent = n_tok = n_empty = n_fallback = n_bad_span = 0
        docs = set()

        for i, (meta, tokens, tags) in enumerate(parse_iob2(path)):
            if len(tokens) != len(tags):
                raise ValueError(f"{fname} 句 {i} token/tag 数不一致")
            text = meta.get("text") or " ".join(tokens)
            offsets = token_offsets(text, tokens)
            if offsets is None:
                text = " ".join(tokens)
                offsets = token_offsets(text, tokens)
                n_fallback += 1
            assert offsets is not None

            entities = build_entities(tokens, tags, offsets)
            for ent in entities:
                ent["text"] = text[ent["start"]:ent["end"]]
                ent_counter[ent["label"]] += 1
            n_ent += len(entities)
            n_tok += len(tokens)
            if not entities:
                n_empty += 1

            doc_id = meta.get("newdoc id") or meta.get("newdoc_id") or ""
            docs.add(doc_id)
            records.append(
                {
                    "id": meta.get("sent_id") or f"{split}_{i:05d}",
                    "doc_id": doc_id,
                    "split": split,
                    "tokens": tokens,
                    "text": text,
                    "entities": entities,
                }
            )

        with (OUT_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n = len(records)
        dup = n - len({r["text"] for r in records})
        texts_by_split[split] = Counter(r["text"] for r in records)
        stats["splits"][split] = {
            "documents": len(docs),
            "sentences": n,
            "entities": n_ent,
            "tokens": n_tok,
            "entities_per_sentence": round(n_ent / n, 4),
            "tokens_per_sentence": round(n_tok / n, 2),
            "no_entity_sentences": n_empty,
            "no_entity_rate": round(n_empty / n, 4),
            "per_class": dict(ent_counter),
            "duplicate_texts": dup,
            "offset_fallbacks": n_fallback,
        }
        print(f"  {split}: {len(docs)} 文档 / {n} 句 / {n_ent} 实体 / {n_tok} token（回退 {n_fallback}）")

    leak = {}
    for a, b in [("train", "test"), ("train", "dev"), ("dev", "test")]:
        if a in texts_by_split and b in texts_by_split:
            overlap = set(texts_by_split[a]) & set(texts_by_split[b])
            leak[f"{a}&{b}"] = {"exact_duplicate_sentences": len(overlap), "sample": list(overlap)[:3]}
    stats["exact_text_overlap"] = leak

    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
