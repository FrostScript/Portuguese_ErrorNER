"""英文对照实验：把 UNER English-EWT 的 .iob2 原始文件转成项目统一 JSONL，并输出统计表。

**与葡语版的关系：** 本文件是 `src/data/prepare_uner.py` 的英文副本，逻辑逐行对应，
只替换了数据源与输出目录。按 `docs/specs/11-english-parallel.md` 的隔离原则，
**不修改、不 import 葡语版**，两侧独立演进。

**为什么选 English-EWT：**
    与 `UNER_Portuguese-Bosque` 同属 UniversalNER 母体、同 IOB2 结构、同标注规范，
    标签集恰好也是 PER / ORG / LOC（train: PER 2874 / ORG 1436 / LOC 2712），
    **不需要任何标签映射或丢弃**，语言成为唯一设计变量。
    已知偏差：EWT 是网络文本（博客/邮件/评论），`pt_bosque` 是新闻，语体不同 → P-EN3。

数据源：
    https://raw.githubusercontent.com/UniversalNER/UNER_English-EWT/master/en_ewt-ud-{train,dev,test}.iob2
    许可证 CC-BY-SA-4.0

原始文件格式：每 token 一行 `id \\t token \\t tag \\t misc \\t annotator`，
句子之间空行分隔，句前有 `# newdoc id` / `# sent_id` / `# text` 元数据。

产物：
    data/en/uner_en/{train,dev,test}.jsonl
    data/en/uner_en/stats.json

原始 iob2 放在仓库外（`D:\\pt_ner_work\\raw_uner_en\\`）：它可随时重下，
且 `.gitignore` 只忽略 `data/raw_uner/`，放仓库内会被 git 看到。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
RAW_DIR = Path("D:/pt_ner_work/raw_uner_en")
OUT_DIR = WORKSPACE / "data" / "en" / "uner_en"

SPLITS = {
    "train": "en_ewt-ud-train.iob2",
    "dev": "en_ewt-ud-dev.iob2",
    "test": "en_ewt-ud-test.iob2",
}
TAG_MAP = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}
RAW_BASE = "https://raw.githubusercontent.com/UniversalNER/UNER_English-EWT/master"


def download() -> None:
    """缺哪个下哪个；已存在则跳过（幂等）。"""
    import urllib.request

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in SPLITS.values():
        target = RAW_DIR / fname
        if target.exists() and target.stat().st_size > 0:
            print(f"  已存在 {target.name}（{target.stat().st_size} B）")
            continue
        url = f"{RAW_BASE}/{fname}"
        print(f"  下载 {url}")
        urllib.request.urlretrieve(url, target)
        print(f"    → {target}（{target.stat().st_size} B）")


def parse_iob2(path: Path):
    """产出 (meta, tokens, tags) 三元组。

    与葡语版一致：`# newdoc id` 只出现在每个文档的第一句，需要向前传递到同文档的后续句子，
    否则 doc_id 会大量为空，导致文档级重采样与泄漏审计失真。
    """
    meta: dict[str, str] = {}
    tokens: list[str] = []
    tags: list[str] = []
    current_doc: str = ""

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    if "newdoc id" not in meta and current_doc:
                        meta["newdoc id"] = current_doc
                    yield dict(meta), list(tokens), list(tags)
                    meta.clear()
                    tokens.clear()
                    tags.clear()
                continue
            if line.startswith("#"):
                body = line.lstrip("#").strip()
                if " = " in body:
                    key, _, val = body.partition(" = ")
                    key = key.strip()
                    meta[key] = val
                    if key == "newdoc id":
                        current_doc = val
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
        if "newdoc id" not in meta and current_doc:
            meta["newdoc id"] = current_doc
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
    if "--no-download" not in sys.argv:
        download()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {"source": "UniversalNER/UNER_English-EWT@master", "splits": {}}
    texts_by_split: dict[str, Counter] = {}

    for split, fname in SPLITS.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"  ! 缺少 {path}")
            continue

        records = []
        ent_counter: Counter = Counter()
        n_ent = n_tok = n_empty = n_fallback = 0
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
            # 三个 split 的文档编号各自从 0 开始，必须加命名空间，否则泄漏审计误报
            if doc_id:
                doc_id = f"{split}:{doc_id}"
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
