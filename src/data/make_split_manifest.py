"""生成三集划分清单（文档级），作为泄漏审计与复现证据。

背景（2026-11-04 决策）：
    - 取消 dev 的 872 / 300 二次切分。理由是原始设计里 Dev Feedback 只服务于
      已删除的 Error-Aware 实验，Dev Selection 的"搜阈值"用途也因阈值统一为 0.80 而消失；
      再切只会把报告集从 1,172 句缩到 300 句，使置信区间宽度翻倍。
    - 完整 dev 同时用于"选 checkpoint"与"报告指标"，该轻微乐观偏差在 limitations 中披露。
    - 本清单仍要落盘：它证明三段之间没有文档级泄漏，且划分可复现。

产物：
    data/pt_bosque/split_manifest.json
    data/pt_bosque/split_manifest.md

用法：
    & "...\\.venv\\Scripts\\python.exe" src/data/make_split_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: F401  统一环境

DATA = WORKSPACE / "data" / "pt_bosque"
SPLITS = ["train", "dev", "test"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    info: dict[str, dict] = {}
    texts: dict[str, list[str]] = {}

    for split in SPLITS:
        path = DATA / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        doc_counts = Counter(r["doc_id"] for r in rows)
        texts[split] = [r["text"] for r in rows]
        info[split] = {
            "file": str(path),
            "sha256": sha256(path),
            "sentences": len(rows),
            "entities": sum(len(r["entities"]) for r in rows),
            "documents": len(doc_counts),
            "document_ids": sorted(doc_counts),
            "sentences_per_document": dict(sorted(doc_counts.items())),
            "internal_duplicate_sentences": len(rows) - len(set(texts[split])),
        }

    # 文档级互斥检查
    doc_overlap = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            common = set(info[a]["document_ids"]) & set(info[b]["document_ids"])
            doc_overlap[f"{a}&{b}"] = {"count": len(common), "documents": sorted(common)}

    # 句级完全重复检查
    text_overlap = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            common = set(texts[a]) & set(texts[b])
            text_overlap[f"{a}&{b}"] = {"count": len(common), "samples": sorted(common)[:5]}

    total_dup = sum(v["count"] for v in text_overlap.values()) + sum(
        info[s]["internal_duplicate_sentences"] for s in SPLITS
    )

    manifest = {
        "created_by": "src/data/make_split_manifest.py",
        "policy": {
            "dev_split": "none — 完整 dev 同时用于 checkpoint 选择与指标报告（2026-11-04 决策）",
            "threshold": 0.80,
            "note": "Dev Feedback / Dev Selection 的 872/300 二次切分已取消",
        },
        "splits": info,
        "document_level_overlap": doc_overlap,
        "text_level_overlap": text_overlap,
        "duplicate_summary": {
            "total_duplicate_sentences": total_dup,
            "internal": {s: info[s]["internal_duplicate_sentences"] for s in SPLITS},
            "cross": {k: v["count"] for k, v in text_overlap.items()},
            "handling": "E1 训练集 data/curated/real_300.jsonl 生成时已剔除（内部 11 条、跨集 3 条）；"
                        "合成数据由校验器保证为 0；dev/test 内部重复为 0",
        },
    }

    (DATA / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 三集划分清单",
        "",
        f"生成脚本 `src/data/make_split_manifest.py`　生成日期 2026-11-04",
        "",
        "## 划分政策",
        "",
        "- **dev 不做二次切分**：完整 dev 同时用于 checkpoint 选择与指标报告（2026-11-04 决策）",
        "- 阈值全项目统一 **0.80**",
        "- 取消 872/300 切分的原因：Dev Feedback 只服务于已删除的 Error-Aware；Dev Selection 的阈值搜索用途也已消失",
        "",
        "## 规模",
        "",
        "| Split | 句子 | 实体 | 文档 | 内部重复句 | sha256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for s in SPLITS:
        d = info[s]
        lines.append(f"| {s} | {d['sentences']} | {d['entities']} | {d['documents']} | {d['internal_duplicate_sentences']} | `{d['sha256'][:16]}…` |")

    lines += ["", "## 文档级互斥（泄漏审计）", "", "| 组合 | 交叠文档数 |", "|---|---:|"]
    for k, v in doc_overlap.items():
        lines.append(f"| {k} | **{v['count']}** |")

    lines += ["", "## 句级完全重复", "", "| 组合 | 重复句数 | 示例 |", "|---|---:|---|"]
    for k, v in text_overlap.items():
        sample = "; ".join(v["samples"][:2]) if v["samples"] else "—"
        lines.append(f"| {k} | {v['count']} | {sample} |")

    lines += [
        "",
        "## 重复句处理",
        "",
        f"- 全库重复句总数：**{total_dup}**（train 内部 {info['train']['internal_duplicate_sentences']} + 跨集 "
        f"{sum(v['count'] for v in text_overlap.values())}）",
        "- E1 训练集生成时已剔除（`data/curated/real_300.jsonl`，剔除内部 11 条、跨集 3 条）",
        "- 合成数据由校验器保证重复率为 0",
        "- dev / test 内部重复为 0",
        "",
        "## 文档 ID 清单",
        "",
    ]
    for s in SPLITS:
        lines.append(f"### {s}（{info[s]['documents']} 篇）")
        lines.append("")
        lines.append("```text")
        lines.append(", ".join(info[s]["document_ids"]))
        lines.append("```")
        lines.append("")

    (DATA / "split_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[输出] {DATA / 'split_manifest.json'}")
    print(f"[输出] {DATA / 'split_manifest.md'}")
    print(f"文档级交叠: { {k: v['count'] for k, v in doc_overlap.items()} }")
    print(f"句级重复总数: {total_dup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
