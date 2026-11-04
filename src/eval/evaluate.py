"""E0 全量评测：严格实体匹配 + 阈值网格搜索 + 错误分类 + bootstrap CI。

设计（见 docs/specs/07-evaluation-protocol.md）：
1. 候选收集：用 threshold=0.0 一次前向拿到每句全部候选 span 与分数，落盘缓存；
2. 离线扫阈值：0.10–0.90 步长 0.05，在缓存上重算，不重复推理；
3. 指标：严格实体级 Micro P/R/F1、Macro-F1、per-class、FP/FN、Type Error、Boundary Error；
4. 置信区间：按文档重采样 1000 次。

用法：
    & "...\\.venv\\Scripts\\python.exe" src/eval/evaluate.py \
        --model models/gliner_multi-v2.1 --split dev \
        --cache D:/pt_ner_work/results/e0_dev_candidates.jsonl \
        --out   D:/pt_ner_work/results/e0_dev
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: E402,F401  统一环境（HF 镜像、临时目录），需在 import torch 前生效

LABELS = ["person", "organization", "location"]
LABEL2GOLD = {"person": "PER", "organization": "ORG", "location": "LOC"}
GOLD_CLASSES = ["PER", "ORG", "LOC"]
THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(17)]


# ---------------------------------------------------------------- 候选收集
def collect_candidates(model_dir: Path, split: str, cache: Path) -> list[dict]:
    from gliner import GLiNER

    data = WORKSPACE / "data" / "pt_bosque" / f"{split}.jsonl"
    records = [json.loads(line) for line in data.open(encoding="utf-8")]

    print(f"[候选] 加载模型 {model_dir}")
    t0 = time.time()
    model = GLiNER.from_pretrained(str(model_dir), map_location="cuda")
    print(f"       加载完成 {time.time() - t0:.1f}s，句子 {len(records)} 条")

    cache.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    with cache.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            preds = model.predict_entities(rec["text"], LABELS, threshold=0.0)
            cands = [
                [p["start"], p["end"], LABEL2GOLD.get(p["label"], p["label"]), round(float(p["score"]), 6)]
                for p in preds
            ]
            gold = [[e["start"], e["end"], e["label"]] for e in rec["entities"]]
            fh.write(
                json.dumps(
                    {"id": rec["id"], "doc_id": rec["doc_id"], "gold": gold, "cands": cands},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 200 == 0:
                print(f"       进度 {i + 1}/{len(records)}  已用 {time.time() - t1:.0f}s", flush=True)
    print(f"[候选] 完成，用时 {time.time() - t1:.1f}s，缓存 {cache}")
    return records


# ---------------------------------------------------------------- 指标
def preds_at(cands: list[list], thr: float) -> set[tuple[int, int, str]]:
    return {(s, e, lab) for s, e, lab, sc in cands if sc >= thr}


def score_sentence(gold: list[list], cands: list[list], thr: float) -> tuple[int, int, int]:
    g = {(s, e, lab) for s, e, lab in gold}
    p = preds_at(cands, thr)
    tp = len(g & p)
    return tp, len(p) - tp, len(g) - tp


def micro_metrics(totals: dict) -> dict:
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def sweep(rows: list[dict]) -> list[dict]:
    out = []
    for thr in THRESHOLDS:
        totals = {"tp": 0, "fp": 0, "fn": 0}
        per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in GOLD_CLASSES}
        for row in rows:
            g = {(s, e, lab) for s, e, lab in row["gold"]}
            p = preds_at(row["cands"], thr)
            tp = g & p
            fp = p - g
            fn = g - p
            totals["tp"] += len(tp)
            totals["fp"] += len(fp)
            totals["fn"] += len(fn)
            for c in GOLD_CLASSES:
                per_class[c]["tp"] += sum(1 for x in tp if x[2] == c)
                per_class[c]["fp"] += sum(1 for x in fp if x[2] == c)
                per_class[c]["fn"] += sum(1 for x in fn if x[2] == c)
        m = micro_metrics(totals)
        f1s = []
        for c in GOLD_CLASSES:
            d = per_class[c]
            pc = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
            rc = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
            f1s.append(2 * pc * rc / (pc + rc) if pc + rc else 0.0)
        out.append({"threshold": thr, **m, "macro_f1": sum(f1s) / len(f1s), "per_class": per_class})
    return out


def error_breakdown(rows: list[dict], thr: float) -> dict:
    """Type Error = 边界正确但类别错；Boundary Error = 类别正确但边界不完全一致。"""
    type_err = boundary_err = 0
    fp_examples, fn_examples = [], []
    for row in rows:
        g = {(s, e, lab) for s, e, lab in row["gold"]}
        p = preds_at(row["cands"], thr)
        fp = p - g
        fn = g - p
        for span in fn:
            same_span_other_label = [x for x in fp if x[0] == span[0] and x[1] == span[1]]
            if same_span_other_label:
                type_err += 1
                fp.discard(same_span_other_label[0])
            else:
                overlap_same_label = [x for x in fp if x[2] == span[2] and not (x[1] <= span[0] or x[0] >= span[1])]
                if overlap_same_label:
                    boundary_err += 1
                    fp.discard(overlap_same_label[0])
                elif len(fn_examples) < 5:
                    fn_examples.append({"id": row["id"], "gold": span})
        for x in fp:
            if len(fp_examples) < 5:
                fp_examples.append({"id": row["id"], "pred": list(x)})
    return {
        "type_error": type_err,
        "boundary_error": boundary_err,
        "unmatched_fn": sum(len(set(map(tuple, r["gold"])) - preds_at(r["cands"], thr)) for r in rows) - type_err - boundary_err,
        "unmatched_fp": sum(len(preds_at(r["cands"], thr) - set(map(tuple, r["gold"]))) for r in rows) - type_err - boundary_err,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
    }


def bootstrap_ci(rows: list[dict], thr: float, n_iter: int = 1000, seed: int = 13) -> dict:
    """按文档重采样，返回 Micro-F1 的 95% 置信区间。"""
    by_doc: dict[str, list[tuple[int, int, int]]] = {}
    for row in rows:
        by_doc.setdefault(row["doc_id"], []).append(score_sentence(row["gold"], row["cands"], thr))
    docs = list(by_doc)
    if not docs:
        return {"ci_low": 0.0, "ci_high": 0.0, "documents": 0}

    rng = random.Random(seed)
    f1s = []
    for _ in range(n_iter):
        sample = [by_doc[rng.choice(docs)] for _ in docs]
        tp = fp = fn = 0
        for sent_scores in sample:
            for a, b, c in sent_scores:
                tp += a
                fp += b
                fn += c
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    f1s.sort()
    return {
        "ci_low": f1s[int(0.025 * n_iter)],
        "ci_high": f1s[int(0.975 * n_iter) - 1],
        "mean": sum(f1s) / len(f1s),
        "documents": len(docs),
        "iterations": n_iter,
    }


# ---------------------------------------------------------------- 报告
def to_markdown(res: dict) -> str:
    best = res["best"]
    sweep_best = res.get("sweep_best") or {"threshold": best["threshold"], "f1": best["f1"]}
    lines = [
        f"# {res['split']} 评测报告 — {res.get('arm', '')} / {res['model']}",
        "",
        f"- 阈值模式：**{res.get('mode', '扫描最优')}**",
        f"- 句子数：{res['n_sentences']}，文档数：{res['n_documents']}，gold 实体 {res.get('gold_entities', '?')}，预测实体 {res.get('pred_entities', '?')}",
        f"- 本次采用阈值：**{best['threshold']:.2f}**（Micro-F1 = {best['f1']:.4f}）",
        f"- 扫描最优参考：{sweep_best['threshold']:.2f}（Micro-F1 = {sweep_best['f1']:.4f}）",
        f"- 95% CI（按文档 bootstrap，{res['ci']['iterations']} 次）：[{res['ci']['ci_low']:.4f}, {res['ci']['ci_high']:.4f}]",
        "",
        "## 主表",
        "",
        "| 阈值 | Precision | Recall | Micro-F1 | Macro-F1 | TP | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in res["sweep"]:
        mark = " **←**" if row["threshold"] == best["threshold"] else ""
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['macro_f1']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} |{mark}"
        )
    lines += ["", "## 分类别（最优阈值）", "", "| 类别 | Precision | Recall | F1 | TP | FP | FN | 支持数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in GOLD_CLASSES:
        d = best["per_class"][c]
        pc = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        rc = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        f1 = 2 * pc * rc / (pc + rc) if pc + rc else 0.0
        support = d["tp"] + d["fn"]
        lines.append(f"| {c} | {pc:.4f} | {rc:.4f} | {f1:.4f} | {d['tp']} | {d['fp']} | {d['fn']} | {support} |")
    err = res["errors"]
    lines += [
        "",
        "## 错误分类（最优阈值）",
        "",
        "| 类型 | 数量 |",
        "|---|---:|",
        f"| Type Error（边界对、类别错） | {err['type_error']} |",
        f"| Boundary Error（类别对、边界错） | {err['boundary_error']} |",
        f"| 其余 FP | {err['unmatched_fp']} |",
        f"| 其余 FN | {err['unmatched_fn']} |",
    ]
    if err["fp_examples"]:
        lines += ["", "FP 示例：" + "; ".join(f"{x['id']} {x['pred']}" for x in err["fp_examples"][:3])]
    if err["fn_examples"]:
        lines += ["", "FN 示例：" + "; ".join(f"{x['id']} {x['gold']}" for x in err["fn_examples"][:3])]
    lines += ["", "> 说明：本报告在完整 dev 上生成；dev 二次切分（872 / 300）落地后需重跑才能作为 G2 闸门数字。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="固定阈值（项目统一 0.80）。给了就按该阈值出最终数字，扫描表仅作参考")
    ap.add_argument("--arm", default="E0", help="实验组标识，写入台账")
    ap.add_argument("--seed", default="", help="训练随机种子（E0 留空）")
    ap.add_argument("--record", default=None, help="把最终指标追加到该 CSV（默认 results/experiments.csv）")
    args = ap.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_absolute():
        model_dir = WORKSPACE / model_dir
    cache = Path(args.cache)
    out_prefix = Path(args.out)

    if cache.exists():
        print(f"[候选] 复用缓存 {cache}")
        rows = [json.loads(line) for line in cache.open(encoding="utf-8")]
    else:
        collect_candidates(model_dir, args.split, cache)
        rows = [json.loads(line) for line in cache.open(encoding="utf-8")]

    print(f"[评测] 阈值扫描 {len(THRESHOLDS)} 档")
    sweep_rows = sweep(rows)
    sweep_best = max(sweep_rows, key=lambda r: r["f1"])

    if args.threshold is None:
        best = sweep_best
        mode = "扫描最优"
    else:
        matched = [r for r in sweep_rows if abs(r["threshold"] - args.threshold) < 1e-9]
        best = matched[0] if matched else None
        if best is None:
            raise SystemExit(f"阈值 {args.threshold} 不在网格内，可选：{THRESHOLDS}")
        mode = f"固定 {args.threshold:.2f}"
    print(f"       {mode} → Micro-F1 {best['f1']:.4f}（扫描最优为 {sweep_best['threshold']} / {sweep_best['f1']:.4f}）")

    print("[评测] 错误分类")
    errors = error_breakdown(rows, best["threshold"])

    print("[评测] bootstrap CI（1000 次，按文档重采样）")
    ci = bootstrap_ci(rows, best["threshold"])

    gold_total = sum(len(r["gold"]) for r in rows)
    pred_total = sum(len(preds_at(r["cands"], best["threshold"])) for r in rows)

    res = {
        "model": model_dir.name,
        "split": args.split,
        "arm": args.arm,
        "mode": mode,
        "n_sentences": len(rows),
        "n_documents": len({r["doc_id"] for r in rows}),
        "gold_entities": gold_total,
        "pred_entities": pred_total,
        "sweep": sweep_rows,
        "sweep_best": {"threshold": sweep_best["threshold"], "f1": sweep_best["f1"]},
        "best": best,
        "errors": errors,
        "ci": ci,
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".json")).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_prefix.with_suffix(".md")).write_text(to_markdown(res), encoding="utf-8")
    print(f"[输出] {out_prefix.with_suffix('.json')}")
    print(f"[输出] {out_prefix.with_suffix('.md')}")

    if args.record is not None:
        sys.path.insert(0, str(WORKSPACE))
        from src.eval.record import append_row

        per_class = best["per_class"]

        def f1_of(cls: str) -> float:
            d = per_class[cls]
            p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
            r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
            return 2 * p * r / (p + r) if p + r else 0.0

        path = append_row(
            {
                "arm": args.arm,
                "split": args.split,
                "checkpoint": str(model_dir),
                "seed": args.seed,
                "threshold": best["threshold"],
                "precision": best["precision"],
                "recall": best["recall"],
                "micro_f1": best["f1"],
                "macro_f1": best["macro_f1"],
                "per_f1": f1_of("PER"),
                "org_f1": f1_of("ORG"),
                "loc_f1": f1_of("LOC"),
                "tp": best["tp"],
                "fp": best["fp"],
                "fn": best["fn"],
                "type_error": errors["type_error"],
                "boundary_error": errors["boundary_error"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "gold_entities": gold_total,
                "pred_entities": pred_total,
                "sentences": len(rows),
                "documents": len({r["doc_id"] for r in rows}),
                "notes": f"mode={mode}; sweep_best={sweep_best['threshold']}@{sweep_best['f1']:.4f}",
            },
            args.record,
        )
        print(f"[台账] 已追加一行到 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
