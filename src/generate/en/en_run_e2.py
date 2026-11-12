"""英文对照实验 E2en 入口：普通受约束生成 → 165 含实体 + 135 无实体 = 300 条。

**与葡语版 `src/generate/run_e2.py` 的关系：** 流程、目标条数、每批条数、重试策略、
校验与度量全部一致；只有四处必须不同：
    1. 提示词目录 → `prompts/templates_en/`（经 `en_pipeline`）；
    2. 产物路径 → 英文命名空间（`data/en/synthetic/`、`D:/pt_ner_work/data/synthetic/e2en`）；
    3. arm 标识 → `E2en`；
    4. 对齐检查与抄袭检查的对象 → 英文种子与英文 E1 训练集。

token 切分**复用 `src/generate/pt_tokenize.py`**：这是 `specs/11` P-EN7 的既定处理，
两侧口径一致优先于单语言最优。

用法：
    & python src/generate/en/en_run_e2.py --dry-run     # 只渲染，不花钱
    & python src/generate/en/en_run_e2.py --smoke       # 各 1 批
    & python src/generate/en/en_run_e2.py --budget-usd 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))

from src.common import env  # noqa: E402,F401
from src.generate import metrics  # noqa: E402
from src.generate.en import en_pipeline as pipeline  # noqa: E402

OUT_DIR = Path("D:/pt_ner_work/data/synthetic/e2en")
LEDGER = Path("D:/pt_ner_work/results/gen.csv")
FINAL_DIR = WORKSPACE / "data" / "en" / "synthetic"
FINAL_POS = FINAL_DIR / "e2_pos_en.jsonl"
FINAL_NEG = FINAL_DIR / "e2_neg_en.jsonl"
SEEDS_JSONL = WORKSPACE / "data" / "en" / "curated" / "seeds_20_en.jsonl"
E1_JSONL = WORKSPACE / "data" / "en" / "curated" / "real_300_en.jsonl"

TARGET_POS, TARGET_NEG = 165, 135
PER_BATCH_POS, PER_BATCH_NEG = 11, 9
ARM = "E2en"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只渲染 Prompt，不调 API")
    ap.add_argument("--smoke", action="store_true", help="试产：含实体与无实体各 1 批")
    ap.add_argument("--max-calls", type=int, default=60, help="调用次数上限（额度保护）")
    ap.add_argument("--budget-usd", type=float, default=1.0, help="花费上限（USD）")
    ap.add_argument("--price-in", type=float, default=0.0, help="输入价格 USD/1M token")
    ap.add_argument("--price-out", type=float, default=0.0, help="输出价格 USD/1M token")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="单次输出上限。deepseek-flash 是推理模型，thinking 会占掉大部分配额")
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--only", choices=["both", "pos", "neg"], default="both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    target_pos = PER_BATCH_POS if args.smoke else TARGET_POS
    target_neg = PER_BATCH_NEG if args.smoke else TARGET_NEG

    if args.dry_run:
        print("=== dry-run：只渲染 Prompt ===")
        p_pos = pipeline.prepare_prompt.render("e2_generation", batch="e2en_001", n=PER_BATCH_POS)
        p_neg = pipeline.prepare_prompt.render("e2_negative", batch="e2en_101", n=PER_BATCH_NEG)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dryrun_pos.txt").write_text(p_pos, encoding="utf-8")
        (out_dir / "dryrun_neg.txt").write_text(p_neg, encoding="utf-8")
        print(f"含实体 Prompt {len(p_pos)} 字符 → {out_dir / 'dryrun_pos.txt'}")
        print(f"无实体 Prompt {len(p_neg)} 字符 → {out_dir / 'dryrun_neg.txt'}")
        return 0

    from src.generate.client import DeepSeekClient

    client = DeepSeekClient(model=args.model, base_url=args.base_url, temperature=args.temperature,
                            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
    print(f"模型 {client.model} @ {client.base_url}，温度 {client.temperature}，"
          f"max_tokens {client.max_tokens}，reasoning_effort {client.reasoning_effort}")
    print(f"目标：{target_pos} 含实体 + {target_neg} 无实体")

    if args.only in ("both", "pos"):
        print("\n[1/2] 含实体批次")
        pos = pipeline.collect(
            client, stage="e2_generation", expect="entity", target=target_pos, per_batch=PER_BATCH_POS,
            arm=ARM, out_dir=out_dir, ledger_path=LEDGER, batch_prefix="e2en_2",
            max_calls=args.max_calls, budget_usd=args.budget_usd, price_in=args.price_in, price_out=args.price_out,
        )
        p = pipeline.merge_write(pos["rows"], FINAL_POS)
        print(f"含实体合并落盘：{p}（{len(pos['rows'])} 条），调用 {pos['calls']} 次")
    else:
        print("\n[1/2] 含实体批次：跳过（--only neg），保留已有 e2_pos_en.jsonl")

    if args.only in ("both", "neg"):
        print("\n[2/2] 无实体批次")
        neg = pipeline.collect(
            client, stage="e2_negative", expect="none", target=target_neg, per_batch=PER_BATCH_NEG,
            arm=ARM, out_dir=out_dir, ledger_path=LEDGER, batch_prefix="e2en_1",
            max_calls=args.max_calls, budget_usd=args.budget_usd, price_in=args.price_in, price_out=args.price_out,
        )
        n = pipeline.merge_write(neg["rows"], FINAL_NEG)
        print(f"无实体合并落盘：{n}（{len(neg['rows'])} 条），调用 {neg['calls']} 次")
    else:
        print("\n[2/2] 无实体批次：跳过（--only pos），保留已有 e2_neg_en.jsonl")

    # 合并成训练用文件（补 tokens / split），token 切分复用葡语侧 pt_tokenize（P-EN7）
    from src.generate.pt_tokenize import tokenize

    all_rows = []
    for f in (FINAL_POS, FINAL_NEG):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            toks, _ = tokenize(r["text"])
            all_rows.append({"id": r["id"], "doc_id": "synthetic_e2en", "split": "synthetic",
                             "tokens": toks, "text": r["text"], "entities": r["entities"]})
    merged_path = FINAL_DIR / "e2_en_300.jsonl"
    merged_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")
    print(f"训练用合并：{merged_path}（{len(all_rows)} 条）")

    # 项目级检查（复用葡语侧 validator，阈值常量与英文侧取值相同，见 specs/11 §8）
    from src.generate.validator import check_file

    seeds = [json.loads(l)["text"] for l in SEEDS_JSONL.open(encoding="utf-8")]
    for label, path, expect in (("含实体", FINAL_POS, "entity"), ("无实体", FINAL_NEG, "none")):
        r = check_file(path, seeds, expect)
        print(f"[项目检查-{label}] accepted {r['accepted']} / rejected {r['rejected']} / "
              f"no_entity_ratio {r['no_entity_ratio']} / per_class {r['per_class']}")

    # 与 E1en 的分布对照
    e1 = [json.loads(l) for l in E1_JSONL.open(encoding="utf-8")]

    def dist(rows):
        ents = [len(r["entities"]) for r in rows if r["entities"]]
        cls = {}
        for r in rows:
            for e in r["entities"]:
                cls[e["label"]] = cls.get(e["label"], 0) + 1
        return {"entities": sum(ents), "per_entity_sentence": round(sum(ents) / len(ents), 2) if ents else 0,
                "classes": cls}

    d_e2, d_e1 = dist(all_rows), dist(e1)
    print(f"\n[对齐检查] E2en 实体 {d_e2['entities']} / 每含实体句 {d_e2['per_entity_sentence']} / {d_e2['classes']}")
    print(f"[对齐检查] E1en 实体 {d_e1['entities']} / 每含实体句 {d_e1['per_entity_sentence']} / {d_e1['classes']}")
    dev = (d_e2["entities"] - d_e1["entities"]) / d_e1["entities"] * 100
    print(f"[对齐检查] 实体总量偏差 {dev:+.1f}%（specs/05 目标 ±5%）")

    res = metrics.report(FINAL_POS, FINAL_NEG)
    md = metrics.to_markdown(res)
    report = Path("D:/pt_ner_work/results/e2en_metrics.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    report.with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"\n质量度量 → {report}")
    print(md)
    return 0


if __name__ == "__main__":
    from src.common import runlog

    with runlog.capture("run_e2en"):
        raise SystemExit(main())
