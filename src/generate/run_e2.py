"""E2 生成入口：普通受约束生成 → 165 含实体 + 135 无实体 = 300 条。

流程（每类分别"只补不删"直到凑够）：
    pos: e2_generation（每批 11 条）→ convert_validate --expect entity
    neg: e2_negative  （每批 9 条） → convert_validate --expect none
    合并 → 项目级检查（src/generate/validator.py）→ 质量度量（src/generate/metrics.py）

用法：
    # 只看渲染出来的 Prompt，不花钱
    & python src/generate/run_e2.py --dry-run

    # 试产：各 1 批
    & python src/generate/run_e2.py --smoke

    # 正式：各凑够目标条数
    & python src/generate/run_e2.py --budget-usd 1.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
TEMPLATES = WORKSPACE / "prompts" / "templates"
sys.path.insert(0, str(TEMPLATES))

from src.common import env  # noqa: E402,F401
from src.generate import pipeline, metrics  # noqa: E402

OUT_DIR = Path("D:/pt_ner_work/data/synthetic/e2")
LEDGER = Path("D:/pt_ner_work/results/gen.csv")
FINAL_POS = WORKSPACE / "data" / "synthetic" / "e2" / "e2_pos.jsonl"
FINAL_NEG = WORKSPACE / "data" / "synthetic" / "e2" / "e2_neg.jsonl"

TARGET_POS, TARGET_NEG = 165, 135
PER_BATCH_POS, PER_BATCH_NEG = 11, 9


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
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    target_pos = PER_BATCH_POS if args.smoke else TARGET_POS
    target_neg = PER_BATCH_NEG if args.smoke else TARGET_NEG

    if args.dry_run:
        print("=== dry-run：只渲染 Prompt ===")
        p_pos = pipeline.prepare_prompt.render("e2_generation", batch="e2_001", n=PER_BATCH_POS)
        p_neg = pipeline.prepare_prompt.render("e2_negative", batch="e2_101", n=PER_BATCH_NEG)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dryrun_pos.txt").write_text(p_pos, encoding="utf-8")
        (out_dir / "dryrun_neg.txt").write_text(p_neg, encoding="utf-8")
        print(f"含实体 Prompt {len(p_pos)} 字符 → {out_dir / 'dryrun_pos.txt'}")
        print(f"无实体 Prompt {len(p_neg)} 字符 → {out_dir / 'dryrun_neg.txt'}")
        return 0

    from src.generate.client import DeepSeekClient

    client = DeepSeekClient(model=args.model, base_url=args.base_url, temperature=args.temperature)
    print(f"模型 {client.model} @ {client.base_url}，温度 {client.temperature}")
    print(f"目标：{target_pos} 含实体 + {target_neg} 无实体")

    print("\n[1/2] 含实体批次")
    pos = pipeline.collect(
        client, stage="e2_generation", expect="entity", target=target_pos, per_batch=PER_BATCH_POS,
        arm="E2", out_dir=out_dir, ledger_path=LEDGER, batch_prefix="e2_0",
        max_calls=args.max_calls, budget_usd=args.budget_usd, price_in=args.price_in, price_out=args.price_out,
    )
    print("\n[2/2] 无实体批次")
    neg = pipeline.collect(
        client, stage="e2_negative", expect="none", target=target_neg, per_batch=PER_BATCH_NEG,
        arm="E2", out_dir=out_dir, ledger_path=LEDGER, batch_prefix="e2_1",
        max_calls=args.max_calls, budget_usd=args.budget_usd, price_in=args.price_in, price_out=args.price_out,
    )

    print(f"\n采集完成：含实体 {pos['accepted']} / 无实体 {neg['accepted']}；调用 {pos['calls'] + neg['calls']} 次；"
          f"花费 ${pos['cost_usd'] + neg['cost_usd']:.4f}")

    if pos["accepted"] < target_pos or neg["accepted"] < target_neg:
        print("⚠ 未凑够目标条数：先保留已产出的部分，可再跑一次续采（已产文件不会重复覆盖）")

    p = pipeline.merge_write(pos["rows"], FINAL_POS)
    n = pipeline.merge_write(neg["rows"], FINAL_NEG)
    print(f"合并落盘：{p}（{len(pos['rows'])} 条）、{n}（{len(neg['rows'])} 条）")

    # 项目级检查（第二阶段）——直接调用模块，不起子进程
    from src.generate.validator import check_file

    seeds = [json.loads(l)["text"] for l in (WORKSPACE / "data/curated/seeds_20.jsonl").open(encoding="utf-8")]
    for label, path, expect in (("含实体", FINAL_POS, "entity"), ("无实体", FINAL_NEG, "none")):
        r = check_file(path, seeds, expect)
        print(f"[项目检查-{label}] accepted {r['accepted']} / rejected {r['rejected']} / "
              f"no_entity_ratio {r['no_entity_ratio']} / per_class {r['per_class']}")

    res = metrics.report(FINAL_POS, FINAL_NEG)
    md = metrics.to_markdown(res)
    report = Path("D:/pt_ner_work/results/e2_metrics.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    report.with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"\n质量度量 → {report}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
