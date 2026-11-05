"""E3 入口：R-GDA 四阶段 + 双路径，按论文 Algorithm 1 实现打分与循环。

流程（每阶段都落盘、每阶段都校验）：
    1. Guidance    —— 由 20 条种子产生初始指导（JSON 五数组）
    2. GR loop     —— Guidance Refiner 打分（100 分制五准则），
                      `score < 90 且 round < 3` 就再 refine 一轮（Algorithm 1）
    3. Generation  —— GDA Composer 按 refined guidance 产批（含实体 11/批、无实体 9/批）
    4. AC loop     —— Augmentation Calibrator 逐条打分（100 分制四准则），
                      `平均分 < 90 且 round < 3` 就对 keep/revise 后的样本再校准一轮
    5. 双层校验 → 合并 300 条 → 度量

依据：docs/specs/10-rgda-adaptation.md（论文 §3.1–§3.3、Figure 8/9、Algorithm 1）

用法：
    & python src/generate/run_e3.py --dry-run          # 只渲染，不花钱
    & python src/generate/run_e3.py --smoke            # 各 1 批，验证全链路
    & python src/generate/run_e3.py --budget-usd 3.0   # 正式
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
TEMPLATES = WORKSPACE / "prompts" / "templates"
sys.path.insert(0, str(TEMPLATES))

from src.common import env, runlog  # noqa: E402,F401
from src.generate import ledger, metrics, pipeline  # noqa: E402
from src.generate import e3_schema  # noqa: E402

import convert_validate as cv  # noqa: E402
import prepare_prompt  # noqa: E402

OUT_DIR = Path("D:/pt_ner_work/data/synthetic/e3")
LEDGER = Path("D:/pt_ner_work/results/gen.csv")
FINAL_DIR = WORKSPACE / "data" / "synthetic" / "e3"

TARGET_POS, TARGET_NEG = 165, 135
PER_BATCH_POS, PER_BATCH_NEG = 11, 9


def _save(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _jlog(row: dict) -> None:
    ledger.append_row(row, LEDGER)


def run_guidance(client, run_tag: str, price_in: float, price_out: float, out_dir: Path) -> str:
    """阶段 1：初始指导。返回规范化的 guidance JSON 文本。"""
    prompt = prepare_prompt.render("e3_guidance")
    comp = client.complete(prompt, tag=f"{run_tag}_E3_guidance")
    raw = _save(out_dir / f"{run_tag}_e3_guidance.raw.txt", comp.text)
    issues = e3_schema.check_guidance(comp.text)
    cost = ledger.estimate_cost(comp.prompt_tokens, comp.completion_tokens, price_in, price_out)
    _jlog({"run_id": run_tag, "arm": "E3", "stage": "e3_guidance", "batch_id": "guidance",
           "expect": "-", "model": comp.model, "temperature": client.temperature,
           "prompt_chars": len(prompt), "prompt_tokens": comp.prompt_tokens,
           "completion_tokens": comp.completion_tokens, "total_tokens": comp.total_tokens,
           "reasoning_tokens": comp.reasoning_tokens, "candidates": 1, "accepted": 0 if issues else 1,
           "rejected": 1 if issues else 0, "cost_usd": round(cost, 6), "raw_path": str(raw),
           "notes": "guidance 校验失败: " + "; ".join(issues[:3]) if issues else "ok"})
    if issues:
        raise SystemExit("guidance 校验失败：" + "; ".join(issues[:5]))
    obj = json.loads(comp.text.strip())
    return json.dumps(obj, ensure_ascii=False, indent=1)


def run_gr_loop(client, guidance_text: str, run_tag: str, price_in: float,
                price_out: float, out_dir: Path) -> tuple[str, list[dict]]:
    """阶段 2：Guidance Refiner 循环（Algorithm 1，最多 3 轮，阈值 90）。

    返回 (供生成阶段使用的指导文本, 每轮轨迹)。
    """
    trace: list[dict] = []
    current = guidance_text
    last_obj: dict = {}
    for round_no in range(1, e3_schema.MAX_ROUNDS + 1):
        prompt = prepare_prompt.render("e3_refiner", initial=current)
        comp = client.complete(prompt, tag=f"{run_tag}_E3_gr_r{round_no}")
        raw = _save(out_dir / f"{run_tag}_e3_gr_r{round_no}.raw.txt", comp.text)
        issues = e3_schema.check_refiner(comp.text)
        cost = ledger.estimate_cost(comp.prompt_tokens, comp.completion_tokens, price_in, price_out)
        score = 0.0
        try:
            last_obj = json.loads(comp.text.strip())
            score = float(last_obj.get("score", 0))
        except Exception:  # noqa: BLE001
            issues = issues + ["JSON 解析失败"]
        _jlog({"run_id": run_tag, "arm": "E3", "stage": "e3_refiner", "batch_id": f"gr_r{round_no}",
               "expect": "-", "model": comp.model, "temperature": client.temperature,
               "prompt_chars": len(prompt), "prompt_tokens": comp.prompt_tokens,
               "completion_tokens": comp.completion_tokens, "total_tokens": comp.total_tokens,
               "reasoning_tokens": comp.reasoning_tokens, "candidates": 1,
               "accepted": 0 if issues else 1, "rejected": 1 if issues else 0,
               "cost_usd": round(cost, 6), "raw_path": str(raw),
               "notes": f"score={score}" + ("; " + "; ".join(issues[:2]) if issues else "")})
        trace.append({"round": round_no, "score": score, "issues": issues[:5], "raw": str(raw)})
        print(f"  [GR r{round_no}] score={score}" + (f" 问题 {len(issues)} 项" if issues else ""), flush=True)
        if issues:
            raise SystemExit(f"GR 第 {round_no} 轮输出不合规：{'; '.join(issues[:5])}")
        current = json.dumps(last_obj.get("revised_guidance", []), ensure_ascii=False, indent=1)
        if not e3_schema.should_continue_loop(score, round_no):
            print(f"  [GR] 终止：score={score} ≥ {e3_schema.SCORE_THRESHOLD} 或已满 {round_no} 轮", flush=True)
            break

    # 生成阶段要"改写后的指导原文"，不是评分对象
    guidance_for_gen = json.dumps(
        {"revised_guidance": last_obj.get("revised_guidance", []),
         "batch_coverage_plan": last_obj.get("batch_coverage_plan", []),
         "annotation_checklist": last_obj.get("annotation_checklist", [])},
        ensure_ascii=False, indent=1)
    return guidance_for_gen, trace


def run_ac_loop(client, samples: list[dict], refined_guidance: str, expect: str,
                run_tag: str, price_in: float, price_out: float, out_dir: Path) -> list[dict]:
    """阶段 4：Augmentation Calibrator 循环，返回 keep/revise 后的样本。"""
    stage = "e3_calibrator" if expect == "entity" else "e3_calibrator_negative"
    current = samples
    for round_no in range(1, e3_schema.MAX_ROUNDS + 1):
        batch_text = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in current)
        prompt = prepare_prompt.render(stage, refined=refined_guidance, generated=batch_text)
        comp = client.complete(prompt, tag=f"{run_tag}_E3_ac_{expect}_r{round_no}")
        raw = _save(out_dir / f"{run_tag}_e3_ac_{expect}_r{round_no}.raw.txt", comp.text)
        issues, stats = e3_schema.check_calibrator(comp.text)
        cost = ledger.estimate_cost(comp.prompt_tokens, comp.completion_tokens, price_in, price_out)
        _jlog({"run_id": run_tag, "arm": "E3", "stage": stage, "batch_id": f"ac_{round_no}",
               "expect": expect, "model": comp.model, "temperature": client.temperature,
               "prompt_chars": len(prompt), "prompt_tokens": comp.prompt_tokens,
               "completion_tokens": comp.completion_tokens, "total_tokens": comp.total_tokens,
               "reasoning_tokens": comp.reasoning_tokens, "candidates": stats["lines"],
               "accepted": stats["keep"] + stats["revise"], "rejected": stats["reject"],
               "cost_usd": round(cost, 6), "raw_path": str(raw),
               "notes": f"avg_score={stats['avg_score']}; round={round_no}"
                        + ("; " + "; ".join(issues[:2]) if issues else "")})
        print(f"  [AC {expect} r{round_no}] 平均分 {stats['avg_score']} "
              f"keep {stats['keep']} / revise {stats['revise']} / reject {stats['reject']}", flush=True)
        if issues:
            print(f"  [AC] 有 {len(issues)} 项格式问题，仅采用能解析的行（示例：{issues[0][:80]}）")
        kept = []
        for line in comp.text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") in ("keep", "revise") and isinstance(obj.get("sample"), dict):
                kept.append(obj["sample"])
        if not e3_schema.should_continue_loop(stats["avg_score"], round_no) or not kept:
            break
        current = kept
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--budget-usd", type=float, default=3.0)
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-out", type=float, default=0.0)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    target_pos = PER_BATCH_POS if args.smoke else TARGET_POS
    target_neg = PER_BATCH_NEG if args.smoke else TARGET_NEG

    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for stage, kw in (("e3_guidance", {}),
                          ("e3_refiner", {"initial": "（初始指导占位）"}),
                          ("e3_generation", {"refined": "（指导占位）"}),
                          ("e3_calibrator", {"refined": "（指导占位）",
                                             "generated": '{"id":"x","text":"t","entities":[]}\n'})):
            p = prepare_prompt.render(stage, batch="e3_001", n=PER_BATCH_POS, **kw)
            _save(out_dir / f"dryrun_{stage}.txt", p)
            print(f"{stage:<18} {len(p):>6} 字符  残留占位符: {'有' if '{' in p.split('Hard constraints')[0][-200:] else '无'}")
        return 0

    from src.generate.client import DeepSeekClient

    client = DeepSeekClient(model=args.model, base_url=args.base_url, temperature=args.temperature,
                            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
    run_tag = f"e3_{datetime.now():%H%M%S}"
    print(f"模型 {client.model}，温度 {client.temperature}，max_tokens {client.max_tokens}，"
          f"reasoning_effort {client.reasoning_effort}")
    print(f"目标：{target_pos} 含实体 + {target_neg} 无实体；run_tag={run_tag}")

    print("\n[1/4] Guidance")
    guidance_text = run_guidance(client, run_tag, args.price_in, args.price_out, out_dir)
    _save(out_dir / f"{run_tag}_guidance.json", guidance_text)
    print(f"  已生成指导 {len(guidance_text)} 字符")

    print("\n[2/4] Guidance Refiner 循环（≤3 轮，阈值 90）")
    refined, gr_trace = run_gr_loop(client, guidance_text, run_tag, args.price_in,
                                    args.price_out, out_dir)
    _save(out_dir / f"{run_tag}_refined.json", refined)
    _save(out_dir / f"{run_tag}_gr_trace.json", json.dumps(gr_trace, ensure_ascii=False, indent=1))

    print("\n[3/4] 生成（含实体 / 无实体）")
    pos = pipeline.collect(
        client, stage="e3_generation", expect="entity", target=target_pos, per_batch=PER_BATCH_POS,
        arm="E3", out_dir=out_dir, ledger_path=LEDGER, batch_prefix="e3_0",
        max_calls=args.max_calls, budget_usd=args.budget_usd, price_in=args.price_in,
        price_out=args.price_out, extra={"refined": refined}, run_tag=run_tag)
    neg = pipeline.collect(
        client, stage="e3_negative", expect="none", target=target_neg, per_batch=PER_BATCH_NEG,
        arm="E3", out_dir=out_dir, ledger_path=LEDGER, batch_prefix="e3_1",
        max_calls=args.max_calls, budget_usd=args.budget_usd, price_in=args.price_in,
        price_out=args.price_out, extra={"refined": refined}, run_tag=run_tag)
    print(f"  含实体 {pos['accepted']} / 无实体 {neg['accepted']}")

    print("\n[4/4] Augmentation Calibrator 循环（≤3 轮，阈值 90）")
    final = {}
    for label, rows, expect in (("pos", pos["rows"], "entity"), ("neg", neg["rows"], "none")):
        if not rows:
            print(f"  {label}: 无候选，跳过")
            continue
        kept = run_ac_loop(client, rows, refined, expect, run_tag, args.price_in,
                           args.price_out, out_dir)
        final[label] = kept
        print(f"  {label}: 校准后保留 {len(kept)} 条")

    # 双层校验 + 落盘
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for label, expect, name in (("pos", "entity", "e3_pos.jsonl"), ("neg", "none", "e3_neg.jsonl")):
        rows = final.get(label, [])
        staged = out_dir / f"{run_tag}_{label}_final.jsonl"
        _save(staged, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        out_path = FINAL_DIR / name
        if out_path.exists():
            out_path.unlink()
        try:
            r = cv.convert_validate(staged, out_path, expect=expect)
            print(f"  [{name}] 通过 {r['passed']} 条")
        except ValueError as exc:
            print(f"  [{name}] 校验失败：{exc}")

    # 合并训练用文件（tokens 用词+独立标点口径）
    from src.generate.tokenize import tokenize

    merged = []
    for name in ("e3_pos.jsonl", "e3_neg.jsonl"):
        p = FINAL_DIR / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            toks, _ = tokenize(r["text"])
            merged.append({"id": r["id"], "doc_id": "synthetic_e3", "split": "synthetic",
                           "tokens": toks, "text": r["text"], "entities": r["entities"]})
    _save(FINAL_DIR / "e3_300.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    print(f"\n训练用合并：{FINAL_DIR / 'e3_300.jsonl'}（{len(merged)} 条）")

    res = metrics.report(FINAL_DIR / "e3_pos.jsonl", FINAL_DIR / "e3_neg.jsonl")
    md = metrics.to_markdown(res)
    rep = Path("D:/pt_ner_work/results/e3_metrics.json")
    _save(rep, json.dumps(res, ensure_ascii=False, indent=2))
    _save(rep.with_suffix(".md"), md)
    print(md)
    return 0


if __name__ == "__main__":
    with runlog.capture("run_e3"):
        raise SystemExit(main())
