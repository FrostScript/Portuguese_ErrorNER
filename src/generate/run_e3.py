"""E3 入口：R-GDA 四阶段，按论文 Algorithm 1 实现打分与循环。

**结构要点（2026-11-05 修订）**：AC 校准按批进行，不是把整侧样本一次性丢进去。
    整侧校准会有两个问题：① 165 条 × 约 150 字符的输入 + 同规模输出必然超过 max_tokens；
    ② 一次给 165 条打分，评分质量会垮。改为「生成为校准服务」：
        每批生成 → AC 循环（最多 3 轮）→ 收下合格样本 → 不够再产下一批。

流程：
    1. Guidance    1 次调用，输出 JSON 五数组
    2. GR loop     ≤3 轮，score < 90 且未满 3 轮则带改写后的指导再评（Algorithm 1）
    3. 生成 + AC   （含实体 11 条/批、无实体 9 条/批）
                   每批：生成 → AC 循环（≤3 轮，平均分 < 90 继续）→ 双层校验 → 收下
                   直到凑够 165 / 135
    4. 合并 e3_300.jsonl → 质量度量

依据：docs/specs/10-rgda-adaptation.md（论文 §3.1–§3.3、Figure 8/9、Algorithm 1）

用法：
    & python src/generate/run_e3.py --dry-run      # 只渲染，不花钱
    & python src/generate/run_e3.py --smoke        # 各 1 批，验证全链路
    & python src/generate/run_e3.py --budget-usd 3 # 正式
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
from src.generate import e3_schema, ledger, metrics, pipeline  # noqa: E402

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


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _to_model_format(rec: dict) -> dict:
    """内部格式（start/end）→ 模型格式（occurrence）。

    校准器的输入输出都必须是 occurrence 格式：convert_validate 的输出只有 start/end，
    直接喂给模型会让它无从填写 occurrence，回收的样本再校验必然失败。
    """
    ents, text = [], rec["text"]
    for e in rec.get("entities", []):
        occ = sum(1 for f in rec["entities"] if f["text"] == e["text"] and f["start"] <= e["start"])
        ents.append({"text": e["text"], "label": e["label"], "occurrence": max(1, occ)})
    return {"id": rec["id"], "text": text, "entities": ents}


def _sample_ok(sample) -> bool:
    """校准器返回的样本是否具备 convert_validate 需要的字段（occurrence 格式）。"""
    if not isinstance(sample, dict):
        return False
    if not isinstance(sample.get("id"), str) or not isinstance(sample.get("text"), str):
        return False
    if not isinstance(sample.get("entities"), list):
        return False
    for e in sample["entities"]:
        if not isinstance(e, dict):
            return False
        if not isinstance(e.get("text"), str) or e.get("label") not in ("PER", "ORG", "LOC"):
            return False
        if type(e.get("occurrence")) is not int or e["occurrence"] < 1:
            return False
    return True


def run_guidance(client, run_tag: str, out_dir: Path, price_in: float, price_out: float) -> str:
    """阶段 1：初始指导 → 规范化 JSON 文本。"""
    prompt = prepare_prompt.render("e3_guidance")
    comp = client.complete(prompt, tag=f"{run_tag}_E3_guidance")
    raw = _save(out_dir / f"{run_tag}_e3_guidance.raw.txt", comp.text)
    issues = e3_schema.check_guidance(comp.text)
    cost = ledger.estimate_cost(comp.prompt_tokens, comp.completion_tokens, price_in, price_out)
    ledger.append_row({
        "run_id": run_tag, "arm": "E3", "stage": "e3_guidance", "batch_id": "guidance",
        "expect": "-", "model": comp.model, "temperature": client.temperature,
        "prompt_chars": len(prompt), "prompt_tokens": comp.prompt_tokens,
        "completion_tokens": comp.completion_tokens, "total_tokens": comp.total_tokens,
        "reasoning_tokens": comp.reasoning_tokens, "candidates": 1,
        "accepted": 0 if issues else 1, "rejected": 1 if issues else 0,
        "cost_usd": round(cost, 6), "raw_path": str(raw),
        "notes": ("guidance 校验失败: " + "; ".join(issues[:3])) if issues else "ok",
    }, LEDGER)
    if issues:
        raise SystemExit("guidance 校验失败：" + "; ".join(issues[:5]))
    return json.dumps(json.loads(comp.text.strip()), ensure_ascii=False, indent=1)


def run_gr_loop(client, guidance_text: str, run_tag: str, out_dir: Path,
                price_in: float, price_out: float) -> tuple[str, list[dict]]:
    """阶段 2：GR 循环（≤3 轮，阈值 90）。返回 (供生成用的指导, 轨迹)。"""
    trace, current, last_obj = [], guidance_text, {}
    guidance_for_gen = guidance_text
    for round_no in range(1, e3_schema.MAX_ROUNDS + 1):
        prompt = prepare_prompt.render("e3_refiner", initial=current)
        comp = client.complete(prompt, tag=f"{run_tag}_E3_gr_r{round_no}")
        raw = _save(out_dir / f"{run_tag}_e3_gr_r{round_no}.raw.txt", comp.text)
        parse_ok = True
        try:
            obj = json.loads(comp.text.strip())
        except json.JSONDecodeError as exc:
            # 多为输出超 max_tokens 被截断：不中止整轮实验，沿用上一轮指导继续
            parse_ok = False
            print(f"  [GR r{round_no}] 输出无法解析（{str(exc)[:60]}），沿用上一轮指导", flush=True)
            ledger.append_row({
                "run_id": run_tag, "arm": "E3", "stage": "e3_refiner", "batch_id": f"gr_r{round_no}",
                "expect": "-", "model": comp.model, "temperature": client.temperature,
                "prompt_chars": len(prompt), "prompt_tokens": comp.prompt_tokens,
                "completion_tokens": comp.completion_tokens, "total_tokens": comp.total_tokens,
                "reasoning_tokens": comp.reasoning_tokens, "candidates": 1, "accepted": 0,
                "rejected": 1, "cost_usd": round(
                    ledger.estimate_cost(comp.prompt_tokens, comp.completion_tokens, price_in, price_out), 6),
                "raw_path": str(raw), "notes": f"JSON 解析失败: {type(exc).__name__}",
            }, LEDGER)
            trace.append({"round": round_no, "score": None, "issues": ["JSON 解析失败"], "raw": str(raw)})
            break

        issues = e3_schema.check_refiner(comp.text)
        cost = ledger.estimate_cost(comp.prompt_tokens, comp.completion_tokens, price_in, price_out)
        score = 0.0
        try:
            score = float(obj.get("score", 0))
        except (TypeError, ValueError):
            issues = issues + ["score 不是数值"]
        ledger.append_row({
            "run_id": run_tag, "arm": "E3", "stage": "e3_refiner", "batch_id": f"gr_r{round_no}",
            "expect": "-", "model": comp.model, "temperature": client.temperature,
            "prompt_chars": len(prompt), "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens, "total_tokens": comp.total_tokens,
            "reasoning_tokens": comp.reasoning_tokens, "candidates": 1,
            "accepted": 0 if issues else 1, "rejected": 1 if issues else 0,
            "cost_usd": round(cost, 6), "raw_path": str(raw),
            "notes": f"score={score}" + ("; " + "; ".join(issues[:2]) if issues else ""),
        }, LEDGER)
        trace.append({"round": round_no, "score": score, "issues": issues[:5], "raw": str(raw)})
        print(f"  [GR r{round_no}] score={score}" + (f"，格式问题 {len(issues)} 项" if issues else ""), flush=True)
        if issues:
            # 评分/结构不合规：保留上一轮指导，不再继续 refine
            print(f"  [GR] 第 {round_no} 轮不合规，沿用上一轮指导：{'; '.join(issues[:3])}", flush=True)
            break

        last_obj = obj
        guidance_for_gen = json.dumps({
            "revised_guidance": obj.get("revised_guidance", []),
            "batch_coverage_plan": obj.get("batch_coverage_plan", []),
            "annotation_checklist": obj.get("annotation_checklist", []),
        }, ensure_ascii=False, indent=1)
        # 下一轮把上一轮的完整评分与反馈一起喂回去，避免只给 revised_guidance 导致上下文丢失
        current = json.dumps(obj, ensure_ascii=False)
        if not e3_schema.should_continue_loop(score, round_no):
            print(f"  [GR] 终止：score={score}，共 {round_no} 轮", flush=True)
            break

    return guidance_for_gen, trace


def run_ac_loop(client, samples: list[dict], refined: str, expect: str, batch_id: str,
                run_tag: str, out_dir: Path, price_in: float, price_out: float) -> tuple[list[dict], float, int]:
    """阶段 4：对**单批**样本做 AC 校准循环（≤3 轮，平均分 < 90 继续）。

    返回 (keep/revise 后的样本, 本批花费, 调用次数)。
    """
    stage = "e3_calibrator" if expect == "entity" else "e3_calibrator_negative"
    # 进来的是内部格式（start/end），先转成模型格式（occurrence）再送去校准
    current = [_to_model_format(s) for s in samples]
    spent, calls = 0.0, 0
    kept: list[dict] = []
    for round_no in range(1, e3_schema.MAX_ROUNDS + 1):
        batch_text = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in current)
        prompt = prepare_prompt.render(stage, refined=refined, generated=batch_text)
        comp = client.complete(prompt, tag=f"{run_tag}_E3_ac_{expect}_{batch_id}_r{round_no}")
        raw = _save(out_dir / f"{run_tag}_e3_ac_{expect}_{batch_id}_r{round_no}.raw.txt", comp.text)
        calls += 1
        cost = ledger.estimate_cost(comp.prompt_tokens, comp.completion_tokens, price_in, price_out)
        spent += cost
        issues, stats = e3_schema.check_calibrator(comp.text)
        ledger.append_row({
            "run_id": run_tag, "arm": "E3", "stage": stage, "batch_id": f"{batch_id}_r{round_no}",
            "expect": expect, "model": comp.model, "temperature": client.temperature,
            "prompt_chars": len(prompt), "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens, "total_tokens": comp.total_tokens,
            "reasoning_tokens": comp.reasoning_tokens, "candidates": stats["lines"],
            "accepted": stats["keep"] + stats["revise"], "rejected": stats["reject"],
            "cost_usd": round(cost, 6), "raw_path": str(raw),
            "notes": f"avg_score={stats['avg_score']}; round={round_no}"
                     + ("; " + "; ".join(issues[:2]) if issues else ""),
        }, LEDGER)
        print(f"  [AC {batch_id} r{round_no}] 输入 {stats['lines']} 条，平均分 {stats['avg_score']}"
              f"，keep {stats['keep']} / revise {stats['revise']} / reject {stats['reject']}"
              + (f"，格式问题 {len(issues)} 项" if issues else ""), flush=True)

        kept = []
        origin = {s["id"]: s for s in current}
        for line in comp.text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = obj.get("status")
            if st == "reject":
                continue
            if st == "keep":
                # keep 一定用原样本：模型复述时容易丢 id/occurrence，原样本是校验过的
                src = origin.get(obj.get("id"))
                if src:
                    kept.append(src)
                continue
            if st == "revise":
                cand = obj.get("sample")
                if _sample_ok(cand):
                    kept.append(cand)
                else:
                    # revise 输出缺字段（实测常丢 occurrence）→ 退回原样本，不丢数据
                    src = origin.get(obj.get("id"))
                    if src:
                        kept.append(src)
                        print(f"     revise 输出不完整（{obj.get('id')}），退回原样本", flush=True)
        if not e3_schema.should_continue_loop(stats["avg_score"], round_no) or not kept:
            print(f"  [AC {batch_id}] 终止：平均分 {stats['avg_score']}，共 {round_no} 轮", flush=True)
            break
        current = kept
    return kept, round(spent, 6), calls


def collect_calibrated(client, *, stage: str, expect: str, target: int, per_batch: int,
                       prefix: str, run_tag: str, out_dir: Path, refined: str,
                       max_calls: int, budget_usd: float, price_in: float,
                       price_out: float) -> dict:
    """生成 + 校准循环：每批生成后立即校准，凑够 target 条合格样本。"""
    accepted: list[dict] = []
    seen_text: set[str] = set()
    stats = {"calls": 0, "cost": 0.0, "batches": 0, "rejected_by_ac": 0, "rejected_by_check": 0}
    idx = 0
    while len(accepted) < target and stats["calls"] < max_calls and stats["cost"] < budget_usd:
        idx += 1
        batch_id = f"{prefix}{idx:03d}"
        need = min(per_batch, target - len(accepted))
        r = pipeline.generate_one_batch(
            client, stage=stage, batch_id=batch_id, n=per_batch, expect=expect,
            out_dir=out_dir, arm="E3", ledger_path=LEDGER, price_in=price_in,
            price_out=price_out, extra={"refined": refined}, run_tag=run_tag)
        stats["calls"] += 1
        stats["cost"] = round(stats["cost"] + r.get("cost_usd", 0.0), 6)
        stats["batches"] += 1
        rows = _read_jsonl(out_dir / f"{run_tag}_E3_{stage}_{batch_id}.checked.jsonl")
        if not rows:
            print(f"  [{stage} {batch_id}] 本批无有效样本，跳过", flush=True)
            continue

        kept, ac_cost, ac_calls = run_ac_loop(client, rows, refined, expect, batch_id,
                                              run_tag, out_dir, price_in, price_out)
        stats["calls"] += ac_calls
        stats["cost"] = round(stats["cost"] + ac_cost, 6)
        stats["rejected_by_ac"] += len(rows) - len(kept)

        # 校准后的样本再走一遍格式校验，防止 revise 改坏偏移
        if kept:
            staged = out_dir / f"{run_tag}_E3_{stage}_{batch_id}_ac.jsonl"
            _save(staged, "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in kept))
            out_path = out_dir / f"{run_tag}_E3_{stage}_{batch_id}_ac_checked.jsonl"
            if out_path.exists():
                out_path.unlink()
            try:
                cv.convert_validate(staged, out_path, expect=expect)
                checked = _read_jsonl(out_path)
            except ValueError as exc:
                print(f"  [{stage} {batch_id}] 校准后校验失败：{str(exc)[:120]}", flush=True)
                checked = []
            stats["rejected_by_check"] += len(kept) - len(checked)
            for rec in checked:
                if rec["text"] in seen_text:
                    continue
                seen_text.add(rec["text"])
                accepted.append(rec)

        print(f"  [{stage} {batch_id}] 收 {len(kept)} 条，累计 {len(accepted)}/{target}"
              f"（目标本批 {need}），花费 ${stats['cost']:.4f}", flush=True)

    stats["accepted"] = len(accepted)
    stats["rows"] = accepted[:target]
    return stats


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
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="E3 的 refiner/calibrator 输出比 E2 长，8192 实测会被截断")
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    target_pos = PER_BATCH_POS if args.smoke else TARGET_POS
    target_neg = PER_BATCH_NEG if args.smoke else TARGET_NEG

    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        specs = [
            ("e3_guidance", {}),
            ("e3_refiner", {"initial": "（初始指导占位）"}),
            ("e3_generation", {"refined": "（指导占位）"}),
            ("e3_calibrator", {"refined": "（指导占位）",
                               "generated": '{"id":"x","text":"t","entities":[]}\n'}),
            ("e3_negative", {"refined": "（指导占位）"}),
            ("e3_calibrator_negative", {"refined": "（指导占位）",
                                        "generated": '{"id":"x","text":"t","entities":[]}\n'}),
        ]
        placeholders = ["{LABEL_SCHEMA}", "{SEED_EXAMPLES}", "{N}", "{BATCH_ID}",
                        "{INITIAL_GUIDANCE}", "{REFINED_GUIDANCE}", "{GENERATED_BATCH}"]
        for stage, kw in specs:
            p = prepare_prompt.render(stage, batch="e3_001", n=PER_BATCH_POS, **kw)
            _save(out_dir / f"dryrun_{stage}.txt", p)
            left = [x for x in placeholders if x in p]
            print(f"{stage:<24} {len(p):>6} 字符  残留占位符: {left or '无'}")
        return 0

    from src.generate.client import DeepSeekClient

    client = DeepSeekClient(model=args.model, base_url=args.base_url, temperature=args.temperature,
                            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
    run_tag = f"e3_{datetime.now():%H%M%S}"
    print(f"模型 {client.model}，温度 {client.temperature}，max_tokens {client.max_tokens}，"
          f"reasoning_effort {client.reasoning_effort}")
    print(f"目标：{target_pos} 含实体 + {target_neg} 无实体；run_tag={run_tag}")
    print(f"额度保护：最多 {args.max_calls} 次调用 / ${args.budget_usd}")

    print("\n[1/4] Guidance")
    guidance_text = run_guidance(client, run_tag, out_dir, args.price_in, args.price_out)
    _save(out_dir / f"{run_tag}_guidance.json", guidance_text)
    print(f"  指导 {len(guidance_text)} 字符")

    print("\n[2/4] Guidance Refiner 循环（≤3 轮，阈值 90）")
    refined, gr_trace = run_gr_loop(client, guidance_text, run_tag, out_dir,
                                    args.price_in, args.price_out)
    _save(out_dir / f"{run_tag}_refined.json", refined)
    _save(out_dir / f"{run_tag}_gr_trace.json", json.dumps(gr_trace, ensure_ascii=False, indent=1))

    print("\n[3/4] 生成 + AC 校准（含实体）")
    pos = collect_calibrated(
        client, stage="e3_generation", expect="entity", target=target_pos, per_batch=PER_BATCH_POS,
        prefix="e3_0", run_tag=run_tag, out_dir=out_dir, refined=refined,
        max_calls=args.max_calls, budget_usd=args.budget_usd,
        price_in=args.price_in, price_out=args.price_out)
    print(f"  含实体：合格 {pos['accepted']} 条，{pos['batches']} 批，AC 淘汰 {pos['rejected_by_ac']}，"
          f"校验淘汰 {pos['rejected_by_check']}")

    print("\n[4/4] 生成 + AC 校准（无实体）")
    neg = collect_calibrated(
        client, stage="e3_negative", expect="none", target=target_neg, per_batch=PER_BATCH_NEG,
        prefix="e3_1", run_tag=run_tag, out_dir=out_dir, refined=refined,
        max_calls=args.max_calls, budget_usd=args.budget_usd,
        price_in=args.price_in, price_out=args.price_out)
    print(f"  无实体：合格 {neg['accepted']} 条，{neg['batches']} 批，AC 淘汰 {neg['rejected_by_ac']}，"
          f"校验淘汰 {neg['rejected_by_check']}")

    print(f"\n合计调用 {pos['calls'] + neg['calls'] + 1 + len(gr_trace)} 次，"
          f"生成+校准花费 ${pos['cost'] + neg['cost']:.4f}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for label, rows, name in (("pos", pos["rows"], "e3_pos.jsonl"),
                              ("neg", neg["rows"], "e3_neg.jsonl")):
        out_path = FINAL_DIR / name
        if out_path.exists():
            out_path.unlink()
        _save(out_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"  {name}: {len(rows)} 条")

    from src.generate.pt_tokenize import tokenize

    merged = []
    for name in ("e3_pos.jsonl", "e3_neg.jsonl"):
        for r in _read_jsonl(FINAL_DIR / name):
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
