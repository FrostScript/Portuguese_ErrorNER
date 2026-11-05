"""合成数据的模板化度量与两类句子分布对比（TODO D14）。

三个指标（docs/specs/04 §4）：

1. **实体复用率** —— 生成样本里的实体串有多少来自种子实体集合。
   高复用率说明模型只是把种子里的名字搬来搬去。
2. **句式重复率** —— 把实体替换成占位符后比较 token 骨架，衡量"同一句型换个名字"。
3. **两类句子的长度分布对比** —— 含实体句 vs 无实体句，检验 §5 要求的"无系统差异"。

用法：
    & "...\\.venv\\Scripts\\python.exe" src/generate/metrics.py data/synthetic/e2_pos_checked.jsonl \
        --neg data/synthetic/e2_neg_checked.jsonl --out D:/pt_ner_work/results/e2_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: F401

WORD_RE = re.compile(r"\S+")
ENTITY_TOKEN = "<ENT>"
SEEDS = WORKSPACE / "data" / "curated" / "seeds_20.jsonl"


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def seed_entities() -> set[str]:
    ents = set()
    for rec in load(SEEDS):
        for e in rec["entities"]:
            ents.add(e["text"].lower())
    return ents


def entity_reuse(records: list[dict], seed_ents: set[str]) -> dict:
    all_ents = [(e["text"], e["label"]) for r in records for e in r["entities"]]
    if not all_ents:
        return {"entities": 0, "reused_from_seeds": 0, "reuse_rate": 0.0, "top_reused": []}
    reused = [t for t, _ in all_ents if t.lower() in seed_ents]
    counter = Counter(t for t, _ in all_ents if t.lower() in seed_ents)
    return {
        "entities": len(all_ents),
        "distinct_entities": len({t.lower() for t, _ in all_ents}),
        "reused_from_seeds": len(reused),
        "reuse_rate": round(len(reused) / len(all_ents), 4),
        "top_reused": counter.most_common(5),
    }


def skeleton(text: str, ents: list[dict]) -> tuple[str, ...]:
    """把实体片段替换成 <ENT>，返回小写 token 骨架。"""
    spans = sorted((e["start"], e["end"]) for e in ents if 0 <= e["start"] < e["end"] <= len(text))
    out, cursor = [], 0
    for s, e in spans:
        if s < cursor:
            continue
        out.append(text[cursor:s])
        out.append(f" {ENTITY_TOKEN} ")
        cursor = e
    out.append(text[cursor:])
    return tuple(w.lower() for w in WORD_RE.findall("".join(out)))


def template_repetition(records: list[dict]) -> dict:
    if not records:
        return {"sentences": 0, "distinct_skeletons": 0, "repetition_rate": 0.0}
    skeletons = [skeleton(r["text"], r["entities"]) for r in records]
    distinct = len(set(skeletons))
    # n-gram（4-gram）骨架重复率：能在别的句子里找到同一 4-gram 骨架的比例
    grams = Counter()
    for sk in skeletons:
        for i in range(len(sk) - 3):
            grams[sk[i:i + 4]] += 1
    shared = sum(1 for g, c in grams.items() if c > 1)
    return {
        "sentences": len(records),
        "distinct_skeletons": distinct,
        "sentence_repetition_rate": round(1 - distinct / len(records), 4),
        "shared_4gram_rate": round(shared / len(grams), 4) if grams else 0.0,
    }


def length_stats(records: list[dict]) -> dict:
    lens = [len(WORD_RE.findall(r["text"])) for r in records]
    if not lens:
        return {"n": 0}
    lens_sorted = sorted(lens)
    n = len(lens)
    return {
        "n": n,
        "mean": round(sum(lens) / n, 2),
        "median": lens_sorted[n // 2] if n % 2 else (lens_sorted[n // 2 - 1] + lens_sorted[n // 2]) / 2,
        "min": min(lens),
        "max": max(lens),
        "std": round((sum((x - sum(lens) / n) ** 2 for x in lens) / n) ** 0.5, 2),
    }


def compare_lengths(pos: list[dict], neg: list[dict]) -> dict:
    """含实体批 vs 无实体批的长度对比。

    判据（specs/04 §5）：**差异方向与幅度要和真实数据一致**，而不是"无显著差异"。
        真实 E1：含实体 29.07 词 / 无实体 15.19 词 → 差 +13.9
        全量 train：25.65 / 15.94 → 差 +9.7
        合格区间：+4 至 +18 词，且含实体批必须更长
    """
    a = [len(WORD_RE.findall(r["text"])) for r in pos]
    b = [len(WORD_RE.findall(r["text"])) for r in neg]
    res = {"pos": length_stats(pos), "neg": length_stats(neg)}
    if not (a and b):
        res["note"] = "缺一组样本，无法比较"
        return res

    delta = round(sum(a) / len(a) - sum(b) / len(b), 2)
    res["delta_mean"] = delta
    res["real_reference"] = {"e1_delta": 13.9, "train_delta": 9.7, "acceptable": [4, 18]}

    if len(a) >= 5 and len(b) >= 5:
        try:
            from scipy.stats import mannwhitneyu

            u, p = mannwhitneyu(a, b, alternative="two-sided")
            res["mannwhitney"] = {"U": round(float(u), 2), "p_value": round(float(p), 4)}
        except ImportError:
            pass

    if delta < 4:
        res["verdict"] = (f"差幅 {delta:+.2f} 词偏小：无实体句被写得太长，与真实数据的 +9.7~+13.9 不一致，需调 Prompt")
        res["pass"] = False
    elif delta > 18:
        res["verdict"] = f"差幅 {delta:+.2f} 词偏大，超出真实数据的 +9.7~+13.9 太多"
        res["pass"] = False
    else:
        res["verdict"] = (f"差幅 {delta:+.2f} 词，方向正确（含实体句更长），"
                          f"落在真实参照 +9.7~+13.9 的合理邻域")
        res["pass"] = True
    return res


def seed_ngram_overlap(records: list[dict], n: int = 8) -> dict:
    seeds = [r["text"] for r in load(SEEDS)]

    def grams(t: str) -> set[str]:
        w = [x.lower() for x in WORD_RE.findall(t)]
        return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

    seed_g: set[str] = set()
    for s in seeds:
        seed_g |= grams(s)
    hits = sum(1 for r in records if grams(r["text"]) & seed_g)
    return {"sentences": len(records), "with_seed_ngram_overlap": hits,
            "overlap_rate": round(hits / len(records), 4) if records else 0.0}


def report(pos_path: str | Path | None, neg_path: str | Path | None = None) -> dict:
    pos = load(pos_path) if pos_path else []
    neg = load(neg_path) if neg_path else []
    seeds_e = seed_entities()
    out = {
        "pos_file": str(pos_path) if pos_path else None,
        "neg_file": str(neg_path) if neg_path else None,
        "pos": {
            "template": template_repetition(pos),
            "entity_reuse": entity_reuse(pos, seeds_e),
            "seed_ngram": seed_ngram_overlap(pos),
        },
        "neg": {"template": template_repetition(neg), "seed_ngram": seed_ngram_overlap(neg)},
    }
    if pos and neg:
        out["length_comparison"] = compare_lengths(pos, neg)
    return out


def to_markdown(res: dict) -> str:
    lines = ["# 合成数据质量度量（D14）", ""]
    for key in ("pos", "neg"):
        d = res.get(key) or {}
        if not d:
            continue
        t = d.get("template", {})
        lines += [f"## {key} 批", "", "| 项 | 值 |", "|---|---:|",
                  f"| 句子数 | {t.get('sentences')} |",
                  f"| 不同句式骨架 | {t.get('distinct_skeletons')} |",
                  f"| 句式重复率 | {t.get('sentence_repetition_rate')} |",
                  f"| 4-gram 骨架共享率 | {t.get('shared_4gram_rate')} |",
                  f"| 与种子 8-gram 重叠条数 | {d['seed_ngram']['with_seed_ngram_overlap']} |", ""]
        if key == "pos" and "entity_reuse" in d:
            r = d["entity_reuse"]
            lines += ["| 实体总数 | %s |" % r["entities"], "| 不同实体 | %s |" % r["distinct_entities"],
                      "| 来自种子实体 | %s |" % r["reused_from_seeds"], "| 实体复用率 | %s |" % r["reuse_rate"], ""]
    lc = res.get("length_comparison")
    if lc and "verdict" in lc:
        lines += ["## 长度分布对比", "", "| 批 | n | 均值 | 中位 | 最小 | 最大 | 标准差 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key in ("pos", "neg"):
            s = lc[key]
            lines.append(f"| {key} | {s['n']} | {s['mean']} | {s['median']} | {s['min']} | {s['max']} | {s['std']} |")
        lines += ["", f"**差幅 {lc['delta_mean']:+.2f} 词**；真实参照 E1 +13.9 / train +9.7；合格区间 +4~+18"]
        if "mannwhitney" in lc:
            m = lc["mannwhitney"]
            lines.append(f"（Mann-Whitney U = {m['U']}，p = {m['p_value']}，仅作参考）")
        lines += ["", f"**{'通过' if lc.get('pass') else '不通过'}：{lc['verdict']}**"]
    elif lc and "note" in lc:
        lines += ["## 长度分布对比", "", lc["note"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pos", nargs="?", help="含实体批 JSONL；省略则跑自检")
    ap.add_argument("--neg", default=None, help="无实体批 JSONL")
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    args = ap.parse_args()

    if not args.pos:
        # 自检：用种子句与人工构造句跑一遍
        demo_pos = load(SEEDS)[:5]
        demo_neg = [{"id": f"n{i}", "text": t, "entities": []} for i, t in enumerate([
            "O indice de precos ao consumidor subiu dois por cento no mes passado segundo dados do instituto nacional de pesquisa.",
            "A producao industrial cresceu pelo terceiro mes consecutivo e o ritmo de atividade economica deve seguir estavel.",
            "O relatorio aponta melhora no emprego formal e queda no numero de contratos temporarios em todas as regioes.",
            "Os investimentos em saneamento basico foram ampliados e a cobertura de agua tratada chegou a setenta por cento.",
            "A arrecadacao de impostos cresceu acima da inflacao e o superavit primario ficou dentro da meta do governo.",
        ])]
        res = report_from(demo_pos, demo_neg)
        print(to_markdown(res))
        return 0

    res = report(args.pos, args.neg)
    md = to_markdown(res)
    print(md)
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")
        print(f"结果已写入 {args.out}")
    return 0


def report_from(pos: list[dict], neg: list[dict]) -> dict:
    seeds_e = seed_entities()
    out = {"pos": {"template": template_repetition(pos), "entity_reuse": entity_reuse(pos, seeds_e),
                   "seed_ngram": seed_ngram_overlap(pos)},
           "neg": {"template": template_repetition(neg), "seed_ngram": seed_ngram_overlap(neg)}}
    if pos and neg:
        out["length_comparison"] = compare_lengths(pos, neg)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
