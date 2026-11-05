"""步数 × 种子 网格扫描：把训练步数当作实验变量之一。

背景（2026-11-05）：E1（真实数据）在 S=500 仍未见顶，而 E2（合成数据）在 S=500 明显过拟合
（500 步 0.7578 → 200 步 0.7913）。单一 S 无法同时公平服务两组，因此把步数纳入结果表。

用法：
    & python src/train/sweep_steps.py --arm E1 --data data/curated/real_300.jsonl
    & python src/train/sweep_steps.py --arm E2 --data data/synthetic/e2/e2_300.jsonl

已存在的 (arm, seed, max_steps) 组合会自动跳过，可反复运行补漏。
结果写入 results/experiments.csv（与主表同一张，max_steps 列已有）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: F401

PY = r"C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
LEDGER = WORKSPACE / "results" / "experiments.csv"
CKPT_ROOT = Path("D:/pt_ner_work/checkpoints")
RESULT_ROOT = Path("D:/pt_ner_work/results")


def done_combos(arm: str) -> set[tuple[str, str]]:
    if not LEDGER.exists():
        return set()
    with LEDGER.open(encoding="utf-8", newline="") as fh:
        return {
            (r["seed"], r["max_steps"])
            for r in csv.DictReader(fh)
            if r["arm"] == arm and r.get("max_steps")
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["E1", "E2", "E3"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", default="100,200,500")
    ap.add_argument("--seeds", default="13,23,33")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    done = done_combos(args.arm)
    todo = [(s, st) for st in steps for s in seeds if (str(s), str(st)) not in done]

    print(f"[{args.arm}] 网格 {len(steps)} 步数 × {len(seeds)} 种子 = {len(steps) * len(seeds)} 组")
    print(f"[{args.arm}] 已完成 {len(done)} 组，待跑 {len(todo)} 组")
    if not todo:
        print("全部已完成")
        return 0

    import subprocess

    for seed, st in todo:
        ckpt = CKPT_ROOT / f"{args.arm}_sweep" / f"seed{seed}_S{st}"
        cache = RESULT_ROOT / f"{args.arm.lower()}_sweep_s{seed}_S{st}_dev_candidates.jsonl"
        out = RESULT_ROOT / f"{args.arm.lower()}_sweep_s{seed}_S{st}"
        print(f"\n=== {args.arm} seed={seed} S={st} 训练 ===", flush=True)
        r1 = subprocess.run(
            [PY, str(WORKSPACE / "src/train/train_arm.py"), "--arm", args.arm, "--data", str(args.data),
             "--seed", str(seed), "--max-steps", str(st), "--batch", str(args.batch), "--lr", str(args.lr),
             "--out", str(ckpt)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r1.returncode != 0:
            print(f"训练失败：{r1.stderr[-400:]}")
            continue
        for line in r1.stdout.splitlines():
            if "train_runtime" in line or "完成" in line:
                print("  " + line.strip(), flush=True)

        print(f"=== {args.arm} seed={seed} S={st} 评测 ===", flush=True)
        r2 = subprocess.run(
            [PY, str(WORKSPACE / "src/eval/evaluate.py"), "--model", str(ckpt / "final"),
             "--split", "dev", "--cache", str(cache), "--out", str(out),
             "--threshold", "0.80", "--arm", args.arm, "--seed", str(seed),
             "--max-steps", str(st), "--record", str(LEDGER)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r2.returncode != 0:
            print(f"评测失败：{r2.stderr[-400:]}")
            continue
        for line in r2.stdout.splitlines():
            if "固定" in line or "台账" in line:
                print("  " + line.strip(), flush=True)
    return 0


if __name__ == "__main__":
    from src.common import runlog

    with runlog.capture(f"sweep_{sys.argv[1] if len(sys.argv) > 1 else 'steps'}"):
        raise SystemExit(main())
