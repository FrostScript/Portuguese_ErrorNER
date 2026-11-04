"""E0 环境冒烟测试：加载 GLiNER 多语言权重，在 pt_bosque dev 上跑零样本推理。

用法：
    $env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HOME="D:\\pt_ner_work\\hf_cache"
    & "...\\.venv\\Scripts\\python.exe" src/smoke_test_e0.py

不做评测，只验证：权重能下载并加载、GPU 能用、三类标签能出预测、
预测与 gold 的字符区间口径一致。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: E402,F401  统一环境（HF 镜像、临时目录），需在 import torch 前生效

MODEL_ID = "urchade/gliner_multi-v2.1"
MODEL_DIR = WORKSPACE / "models" / "gliner_multi-v2.1"
DEV = WORKSPACE / "data" / "pt_bosque" / "dev.jsonl"

LABELS = ["person", "organization", "location"]
LABEL2GOLD = {"person": "PER", "organization": "ORG", "location": "LOC"}
N_SENTENCES = 20


def main() -> int:
    import torch
    import transformers
    import gliner
    from gliner import GLiNER

    print(f"torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"transformers {transformers.__version__}")
    print(f"gliner       {getattr(gliner, '__version__', '?')}")

    t0 = time.time()
    target = str(MODEL_DIR) if MODEL_DIR.exists() else MODEL_ID
    print(f"加载来源: {target}")
    model = GLiNER.from_pretrained(target, map_location="cuda")
    print(f"模型加载完成，用时 {time.time() - t0:.1f}s")

    records = [json.loads(line) for line in DEV.open(encoding="utf-8")]
    # 优先取含实体的句子，便于对照
    sample = [r for r in records if r["entities"]][:N_SENTENCES]

    stats = Counter()
    shown = 0
    t1 = time.time()
    for rec in sample:
        preds = model.predict_entities(rec["text"], LABELS, threshold=0.5)
        pred_set = {(p["start"], p["end"], LABEL2GOLD.get(p["label"], p["label"])) for p in preds}
        gold_set = {(e["start"], e["end"], e["label"]) for e in rec["entities"]}

        # 区间口径校验：预测字符区间必须能在原文还原出实体串
        for p in preds:
            if rec["text"][p["start"]:p["end"]] != p["text"]:
                stats["span_mismatch"] += 1
        stats["pred"] += len(pred_set)
        stats["gold"] += len(gold_set)
        stats["tp"] += len(pred_set & gold_set)
        stats["fp"] += len(pred_set - gold_set)
        stats["fn"] += len(gold_set - pred_set)

        if shown < 3:
            shown += 1
            print("\n--- 句子 ---")
            print(rec["text"])
            print("gold:", [(e["text"], e["label"]) for e in rec["entities"]])
            print("pred:", [(p["text"], LABEL2GOLD.get(p["label"], p["label"]), round(p["score"], 3)) for p in preds])

    dt = time.time() - t1
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"\n=== {len(sample)} 句、阈值 0.5 的粗算（仅供冒烟，非正式评测） ===")
    print(f"gold={stats['gold']} pred={stats['pred']} TP={tp} FP={fp} FN={fn}")
    print(f"P={p:.4f} R={r:.4f} F1={f1:.4f}  用时 {dt:.1f}s（{dt / len(sample) * 1000:.0f} ms/句）")
    print(f"区间还原失败: {stats['span_mismatch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
