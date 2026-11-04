"""A1 冒烟验证：Python 3.14 + transformers 5.17 + gliner 0.2.30 能否完成微调。

目的不是训练出可用模型，而是确认微调链路（数据格式 → collator → Trainer → 反向传播 →
保存 → 重新推理）在本机这套版本组合下是否存在硬性故障（重点怀疑 torch.jit 与 Trainer 接口）。

用法：
    $env:PYTHONIOENCODING="utf-8"
    & "...\\.venv\\Scripts\\python.exe" src/train/smoke_finetune.py

产物：D:/pt_ner_work/checkpoints/smoke/（2 步的中间产物，可随时删）
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: E402,F401  统一环境（HF 镜像、临时目录），需在 import torch 前生效

TRAIN_JSONL = WORKSPACE / "data" / "pt_bosque" / "train.jsonl"
MODEL_DIR = WORKSPACE / "models" / "gliner_multi-v2.1"
OUT_DIR = Path("D:/pt_ner_work/checkpoints/smoke")

N_EXAMPLES = 32
MAX_STEPS = 2
BATCH = 2
LABEL_MAP = {"PER": "person", "ORG": "organization", "LOC": "location"}
PROMPT_LABELS = ["person", "organization", "location"]


def load_examples(n: int) -> list[dict]:
    """把项目 JSONL（字符区间）转成 GLiNER 训练格式（词级区间）。"""
    examples = []
    with TRAIN_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            tokens = rec["tokens"]
            text = rec["text"]

            # 重建 token 在 text 中的字符偏移
            offsets = []
            cursor = 0
            ok = True
            for tok in tokens:
                idx = text.find(tok, cursor)
                if idx < 0:
                    ok = False
                    break
                offsets.append(idx)
                cursor = idx + len(tok)
            if not ok:
                continue

            ner = []
            for ent in rec["entities"]:
                s_candidates = [i for i, off in enumerate(offsets) if off == ent["start"]]
                e_candidates = [i for i, off in enumerate(offsets) if off + len(tokens[i]) == ent["end"]]
                if not s_candidates or not e_candidates:
                    continue
                ner.append([s_candidates[0], e_candidates[0], LABEL_MAP[ent["label"]]])

            examples.append({"tokenized_text": tokens, "ner": ner})
            if len(examples) >= n:
                break
    return examples


def main() -> int:
    import torch
    import transformers
    import gliner
    from gliner import GLiNER

    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"transformers {transformers.__version__}")
    print(f"gliner       {getattr(gliner, '__version__', '?')}")

    examples = load_examples(N_EXAMPLES)
    print(f"训练样本 {len(examples)} 条，示例 ner={examples[0]['ner'][:3]}")
    if not examples:
        print("FAIL: 没有可用的训练样本")
        return 1

    print("\n[1/4] 加载权重")
    t0 = time.time()
    model = GLiNER.from_pretrained(str(MODEL_DIR), map_location="cuda")
    print(f"    class = {type(model).__name__}, {time.time() - t0:.1f}s")
    print(f"    has train_model = {hasattr(model, 'train_model')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[2/4] 微调 max_steps=%d batch=%d" % (MAX_STEPS, BATCH))
    t1 = time.time()
    trainer = model.train_model(
        train_dataset=examples,
        eval_dataset=None,
        output_dir=str(OUT_DIR),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=10_000,
        report_to=[],
        bf16=False,
        fp16=False,
    )
    print(f"    微调完成 {time.time() - t1:.1f}s")
    print(f"    train_loss = {trainer.state.log_history}")

    print("[3/4] 保存")
    save_dir = OUT_DIR / "final"
    model.save_pretrained(str(save_dir))
    sizes = [(p.name, p.stat().st_size) for p in save_dir.iterdir() if p.is_file()]
    print(f"    保存到 {save_dir}，文件 {len(sizes)} 个，最大 {max(s for _, s in sizes) / 1e6:.1f} MB")

    print("[4/4] 微调后推理")
    text = "Vera Lúcia Barbosa morreu em São Paulo, no rio Tietê."
    ents = model.predict_entities(text, PROMPT_LABELS, threshold=0.5)
    print("    " + json.dumps([(e["text"], e["label"], round(e["score"], 3)) for e in ents], ensure_ascii=False))

    print("\n结论：A1 通过 —— Python 3.14 上微调链路可用")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("\n结论：A1 失败 —— 微调链路在本机版本组合下不可用")
        traceback.print_exc()
        raise SystemExit(2)
