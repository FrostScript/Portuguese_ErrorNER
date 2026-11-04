"""通用微调入口：E1 / E2 / E3 共用同一个训练协议。

**本脚本不会自动运行**：需人工显式调用，写入命令见 docs/specs/06-training-protocol.md。

用法：
    & "...\\.venv\\Scripts\\python.exe" src/train/train_arm.py `
        --arm E1 --data data/curated/real_300.jsonl --seed 13 `
        --max-steps 300 --batch 2 --lr 2e-5 `
        --out D:/pt_ner_work/checkpoints/E1/seed13

约定：
    - 所有 arm 从同一预训练权重 independent 初始化，不在上一组基础上续训；
    - 同一 S、同一 batch、同一优化器；
    - 产出 checkpoint + 训练配置快照，供评测阶段与复现使用。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: E402,F401  统一环境，必须在 import torch 前生效

from src.data.to_gliner import load_examples  # noqa: E402

MODEL_DIR = WORKSPACE / "models" / "gliner_multi-v2.1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="E1 / E2 / E3")
    ap.add_argument("--data", required=True, help="训练数据 JSONL（字符区间格式）")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", required=True, help="checkpoint 输出目录")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    args = ap.parse_args()

    import torch
    from gliner import GLiNER

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = WORKSPACE / data_path
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples, stats = load_examples(data_path)
    print(f"[数据] {stats}")
    if not examples:
        print("FAIL: 无可用训练样本")
        return 1

    print(f"[模型] {args.model_dir}")
    t0 = time.time()
    model = GLiNER.from_pretrained(args.model_dir, map_location="cuda")
    print(f"       {type(model).__name__}，加载 {time.time() - t0:.1f}s")

    config = {
        "arm": args.arm,
        "data": str(data_path),
        "data_stats": stats,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "batch": args.batch,
        "lr": args.lr,
        "model_dir": args.model_dir,
        "out_dir": str(out_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    t1 = time.time()
    trainer = model.train_model(
        train_dataset=examples,
        eval_dataset=None,
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=max(1, args.max_steps // 20),
        save_steps=10_000,
        report_to=[],
        bf16=False,
        fp16=False,
        seed=args.seed,
    )
    train_seconds = time.time() - t1

    save_dir = out_dir / "final"
    model.save_pretrained(str(save_dir))

    config["finished_at"] = datetime.now().isoformat(timespec="seconds")
    config["train_seconds"] = round(train_seconds, 1)
    config["log_history"] = trainer.state.log_history
    config["checkpoint"] = str(save_dir)
    (out_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[完成] {save_dir}，训练 {train_seconds:.1f}s")
    print(f"[配置] {out_dir / 'train_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
