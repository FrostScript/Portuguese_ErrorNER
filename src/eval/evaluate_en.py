"""英文对照实验的评测入口。

**做法：** 复用葡语 `src/eval/evaluate.py` 的全部实现——指标、17 档阈值网格、错误分类、
bootstrap CI、台账写入——只把"数据集路径"这一处替换成英文。

**为什么这样写：** 按 `docs/specs/11-english-parallel.md` 的隔离原则，不能修改
`evaluate.py`；而评测口径又必须与葡语**逐行相同**（对齐原则）。因此这里以
"加载模块 + 替换 `collect_candidates`" 的方式接入：`main()` 内部对
`collect_candidates` 的调用是模块全局查找，替换后其余逻辑原样执行。

结果：英文侧的阈值扫描、错误分类、CI 计算、台账字段与葡语侧是同一份代码，
不存在"两套实现算出两个数"的风险。

用法与葡语版完全一致，只把 `--cache` / `--out` 指向英文命名空间：

    & python src/eval/evaluate_en.py --model models/gliner_multi-v2.1 --split dev \\
        --cache D:/pt_ner_work/results/e0en_dev_candidates.jsonl \\
        --out   D:/pt_ner_work/results/e0en_dev \\
        --threshold 0.80 --arm E0en --record results/experiments.csv
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: E402,F401  统一环境（HF 镜像、临时目录）

DATA_DIR = WORKSPACE / "data" / "en" / "uner_en"
PT_EVALUATE = WORKSPACE / "src" / "eval" / "evaluate.py"


def _load_pt_module():
    """把葡语评测器当普通模块加载（其 __main__ 保护会阻止自动执行）。"""
    spec = importlib.util.spec_from_file_location("dsh_evaluate_pt", PT_EVALUATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {PT_EVALUATE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pt = _load_pt_module()


def collect_candidates(model_dir: Path, split: str, cache: Path) -> list[dict]:
    """与葡语版唯一不同之处：数据目录指向 `data/en/uner_en/`。"""
    from gliner import GLiNER

    data = DATA_DIR / f"{split}.jsonl"
    if not data.exists():
        raise SystemExit(f"缺少英文数据集 {data}，先跑 src/data/prepare_uner_en.py")
    records = [json.loads(line) for line in data.open(encoding="utf-8")]

    print(f"[候选] 加载模型 {model_dir}")
    t0 = time.time()
    model = GLiNER.from_pretrained(str(model_dir), map_location="cuda")
    print(f"       加载完成 {time.time() - t0:.1f}s，句子 {len(records)} 条")

    cache.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    with cache.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            preds = model.predict_entities(rec["text"], pt.LABELS, threshold=0.0)
            cands = [
                [p["start"], p["end"], pt.LABEL2GOLD.get(p["label"], p["label"]), round(float(p["score"]), 6)]
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


# 替换后，pt.main() 内部调用 collect_candidates 时会解析到本函数的引用
pt.collect_candidates = collect_candidates


if __name__ == "__main__":
    raise SystemExit(pt.main())
