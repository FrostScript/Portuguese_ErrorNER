"""提示词渲染（可 import，也可命令行调用）。

占位符注入规则：
    {LABEL_SCHEMA}       ← data/label_schema.json 原文
    {SEED_EXAMPLES}      ← 20 条种子，**只注入 text 与 {text,label}**（剥掉 start/end）
    {N} {BATCH_ID}       ← 调用参数
    {INITIAL_GUIDANCE} {REFINED_GUIDANCE} {GENERATED_BATCH} ← 由命令行或调用方提供

命令行：
    python prepare_prompt.py <stage> --batch e2_001 --n 11 --out req.txt
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = ROOT / "data" / "label_schema.json"
DEFAULT_SEEDS = ROOT / "data" / "seeds_20.json"
STAGES = [
    "e2_generation",
    "e2_negative",
    "e3_guidance",
    "e3_refiner",
    "e3_generation",
    "e3_negative",
    "e3_calibrator",
    "e3_calibrator_negative",
]


def load_seeds(path=DEFAULT_SEEDS):
    seeds = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # 只保留 text 与实体文本/标签：模型不需要偏移，带上反而与"Do not output character offsets"矛盾
    return [
        {"text": x["text"], "entities": [{"text": e["text"], "label": e["label"]} for e in x["entities"]]}
        for x in seeds
    ]


def render(stage, batch="batch001", n=20, initial=None, refined=None, generated=None,
           schema=DEFAULT_SCHEMA, seeds=DEFAULT_SEEDS):
    if stage not in STAGES:
        raise ValueError(f"未知 stage: {stage}")
    template = (ROOT / "prompts" / f"{stage}.txt").read_text(encoding="utf-8")

    values = {
        "LABEL_SCHEMA": Path(schema).read_text(encoding="utf-8-sig"),
        "SEED_EXAMPLES": json.dumps(load_seeds(seeds), ensure_ascii=False, separators=(",", ":")),
        "N": str(n),
        "BATCH_ID": batch,
    }
    for key, val in (("INITIAL_GUIDANCE", initial), ("REFINED_GUIDANCE", refined), ("GENERATED_BATCH", generated)):
        if "{" + key + "}" in template:
            if val is None:
                raise ValueError(f"{stage} 缺少 {key}")
            values[key] = Path(val).read_text(encoding="utf-8-sig") if Path(str(val)).exists() else str(val)

    for k, v in values.items():
        template = template.replace("{" + k + "}", v)
    return template


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--batch", default="batch001")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--initial")
    p.add_argument("--refined")
    p.add_argument("--generated")
    p.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    p.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    p.add_argument("--out", default="ready_prompt.txt")
    a = p.parse_args()
    text = render(a.stage, a.batch, a.n, a.initial, a.refined, a.generated, a.schema, a.seeds)
    Path(a.out).write_text(text, encoding="utf-8")
    print("已生成", a.out)


if __name__ == "__main__":
    main()
