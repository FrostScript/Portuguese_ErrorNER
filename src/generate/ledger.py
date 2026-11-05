"""生成侧成本与产量台账。

写到 `D:\\pt_ner_work\\results\\gen.csv`，每次 API 调用一行（或每批一行），
用于回答"这 300 条花了多少钱、失败多少次、重试几次"。

与评测台账 `results/experiments.csv` 分开：那张表记模型指标，这张记生成开销。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CSV = Path("D:/pt_ner_work/results/gen.csv")

COLUMNS = [
    "run_id",
    "timestamp",
    "arm",
    "stage",
    "batch_id",
    "expect",
    "model",
    "temperature",
    "prompt_chars",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "candidates",
    "accepted",
    "rejected",
    "retries",
    "cost_usd",
    "raw_path",
    "out_file",
    "notes",
]


def append_row(row: dict, csv_path: str | Path = DEFAULT_CSV) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {k: row.get(k) for k in COLUMNS}
    if not record.get("timestamp"):
        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: ("" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))) for k, v in record.items()})
    return path


def estimate_cost(prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float) -> float:
    """price 单位：USD / 1M tokens。价格未配置时返回 0，token 数仍照实记录。"""
    return prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out


def summarize(csv_path: str | Path = DEFAULT_CSV) -> dict:
    path = Path(csv_path)
    if not path.exists():
        return {"rows": 0}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    tot = lambda k: sum(float(r[k] or 0) for r in rows)  # noqa: E731
    return {
        "rows": len(rows),
        "prompt_tokens": int(tot("prompt_tokens")),
        "completion_tokens": int(tot("completion_tokens")),
        "candidates": int(tot("candidates")),
        "accepted": int(tot("accepted")),
        "rejected": int(tot("rejected")),
        "cost_usd": round(tot("cost_usd"), 6),
        "accept_rate": round(tot("accepted") / tot("candidates"), 3) if tot("candidates") else 0.0,
    }
