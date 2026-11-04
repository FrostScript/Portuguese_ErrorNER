"""统一实验结果台账：把每次实验的最终指标追加到 results/experiments.csv。

为什么要有：所有 arm（E0 / E1 / E2 / E3）的最终指标落在同一张表，便于直接对比、
便于画图、便于论文取数。CSV 只存"最终数字"，中间过程写 docs/runs/。

用法：
    from src.eval.record import append_row
    append_row(row_dict)

或命令行追加（少见，一般由 evaluate.py --record 自动调用）。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CSV = WORKSPACE / "results" / "experiments.csv"

COLUMNS = [
    "run_id",
    "timestamp",
    "arm",
    "split",
    "checkpoint",
    "seed",
    "threshold",
    "precision",
    "recall",
    "micro_f1",
    "macro_f1",
    "per_f1",
    "org_f1",
    "loc_f1",
    "tp",
    "fp",
    "fn",
    "type_error",
    "boundary_error",
    "ci_low",
    "ci_high",
    "gold_entities",
    "pred_entities",
    "sentences",
    "documents",
    "no_entity_ratio",
    "notes",
]


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def append_row(row: dict, csv_path: str | Path = DEFAULT_CSV) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {k: row.get(k) for k in COLUMNS}
    record.setdefault("run_id", "")
    if not record.get("run_id"):
        record["run_id"] = f"{record.get('arm', 'run')}_{record.get('split', '')}_{datetime.now():%Y%m%d_%H%M%S}"
    if not record.get("timestamp"):
        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: _fmt(v) for k, v in record.items()})
    return path
