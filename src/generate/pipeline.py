"""通用生成流水线：渲染 Prompt → 调 API → 解析 → 两级校验 → 合并 → 度量。

本模块只负责"一批怎么产"，具体跑 E2 还是 E3 由上层脚本决定。

关键约定：
    - **只补不删**：目标条数固定（含实体 TO_POS 条、无实体 TO_NEG 条），不足就继续请求；
    - **一行坏不废整批**：先逐行丢解析失败的，再用 convert_validate 严格校验；
    - **每次都落盘**：原始响应、校验后样本、台账行，任一时刻中断都能续；
    - **额度保护**：`--max-calls` 与 `--budget-usd` 任一触顶即停。

用法见 `run_e2.py`。
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
TEMPLATES = WORKSPACE / "prompts" / "templates"
sys.path.insert(0, str(TEMPLATES))

from src.common import env  # noqa: E402,F401
from src.generate import ledger  # noqa: E402

import prepare_prompt  # noqa: E402  (prompts/templates/prepare_prompt.py)
import convert_validate as cv  # noqa: E402  (prompts/templates/convert_validate.py)

JSON_LINE = re.compile(r"^\s*\{.*\}\s*$")


def parse_jsonl(text: str) -> tuple[list[dict], int]:
    """从模型输出里抽 JSON 行。容忍 ``` 围栏与前后废话。返回 (对象列表, 丢弃行数)。"""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    objs, dropped = [], 0
    for line in s.splitlines():
        if not line.strip():
            continue
        if not JSON_LINE.match(line):
            dropped += 1
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            dropped += 1
    return objs, dropped


def generate_one_batch(
    client,
    *,
    stage: str,
    batch_id: str,
    n: int,
    expect: str,
    out_dir: Path,
    arm: str,
    ledger_path: Path,
    price_in: float = 0.0,
    price_out: float = 0.0,
    extra: dict | None = None,
    dry_run: bool = False,
    run_tag: str = "",
) -> dict:
    """产一批：渲染 → 调用 → 解析 → 校验。返回统计 dict。

    `run_tag` 会拼进文件名，避免重跑时覆盖/复用上一轮的产物
    （convert_validate 拒绝覆盖已存在的输出文件，旧文件若残留会被误读成新产出）。
    """
    prompt = prepare_prompt.render(stage, batch=batch_id, n=n, **(extra or {}))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run_tag}_{arm}_{stage}_{batch_id}" if run_tag else f"{arm}_{stage}_{batch_id}"

    if dry_run:
        prompt_path = out_dir / f"{tag}.prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        return {"stage": stage, "batch_id": batch_id, "dry_run": True, "prompt_chars": len(prompt),
                "prompt_path": str(prompt_path), "accepted": 0, "candidates": 0}

    comp = client.complete(prompt, tag=tag)

    raw_path = out_dir / f"{tag}.raw.txt"
    raw_path.write_text(comp.text, encoding="utf-8")

    objs, dropped = parse_jsonl(comp.text)
    candidates = len(objs)
    accepted = 0
    rejected = dropped
    note = ""

    if objs:
        staged = out_dir / f"{tag}.staged.jsonl"
        staged.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs), encoding="utf-8")
        checked = out_dir / f"{tag}.checked.jsonl"
        try:
            r = cv.convert_validate(staged, checked, expect=expect)
            accepted = r["passed"]
            rejected += len(objs) - accepted
        except ValueError as exc:
            note = str(exc)[:200]
            rejected += len(objs)

    cost = ledger.estimate_cost(comp.prompt_tokens, comp.completion_tokens, price_in, price_out)
    ledger.append_row(
        {
            "run_id": f"{arm}_{datetime.now():%Y%m%d}",
            "arm": arm,
            "stage": stage,
            "batch_id": batch_id,
            "expect": expect,
            "model": comp.model,
            "temperature": client.temperature,
            "prompt_chars": len(prompt),
            "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens,
            "total_tokens": comp.total_tokens,
            "reasoning_tokens": getattr(comp, "reasoning_tokens", 0),
            "candidates": candidates,
            "accepted": accepted,
            "rejected": rejected,
            "retries": 0,
            "cost_usd": round(cost, 6),
            "raw_path": raw_path,
            "out_file": str(out_dir / f"{tag}.checked.jsonl"),
            "notes": note or f"finish={comp.finish_reason}",
        },
        ledger_path,
    )
    return {
        "stage": stage, "batch_id": batch_id, "candidates": candidates, "accepted": accepted,
        "rejected": rejected, "prompt_tokens": comp.prompt_tokens, "completion_tokens": comp.completion_tokens,
        "reasoning_tokens": getattr(comp, "reasoning_tokens", 0),
        "cost_usd": round(cost, 6), "note": note,
    }


def collect(
    client,
    *,
    stage: str,
    expect: str,
    target: int,
    per_batch: int,
    arm: str,
    out_dir: Path,
    ledger_path: Path,
    batch_prefix: str,
    max_calls: int,
    budget_usd: float,
    price_in: float = 0.0,
    price_out: float = 0.0,
    extra: dict | None = None,
    run_tag: str = "",
) -> dict:
    """反复产批，直到凑够 target 条有效样本。返回汇总。

    `run_tag` 由调用方给定（通常是一次运行的时间戳），用于隔离不同轮次的产物文件；
    为空时自动按当前时间生成，避免读到上一轮的 `.checked.jsonl`。
    """
    run_tag = run_tag or datetime.now().strftime("%H%M%S")
    accepted_rows: list[dict] = []
    stats = {"calls": 0, "candidates": 0, "accepted": 0, "cost_usd": 0.0, "batches": [], "run_tag": run_tag}
    idx = 0
    while len(accepted_rows) < target and stats["calls"] < max_calls and stats["cost_usd"] < budget_usd:
        idx += 1
        batch_id = f"{batch_prefix}{idx:03d}"
        try:
            r = generate_one_batch(
                client, stage=stage, batch_id=batch_id, n=per_batch, expect=expect,
                out_dir=out_dir, arm=arm, ledger_path=ledger_path,
                price_in=price_in, price_out=price_out, extra=extra, run_tag=run_tag,
            )
        except Exception as exc:  # noqa: BLE001  截断/网络等：记一行失败，继续下一批
            stats["calls"] += 1
            stats["failed"] = stats.get("failed", 0) + 1
            note = f"{type(exc).__name__}: {str(exc)[:160]}"
            ledger.append_row(
                {"run_id": f"{arm}_{datetime.now():%Y%m%d}", "arm": arm, "stage": stage,
                 "batch_id": batch_id, "expect": expect, "model": getattr(client, "model", ""),
                 "temperature": getattr(client, "temperature", ""), "candidates": 0, "accepted": 0,
                 "rejected": 0, "notes": note},
                ledger_path,
            )
            print(f"  [{stage} {batch_id}] 失败：{note[:100]}", flush=True)
            stats["batches"].append({"stage": stage, "batch_id": batch_id, "error": note,
                                     "accepted": 0, "candidates": 0, "cost_usd": 0.0})
            continue
        stats["calls"] += 1
        stats["candidates"] += r.get("candidates", 0)
        stats["cost_usd"] = round(stats["cost_usd"] + r.get("cost_usd", 0.0), 6)
        stats["batches"].append(r)
        checked = out_dir / f"{run_tag}_{arm}_{stage}_{batch_id}.checked.jsonl"
        if checked.exists():
            rows = [json.loads(line) for line in checked.read_text(encoding="utf-8").splitlines() if line.strip()]
            accepted_rows.extend(rows)
        print(f"  [{stage} {batch_id}] 收 {r.get('accepted', 0)} 条，累计 {len(accepted_rows)}/{target}，"
              f"花费 ${stats['cost_usd']:.4f}", flush=True)
    stats["accepted"] = len(accepted_rows)
    stats["rows"] = accepted_rows[:target]
    return stats


def merge_write(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    out = []
    for r in rows:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        out.append(r)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    return path
