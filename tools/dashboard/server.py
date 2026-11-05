"""本地实验看板服务（仅标准库，不依赖任何实验代码）。

设计原则：
    1. **不修改任何实验脚本**：页面只读运行产物（日志文件、experiments.csv、
       train_config.json、docs/runs/），实验代码一行都不用改。
    2. **只绑 127.0.0.1**：不对外暴露。
    3. **路径白名单**：接口只能读 logs/、results/、docs/runs/ 与 D:/pt_ner_work/ 下的文件。

启动：
    & "...\\.venv\\Scripts\\python.exe" tools/dashboard/server.py --port 8765

接口：
    GET /                    页面
    GET /api/state           汇总状态（运行列表、台账、checkpoint、运行记录）
    GET /api/log?name=<file> 某个日志文件的尾部
    GET /api/progress?ckpt=<相对路径>  checkpoint 目录下的 train_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WORKSPACE = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
LOGS = WORKSPACE / "logs"
RESULTS = WORKSPACE / "results"
RUNS_DOC = WORKSPACE / "docs" / "runs"
D_WORK = Path("D:/pt_ner_work")
CHECKPOINTS = D_WORK / "checkpoints"

ALLOWED_ROOTS = [LOGS, RESULTS, RUNS_DOC, CHECKPOINTS, D_WORK / "results"]
TAIL_LINES = 400


def _safe_path(raw: str, roots: list[Path]) -> Path | None:
    """把用户给的路径限制在白名单目录内。"""
    try:
        p = Path(raw)
        if not p.is_absolute():
            p = WORKSPACE / p
        p = p.resolve()
    except OSError:
        return None
    for root in roots:
        try:
            root_r = root.resolve()
        except OSError:
            continue
        if p == root_r or root_r in p.parents:
            return p
    return None


def _read_text_any(path: Path) -> str:
    """按 BOM 自动判定编码。

    Windows PowerShell 5.1 的 Tee-Object 只能写 UTF-16LE（没有 -Encoding 参数），
    而 Python 日志一般是 UTF-8，两种文件都要能读。
    """
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _tail(path: Path, limit: int = TAIL_LINES) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = _read_text_any(path)
    except OSError as exc:
        return f"[读取失败] {exc}"
    lines = text.splitlines(keepends=True)
    return "".join(lines[-limit:])


def _read_runs() -> list[dict]:
    """logs/runs.jsonl：由 scripts/run.ps1 追加。

    每次运行会写两行（开始 status=running、结束 status=done/failed，同 id），
    这里按 id 去重，保留最后一条状态。
    """
    path = LOGS / "runs.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[rec.get("id") or rec.get("start") or line] = rec
    runs = list(latest.values())
    runs.sort(key=lambda r: r.get("start", ""), reverse=True)
    return runs


def _read_experiments() -> list[dict]:
    path = RESULTS / "experiments.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _scan_checkpoints(limit: int = 20) -> list[dict]:
    if not CHECKPOINTS.exists():
        return []
    items = []
    for cfg in sorted(CHECKPOINTS.glob("*/*/train_config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        final = cfg.parent / "final"
        size = 0
        if final.exists():
            size = sum(f.stat().st_size for f in final.rglob("*") if f.is_file())
        items.append(
            {
                "path": str(cfg.parent.relative_to(CHECKPOINTS)).replace("\\", "/"),
                "arm": data.get("arm"),
                "seed": data.get("seed"),
                "max_steps": data.get("max_steps"),
                "batch": data.get("batch"),
                "lr": data.get("lr"),
                "train_seconds": data.get("train_seconds"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
                "data_stats": data.get("data_stats"),
                "checkpoint": data.get("checkpoint"),
                "size_mb": round(size / 1e6, 1),
                "log_history": data.get("log_history", [])[-5:],
            }
        )
        if len(items) >= limit:
            break
    return items


def _list_run_docs() -> list[dict]:
    if not RUNS_DOC.exists():
        return []
    docs = []
    for p in sorted(RUNS_DOC.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.name.lower() == "readme.md":
            continue
        docs.append(
            {
                "name": p.name,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "size": p.stat().st_size,
            }
        )
    return docs


def _recent_activity() -> dict:
    """最近活动：把关键文件的 mtime 汇总，用于判断"页面显示的东西是不是新的"。

    看板的日志面板只读 logs/*.log；若脚本没有 tee 到这里，面板会停在旧文件上。
    这里显式给出各产物的最后更新时间，便于一眼看出是哪一环没写。
    """
    targets = {
        "实验台账": RESULTS / "experiments.csv",
        "生成台账": D_WORK / "results" / "gen.csv",
        "项目日志": RUNS_DOC.parent / "logs" / "project_log.md",
        "最新运行日志": None,
        "最新 checkpoint": None,
    }
    logs = sorted(LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if LOGS.exists() else []
    if logs:
        targets["最新运行日志"] = logs[0]
    ckpts = sorted((p for p in CHECKPOINTS.glob("*/*/train_config.json")), key=lambda p: p.stat().st_mtime, reverse=True) if CHECKPOINTS.exists() else []
    if ckpts:
        targets["最新 checkpoint"] = ckpts[0]

    out = {}
    now = datetime.now().timestamp()
    for label, path in targets.items():
        if path is None or not Path(path).exists():
            out[label] = {"path": None, "mtime": None, "age_minutes": None}
            continue
        m = Path(path).stat().st_mtime
        out[label] = {
            "path": str(path),
            "mtime": datetime.fromtimestamp(m).strftime("%Y-%m-%d %H:%M"),
            "age_minutes": round((now - m) / 60, 1),
        }
    return out


def state() -> dict:
    runs = _read_runs()
    return {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "workspace": str(WORKSPACE),
        "recent": _recent_activity(),
        "runs": runs[:30],
        "running": [r for r in runs if r.get("status") == "running"],
        "experiments": _read_experiments(),
        "checkpoints": _scan_checkpoints(),
        "run_docs": _list_run_docs(),
        "logs": sorted(
            ({"name": p.name, "size": p.stat().st_size,
              "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")}
             for p in LOGS.glob("*.log")),
            key=lambda x: x["mtime"],
            reverse=True,
        )[:40],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "dsh-lab-dashboard/1.0"

    def log_message(self, fmt, *args):  # 静默，避免刷屏
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        if path == "/app.js":
            return self._send(200, (STATIC / "app.js").read_bytes(), "application/javascript; charset=utf-8")
        if path == "/style.css":
            return self._send(200, (STATIC / "style.css").read_bytes(), "text/css; charset=utf-8")

        if path == "/api/state":
            return self._json(state())

        if path == "/api/log":
            name = (query.get("name") or [""])[0]
            target = _safe_path(str(LOGS / name), [LOGS])
            if target is None:
                return self._json({"error": "路径不允许"}, 403)
            return self._json({"name": name, "text": _tail(target)})

        if path == "/api/progress":
            ckpt = (query.get("ckpt") or [""])[0]
            target = _safe_path(str(CHECKPOINTS / ckpt / "train_config.json"), [CHECKPOINTS])
            if target is None or not target.exists():
                return self._json({"error": "找不到该 checkpoint"}, 404)
            try:
                return self._json(json.loads(target.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, 500)

        if path == "/api/doc":
            name = (query.get("name") or [""])[0]
            target = _safe_path(str(RUNS_DOC / name), [RUNS_DOC])
            if target is None:
                return self._json({"error": "路径不允许"}, 403)
            return self._json({"name": name, "text": _tail(target, 800)})

        return self._json({"error": "not found"}, 404)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[看板] http://{args.host}:{args.port}/", flush=True)
    print(f"[看板] 工作区 {WORKSPACE}", flush=True)
    print(f"[看板] 日志目录 {LOGS}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[看板] 已停止", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
