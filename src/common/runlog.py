"""统一的运行日志：把 stdout/stderr tee 到 logs/<时间戳>_<名字>.log，并登记 runs.jsonl。

为什么需要：早期只有 `scripts/run.ps1` 做 tee，而它不能在本会话（ConstrainedLanguage）里执行；
改用后台任务后很长一段时间的运行都没进 logs/，看板的日志面板就停在旧文件上。
把这个能力放进 Python，脚本自己负责写日志，与用什么方式启动无关。

用法：
    from src.common import runlog
    with runlog.capture("sweep_E1"):
        main()
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
LOG_DIR = WORKSPACE / "logs"
RUNS = LOG_DIR / "runs.jsonl"


class _Tee:
    """同时写原流与日志文件。"""

    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, data):
        try:
            self.stream.write(data)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.handle.write(data)
            self.handle.flush()
        except Exception:  # noqa: BLE001
            pass
        return len(data)

    def flush(self):
        for s in (self.stream, self.handle):
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass

    def isatty(self):
        return False


@contextmanager
def capture(name: str):
    """在 with 块内把 stdout/stderr 同时写到 logs/ 并登记运行状态。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{stamp}_{name}.log"
    path = LOG_DIR / fname
    run_id = f"{stamp}_{name}"
    start = datetime.now().isoformat(timespec="seconds")

    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"=== {name} ===\nid    : {run_id}\nstart : {start}\nworkdir: {WORKSPACE}\n{'-' * 40}\n")
        fh.flush()
        _register({"id": run_id, "name": name, "file": fname, "cmd": " ".join(sys.argv)[:400],
                   "start": start, "end": None, "exit": None, "status": "running"})
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, fh), _Tee(old_err, fh)
        t0 = time.time()
        code = 0
        try:
            yield path
        except SystemExit as exc:  # 保留退出码
            code = int(exc.code or 0)
            raise
        except BaseException:
            code = 1
            raise
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            end = datetime.now().isoformat(timespec="seconds")
            dur = round(time.time() - t0, 1)
            fh.write(f"{'-' * 40}\nend   : {end}\ntime  : {dur}s\nexit  : {code}\n")
            fh.flush()
    _register({"id": run_id, "name": name, "file": fname, "cmd": " ".join(sys.argv)[:400],
               "start": start, "end": end, "exit": code, "status": "done" if code == 0 else "failed"})


def _register(entry: dict) -> None:
    with RUNS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
