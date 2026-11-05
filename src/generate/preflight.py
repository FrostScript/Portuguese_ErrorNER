"""API key 预检：确认凭证已就位，且**不打印 key 内容**。

来源优先级：
    1. 环境变量 DEEPSEEK_API_KEY
    2. D:\\pt_ner_work\\.secrets\\deepseek.key（方法 B，仓库外）

用法：
    & "...\\.venv\\Scripts\\python.exe" src/generate/preflight.py           # 只检查本地文件
    & "...\\.venv\\Scripts\\python.exe" src/generate/preflight.py --ping    # 额外发一次最小请求验证额度

安全约定：任何情况下都不输出 key 的明文片段，只输出长度与格式判断。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: F401

SECRETS_FILE = Path("D:/pt_ner_work/.secrets/deepseek.key")
DEFAULT_BASE = "https://api.deepseek.com"
PLACEHOLDER = "sk-REPLACE_ME"
MIN_LEN = 20


def load_key() -> tuple[str | None, str]:
    """返回 (key, 来源描述)。"""
    v = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if v:
        return v, "环境变量 DEEPSEEK_API_KEY"
    if SECRETS_FILE.exists():
        raw = SECRETS_FILE.read_text(encoding="utf-8", errors="replace").strip()
        # 允许误写成 KEY=xxx 或带引号的情况，做一次容错
        if "=" in raw and not raw.startswith("sk-"):
            raw = raw.split("=", 1)[1]
        raw = raw.strip().strip('"').strip("'")
        return (raw or None), f"文件 {SECRETS_FILE}"
    return None, "未找到"


def check_shape(key: str) -> list[str]:
    problems = []
    if key == PLACEHOLDER or "REPLACE" in key.upper():
        problems.append("仍是占位符，请把 sk-REPLACE_ME 替换成真实 key")
    if len(key) < MIN_LEN:
        problems.append(f"长度只有 {len(key)}，不像有效 key（通常 ≥{MIN_LEN}）")
    if not key.startswith("sk-"):
        problems.append("不以 sk- 开头，请确认复制完整")
    if any(c.isspace() for c in key):
        problems.append("含空白字符，可能复制时带入了换行或空格")
    return problems


def ping(base: str, key: str) -> tuple[bool, str]:
    url = base.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read(400).decode("utf-8", errors="replace")
            return True, f"HTTP {resp.status}：{body[:200]}"
    except urllib.error.HTTPError as e:
        detail = e.read(300).decode("utf-8", errors="replace")
        return False, f"HTTP {e.code}：{detail[:200]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true", help="额外发一次最小请求验证 key 可用")
    ap.add_argument("--base", default=DEFAULT_BASE, help="API base url")
    args = ap.parse_args()

    key, source = load_key()
    print(f"来源   : {source}")

    if not key:
        print("状态   : 未找到 key")
        print(f"做法   : 把 key 写进 {SECRETS_FILE}（只放 key 本身，不要前缀与引号）")
        return 2

    print(f"长度   : {len(key)}")
    print(f"前缀   : {'sk- 开头 ✓' if key.startswith('sk-') else '不是 sk- 开头 ✗'}")

    problems = check_shape(key)
    if problems:
        print("状态   : 格式有问题")
        for p in problems:
            print(f"  - {p}")
        return 3

    print("状态   : 格式检查通过（未验证额度）")

    if args.ping:
        ok, msg = ping(args.base, key)
        print(f"连通性 : {'可用 ✓' if ok else '失败 ✗'}")
        print(f"        {msg}")
        return 0 if ok else 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
