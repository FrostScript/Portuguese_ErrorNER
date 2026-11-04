"""绕过 HF Hub API（镜像限流）直连下载 GLiNER 权重与所需分词器文件。

背景：hf-mirror 的 /api 接口对本机返回 429（配额耗尽），但 /resolve 直链可用。
GLiNER 仓库本身不含分词器，分词器来自 gliner_config.json 里声明的基座模型
microsoft/mdeberta-v3-base，因此一并下载到同一目录，供本地加载。

用法：
    python src/data/fetch_model.py
产物：
    D:/pt_ner_work/models/gliner_multi-v2.1/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

MIRROR = "https://hf-mirror.com"
WORKSPACE = Path(__file__).resolve().parents[2]
DEST = WORKSPACE / "models" / "gliner_multi-v2.1"

FILES = [
    ("urchade/gliner_multi-v2.1", "gliner_config.json"),
    ("urchade/gliner_multi-v2.1", "model.safetensors"),
    ("microsoft/mdeberta-v3-base", "config.json"),
    ("microsoft/mdeberta-v3-base", "spm.model"),
    ("microsoft/mdeberta-v3-base", "tokenizer_config.json"),
    # mdeberta-v3-base 仓库没有 special_tokens_map.json（实测 404），不需要
]


def fetch(repo: str, name: str) -> None:
    url = f"{MIRROR}/{repo}/resolve/main/{name}"
    out = DEST / name
    if out.exists() and out.stat().st_size > 0:
        print(f"skip (exists) {name} {out.stat().st_size / 1e6:.1f} MB")
        return

    tmp = out.with_suffix(out.suffix + ".part")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        last = 0.0
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    last = time.time()
                    pct = f"{done / total * 100:.1f}%" if total else "?"
                    print(f"  {name}: {done / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct})", flush=True)
    tmp.replace(out)
    print(f"done {name}: {out.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f}s", flush=True)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for repo, name in FILES:
        try:
            fetch(repo, name)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {repo}/{name}: {type(exc).__name__} {exc}")
            return 1
    total = sum(f.stat().st_size for f in DEST.iterdir() if f.is_file())
    print(f"\n目录 {DEST} 共 {total / 1e6:.1f} MB")
    for f in sorted(DEST.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:10.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
