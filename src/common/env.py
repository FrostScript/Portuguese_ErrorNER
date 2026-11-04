"""统一的运行环境设置，所有实验脚本第一行导入。

为什么需要：
1. `huggingface.co` 在本机被 DNS 污染，必须走镜像；GLiNER 会按 gliner_config.json
   里的 model_name 去取基座配置，缓存位置也要指定，否则报连接失败。
2. 受限沙箱下 `%TEMP%` 不可写，torch/triton 会把编译缓存和临时目录建到**当前工作目录**，
   在工作区根目录留下一堆 `tmpXXXXXXXX/` 与 `torchinductor_*/`。这里把临时目录
   显式指到 D 盘，避免污染仓库。

用法：
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from src.common import env  # noqa: F401  仅需副作用
"""

from __future__ import annotations

import os
from pathlib import Path

WORK_ROOT = Path("D:/pt_ner_work")
TMP_DIR = WORK_ROOT / "tmp"

# --- HuggingFace
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(WORK_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- 临时目录（必须在 import torch 之前生效）
TMP_DIR.mkdir(parents=True, exist_ok=True)
for _key in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_key] = str(TMP_DIR)

# --- 控制台编码（Windows 下打印葡语需要）
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def describe() -> dict[str, str]:
    return {
        "HF_ENDPOINT": os.environ["HF_ENDPOINT"],
        "HF_HOME": os.environ["HF_HOME"],
        "TMPDIR": os.environ["TMPDIR"],
    }
