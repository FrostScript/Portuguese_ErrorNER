"""由葡语 `src/generate/run_e3.py` 派生英文 `en_run_e3.py`。

**为什么用脚本派生：** E3 的编排逻辑（四阶段、GR 3 轮、AC 3 轮、双层校验）必须与葡语侧
逐行一致；手抄会引入差异。脚本只改"落脚点"（路径、arm 名、模块导入），其余原样保留。

替换点（全部列在 REPLACEMENTS 里，可逐条核对）：
    1. 模块文档字符串 → 英文对照版说明
    2. WORKSPACE 层级 parents[2] → parents[3]
    3. 提示词目录 templates → templates_en，并追加 templates 供 convert_validate 复用
    4. pipeline → en_pipeline；prepare_prompt → en_prepare_prompt
    5. OUT_DIR / FINAL_DIR → 英文命名空间
    6. arm 标识 "E3" → "E3en"；文件 tag _E3_ → _E3en_
    7. 产物文件名 e3_pos/neg/300 → *_en；doc_id、metrics 文件名同步

用法：
    & python src/generate/en/make_en_run_e3.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
SRC = WORKSPACE / "src" / "generate" / "run_e3.py"
DST = WORKSPACE / "src" / "generate" / "en" / "en_run_e3.py"

DOCSTRING = '''"""英文对照实验 E3en 入口：R-GDA 四阶段（由葡语 `src/generate/run_e3.py` 派生）。

**与葡语版的差别只有落脚点：**
    1. 提示词目录 → `prompts/templates_en/`（经 `en_prepare_prompt` / `en_pipeline`）；
    2. 产物路径 → 英文命名空间（`data/en/synthetic/`、`D:/pt_ner_work/data/synthetic/e3en`）；
    3. arm 标识 → `E3en`，合并文件名 `e3_en_300.jsonl`。
流程、轮数上限、阈值、批大小、双层校验与度量全部沿用葡语版：
    Guidance 1 次 → GR（100 分制 / 阈值 90 / ≤3 轮）→ 每批生成 + AC（三态、不评分、≤3 轮）
    → 校准后再过 convert_validate → 收下 → 凑够 165 / 135 → 合并 → 度量。
token 切分复用 `src/generate/pt_tokenize.py`（P-EN7 的既定处理）。

派生脚本：`src/generate/en/make_en_run_e3.py`（改葡语侧后需重跑）

用法：
    & python src/generate/en/en_run_e3.py --dry-run      # 只渲染，不花钱
    & python src/generate/en/en_run_e3.py --smoke        # 各 1 批
    & python src/generate/en/en_run_e3.py --budget-usd 3
"""'''

REPLACEMENTS = [
    # 3. 模板目录 + 复用 convert_validate
    (
        'TEMPLATES = WORKSPACE / "prompts" / "templates"\nsys.path.insert(0, str(TEMPLATES))',
        'TEMPLATES = WORKSPACE / "prompts" / "templates_en"\nTEMPLATES_PT = WORKSPACE / "prompts" / "templates"\n'
        'sys.path.insert(0, str(TEMPLATES))\nsys.path.insert(0, str(TEMPLATES_PT))',
    ),
    # 4. 模块导入
    (
        "from src.generate import e3_schema, ledger, metrics, pipeline  # noqa: E402",
        "from src.generate import e3_schema, ledger, metrics  # noqa: E402\n"
        "from src.generate.en import en_pipeline as pipeline  # noqa: E402",
    ),
    (
        "import prepare_prompt  # noqa: E402",
        "import en_prepare_prompt as prepare_prompt  # noqa: E402",
    ),
    # 5. 产物目录
    (
        'OUT_DIR = Path("D:/pt_ner_work/data/synthetic/e3")',
        'OUT_DIR = Path("D:/pt_ner_work/data/synthetic/e3en")',
    ),
    (
        'FINAL_DIR = WORKSPACE / "data" / "synthetic" / "e3"',
        'FINAL_DIR = WORKSPACE / "data" / "en" / "synthetic"',
    ),
    # 6. arm 与文件 tag
    ('"arm": "E3"', '"arm": "E3en"'),
    ('f"{run_tag}_E3_', 'f"{run_tag}_E3en_'),
    # 7. 产物文件名 / doc_id / 度量报告
    ('"e3_pos.jsonl"', '"e3_pos_en.jsonl"'),
    ('"e3_neg.jsonl"', '"e3_neg_en.jsonl"'),
    ('"e3_300.jsonl"', '"e3_en_300.jsonl"'),
    ("'e3_300.jsonl'", "'e3_en_300.jsonl'"),
    ('"doc_id": "synthetic_e3"', '"doc_id": "synthetic_e3en"'),
    ('D:/pt_ner_work/results/e3_metrics.json', 'D:/pt_ner_work/results/e3en_metrics.json'),
    # 批次前缀与日志名
    ('prefix="e3_0"', 'prefix="e3en_0"'),
    ('prefix="e3_1"', 'prefix="e3en_1"'),
    ('runlog.capture("run_e3")', 'runlog.capture("run_e3en")'),
    # 2. 工作区层级
    ("WORKSPACE = Path(__file__).resolve().parents[2]",
     "WORKSPACE = Path(__file__).resolve().parents[3]"),
]


def main() -> int:
    text = SRC.read_text(encoding="utf-8")

    new_text, n = re.subn(r'^"""[\s\S]*?"""', lambda _m: DOCSTRING, text, count=1)
    if n != 1:
        print("FAIL: 未找到模块文档字符串")
        return 1
    text = new_text

    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count == 0:
            print(f"  ! 未命中：{old[:70]!r}")
        text = text.replace(old, new)
        print(f"  替换 {count:>2} 处：{old[:60]!r}")

    DST.write_text(text, encoding="utf-8")
    print(f"\n[输出] {DST}（{len(text.splitlines())} 行）")

    # 残留检查：不应再出现葡语侧的路径与模块名
    leftovers = [s for s in (
        "src.generate import e3_schema, ledger, metrics, pipeline",
        "import prepare_prompt  # noqa",
        'synthetic/e3"', '"arm": "E3"', "parents[2]",
    ) if s in text]
    if leftovers:
        print(f"  ! 仍有残留：{leftovers}")
        return 1
    print("  残留检查：通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
