"""从葡语模板生成英文模板（保证"与葡语版基本相同"）。

**为什么用脚本而不是手写：** 要求是"英文模板与葡语版逐行相同，只替换语言词"。
手写无法保证这一点；机器替换 + 归一化比对可以证明它。

替换表只有三项：
    Portuguese                    → English
    葡语                           → 英文
    o presidente, a empresa       → the president, the company   （泛指示例必须换成英文）

产物：`prompts/templates_en/prompts/*.txt`（8 个）
校验：把产物反向归一化（English→Portuguese、英文→葡语、示例换回）后与葡语原文逐行比对，必须 0 差异。

用法：
    & python src/generate/en/make_en_templates.py
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
SRC = WORKSPACE / "prompts" / "templates" / "prompts"
DST = WORKSPACE / "prompts" / "templates_en" / "prompts"

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

FORWARD = [
    ("Portuguese", "English"),
    ("葡语", "英文"),
    ("o presidente, a empresa", "the president, the company"),
]
BACKWARD = [(b, a) for a, b in FORWARD]


def normalise_back(text: str) -> list[str]:
    for a, b in BACKWARD:
        text = text.replace(a, b)
    return text.splitlines()


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    failed = 0

    for stage in STAGES:
        src_path = SRC / f"{stage}.txt"
        if not src_path.exists():
            print(f"  ! 缺少葡语模板 {src_path}")
            failed += 1
            continue

        text = src_path.read_text(encoding="utf-8")
        out = text
        for a, b in FORWARD:
            out = out.replace(a, b)
        (DST / f"{stage}.txt").write_text(out, encoding="utf-8")

        # 反向归一化后与原文逐行比对
        pt_lines = text.splitlines()
        en_norm = normalise_back(out)
        diff = [
            x
            for x in difflib.unified_diff(pt_lines, en_norm, lineterm="", n=0)
            if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))
        ]
        status = "OK" if not diff else f"DIFF {len(diff)}"
        print(f"  {stage:<26} {len(out):>5} 字符  归一化差异 {status}")
        for line in diff[:4]:
            print(f"      {line[:150]}")
        failed += bool(diff)

    # 残留葡语字符检查（重音符）
    accents = "ãõçáàâéêíóôúÃÕÇÁÀÂÉÊÍÓÔÚ"
    for stage in STAGES:
        p = DST / f"{stage}.txt"
        if not p.exists():
            continue
        bad = [
            (i, line)
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if any(ch in accents for ch in line)
        ]
        if bad:
            print(f"  ! {stage} 仍有带葡语重音的行：")
            for i, line in bad[:3]:
                print(f"      L{i}: {line[:120]}")
            failed += 1

    print(f"\n结果：{'全部通过' if not failed else f'{failed} 项需处理'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
