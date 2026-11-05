"""格式与偏移校验（可 import，也可命令行调用）。

本文件同时是项目的**第一阶段校验器**：
    - 负责 ID/文本唯一、occurrence 定位、区间不重叠、E3 的 keep/revise/reject；
    - 任一行失败则整批不写出（fail closed）；
    - 第二阶段的项目级检查在 `src/generate/validator.py`。

命令行（与原版一致，仅新增 --expect）：
    python convert_validate.py input.jsonl output.jsonl [--expect auto|entity|none]
"""

import argparse
import json
from pathlib import Path


def convert_validate(input_path, output_path, expect="auto"):
    """校验并转换。返回统计 dict；任一行失败抛 ValueError（不写出文件）。"""
    rows = []
    ids = set()
    texts = set()
    rejected = 0

    for line_no, line in enumerate(Path(input_path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            x = json.loads(line)
            if "status" in x:
                assert x["status"] in ["keep", "revise", "reject"], "非法status"
                if x["status"] == "reject":
                    rejected += 1
                    continue
                assert x["sample"]["id"] == x["id"], "校准ID不一致"
                x = x["sample"]
            assert isinstance(x["id"], str) and x["id"] and x["id"] not in ids, "ID为空或重复"
            t = x["text"]
            assert isinstance(t, str) and t.strip() and t not in texts, "文本为空或重复"
            assert isinstance(x["entities"], list), "entities不是列表"
            if expect == "none":
                assert not x["entities"], "本批次要求无实体，但给出了实体"
            else:
                assert x["entities"], "没有实体"
            es = []
            for e in x["entities"]:
                v = e["text"]
                label = e["label"]
                n = e["occurrence"]
                assert isinstance(v, str) and v and label in ["PER", "ORG", "LOC"], "实体或标签不合法"
                assert type(n) == int and n > 0, "occurrence不合法"
                pos = -len(v)
                for _ in range(n):
                    pos = t.find(v, pos + len(v))
                    assert pos >= 0, "找不到指定实体出现位置"
                es.append({"start": pos, "end": pos + len(v), "text": v, "label": label})
            es.sort(key=lambda e: e["start"])
            assert all(es[i]["end"] <= es[i + 1]["start"] for i in range(len(es) - 1)), "实体重叠或重复"
            rows.append({"id": x["id"], "text": t, "entities": es})
            ids.add(x["id"])
            texts.add(t)
        except Exception as ex:
            raise ValueError(f"第{line_no}行失败：{ex}") from ex

    out = Path(output_path)
    if out.resolve() == Path(input_path).resolve():
        raise ValueError("不得覆盖输入文件")
    if out.exists():
        raise ValueError("输出文件已存在，请换文件名")
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return {"passed": len(rows), "calibration_rejected": rejected, "expect": expect, "output": str(out)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--expect", choices=["auto", "entity", "none"], default="auto",
                   help="auto=沿用原规则(必须有实体); entity=强制每条至少1个实体; none=强制每条0个实体(无实体句批次用)")
    a = p.parse_args()
    try:
        r = convert_validate(a.input, a.output, a.expect)
    except ValueError as ex:
        raise SystemExit(f"{ex}；未写出结果")
    print(f"通过 {r['passed']} 条；校准拒绝 {r['calibration_rejected']} 条；模式 {r['expect']}。"
          f"仅验证结构和偏移，不保证语义完整正确。")


if __name__ == "__main__":
    main()
