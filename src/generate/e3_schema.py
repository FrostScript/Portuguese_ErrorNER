"""E3 中间产物的 schema 校验：guidance / refiner / calibrator 三类 JSON。

为什么需要：这三个步骤的输出不是样本 JSONL，而是结构化 JSON。格式一错，后面所有步骤都会崩，
而且要等到训练前才暴露。这里在每一步之后立即校验，失败即中止并保留原始响应。

依据：docs/specs/10-rgda-adaptation.md（R-GDA 论文 §3.1–§3.3、Figure 8/9、Algorithm 1）

用法：
    & python src/generate/e3_schema.py guidance  <file.json>
    & python src/generate/e3_schema.py refiner   <file.json>
    & python src/generate/e3_schema.py calibrator <file.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

# 论文 Figure 8：GR 五准则与权重（合计 100）
GR_CRITERIA = {
    "context_abstraction": 30,
    "keyword_detection": 15,
    "structure_relationship": 15,
    "entity_explanation": 20,
    "correct_relationship": 20,
}
# 论文 Figure 9 的 AC 四准则（30/30/25/15）**已废弃（2026-11-10）**：
# AC 改为不打分，四项维度降级为模板里的检查清单，代码侧只看 keep/revise/reject。
# 理由见 docs/specs/10-rgda-adaptation.md「本项目实现偏差」。
SCORE_THRESHOLD = 90   # 仅 GR 使用：score < 90 且 round < 3 才继续 refine
MAX_ROUNDS = 3         # GR 与 AC 共用：最多 3 轮

# guidance 步骤的五个数组（本项目 e3_guidance.txt）
GUIDANCE_KEYS = ["observations", "generation_guidance", "coverage_plan", "annotation_risks", "avoid_until_clarified"]
# refiner 步骤的四个数组（本项目 e3_refiner.txt）+ 论文要求的评分
REFINER_KEYS = ["issues_found", "revised_guidance", "batch_coverage_plan", "annotation_checklist"]


def _load_json(text: str):
    s = text.strip()
    if s.startswith("```"):
        import re

        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


def _check_score(obj: dict, criteria: dict, where: str, feedback_key: str = "feedback") -> list[str]:
    issues = []
    score = obj.get("score")
    if not isinstance(score, (int, float)):
        issues.append(f"{where}: 缺少数值字段 score")
    elif not (0 <= score <= 100):
        issues.append(f"{where}: score={score} 不在 0–100")
    crit = obj.get("criteria")
    if not isinstance(crit, dict):
        issues.append(f"{where}: 缺少 criteria 对象")
        return issues
    for name, weight in criteria.items():
        v = crit.get(name)
        if not isinstance(v, (int, float)):
            issues.append(f"{where}: criteria.{name} 缺失或非数值")
        elif not (0 <= v <= weight):
            issues.append(f"{where}: criteria.{name}={v} 超出 0–{weight}")
    total = sum(v for v in crit.values() if isinstance(v, (int, float)))
    if isinstance(score, (int, float)) and abs(total - score) > 1:
        issues.append(f"{where}: criteria 合计 {total} 与 score {score} 不一致")
    # guidance/refiner 用 feedback，逐样本校准用 issues
    if not isinstance(obj.get(feedback_key), (list, str)):
        issues.append(f"{where}: 缺少 {feedback_key}")
    return issues


def check_guidance(text: str) -> list[str]:
    obj = _load_json(text)
    if not isinstance(obj, dict):
        return ["guidance: 顶层不是 JSON 对象"]
    return [f"guidance: 缺少数组 {k}" for k in GUIDANCE_KEYS
            if not isinstance(obj.get(k), list)]


def check_refiner(text: str) -> list[str]:
    obj = _load_json(text)
    if not isinstance(obj, dict):
        return ["refiner: 顶层不是 JSON 对象"]
    issues = [f"refiner: 缺少数组 {k}" for k in REFINER_KEYS if not isinstance(obj.get(k), list)]
    issues += _check_score(obj, GR_CRITERIA, "refiner")
    if not isinstance(obj.get("revised_guidance"), list) or not obj.get("revised_guidance"):
        issues.append("refiner: revised_guidance 为空，无法进入生成步骤")
    return issues


def check_calibrator(text: str) -> tuple[list[str], dict]:
    """校验 AC 输出（逐行 JSON）。**2026-11-10 改版：不再校验 score / criteria。**

    判定只看「这一行有没有问题」：
        keep   —— 没问题，issues 必须是空数组，sample 为 null
        revise —— 有问题且已改，issues 非空，sample 是修正稿
        reject —— 有问题但改不了，issues 非空，sample 为 null

    返回 (问题列表, 统计)。统计只含行数与三态计数，不再有平均分。
    """
    issues, stats = [], {"lines": 0, "keep": 0, "revise": 0, "reject": 0}
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        stats["lines"] += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"calibrator 第{i}行不是 JSON: {exc}")
            continue
        if not isinstance(obj.get("id"), str):
            issues.append(f"calibrator 第{i}行缺少字符串 id")
        st = obj.get("status")
        if st not in ("keep", "revise", "reject"):
            issues.append(f"calibrator 第{i}行 status 非法: {st!r}")
            continue
        stats[st] += 1
        found = obj.get("issues")
        if not isinstance(found, list):
            issues.append(f"calibrator 第{i}行 issues 不是数组")
            continue
        found = [x for x in found if isinstance(x, str) and x.strip()]
        # 三态与 issues 必须自洽，否则「有没有问题」这个唯一信号就失效了
        if st == "keep" and found:
            issues.append(f"calibrator 第{i}行 status=keep 但 issues 有 {len(found)} 条")
        if st in ("revise", "reject") and not found:
            issues.append(f"calibrator 第{i}行 status={st} 但 issues 为空，说不清问题在哪")
        if st == "revise" and not isinstance(obj.get("sample"), dict):
            issues.append(f"calibrator 第{i}行 status=revise 但缺 sample")
        if st == "reject" and obj.get("sample") is not None:
            issues.append(f"calibrator 第{i}行 status=reject 但 sample 不是 null")
    return issues, stats


def should_continue_loop(avg_score: float, round_no: int) -> bool:
    """Algorithm 1：score < 90 且 round < 3 时继续 refine。

    **仅用于 GR（refiner）**：AC 自 2026-11-10 起不打分，终止条件改为「本轮无问题样本」，
    由 run_e3.run_ac_loop 自己判断，不再走这里。
    """
    return avg_score < SCORE_THRESHOLD and round_no < MAX_ROUNDS


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    kind, path = sys.argv[1], Path(sys.argv[2])
    text = path.read_text(encoding="utf-8-sig")
    try:
        if kind == "guidance":
            issues, extra = check_guidance(text), {}
        elif kind == "refiner":
            issues, extra = check_refiner(text), {}
        elif kind == "calibrator":
            issues, extra = check_calibrator(text)
        else:
            print(f"未知类型 {kind}")
            return 2
    except json.JSONDecodeError as exc:
        print(f"JSON 解析失败：{exc}")
        return 1
    if issues:
        print(f"❌ {kind} 校验失败（{len(issues)} 项）：")
        for x in issues[:12]:
            print("  -", x)
        return 1
    print(f"✅ {kind} 校验通过")
    if extra:
        print("  ", json.dumps(extra, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
