# 模板版（本项目已适配）

五个原始模板（`e2_generation` / `e3_guidance` / `e3_refiner` / `e3_generation` / `e3_calibrator`）仅保留占位符，不内嵌种子、标签清单或具体样本量。数据独立放在 `data/seeds_20.json`，标签独立放在 `data/label_schema.json`。

## 本项目改动（2026-11-04）

| 改动 | 说明 |
|---|---|
| 新增 `prompts/e2_negative.txt` | 原模板要求"每条至少一个实体"，与项目 `specs/05` 的 **45% 无实体句**冲突。新增 E2 的无实体变体 |
| 新增 `prompts/e3_negative.txt` | E3 同样需要 45% 无实体句，切在 `e3_generation` 之后、沿用同一份 refined guidance |
| 新增 `prompts/e3_calibrator_negative.txt` | AC 校准无实体批次：判据是"entities 为空 + 无任何专名 + 不是被挖空的残缺句" |
| `prepare_prompt.py` | `stage` 增加 `e2_negative` / `e3_negative` / `e3_calibrator_negative` |
| `convert_validate.py` | 新增 `--expect {auto,entity,none}`：含实体批次用 `entity`，无实体批次用 `none`；不指定保持原严格规则 |

**E2 与 E3 都使用本模板集**，唯一的差别是 E3 多了 guidance → refiner 两个前置步骤。

**注意：** `convert_validate.py` 一次调用只能处理同类批次——同一文件里混有无实体句和含实体句会被整批拒绝（这是刻意的，防止批次口径混乱）。

## 八个可用 stage

| stage | 用在哪 | 需要额外参数 |
|---|---|---|
| `e2_generation` | E2 含实体批次（11 条/批） | — |
| `e2_negative` | E2 无实体批次（9 条/批） | — |
| `e3_guidance` | E3 第 1 步：初始指导 | — |
| `e3_refiner` | E3 第 2 步：审查并优化指导 | `--initial` |
| `e3_generation` | E3 第 3 步：含实体批次（11 条/批） | `--refined` |
| `e3_negative` | E3 第 3 步：无实体批次（9 条/批） | `--refined` |
| `e3_calibrator` | E3 第 4 步：校准含实体批次 | `--refined --generated` |
| `e3_calibrator_negative` | E3 第 4 步：校准无实体批次 | `--refined --generated` |

## 用法

需要 Python 3，无第三方依赖。在 `prompts/templates/` 目录中执行。

### E2（两次请求 / 批）

```bash
python prepare_prompt.py e2_generation --batch e2_001 --n 11 --out e2_pos_request.txt
python prepare_prompt.py e2_negative   --batch e2_101 --n 9  --out e2_neg_request.txt

# 把请求发给模型，回复分别保存为 e2_pos_raw.jsonl / e2_neg_raw.jsonl
python convert_validate.py e2_pos_raw.jsonl e2_pos_checked.jsonl --expect entity
python convert_validate.py e2_neg_raw.jsonl e2_neg_checked.jsonl --expect none

# 项目级规范检查（句长、密度、抄种子、跨批重复）
python ../../src/generate/validator.py e2_pos_checked.jsonl --expect entity
python ../../src/generate/validator.py e2_neg_checked.jsonl --expect none
```

### E3（四个阶段，每步落盘后再进下一步）

```bash
# 1) 初始指导
python prepare_prompt.py e3_guidance --out step1.txt
# 2) 审查优化指导
python prepare_prompt.py e3_refiner --initial initial_guidance.json --out step2.txt
# 3a) 含实体批次生成
python prepare_prompt.py e3_generation --refined refined_guidance.json --batch e3_001 --n 11 --out step3a.txt
# 3b) 无实体批次生成
python prepare_prompt.py e3_negative  --refined refined_guidance.json --batch e3_101 --n 9  --out step3b.txt
# 4a) 校准含实体批次
python prepare_prompt.py e3_calibrator          --refined refined_guidance.json --generated e3_pos_raw.jsonl --out step4a.txt
# 4b) 校准无实体批次
python prepare_prompt.py e3_calibrator_negative --refined refined_guidance.json --generated e3_neg_raw.jsonl --out step4b.txt

python convert_validate.py e3_pos_calibrated.jsonl e3_pos_checked.jsonl --expect entity
python convert_validate.py e3_neg_calibrated.jsonl e3_neg_checked.jsonl --expect none
```

中间产物：`initial_guidance.json`、`refined_guidance.json`、`e3_{pos,neg}_raw.jsonl`、`e3_{pos,neg}_calibrated.jsonl`。
可用 `--schema` / `--seeds` 替换外部配置；`--n` 是每批生成数量，由调用时指定。

## 仍需遵守的约定

- 两组共用相同种子、模型和基础校验；先各试 20 条（G4 闸门），最终各保留 300 条，**种子不混入训练**；
- **批次 ID 不能重复**（含实体用 `e2_001…`，无实体用 `e2_101…` 号段）；
- 跨批合并原始或校准 JSONL 再统一转换，以便检查重复；
- 校验器只检查结构和偏移，**不保证语义与漏标正确**，需人工抽检；
- 当前校验器按项目 PER/ORG/LOC 校验；
- E3 为四阶段项目适配草案，不声称严格复现论文；
- 自然葡语若需额外预处理，两组须统一并重新计算偏移。
