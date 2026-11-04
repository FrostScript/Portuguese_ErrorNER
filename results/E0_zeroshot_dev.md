# E0 零样本基线（阈值 0.80）

**状态：** 最终成果
**评测集：** 完整 dev（1,172 句 / 40 文档 / 1,401 个 gold 实体）
**模型：** `urchade/gliner_multi-v2.1`（本地 `models/gliner_multi-v2.1`，1,155.9 MB）
**阈值：** 0.80（全项目统一）
**日期：** 2026-11-04

## 主结果

| 指标 | 值 |
|---|---:|
| Precision | 0.7680 |
| Recall | 0.7323 |
| **Micro-F1** | **0.7497** |
| Macro-F1 | 0.7354 |
| 95% CI（按文档 bootstrap，1000 次） | **[0.7148, 0.7814]** |
| TP / FP / FN | 1026 / 310 / 375 |
| gold 实体 / 预测实体 | 1401 / 1336 |

## 分类别

| 类别 | P | R | F1 | 支持数 |
|---|---:|---:|---:|---:|
| PER | 0.8029 | 0.8719 | **0.8360** | 570 |
| ORG | 0.6700 | 0.6427 | **0.6561** | 417 |
| LOC | 0.8233 | 0.6304 | **0.7141** | 414 |

**ORG 比 PER 低 18.0 个点，是三类中的明确短板**；LOC 则是"高精度低召回"（P 0.8233 / R 0.6304，漏检 153 个）。

## 错误结构（阈值 0.80）

| 类型 | 数量 |
|---|---:|
| Type Error（边界对、类别错） | 45 |
| Boundary Error（类别对、边界错） | 45 |
| 其余 FP | 220 |
| 其余 FN | 285 |

## 阈值敏感性

| 阈值 | Micro-F1 |
|---:|---:|
| 0.50（GLiNER 默认） | 0.6843 |
| 0.75 | 0.7442 |
| **0.80** | **0.7497** |
| 0.85 | 0.7427 |
| 0.90 | 0.7080 |

从默认 0.5 调到 0.80 带来 **+6.5 个点**，而这类方法的典型增益只有 1–3 点。仅此一条就说明：不控制阈值，任何"提升"结论都不可信。0.75–0.85 区间平坦（极差 0.7 点），选 0.80 不构成过拟合。

## 复现

```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\eval\evaluate.py `
    --model models/gliner_multi-v2.1 --split dev `
    --cache D:/pt_ner_work/results/e0_dev_candidates.jsonl `
    --out   D:/pt_ner_work/results/e0_dev_thr080 `
    --threshold 0.80 --arm E0 --record results/experiments.csv
```

- 完整报告（含 17 档扫描表）：`D:\pt_ner_work\results\e0_dev_thr080.md`
- 完整 JSON：`D:\pt_ner_work\results\e0_dev_thr080.json`
- 候选缓存：`D:\pt_ner_work\results\e0_dev_candidates.jsonl`
- 运行记录：`docs/runs/2026-11-04_E0_zeroshot_dev.md`
- 台账行：`results/experiments.csv`

## 备注

1. 本报告为**修正版**：初版按 `doc_id` 做 bootstrap 时，`doc_id` 只在每个文档首句存在（其余为空串），导致置信区间被严重高估（初版 [0.5679, 0.7952]）。已修复 `prepare_uner.py` 的文档 ID 传递，现正确值为 [0.7148, 0.7814]；点估计不受影响（同一批预测）。
2. dev 尚未做 872 / 300 的二次切分（TODO A1），切分落地后需重跑一次作为 G2 闸门数字。
3. 这是"不微调"的零样本结果，作为 E1–E3 的比较基准。
