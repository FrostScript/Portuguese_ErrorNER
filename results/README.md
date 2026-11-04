# 成果目录

**用途：** 存放**可直接引用**的实验成果。这里的数字就是最终报告里要用的数字。

## 阈值约定（2026-11-04 定）

**全项目统一使用阈值 0.80。** 所有 arm（E0 / E1 / E2 / E3）都用 0.80 出最终指标，不再逐组搜阈值。

- 依据：E0 在 dev 上扫描 17 档后最优正好是 0.80（Micro-F1 0.7497），且 0.75–0.85 区间平坦（0.7442 / 0.7497 / 0.7427），说明 0.80 不是过拟合出来的尖峰。
- 代价：微调组的分数分布会整体上移，固定 0.80 对它们可能不是最优点。因此每次评测**仍会输出完整扫描表**，扫描最优值一并记入 `experiments.csv` 的 `notes` 列，用于敏感性分析。

## 文件

| 文件 | 内容 |
|---|---|
| `experiments.csv` | **总台账**：每次实验最终指标一行，所有 arm 横向对比用 |
| `E0_zeroshot_dev.md` | E0 零样本基线（阈值 0.80）的成果说明 |

## 与其他目录的分工

| 位置 | 记什么 |
|---|---|
| `results/` | 最终成果（可直接引用） |
| `docs/runs/` | 单次实验的完整运行记录（含命令、环境、过程） |
| `docs/specs/` | 规范：该怎么跑 |
| `docs/logs/project_log.md` | 任务级改动摘要 |
| `D:\pt_ner_work\results\` | 原始输出（完整 JSON、候选缓存、扫描表） |

## experiments.csv 字段

```
run_id, timestamp, arm, split, checkpoint, seed, threshold,
precision, recall, micro_f1, macro_f1, per_f1, org_f1, loc_f1,
tp, fp, fn, type_error, boundary_error, ci_low, ci_high,
gold_entities, pred_entities, sentences, documents, no_entity_ratio, notes
```

- 由 `src/eval/record.py` 统一追加，**不手工编辑**；
- `evaluate.py --record` 会自动写入；
- 同一 arm 多个种子写多行，`seed` 列区分；汇总时按 arm 求均值与标准差。
