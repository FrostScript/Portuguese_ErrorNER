# 2026-11-13 · E2en 普通合成数据生成（300 条）

**状态：** 完成。**但与 E1en 的分布对齐检查未通过（+45.3%，目标 ±5%）。**
**实验标识：** E2en（英文对照组的"普通合成数据"臂，对应葡语 E2）
**数据划分：** 生成阶段，未训练

> 本文记录**实际执行的参数**。模板与标签规范按"与葡语版基本相同"的要求逐行对齐，只替换语言词与一处泛指示例（见 §六）。

---

## 一、运行配置

| 项 | 值 |
|---|---|
| 模型 | `deepseek-flash`（固定版本，与葡语侧相同） |
| temperature / max_tokens | 0.8 / 8192 |
| reasoning_effort | low |
| 每批条数 | 含实体 11 / 无实体 9（同葡语） |
| 目标条数 | 165 / 135（同葡语） |
| 额度保护 | `--budget-usd 2.0` / `--max-calls 60` |
| 提示词 | `prompts/templates_en/prompts/e2_generation.txt`、`e2_negative.txt` |
| 标签规范 | `prompts/templates_en/data/label_schema_en.json`（**与葡语版逐字相同**） |
| 种子 | `prompts/templates_en/data/seeds_20_en.json`（20 条 EWT 真实句） |
| token 切分 | 复用 `src/generate/pt_tokenize.py`（P-EN7 的既定处理） |

### 完整命令

```powershell
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"

# 前置（已执行）
& $py src\data\make_seeds_en.py
& $py src\generate\en\en_run_e2.py --dry-run      # 渲染检查：残留占位符 0
& $py src\generate\en\en_run_e2.py --smoke        # 冒烟：11 + 9 全通过

# 正式
& $py src\generate\en\en_run_e2.py --budget-usd 2.0 --max-calls 60
```

---

## 二、调用与成本

| 项 | 值 |
|---|---:|
| 实际调用 | 含实体 18 次 / 无实体 15 次 = **33 次**（另有 1 次截断失败，1 次冒烟另计） |
| 失败 | 1 次 `TruncatedError`：`reasoning_tokens=8192/8192`，正文为空 |
| prompt tokens | 57,343 |
| completion tokens | 125,555（其中 reasoning 104,633） |
| 合计 tokens | **182,898** |

> 台账 `gen.csv` 的 `cost_usd` 为 0，因为本次未传 `--price-in/--price-out`（与葡语 E2 当时的调用方式一致），成本只能按 token 数换算。

---

## 三、产物

| 类型 | 路径 |
|---|---|
| 含实体 | `data/en/synthetic/e2_pos_en.jsonl`（165 条） |
| 无实体 | `data/en/synthetic/e2_neg_en.jsonl`（135 条） |
| 训练用合并 | `data/en/synthetic/e2_en_300.jsonl`（300 条，含 tokens） |
| 原始响应 | `D:\pt_ner_work\data\synthetic\e2en\` |
| 生成台账 | `D:\pt_ner_work\results\gen.csv`（`arm = E2en`） |
| 质量度量 | `D:\pt_ner_work\results\e2en_metrics.json` / `.md` |

---

## 四、结果

### 4.1 规模与完整性

| 项 | 值 |
|---|---:|
| 句子总数 | 300（165 含实体 + 135 无实体） |
| 实体总数 | **417**（PER 129 / ORG 142 / LOC 146） |
| 每含实体句实体数 | 2.53 |
| 重复句 | 0 |
| 结构校验 | span 自洽 417/417，token 与原文一致 300/300，**问题 0** |

### 4.2 长度

| 批 | n | 均值 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| 含实体 | 165 | 23.3 | 20 | 28 |
| 无实体 | 135 | 17.2 | 14 | 20 |

差幅 **+6.09 词**，方向正确（含实体更长），落在合格区间 +4~+18 内。

### 4.3 质量度量（D14）

| 项 | pos | neg |
|---|---:|---:|
| 句式重复率 | 0.0 | 0.0 |
| 4-gram 骨架共享率 | 0.0551 | 0.0706 |
| 与种子 8-gram 重叠条数 | 0 | 0 |
| 实体复用率 | 0.0 | — |

---

## 五、与 E1en 的对齐检查（**未通过**）

| 项 | E1en（真实） | E2en（合成） | 偏差 |
|---|---:|---:|---:|
| 实体总数 | 287 | **417** | **+45.3%** |
| 每含实体句实体数 | 1.74 | 2.53 | +45.4% |
| PER | 112 | 129 | +15.2% |
| ORG | **62** | **142** | **+129.0%** |
| LOC | 113 | 146 | +29.2% |
| 含实体句均长 | 18.0 | 23.3 | +5.3 词 |

`specs/05` 的实体总量偏差目标是 **±5%**，本次 **+45.3%**，**闸门未过**。

**原因（可解释，非 bug）：**

1. **模板要求"PER/ORG/LOC 三类大致均衡、不要让一类占主导"**。葡语 E1 的分布本就较均衡（134/113/129），这条要求几乎无副作用；**EWT 的 ORG 只有 21.6%**，强行均衡把 ORG 从 62 拉到 142（+129%）。
2. **模板要求 20–30 词**，而 EWT 真实含实体句均长 18.0，导致合成句系统性偏长 +5.3 词（P-EN4 的实际表现）。

两条根因都来自"模板与葡语版保持基本相同"这一约束本身，**不是实现错误**。

---

## 六、模板与葡语版的字面差异

用脚本把英文模板做语言词归一化（`English→Portuguese`、`英文→葡语`）后与葡语版逐行比较：

| 模板 | 归一化后的差异行数 |
|---|---:|
| `e2_generation.txt` | **0** |
| `e2_negative.txt` | **1** |

唯一差异行：

```
- Generic roles (o presidente, a empresa), dates, numbers, ...
+ Generic roles (the president, the company), dates, numbers, ...
```

泛指的示例词必须换成英文，否则在英文提示里出现葡语词。标签规范 `label_schema_en.json` 与葡语版**逐字相同**（中文释义）。

---

## 七、观察

1. **对齐闸门失败是结构性的**：只要模板保持与葡语一致，EWT 这种 ORG 稀疏的语料就一定会被"强制均衡"推高。这不是能靠调 prompt 修好的问题，除非允许英文模板改约束——而那就违反了"基本相同"。
2. **长度偏长是可预期的**：合成 23.3 词 vs 真实 18.0 词，与 `specs/11` P-EN4 的预测一致（当时预测的是"推向均值上方"，实测差值 +5.3 词）。
3. **生成质量本身没问题**：重复率 0、抄袭 0、结构校验 0 问题、33 次调用即凑满配额，说明链路是健康的。
4. **truncation 风险仍在**：1/34 次因 reasoning 吃满 8192 而空输出，属于 deepseek-flash 的已知行为，pipeline 已按"记一行失败继续"处理。

## 八、后续动作

- **待决策**：在 +45.3% 的密度偏差下是否继续 E2en 训练网格。
  - 方案 A：照常训练 9 组，把偏差写进报告（数据仍是"合规生成"的，只是分布与 E1en 不同）；
  - 方案 B：先不动训练，补一个"密度对齐"的消融（改模板让实体数/类别按 E1en 分布采样），但该消融会偏离"模板相同"的前提；
  - 方案 C：按 `specs/05` 的 ±5% 目标重生成（需要改模板或加后处理筛选），同样偏离前提。
- E3en 仍缺 6 个模板与 `en_run_e3.py`。
