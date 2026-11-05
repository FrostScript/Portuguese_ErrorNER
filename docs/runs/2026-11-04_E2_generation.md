# 2026-11-04 · E2 合成数据生成（普通受约束生成）

**状态：** 完成（数据已落盘），但**分布与 E1 有偏差，待决定是否重生成**

## 一、运行配置

| 项 | 值 |
|---|---|
| Arm | E2 — 普通受约束生成（无 GR/AC） |
| 模型 | `deepseek-flash` @ `https://api.deepseek.com` |
| 温度 | 0.8 |
| max_tokens | 8192 |
| **reasoning_effort** | **low**（必须，见 `specs/04` §6.3） |
| 模板 | `e2_generation`（含实体）/ `e2_negative`（无实体） |
| 每批 | 含实体 11 条、无实体 9 条 |
| 目标 | 165 含实体 + 135 无实体 = 300 |
| 调用次数 | **30**（15 + 15） |

**命令**

```powershell
& $py src\generate\run_e2.py --max-calls 45 --budget-usd 3.0
```

## 二、产物（全部已保存）

| 产物 | 位置 | 内容 |
|---|---|---|
| 含实体批 | `data/synthetic/e2/e2_pos.jsonl` | 165 条，含 start/end |
| 无实体批 | `data/synthetic/e2/e2_neg.jsonl` | 135 条，entities 为空 |
| **训练用合并文件** | `data/synthetic/e2/e2_300.jsonl` | **300 条**，已补 `tokens`/`split` 字段，可直接喂 `train_arm.py` |
| 每批原始响应 | `D:\pt_ner_work\data\synthetic\e2\E2_<stage>_<batch>.raw.txt` | 30 个文件 |
| 每批校验后样本 | 同上目录 `.checked.jsonl` | 30 个文件 |
| **API 原始响应（含 usage）** | `D:\pt_ner_work\raw_responses\*.json` | 30 个，含 reasoning_tokens |
| 生成台账 | `D:\pt_ner_work\results\gen.csv` | 30 行 |
| 质量度量 | `D:\pt_ner_work\results\e2_metrics.{json,md}` | 见下 |

## 三、产量与成本

| 项 | 值 |
|---|---:|
| 调用次数 | 30 |
| 候选 / 接受 / 拒绝 | 300 / 300 / **0** |
| 接受率 | **100%** |
| 输入 token | 56,340 |
| 输出 token | 117,277（其中 reasoning 占大头） |
| 总 token | 173,617 |
| 费用 | 未配置单价，台账记 0（token 已记录，可事后换算） |

## 四、质量指标

| 指标 | 值 | 判定 |
|---|---:|---|
| 句式重复率（pos） | 0.0 | ✅ 无模板化 |
| 4-gram 骨架共享率（pos） | 0.084 | ✅ 低 |
| 与种子 8-gram 重叠 | 0 条 | ✅ 无抄袭 |
| 实体复用率 | 1/478 = 0.2% | ✅ 没有搬种子实体 |
| 两类句长（pos / neg） | 23.46 / 24.05 词 | Mann-Whitney p = 0.0609，**无显著差异** ✅ |

## 五、⚠ 与 E1 的分布偏差（必须决定）

| 指标 | E2 合成 | E1 真实 | 偏差 |
|---|---:|---:|---|
| 句子数 | 300 | 300 | ✅ |
| 无实体句 | 135 | 135 | ✅ |
| 含实体句 | 165 | 165 | ✅ |
| **实体总数** | **478** | **376** | **+27%** ❌ |
| 每含实体句实体数 | 2.90 | 2.28 | **+27%** ❌ |
| 实体数直方图 | `1:2, 2:47, 3:85, 4:28, 5:3` | `1:70, 2:44, 3:26, 4:11, 5:7, 6+:7` | 几乎不产 1 实体句 |
| PER / ORG / LOC | **98 / 205 / 175** | **134 / 113 / 129** | **ORG 43% vs 30%，PER 21% vs 36%** ❌ |
| 每句词数 | 23.7 | 22.8 | ✅ |

**后果：** 实体密度高 27% 意味着 E2 每句拿到的监督信号比 E1 多四分之一，会系统性抬高 E2 的表现——这正是 `specs/05` 要避免的混杂。类别偏斜则会让 ORG 的表现被过度训练。

**原因：** 模板写的是 "1–5 entities"，模型倾向取 3；同时模板明确 "Do not hardcode class proportions"，模型自然偏向新闻语体里最常见的机构类实体。

**建议修法（需重新生成 165 条含实体，约 15 次调用）：**

1. 实体数改为「多数 1–3，最多 20% 的样本可到 4–5」，把均值压回 ~2.3；
2. 加一句「PER / ORG / LOC 大致均衡，不要偏向组织类」；
3. 无实体批不受影响，可保留现有 135 条。

## 六、复现

```powershell
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
& $py src\generate\run_e2.py --max-calls 45 --budget-usd 3.0
& $py src\generate\metrics.py data\synthetic\e2\e2_pos.jsonl --neg data\synthetic\e2\e2_neg.jsonl
```
