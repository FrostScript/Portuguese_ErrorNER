# 2026-11-04 · E0 零样本 · dev

**状态：** 成功（基线数字，待 dev 二次切分后重跑）

## 一、运行配置

| 项 | 值 |
|---|---|
| 实验 | E0 — 原始 GLiNER 零样本，不微调 |
| 数据划分 | `data/pt_bosque/dev.jsonl`（1,172 句 / 41 文档 / 1,401 实体） |
| 模型 | `urchade/gliner_multi-v2.1` |
| 模型本地路径 | `models/gliner_multi-v2.1`（1,155.9 MB，含基座分词器 `microsoft/mdeberta-v3-base`） |
| 脚本 | `src/eval/evaluate.py` |
| 推理标签 | `["person", "organization", "location"]` → PER / ORG / LOC |
| 候选收集 | `predict_entities(text, labels, threshold=0.0)`，一次前向取全部候选 |
| 阈值网格 | 0.10–0.90，步长 0.05，共 17 档（离线扫缓存，不重复推理） |
| 匹配规则 | 严格实体级：起止 + 类别全对才算 TP |
| 置信区间 | 按文档（doc_id）重采样 1,000 次，95% |
| 随机种子 | bootstrap seed = 13（推理本身无随机性） |

**完整命令**

```powershell
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\eval\evaluate.py `
    --model models/gliner_multi-v2.1 --split dev `
    --cache D:/pt_ner_work/results/e0_dev_candidates.jsonl `
    --out   D:/pt_ner_work/results/e0_dev
```

**环境**

| 项 | 值 |
|---|---|
| Python | 3.14.7 |
| torch | 2.14.0+cu130（CUDA 可用，RTX 5070 / 12,227 MiB） |
| transformers | 5.17.0 |
| gliner | 0.2.30 |
| HF 镜像 | `HF_ENDPOINT=https://hf-mirror.com` |
| 临时目录 | `D:/pt_ner_work/tmp`（`src/common/env.py` 强制指定） |

## 二、产物

| 文件 | 说明 |
|---|---|
| `D:\pt_ner_work\results\e0_dev.md` | 人读报告 |
| `D:\pt_ner_work\results\e0_dev.json` | 完整结果（17 档阈值全量指标） |
| `D:\pt_ner_work\results\e0_dev_candidates.jsonl` | 候选缓存，后续换阈值无需重新推理 |

## 三、结果

**最优阈值 0.80 → Micro-F1 0.7497**（P 0.7680 / R 0.7323），Macro-F1 0.7354，TP 1026 / FP 310 / FN 375。
95% CI（按文档 bootstrap）：**[0.5679, 0.7952]**。

阈值曲线关键点：

| 阈值 | Precision | Recall | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|---:|
| 0.50（GLiNER 默认） | 0.5871 | 0.8201 | 0.6843 | 0.6800 |
| 0.70 | 0.6973 | 0.7809 | 0.7367 | 0.7274 |
| 0.75 | 0.7290 | 0.7602 | 0.7442 | 0.7333 |
| **0.80** | **0.7680** | **0.7323** | **0.7497** | **0.7354** |
| 0.85 | 0.8198 | 0.6788 | 0.7427 | 0.7239 |
| 0.90 | 0.8827 | 0.5910 | 0.7080 | 0.6788 |

分类别（阈值 0.80）：

| 类别 | P | R | F1 | TP | FP | FN | 支持数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PER | 0.8029 | 0.8719 | **0.8360** | 497 | 122 | 73 | 570 |
| LOC | 0.8233 | 0.6304 | 0.7141 | 261 | 56 | 153 | 414 |
| ORG | 0.6700 | 0.6427 | **0.6561** | 268 | 132 | 149 | 417 |

错误结构（阈值 0.80）：

| 类型 | 数量 |
|---|---:|
| Type Error（边界对、类别错） | 45 |
| Boundary Error（类别对、边界错） | 45 |
| 其余 FP | 220 |
| 其余 FN | 285 |

**耗时：** 模型加载 5.4 s，1,172 句候选收集 21.2 s（≈18 ms/句），阈值扫描 + 错误分类 + 1,000 次 bootstrap 秒级。

## 四、观察

1. **阈值效应极大**：默认 0.5 只有 0.6843，最优 0.80 是 0.7497，**相差 6.5 个点**。任何微调组如果不重搜阈值，与 E0 的对比都不成立。
2. **ORG 是明确短板**：F1 0.6561，比 PER（0.8360）低 18 个点，是三类里唯一 P 和 R 都偏低的。这印证了项目书关于"ORG 边界/类别易混"的动机。
3. **LOC 是"高精度低召回"**：P 0.8233 但 R 0.6304，漏检 153 个，多为长地名与带修饰的机构性地点。
4. **CI 很宽**：41 篇文档重采样，单篇影响大（CI 宽度 0.23）。后续各组差异若小于这个宽度，必须用配对比较和按句 bootstrap，不能只看点估计。
5. Type Error 与 Boundary Error 各 45 个，量级相同 —— 说明错误不是单一边界问题，类别混淆同样重要。

## 五、后续动作

| TODO | 说明 |
|---|---|
| A1 / B3 | dev 二次切分（872 / 300）落地后在 Dev Selection 上重跑，才是 G2 闸门数字 |
| A4 | E1 微调后必须在 Dev Selection 上重搜阈值，不可沿用 0.80 |
| A13 | 报告中需同时给"默认 0.5"与"调优后"两套数字，用于展示阈值敏感性 |
