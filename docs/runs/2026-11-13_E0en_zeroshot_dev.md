# 2026-11-13 · E0en 英文零样本基线（dev）

**状态：** 完成。**闸门 G-EN1 通过。**
**实验标识：** E0en（英文对照组的零样本基线）
**数据划分：** dev
**定位：** `docs/specs/11-english-parallel.md` 的第一个可执行步骤；结果决定英文对照实验是否继续。

> 本文只记录**实际执行的参数**，不回填、不调整。所有参数与葡语 E0 逐项一致，唯一差异是数据语言与语料（见 `specs/11` §6 P-EN1/P-EN3）。

---

## 一、运行配置

| 项 | 值 |
|---|---|
| 实验标识 | `E0en` |
| 数据 | `data/en/uner_en/dev.jsonl`，**2,001 句 / 318 文档 / 966 实体**（PER 343 / ORG 224 / LOC 399） |
| 数据来源 | `UNER_English-EWT`（CC-BY-SA-4.0），由 `src/data/prepare_uner_en.py` 转换 |
| 模型 | `gliner_multi-v2.1`，`models/gliner_multi-v2.1`，1,106.4 MB |
| 脚本 | `src/eval/evaluate_en.py`（复用 `src/eval/evaluate.py` 的全部逻辑，只替换数据路径） |
| 阈值 | **0.80 固定**（扫描网格 0.10–0.90，步长 0.05，共 17 档，仅作参考） |
| bootstrap | 1,000 次，按文档重采样 |
| 标签 | 预测 `person / organization / location` → 比对 `PER / ORG / LOC` |

### 环境

| 项 | 值 |
|---|---|
| Python | 3.14.7 |
| torch | 2.14.0+cu130 |
| transformers | 5.17.0 |
| gliner | 0.2.30 |
| CUDA | 13.0 |
| HF 镜像 | `https://hf-mirror.com` |
| 临时目录 | `D:\pt_ner_work\tmp` |

### 完整命令

```powershell
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"

# 前置（已执行，幂等）
& $py src\data\prepare_uner_en.py

# E0en 评测
& $py src\eval\evaluate_en.py --model models/gliner_multi-v2.1 --split dev `
    --cache D:/pt_ner_work/results/e0en_dev_candidates.jsonl `
    --out   D:/pt_ner_work/results/e0en_dev `
    --threshold 0.80 --arm E0en --record results/experiments.csv
```

---

## 二、产物

| 类型 | 路径 |
|---|---|
| 候选缓存 | `D:\pt_ner_work\results\e0en_dev_candidates.jsonl` |
| 完整指标 | `D:\pt_ner_work\results\e0en_dev.json` |
| 人读报告 | `D:\pt_ner_work\results\e0en_dev.md` |
| 台账 | `results/experiments.csv`（追加一行，`run_id = E0en_dev_20261113_022541`） |

---

## 三、结果（阈值 0.80）

| 指标 | 值 |
|---|---:|
| Precision | 0.5029 |
| Recall | 0.7122 |
| **Micro-F1** | **0.5895** |
| Macro-F1 | 0.5917 |
| TP / FP / FN | 688 / 680 / 278 |
| Type Error | 41 |
| Boundary Error | 65 |
| gold 实体 | 966 |
| 预测实体 | 1,368 |
| 95% CI（按文档 bootstrap） | [0.5419, 0.6307] |
| 扫描最优（参考） | 0.90 → 0.6260 |

### 分类别

| 类别 | F1 |
|---|---:|
| PER | 0.5489 |
| ORG | 0.5586 |
| LOC | 0.6676 |

### 耗时

| 阶段 | 耗时 |
|---|---:|
| 模型加载 | 6.0 s |
| 候选收集（2,001 句，threshold=0.0） | 36.3 s |
| 评测（17 档扫描 + 错误分类 + 1,000 次 bootstrap） | 未单独计时，合计 < 10 s |

---

## 四、与葡语 E0 对照

| | Micro-F1 | P | R | PER | ORG | LOC |
|---|---:|---:|---:|---:|---:|---:|
| E0 葡语 @0.80 | 0.7497 | 0.7680 | 0.7323 | 0.8360 | 0.6561 | 0.7141 |
| **E0en 英文 @0.80** | **0.5895** | 0.5029 | 0.7122 | 0.5489 | 0.5586 | 0.6676 |
| 差 | **−16.0 点** | −26.5 | −2.0 | −28.7 | −9.8 | −4.7 |

---

## 五、观察

1. **闸门 G-EN1 通过，但方向与预期相反。** 设计时担心的是天花板效应（英文零样本太高、提升空间被压扁）；实测是英文**低** 16 点，提升空间比葡语更大。按 `specs/11` §9，该情形可以直接继续。
2. **错误结构不同。** 召回基本相当（0.71 vs 0.73），但精度低 26.5 点 —— 英文侧大量误报（FP 680 对 gold 966，预测数是 gold 的 1.42 倍；葡语 E0 只预测 1.05 倍）。这说明问题不在"漏"，在"乱报"。
3. **PER 塌得最惨**（0.549 vs 葡语 0.836）。EWT 的人名大量出现在非正式语体中（邮箱签名、博客自称），与新闻语体的人名分布不同。
4. **不能归因于语言。** 语体同时从新闻变成了网络文本（P-EN3），本组数据不足以区分"英文更难"与"网络文本更难"。这一点必须写进对照报告的限制。

## 六、后续动作

- 触发：`docs/specs/11-english-parallel.md` §12 第 3 步 → 通过，继续铺 E1en。
- 待办：英文 dev 与葡语 dev 的语体差异需要一个定量描述（后续可加：句子来源子类型分布、平均句长、泛称密度）。
