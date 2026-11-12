# 2026-11-13 · E1en 英文真实数据微调（步数 × 种子网格，dev）

**状态：** 完成。9 组全部成功。
**实验标识：** E1en（英文对照组的"300 条真实数据"臂）
**数据划分：** dev
**定位：** `docs/specs/11-english-parallel.md` §5.3 的网格；与葡语 E1 逐项同协议。

> 本文记录**实际执行的参数**，不改动已跑过的任何设置。与葡语 E1 的唯一实质差异是数据语言/语料，以及由此带来的长度分布差异（见 §六）。

---

## 一、运行配置

| 项 | 值 |
|---|---|
| 实验标识 | `E1en` |
| 训练数据 | `data/en/curated/real_300_en.jsonl`，**300 句 / 170 文档 / 287 实体** |
| 数据构成 | 165 含实体 + 135 无实体（45%）；PER 112 / ORG 62 / LOC 113 |
| 抽样脚本 | `src/data/make_real_300_en.py`（种子 42，同葡语口径） |
| 去重 | 训练池剔内部重复 531 条、跨集重复 345 条，去重后池 11,667 条 |
| 模型 | `gliner_multi-v2.1`，`models/gliner_multi-v2.1`，1,106.4 MB；每个 arm 独立初始化 |
| 脚本 | `src/train/sweep_steps_en.py`（训练调 `src/train/train_arm.py`，评测调 `src/eval/evaluate_en.py`） |
| 步数网格 | `100, 200, 500` |
| 随机种子 | `13, 23, 33` |
| batch / lr | 2 / 2e-5 |
| 评测阈值 | **0.80 固定**（17 档扫描仅作参考） |
| bootstrap | 1,000 次，按文档重采样 |

### 环境

Python 3.14.7 / torch 2.14.0+cu130 / transformers 5.17.0 / gliner 0.2.30 / CUDA 13.0；
HF 镜像 `https://hf-mirror.com`；临时目录 `D:\pt_ner_work\tmp`。

### 完整命令

```powershell
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"

# 前置（已执行，幂等）
& $py src\data\prepare_uner_en.py
& $py src\data\make_real_300_en.py

# 9 组网格（幂等：已完成的组合自动跳过）
& $py src\train\sweep_steps_en.py --arm E1en --data data/en/curated/real_300_en.jsonl
```

单组等价命令（供追溯）：

```powershell
& $py src\train\train_arm.py --arm E1en --data data/en/curated/real_300_en.jsonl `
    --seed 13 --max-steps 500 --batch 2 --lr 2e-5 `
    --out D:/pt_ner_work/checkpoints/E1en_sweep/seed13_S500/final

& $py src\eval\evaluate_en.py --model D:/pt_ner_work/checkpoints/E1en_sweep/seed13_S500/final `
    --split dev --cache D:/pt_ner_work/results/e1en_sweep_s13_S500_dev_candidates.jsonl `
    --out D:/pt_ner_work/results/e1en_sweep_s13_S500 --threshold 0.80 `
    --arm E1en --seed 13 --max-steps 500 --record results/experiments.csv
```

---

## 二、产物

| 类型 | 路径 |
|---|---|
| checkpoint | `D:\pt_ner_work\checkpoints\E1en_sweep\seed{13,23,33}_S{100,200,500}\final`（9 份） |
| 候选缓存 | `D:\pt_ner_work\results\e1en_sweep_s{seed}_S{steps}_dev_candidates.jsonl` |
| 指标 JSON / MD | `D:\pt_ner_work\results\e1en_sweep_s{seed}_S{steps}.{json,md}` |
| 台账 | `results/experiments.csv`（追加 9 行，`arm = E1en`） |

---

## 三、结果（dev，阈值 0.80）

### 3.1 按步数汇总（三种子均值）

| S | n | Micro-F1 | P | R | PER | ORG | LOC | 95% CI |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 100 | 3 | 0.6382 ± 0.0338 | 0.5245 | 0.8171 | 0.7446 | 0.4701 | 0.6910 | [0.5905, 0.6784] |
| 200 | 3 | 0.6914 ± 0.0134 | 0.5943 | 0.8264 | 0.8039 | 0.5194 | 0.7297 | [0.6480, 0.7281] |
| **500** | 3 | **0.7056 ± 0.0107** | 0.6170 | 0.8247 | 0.8024 | 0.5543 | 0.7360 | [0.6622, 0.7433] |

**最优：S=500，0.7056。**

### 3.2 逐种子明细

| S | seed 13 | seed 23 | seed 33 |
|---:|---:|---:|---:|
| 100 | 0.6321 | 0.6079 | 0.6746 |
| 200 | 0.7003 | 0.6759 | 0.6979 |
| 500 | **0.7118** | 0.6932 | 0.7117 |

### 3.3 错误结构（三种子合计均值）

| S | TP | FP | FN | Type Error | Boundary Error | 预测/gold |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 789.3 | 722.7 | 176.7 | 54.7 | 57.3 | 1.53 |
| 200 | 798.3 | 545.3 | 167.7 | 52.0 | 48.0 | 1.35 |
| 500 | 796.7 | 496.3 | 169.3 | 57.7 | 41.3 | 1.28 |

> gold 恒为 966。FP 随步数明显下降（722.7 → 496.3），FN 几乎不变（176.7 → 169.3）——**训练主要是在压制误报，不是在补漏报。**

### 3.4 耗时

| S | 单组训练耗时 | 说明 |
|---:|---|---|
| 100 | 57–61 s | 三次 |
| 200 | 71–186 s | seed 33 异常慢（186 s），其余 71–92 s |
| 500 | 157–182 s | 三次 |

评测耗时未单独记录。

---

## 四、与葡语 E1 对照

| | E0 | E1 最优 | 绝对提升 | E1 最优 S | 收敛形态 |
|---|---|---:|---|---:|---|
| 葡语 | 0.7497 | 0.8487 | **+9.9 点** | 500 | 200→500 仍涨 5.3 点，未饱和 |
| 英文 | 0.5895 | 0.7056 | **+11.6 点** | 500 | 200→500 只涨 1.4 点，接近饱和 |

| 分类别（最优配置） | PER | ORG | LOC |
|---|---:|---:|---:|
| E1 葡语 @500 | 0.9340 | 0.7383 | 0.8527 |
| E1en 英文 @500 | 0.8024 | 0.5543 | 0.7360 |

**精度—召回结构：**

| | P | R |
|---|---:|---:|
| E0 葡语 | 0.7680 | 0.7323 |
| E1 葡语 @500 | 0.8093 | 0.8922 |
| E0en 英文 | 0.5029 | 0.7122 |
| E1en 英文 @500 | 0.6170 | 0.8247 |

---

## 五、观察

1. **真实数据的价值跨语言稳定。** 两组都是"300 条真实句带来约 10–12 点提升"（+9.9 / +11.6）。这是本对照实验目前最坚实的结论。
2. **绝对水平差 14 点，且差距主要来自精度。** E1en 的召回（0.825）与葡语（0.892）只差 7 点，精度差 19 点（0.617 vs 0.809）。英文侧"敢报"的问题从零样本一直带到微调后。
3. **收敛形态不同，可能已经到顶。** 葡语 200→500 还涨 5.3 点；英文只涨 1.4 点。英文若继续加数据，收益未必线性——但本组只到 500 步，不能断言饱和。
4. **ORG 在两种语言里都是最弱**（英文 0.554 / 葡语 0.738），且英文的 ORG 训练信号格外少：E1en 只有 62 个 ORG mention（葡语 113 个）。**E1en 的 ORG 偏低有数据量因素，不能全归因于语言或语体。**
5. **训练在压制误报而不是补漏。** FP 降 31%、FN 几乎不动。这与葡语 E1 的模式不同（葡语是 P 与 R 同时上升）。

---

## 六、已知偏差（引用 `specs/11` §6）

| 编号 | 在本组的具体表现 |
|---|---|
| **P-EN1** | 语言是唯一设计变量 |
| **P-EN3** | 语料从新闻（`pt_bosque`）变为网络文本（EWT），语言与语体同时变化 |
| **P-EN4** | E1en 含实体句均长 **18.0**（中位 17，20–32 词窗口覆盖 30.3%）；葡语 E1 为 **29.1**（中位 26，覆盖 35.2%）。**同一窗口在英文中位于均值上方**，E2en 生成时会产生系统性偏长的样本，已在 spec 中登记并安排消融 |
| P-EN2 | 本组不涉及标签映射（EWT 原标签即 PER/ORG/LOC） |

---

## 七、后续动作

- E1en 完成 → 可进入 E2en（普通合成数据）。
- 前置未建文件：`make_seeds_en.py`、8 个英文模板、`en_prepare_prompt.py`、`en_pipeline.py`、`en_run_e2.py`。
- P-EN4 消融（10–24 词版 E2en）排在 E2en 之后。
