# 英文对照实验（English parallel run）

**状态：** **待定** —— 仅设计稿，未开工。本节所有标「待定」的条目必须在动手前逐条确认。
**定位：** 独立对照实验，用于检验葡语 E0–E3 的结论能否跨语言迁移。**不改变葡语链路、不改变葡语结论、不与葡语数据合并。**
**硬性约束：** **不修改任何现有文件。** 英文侧全部新建，与葡语侧并列存放（见 §3）。
**建立日期：** 2026-11-13
**依赖：** `specs/01`–`specs/10`（英文侧沿用同样的口径，只替换语言相关部分）

---

## 一、目的

葡语侧现在的核心结论是三条：

1. 300 条真实数据 +9.9 点（0.7497 → 0.8487）；
2. 同规模合成数据在低预算下可用（E2 最优 0.8030 @S=200），高预算下过拟合；
3. R-GDA 适配让合成数据对预算不敏感（E3 极差 1.9 点），但峰值未突破（0.8006）。

英文对照要回答的是：**这三条是葡语/巴西新闻域的特性，还是方法本身的特性？**

**不做的事：** 不重跑葡语、不把英文数据混进葡语训练集、不在英文上调葡语已冻结的协议常量后再回头改葡语结论。

---

## 二、必须先定的决策

| 编号 | 决策点 | 候选 | 建议 | 状态 |
|---|---|---|---|---|
| **D-EN1** | 数据集 | UniversalNER 英文子集 / CoNLL-2003 / OntoNotes 5.0 / WikiANN-en | **UniversalNER 英文子集**（如 `UNER_English-EWT`、`UNER_English-CoNLL03`）。理由：与 `pt_bosque` 同母体、同 IOB2 结构、同转换脚本，语言成为唯一变量 | 待定 |
| **D-EN2** | 标签映射 | 英文源标签多于 PER/ORG/LOC | 见 §4 草案。**MISC / NORP / DATE 等非三类必须显式处理**，不得默认丢弃而不声明 | 待定 |
| **D-EN3** | 句长与密度阈值 | 照抄葡语 20–32 词？ | **必须按英文真实分布重定**。葡语的 20–32 是从 E1 实测分布推出来的（含实体 μ=29.07，无实体 μ=15.19），英文分布不同，照抄等于人为扭曲 | 待定 |
| **D-EN4** | 评测阈值 | 统一 0.80 / 每语言各自最优 | 主表**统一 0.80** 保可比；另附「每语言最优阈值」作敏感性分析。跨语言用不同阈值比较会失去意义 | 待定 |
| **D-EN5** | 跑几个 arm | E0–E2 / E0–E3 | 建议 **E0–E3 全跑**，否则"方法可迁移性"回答不了。若预算受限，先跑 E0 + E1 + E2，E3 后补 | 待定 |
| **D-EN6** | arm 命名 | `E0en…E3en` / `L0…L3` | **`E0en…E3en`**：与现有 `experiments.csv` 的 `arm` 列同构，看板无需改动 | 待定 |
| **D-EN7** | 负样本口径 | 英文"无具名实体"如何界定 | 沿用葡语规则（泛称、数字、日期允许），但英文特有的 `Monday`、`Q3`、`the president` 等需逐条明确 | 待定 |
| **D-EN8** | 生成模型 | 与葡语一致 | 必须同一个 DeepSeek Flash 固定版本，否则语言之外又多一个变量 | 待定 |

---

## 三、隔离原则（硬约束）

### 3.1 现有文件：一律只读

下列文件**不允许做任何改动**（包括修 bug、调格式、加参数）：

```
src/generate/*.py            （除新增子目录）
src/train/train_arm.py
src/train/sweep_steps.py
src/eval/evaluate.py
prompts/templates/**         （整棵子树）
data/pt_bosque/**、data/curated/**、data/synthetic/e2|e3/**
docs/specs/01–10
```

英文侧若需要不同行为，**复制一份到 `en/` 命名空间里改**，不要在原文件加 `if lang == 'en'` 分支。

### 3.2 新增目录布局

```
src/
├── data/
│   └── prepare_uner_en.py              ← 新：英文 IOB2 → JSONL（照 prepare_uner.py 结构重写）
├── generate/
│   └── en/                             ← 新：英文生成子包
│       ├── en_prepare_prompt.py        ← 新：ROOT 指向 templates_en/
│       ├── en_pipeline.py              ← 新：TEMPLATES 指向 templates_en/
│       ├── en_validator.py             ← 新：英文长度/密度阈值
│       ├── en_tokenize.py              ← 新：英文分词（缩写、所有格、连字符）
│       ├── en_run_e2.py                ← 新
│       └── en_run_e3.py                ← 新
└── train/
    └── sweep_steps_en.py               ← 新：choices 支持 E0en–E3en

prompts/
└── templates_en/                       ← 新：整棵子树
    ├── prompts/                        ← 8 个模板的英文版
    └── data/
        ├── label_schema_en.json
        └── seeds_20_en.json

data/
└── en/                                 ← 新：英文数据全部落在这里
    ├── uner_en/{train,dev,test}.jsonl、stats.json、split_manifest.{json,md}
    ├── curated/{real_300_en.jsonl、seeds_20_en.jsonl、*_manifest.json}
    └── synthetic/{e2_en_300.jsonl、e3_en_300.jsonl}
```

产物与葡语分开：

| 类别 | 英文侧路径 |
|---|---|
| checkpoint | `D:\pt_ner_work\checkpoints\E{0,1,2,3}en_sweep\seed{n}_S{m}\final` |
| 评测原始输出 | `D:\pt_ner_work\results\e{1,2,3}en_sweep_s{n}_S{m}*` |
| 生成台账 | 复用 `D:\pt_ner_work\results\gen.csv`，靠 `run_id` 前缀区分（`en_`） |
| 实验台账 | 复用 `results/experiments.csv`，`arm` 列写 `E1en` 等（追加行，不改结构） |

### 3.3 允许的"接触"（只读，不算修改）

- `import` 现有的 `convert_validate`、`client`、`ledger`、`metrics`、`e3_schema`、`env`；
- 以 CLI 调用现有 `train_arm.py`、`evaluate.py`（二者的 `--arm` 是自由字符串，无需改）；
- 读取现有 `data/pt_bosque`、`seeds_20.jsonl` 做对照分析。

---

## 四、标签映射草案（待确认）

三类目标：`PER` / `ORG` / `LOC`。

### 4.1 CoNLL-2003 英文

| 源标签 | 映射 | 说明 |
|---|---|---|
| PER | PER | 直接 |
| ORG | ORG | 直接 |
| LOC | LOC | 直接 |
| **MISC** | **丢弃** | 含国籍、赛事、语言、作品名等，与葡语三类对齐时无干净归宿 |

丢弃 MISC 会让英文侧实体密度下降，**必须在 `stats` 里报告丢弃比例**。

### 4.2 OntoNotes v4 英文（若选）

| 源标签 | 映射 |
|---|---|
| PERSON | PER |
| ORG | ORG |
| GPE | LOC（政治实体按地理位置处理，与葡语 `pt_bosque` 的 GPE 处理一致） |
| LOC | LOC |
| FAC | LOC（设施按地点处理） |
| NORP / DATE / TIME / MONEY / PERCENT / CARDINAL / ORDINAL / QUANTITY / EVENT / WORK OF ART / LAW / LANGUAGE / PRODUCT | 丢弃 |

### 4.3 索引与区间

- 沿用 `specs/03`：字符区间左闭右开 `[start, end)`，且 `text[start:end] == 实体文本`；
- 映射后必须重跑交叠检查与边界检查（英文缩写 `U.S.`、`Inc.` 易产生错误边界）。

---

## 五、新建文件清单

| # | 文件 | 职责 | 参照的葡语文件 |
|---|---|---|---|
| 1 | `src/data/prepare_uner_en.py` | 抓取英文 IOB2 → 项目 JSONL + 统计 + 泄漏审计 | `src/data/prepare_uner.py`、`make_split_manifest.py` |
| 2 | `src/data/make_real_300_en.py` | 抽 300 条真实句（165+135，45% 无实体，分层保证类覆盖） | `src/data/make_real_300.py` |
| 3 | `src/data/make_seeds_en.py` | 选 20 条种子（确定性，固定种子） | `src/data/make_seeds.py` |
| 4 | `src/generate/en/en_tokenize.py` | 英文分词，返回 token 与字符偏移 | `src/generate/pt_tokenize.py` |
| 5 | `src/generate/en/en_prepare_prompt.py` | 渲染英文模板 | `prompts/templates/prepare_prompt.py` |
| 6 | `src/generate/en/en_pipeline.py` | 生成流水线（渲染→调用→解析→两级校验） | `src/generate/pipeline.py` |
| 7 | `src/generate/en/en_validator.py` | 项目级校验，阈值按英文重定 | `src/generate/validator.py` |
| 8 | `src/generate/en/en_run_e2.py` | E2en 入口 | `src/generate/run_e2.py` |
| 9 | `src/generate/en/en_run_e3.py` | E3en 入口（GR + AC 去评分化） | `src/generate/run_e3.py` |
| 10 | `src/train/sweep_steps_en.py` | 9 组网格，choices 含 `E0en–E3en` | `src/train/sweep_steps.py` |
| 11 | `prompts/templates_en/prompts/*.txt` | 8 个模板英文版 | `prompts/templates/prompts/*.txt` |
| 12 | `prompts/templates_en/data/label_schema_en.json` | 英文标签规范 | `prompts/templates/data/label_schema.json` |
| 13 | `prompts/templates_en/data/seeds_20_en.json` | 英文 20 条种子 | `prompts/templates/data/seeds_20.json` |
| 14 | `docs/specs/11-english-parallel.md` | 本文件 | — |

**复制而非共享的理由：**

| 现有文件 | 为什么不能直接复用 |
|---|---|
| `prepare_prompt.py` | `ROOT` 与 `STAGES` 在模块内写死，模板目录无法外置 |
| `pipeline.py` | `TEMPLATES` 写死为 `prompts/templates` |
| `validator.py` | `MIN_TOKENS_ENTITY=20 / MAX_TOKENS_ENTITY=32 / NGRAM=8` 是葡语实测值，且自测用例是葡语句 |
| `sweep_steps.py` | `--arm` 的 `choices=["E1","E2","E3"]` 是枚举 |
| `pt_tokenize.py` | 正则针对葡语重音与连字符，英文缩写会切错 |

---

## 六、可零改动复用

| 模块 | 复用方式 | 依据 |
|---|---|---|
| `prompts/templates/convert_validate.py` | `import` | 纯按路径处理输入输出，无语言假设 |
| `src/generate/e3_schema.py` | `import` | 只校验 JSON 结构与三态自洽，不校验语言 |
| `src/generate/client.py` | `import` | DeepSeek 客户端，与语言无关 |
| `src/generate/ledger.py`、`metrics.py` | `import` | 台账与度量（n-gram 重叠基于字符/词，需实测确认） |
| `src/common/env.py` | `import` | 环境变量与路径 |
| `src/train/train_arm.py` | CLI | `--arm` 为自由字符串 |
| `src/eval/evaluate.py` | CLI | `--arm` 为自由字符串，`--split/--cache/--out/--threshold` 均参数化 |
| `tools/dashboard/server.py` | 直接运行 | 只读产物，路径白名单按目录读 |

---

## 七、验收闸门（先跑 E0en，再决定要不要全量投入）

| 闸门 | 判据 | 不通过时的处置 |
|---|---|---|
| **G-EN1** | 英文零样本 dev Micro-F1 | **若 ≥ 0.90：天花板效应**，合成数据的提升空间被压扁，对照实验可能得出"无差异"的假结论。此时应改设计（换更难子集 / 改为比较增量而非绝对值 / 增加低资源语言做第三方对照），不要硬跑 |
| **G-EN2** | 每类 mention ≥ 30（同葡语 G1） | 调整抽样或换子集 |
| **G-EN3** | 文档级交叠 train∩dev、train∩test、dev∩test 全为 0；种子 100% 来自 train 侧 | 重做划分 |
| **G-EN4** | 标签映射后实体数/句密度与葡语量级相当（葡语 E1：376 实体/300 句） | 修正映射或丢弃策略 |

**G-EN1 必须是第 2 个动作**（第 1 个是选数据集）。在零样本基线未知时写其余代码是浪费。

---

## 八、成本与排期（估算）

| 阶段 | 工作量 | 花钱 |
|---|---|---|
| 选数据集 + 标签映射 + 写 spec 实现细节 | 0.5 天 | 0 |
| 数据转换 / 抽样 / 种子 / 泄漏审计 | 0.5–1 天 | 0 |
| **E0en 零样本 dev 评测（闸门）** | 0.5 小时 | 几毛 |
| 英文模板改写 + 分词器 + 校验器 | 1 天 | 0 |
| E2en / E3en 生成（300 条 ×2） | 0.5 天 | $2–4 |
| 训练网格 4 arm × 9 组 | 2–2.5 小时 | 0 |
| 报告与对照分析 | 1 天 | 0 |
| **磁盘** | 36 × 1.16 GB ≈ **42 GB** | — |

---

## 九、风险

1. **天花板效应（最高风险）** — 英文零样本可能已接近实用水平，见 G-EN1。
2. **语体混淆变量** — CoNLL-2003 是新闻，EWT 混合博客/邮件；若英文数据语体与 `pt_bosque` 新闻不一致，"语言"和"领域"就成了两个同时变化的变量。**必须选择语体最接近的子集并写入披露。**
3. **缩写与边界** — 英文 `U.S.`、`Inc.`、`'s`、连字符复合词会同时影响分词与 span 校验；`en_tokenize.py` 必须有独立自测（对齐 `pt_tokenize` 的校验方式）。
4. **双份代码漂移** — 英文侧改动不得回流葡语侧；发现葡语侧 bug 应记录后单独处理，不在本次任务里顺手改（隔离原则）。
5. **许可证** — OntoNotes 需签署协议；CoNLL-2003 分发受限。**选用前先确认可获取性与可引用性**，否则报告里无法公开数据。
6. **预算阈值不可移植** — 葡语 0.80 是在 dev 上选的；英文最优阈值若明显不同，必须两个数都报，不得只报有利的那个。

---

## 十、执行顺序（建议）

```
1. 定 D-EN1（数据集）→ 落 §4 映射表
2. 写 prepare_uner_en.py → 产出 train/dev/test.jsonl + stats + split_manifest
3. 跑 E0en（闸门 G-EN1）
   └─ 不通过 → 回到第 1 步改设计
   └─ 通过 → 继续
4. 定 D-EN3（阈值）、D-EN7（负样本口径）→ 更新本文件为「已定」
5. 种子选取 + 模板改写 + 分词/校验器
6. E1en 归档 → E2en/E3en 生成 → 36 组训练
7. 英文侧独立报告，与葡语报告并列，不做合并结论
```

---

**待办（未做）：** 本文件尚未登记进 `docs/specs/README.md` 的索引表——该文件是现有文件，按隔离原则需单独确认后再加一行。
