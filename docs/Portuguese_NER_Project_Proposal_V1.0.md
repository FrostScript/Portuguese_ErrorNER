# 葡萄牙语命名实体识别优化项目书 V1.0

## Project Proposal

**项目英文名称：** Improving GLiNER for Portuguese Named Entity Recognition through Error-Aware Synthetic Data Generation

**项目中文名称：** 基于错误感知合成数据生成的葡萄牙语命名实体识别优化研究

**研究领域：** Natural Language Processing (NLP)、Named Entity Recognition (NER)、Low-Resource Learning、Synthetic Data Augmentation

**基础模型：** GLiNER — `urchade/gliner_multi-v2.1`

**主数据集：** Universal NER — `pt_bosque`

**核心方法：** Constrained LLM Generation + Error-Aware Sampling + Quality Feedback

**项目性质：** 论文方法复现、跨语言场景适配、对比实验与效果分析

---

# 1. Project Overview — 项目概述

本项目研究如何通过大型语言模型（LLM）生成的合成训练数据，改善 GLiNER 在葡萄牙语命名实体识别任务中的表现。

GLiNER 是一种基于双向 Transformer 的实体识别模型，可以根据指定的实体类别识别文本中的实体。虽然其多语言版本支持葡萄牙语，但在具体非英语数据集上的识别效果仍可能受到语言差异、实体类型和数据分布的影响。

项目以 Universal NER 的葡萄牙语 `pt_bosque` 子集作为主要评估数据集，研究三种数据优化思路：

1. 通过 LLM 受约束生成葡语 NER 训练数据。
2. 通过质量反馈减少合成数据中的实体标注错误。
3. 根据 GLiNER 在葡语数据上的实际错误，定向生成困难实体类型的数据。

最终使用六组实验 E0–E5，对比原始模型、真实数据微调模型与四种合成数据微调模型的识别性能。

本项目不修改 GLiNER 的网络架构，而是通过训练数据构造与优化策略改善模型表现。

# 2. Research Motivation — 研究背景与动机

## 2.1 GLiNER 的多语言适配问题

GLiNER 的优势是无需针对每种实体类别单独训练分类器，并且能够通过标签描述实现开放类别实体识别。

然而，多语言支持不代表不同语言、不同数据集上的性能完全一致。

在葡萄牙语 NER 场景中，我们希望具体分析：

- 人名（PER）是否存在漏检。
- 组织名称（ORG）是否容易出现实体边界识别错误。
- 地点名称（LOC）是否容易发生类别混淆。
- 葡语词汇与上下文结构是否影响 GLiNER 的识别结果。

这些都是待实验验证的问题，不能预先宣称 GLiNER 已经存在某一种具体错误。

## 2.2 人工标注数据的成本

真实 NER 数据需要人工识别实体、确定实体边界并分配正确类别。

当某个目标语言或数据领域缺少足够训练数据时，重新开展人工标注会增加时间成本。

LLM 能够生成自然语言文本，也可以输出实体标签，因此有机会为 NER 模型提供额外训练数据。

但 LLM 生成的句子可能包含不自然表达、错误实体边界、错误实体类别和重复文本。

## 2.3 本项目的核心动机

传统合成数据方法往往按照预先设定的类别分布生成训练样本，并不一定考虑目标模型真正容易犯错的地方。

因此，本项目尝试将模型错误反馈引入数据生成过程：

**先找出 GLiNER 在葡语 NER 中的识别弱点，再重点生成能够训练这些弱点的葡语样本，并利用质量控制提高数据可靠性。**

# 3. Research Questions — 研究问题

**RQ1 — Portuguese NER Performance**

GLiNER 在 `pt_bosque` 中的基础实体识别能力如何？不同实体类别的 Precision、Recall 和 F1 有何差异？

**RQ2 — Synthetic Data Effectiveness**

仅使用受约束 LLM 生成的葡语 NER 数据进行微调，是否能够改善 GLiNER 的表现？

**RQ3 — Error-Aware Generation**

根据模型识别错误决定合成数据类别分布，是否比普通合成数据生成更有效？

**RQ4 — Quality Feedback**

生成指导优化和实体标注质量反馈能否进一步提升模型性能？它们与 Error-Aware 策略是否具有互补效果？

# 4. Related Work — 核心研究论文

## Paper A — GLiNER

**GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer**

Zaratiana et al., NAACL 2024

作用：提供整个项目的实体识别模型和微调方法。

重点复现内容：

- GLiNER 预训练模型加载。
- Zero-shot NER 推理。
- 指定实体类别的预测。
- 葡语 NER 数据格式转换。
- GLiNER 下游任务微调。
- Entity-level F1 评价。

本项目使用预训练权重进行下游适配，不复现原论文的完整预训练流程。

## Paper B — GenLLM

**Overcoming Data Scarcity in Named Entity Recognition: Synthetic Data Generation with Large Language Models**

Dao et al., BioNLP 2025

作用：提供 LLM 合成 NER 数据的基础方法。

参考其实体引导数据生成思想，结合类别定义、实体示例及葡语风格参考，生成带有实体标签的新句子。

本项目将其适配为葡语受约束生成流程，而非完整重现原论文的生物医学实验。

## Paper C — R-GDA

**R-GDA: Reflective Guidance Data Augmentation with Multi-Agent Feedback for Domain-Specific Named Entity Recognition**

Kang et al., Findings of EACL 2026

作用：提供合成数据质量优化的参考机制。

重点借鉴两个组件：

**Guidance Refiner：** 检查并优化生成指令，避免指令过于宽泛。

**Augmentation Calibrator：** 检查合成文本的实体类别、边界和语义一致性。

第一版采用轻量化反馈与校验机制，不要求完整复现原论文的多 Agent 系统。

## Project Adaptation — Error-Aware Generation

本项目沿用之前方案设计的类别级错误感知机制，根据模型在开发集上的实体类别 Recall，动态调整不同类别的生成预算。

它属于项目的主要实验方法，但是否构成新的算法贡献，需要结合文献对比与真实结果判断。

# 5. Dataset — 数据集与划分

## 5.1 主数据集

**Universal NER — pt_bosque**

来源：Mayhew et al., NAACL 2024

Hugging Face：

https://huggingface.co/datasets/universalner/universal_ner

官方数据规模：

| Split | 样本数量 | 作用 |
|---|---:|---|
| Train | 7,018 | 真实样本及葡语风格参考来源 |
| Validation | 1,172 | 错误反馈、模型选择 |
| Test | 1,167 | 最终统一评估 |

## 5.2 实体类别

| 标签 | 全称 | 中文 |
|---|---|---|
| PER | Person | 人物 |
| ORG | Organization | 组织机构 |
| LOC | Location | 地点 |

原始数据采用 BIO 序列标注方式。

例如：

- `B-PER`：人物实体开始。
- `I-PER`：人物实体后续部分。
- `O`：非目标实体 Token。

所有实验使用相同的三类实体标签，并固定模型推理时使用的类别名称及映射关系。

## 5.3 数据使用规则

为了避免数据泄漏，采用以下数据分配方式。

**A. Real-100**

从官方 Train 中固定抽取 100 条真实标注句子。

仅作为 E1 的微调训练数据。

需要确保包含 PER、ORG、LOC，并记录每类实体出现次数。

**B. Style Reference Pool**

官方 Train 中剩余的 6,918 条文本，可作为葡语语言风格参考候选池。

这一部分只允许向生成器提供原始文本，不提供对应的 Gold NER 标签。

E2–E5 使用同一个风格参考池，且不能直接复制参考文本作为合成训练样本。

实体示例采用独立、固定的公开实体种子清单，不从 Test 或 Validation 标签中提取。

如果使用人工整理或带标注来源的实体种子，应在报告中披露其来源及监督成本。

**C. Dev Feedback**

将官方 Validation 划出约 300 条用于 E0 错误分析，为 Error-Aware 策略提供反馈。

该集合允许计算类别 Recall，但不参与 GLiNER 微调，不直接作为生成样本。

**D. Dev Selection**

剩余约 872 条 Validation 样本用于模型 Checkpoint 选择、超参数调整和开发阶段评价。

正式实验开始前固定这两个开发集的划分。

**E. Test**

使用官方完整 Test 集作为最终统一评价集。

Test 的文本和标签不得用于生成 Prompt、合成样本筛选、错误感知策略设计或超参数调整。

# 6. Experimental Design — E0–E5 对照实验

本项目共设计六组实验。

| 实验 | 训练样本 | 核心方法 |
|---|---|---|
| E0 | 0 | 原始 GLiNER Zero-shot |
| E1 | 100 条真实数据 | Few-shot Fine-tuning |
| E2 | 300 条合成数据 | Constrained LLM Generation |
| E3 | 300 条合成数据 | Constrained Generation + Quality Feedback |
| E4 | 300 条合成数据 | Constrained Generation + Error-Aware |
| E5 | 300 条合成数据 | Constrained Generation + Error-Aware + Quality Feedback |

300 条是第一版主实验预算，可在正式实验前根据 Pilot Study 调整。

**关键实验约束：E2、E3、E4、E5 的模型训练集只包含合成数据，不加入 E1 的 100 条真实训练句子。**

所有微调组均从相同的预训练 GLiNER 权重初始化，而不是在 E1 模型上继续训练。

## E0 — Zero-shot Baseline

直接使用预训练 `gliner_multi-v2.1`，不进行任何微调。

目的：

- 测量原始模型在葡语数据集上的性能。
- 获取初始实体错误分布。
- 为 Error-Aware Generation 提供开发阶段反馈。

## E1 — Real-data Baseline

使用固定 100 条真实标注数据微调 GLiNER。

目的：

- 提供少量真实标注条件下的参考表现。
- 衡量 100 条真实标注数据产生的适配效果。
- 与完全使用合成样本微调的 E2–E5 进行对比。

E1 不被预设为最高性能模型。

## E2 — Constrained Synthetic Generation

采用统一的受约束生成 Prompt，让 LLM 生成葡语文本及实体标注。

生成时提供：

- 三种实体类别及定义。
- 独立实体种子示例。
- 葡语新闻文本的风格参考。
- 句子长度及类别覆盖要求。
- JSON 输出和实体标注规则。

E2 仅执行必要的格式、标签及实体区间检查，不加入语义质量反馈或错误感知策略。

目的：验证普通受约束生成是否有助于葡语 NER。

## E3 — Quality-Optimized Generation

在 E2 的基础上加入 R-GDA-inspired Quality Feedback。

通过生成指导检查、实体标注校验、语义一致性判断和有限次数修正，获得质量优化样本。

目的：验证质量优化能否比普通生成更有效。

## E4 — Error-Aware Generation

根据 E0 在 Dev Feedback 上的错误分布，调整实体类别的合成比例。

保持 E2 的基础生成流程，但优先生成较难识别的类别和相关上下文。

不启用 E3 的额外语义质量反馈。

目的：验证定向生成是否比普通类别采样更有效。

## E5 — Error-Aware + Quality Feedback

组合 E4 的错误感知生成机制和 E3 的质量反馈机制。

目的：验证针对困难类别生成数据并优化其质量，是否能够进一步改善模型表现。

# 7. Technical Methodology — 技术方法

## Step 1 — GLiNER 环境部署

建议技术环境：

| 模块 | 技术 |
|---|---|
| Python | 3.11 |
| 深度学习 | PyTorch |
| NER 模型 | GLiNER multi-v2.1 |
| 数据管理 | Hugging Face Datasets |
| LLM | 固定版本的支持葡语生成的 LLM API |
| 评估 | 自定义实体区间匹配 + seqeval 辅助检查 |
| GPU | RTX 5070 |
| 结果保存 | JSON、JSONL、CSV |

首先确保 PyTorch 能正常使用 GPU、GLiNER 能够加载模型，以及 LLM 接口能稳定输出符合 Schema 的 JSON。

## Step 2 — 数据预处理

下载 `pt_bosque`，将原始 BIO 标签转换为 GLiNER 可使用的训练格式。

GLiNER 训练数据的核心字段：

- `tokenized_text`：句子的 Token 列表。
- `ner`：实体开始 Token、结束 Token 和类别。

例如葡语句子：

*Maria visitou Lisboa.*

Token 列表：

`["Maria", "visitou", "Lisboa", "."]`

实体标注：

- `[0, 0, "person"]`
- `[2, 2, "location"]`

GLiNER 训练格式采用包含结束 Token 的区间。评估模块需要明确处理字符位置和 Token 位置的映射。

转换完成后检查实体数量、标签分布、文本长度、重复数据及区间合法性。

## Step 3 — 运行 E0

加载 GLiNER 多语言模型。

对 Dev Feedback 和 Dev Selection 进行预测，统计：

- Micro-F1
- Per-class Precision
- Per-class Recall
- Per-class F1
- False Positive
- False Negative
- Type Error
- Boundary Error

先确定哪些错误确实存在，再设计对应的定向生成策略。

## Step 4 — 训练 E1

使用 100 条真实训练样本微调 GLiNER，保存：

- 模型 Checkpoint。
- Training Loss。
- 验证集 F1。
- 训练参数。
- 随机种子。

后续 E2–E5 不继承 E1 的模型权重。

## Step 5 — Constrained Portuguese Generation

构建统一的葡语 NER 生成 Prompt。

核心要求：

1. 使用自然、连贯的葡萄牙语句子。
2. 模仿葡语新闻语体，不复制参考句子。
3. 仅使用 PER、ORG、LOC 三类标签。
4. 使用允许的实体种子，同时引入合理的词汇和上下文变化。
5. 标注所有符合目标 Schema 的实体。
6. 保持多词实体的完整边界。
7. 返回结构化 JSON。
8. 不生成明显重复或仅替换实体名称的模板化句子。

各组使用相同的 LLM、基础 Prompt 框架及参考池。E4/E5 只在目标类别分布和错误提示上增加差异。

生成结果必须保存模型版本、Prompt 版本、实体种子、生成时间和参数。

## Step 6 — 结构化数据校验

所有 E2–E5 样本均执行基础结构检查，包括：

- JSON 是否可解析。
- 标签是否属于允许的 Schema。
- 实体文字是否能在句子中定位。
- 实体区间是否准确。
- 是否存在不允许的实体重叠。
- 是否存在完全重复样本。
- 是否直接复制真实参考文本。

基础格式校验不是 E3/E5 专属功能，而是保证四个实验组的数据均能够被 GLiNER 正确读取。

## Step 7 — Quality Feedback

E3 和 E5 额外实现：

**Guidance Refiner**

检查生成指令是否足够明确，例如是否规定了葡语语体、实体类别、边界完整性及标签覆盖。

**Augmentation Calibrator**

检查生成结果是否存在：

- 实体类别语义错误。
- 实体遗漏。
- 实体边界错误。
- 过度模板化。
- 不自然的葡语表达。
- 与真实数据风格不符。
- 语义或上下文不一致。

不合格样本允许修正一次，仍不合格则丢弃。

为避免仅由同一个 LLM 自我评价而造成偏差，需要人工抽检一部分样本。

## Step 8 — Error-Aware Sampling

首先根据 E0 在 Dev Feedback 上的真实预测，计算每个实体类别的 Recall。

对于类别 c，使用平滑后的 Recall 构造生成权重：

w_c = (1 − R̃_c + ε)^α

p_c = w_c / Σ w_j

其中：

- R̃_c：平滑后的类别 Recall。
- ε：避免极端权重的小常数。
- α：控制困难类别的采样倾向。
- p_c：目标类别的生成采样概率。

这些数值根据实际开发集结果计算，不能手工指定有利于某类实体的结果。

第一版仅实现类别级 Error-Aware。

针对 Type Error、Boundary Error 生成特定困难结构，作为可选扩展，不影响主实验完成。

## Step 9 — 生成最终合成数据

E2–E5 每组目标保留 300 条有效合成训练句子。

为了公平：

- 统一最终句子数量。
- 统一 LLM 版本。
- 统一基础 Prompt 模板。
- 统一参考文本来源。
- 统一基础格式校验。
- 尽量控制句子长度分布。
- 记录各组的实体数量和类别分布。

质量筛选可能导致 E3/E5 需要生成更多候选样本，因此额外记录候选数量、接受率、Token 使用量及生成成本。

## Step 10 — 训练 E2–E5

E1–E5 均从同一个固定 GLiNER 预训练 Checkpoint 开始独立微调。

建议使用：

- Learning Rate：2e-5，作为初始候选。
- Batch Size：2–4，根据显存调整。
- Mixed Precision：BF16（环境支持时）。
- 固定优化器与训练策略。
- 使用 Validation 选择 Checkpoint。
- 正式实验采用三个随机种子。

具体训练步数及 Early Stopping 规则需要通过小规模实验确定，避免因为 E1 和 E2–E5 数据数量不同而导致不公平的训练预算。

建议同时固定总优化步数，并记录实际训练样本访问次数。

# 8. Evaluation — 评估指标

## 8.1 主要性能指标

**Entity-level Precision**

预测出的实体中，有多少实体的边界与类别完全正确。

**Entity-level Recall**

真实实体中，有多少被完整且正确地识别。

**Micro Entity-level F1**

本项目主要评价指标，使用严格实体匹配：实体类别及边界都正确才计算 True Positive。

**Macro Entity-level F1**

对 PER、ORG、LOC 三类的 F1 取算术平均，用于观察类别之间的差异。

## 8.2 辅助指标

- Per-class F1。
- False Negative Rate。
- Type Confusion。
- Boundary Error。
- Synthetic Label Accuracy。
- Synthetic Data Acceptance Rate。
- Duplicate Rate。
- LLM Generation Cost。
- Training Time。

其中 Synthetic Label Accuracy 使用人工抽检评估，不能直接用 LLM 自评结果替代。

# 9. Experimental Comparisons — 核心比较与消融

| 对比 | 验证目标 |
|---|---|
| E0 vs E1 | 真实标注微调的收益 |
| E0 vs E2 | 普通合成数据能否改善模型 |
| E1 vs E2–E5 | 合成数据方法与真实数据参考的差距 |
| E2 vs E3 | Quality Feedback 的贡献 |
| E2 vs E4 | Error-Aware 的贡献 |
| E4 vs E5 | Error-Aware 下的质量反馈收益 |
| E3 vs E5 | 质量优化条件下错误感知的收益 |

E2–E5 构成一个 2×2 因子实验：

| 方法 | Error-Aware | Quality Feedback |
|---|---|---|
| E2 | No | No |
| E3 | No | Yes |
| E4 | Yes | No |
| E5 | Yes | Yes |

这四组可以用于分析两个机制的独立影响及交互作用。

## 可选消融实验

**A. Synthetic Data Budget**

比较 100、300、500 条合成数据，判断生成数量是否影响提升幅度。

**B. Error-Aware Components**

比较随机类别分配与 Recall-based 分配，必要时加入边界错误定向生成。

**C. Constrained Generation**

比较是否提供葡语风格参考，判断受约束生成对于合成文本质量的影响。

以上属于可选实验，优先级低于完成 E0–E5 主实验及三个随机种子复现。

# 10. Expected Results — 预期实验产出

以下是最终需要填入真实测量结果的主表。

| Method | Precision | Recall | Micro-F1 | Macro-F1 |
|---|---|---|---|---|
| E0 Zero-shot | 待测 | 待测 | 待测 | 待测 |
| E1 Real-100 | 待测 | 待测 | 待测 | 待测 |
| E2 Synthetic | 待测 | 待测 | 待测 | 待测 |
| E3 Quality Feedback | 待测 | 待测 | 待测 | 待测 |
| E4 Error-Aware | 待测 | 待测 | 待测 | 待测 |
| E5 Combined | 待测 | 待测 | 待测 | 待测 |

理想情况是 E2–E5 中至少部分方法优于 E0，而 E4/E5 在部分困难类别上相较 E2 获得进一步改善。

但项目不预设 E5 一定最优，也不保证合成数据一定接近 E1。

如果结果表现为 E3 优于 E5，或者合成数据无法提高 Zero-shot 性能，也需要通过类别错误、标注噪声和数据分布差异解释原因。

正式实验应报告三个随机种子的平均值和标准差，避免仅选择表现最好的一次运行。

# 11. Results Visualization — 实验可视化

至少完成以下图表：

| 图表 | 内容 |
|---|---|
| Figure 1 | 项目完整技术架构 |
| Figure 2 | E0–E5 Micro-F1 对比 |
| Figure 3 | PER、ORG、LOC 的 F1 对比 |
| Figure 4 | Error-Aware 采样比例 |
| Figure 5 | 合成数据质量与接受率 |
| Table 1 | 数据集统计 |
| Table 2 | E0–E5 主实验结果 |
| Table 3 | 消融实验结果 |
| Table 4 | 错误案例及分析 |
| Table 5 | LLM 生成成本 |

其中 Figure 2、Figure 3 和 Table 2 是最重要的结果展示内容。

# 12. Reproducibility — 可复现性

正式实验需要记录并固定：

- GLiNER 模型 ID 及权重版本。
- Python、PyTorch 和 GLiNER 依赖版本。
- UNER 数据集版本。
- 100 条真实样本 ID。
- 风格参考池及开发集划分。
- LLM 名称、版本与生成参数。
- Prompt 与实体种子清单。
- 生成数据快照。
- 微调参数及随机种子。
- 验证和测试脚本。
- 每组最佳 Checkpoint。

对于 API 生成模型，应保留全部合成数据，避免不同时间重新调用 LLM 导致实验输入不一致。

# 13. Project Deliverables — 项目最终交付

## 13.1 数据成果

- `real_100.jsonl`
- `synthetic_E2.jsonl`
- `synthetic_E3.jsonl`
- `synthetic_E4.jsonl`
- `synthetic_E5.jsonl`
- 数据集划分与 Schema 配置。
- 合成数据质量报告。

## 13.2 模型成果

- E0 预训练模型引用及固定配置。
- E1–E5 微调模型 Checkpoint。
- 训练日志。
- 推理脚本。

## 13.3 实验成果

- E0–E5 的完整性能统计。
- 各类别 Precision、Recall、F1。
- 质量优化与错误感知消融。
- LLM 生成成本。
- 错误分析。
- 测试预测 JSONL 文件。

## 13.4 课程成果

- 可运行的项目代码。
- README 与运行说明。
- Final Report（ACL 风格）。
- Presentation PPT。
- 课程要求的展示材料。
- 可选的简单 NER 演示界面。

UI 不属于主研究目标，只有完成实验和报告后才考虑开发。

# 14. Implementation Schedule — 实施顺序

## Phase 1 — 环境与数据准备

- 安装并验证 GLiNER。
- 下载 `pt_bosque`。
- 完成 BIO 转换。
- 确定 Real-100、风格参考池、Dev Feedback 与 Dev Selection。
- 跑通 E0 并完成初步错误统计。

**验收：** E0 能正确输出葡语实体预测和严格实体级 F1。

## Phase 2 — Baseline 与生成可行性验证

- 训练 E1。
- 建立葡语受约束生成 Prompt。
- 生成 30–50 条测试样本。
- 人工检查葡语自然性及实体标注正确性。
- 验证合成样本能用于 GLiNER 微调。

**验收：** 合成数据能成功通过校验、完成微调，且能够识别主要质量问题。

## Phase 3 — E2–E5 方法实现

- 实现普通受约束生成。
- 实现 Guidance Refiner。
- 实现 Augmentation Calibrator。
- 实现 Error-Aware 权重计算。
- 生成四组各 300 条有效样本。

**验收：** 四组数据独立、格式正确，质量控制和错误感知差异可追踪。

## Phase 4 — 正式训练与评估

- 完成 E1–E5 独立训练。
- 运行三个随机种子。
- 汇总开发集表现。
- 固定各组模型与评价配置。
- 对最终 Test 进行统一评价。

**验收：** 得到完整 E0–E5 主实验结果。

## Phase 5 — 结果分析及交付

- 绘制比较图表。
- 完成错误分析与消融。
- 整理数据生成成本。
- 编写 ACL 格式 Final Report。
- 导出模型、JSONL 及复现说明。
- 制作 Presentation PPT。

**验收：** 项目研究问题都有对应实验和真实结果支撑。

# 15. Risks and Limitations — 风险与局限

## Risk 1 — 葡语合成文本不自然

应对：统一真实葡语风格参考、限制模板化生成，并进行人工质量抽检。

## Risk 2 — LLM 生成的实体标注不完整

应对：结构检查、实体完整性检查，以及 E3/E5 的额外语义质量反馈。

## Risk 3 — 原始 GLiNER 已经具有较高葡语识别能力

应对：重点分析各类别 Recall、Boundary Error 和具体错误案例，而不是只追求整体 F1 提升。

## Risk 4 — 300 条合成数据不足以产生稳定提升

应对：优先完成主实验，再根据结果开展 100/300/500 数据规模消融。

## Risk 5 — Error-Aware 的反馈样本过少

应对：报告每类实体支持数量，对 Recall 进行平滑处理，并设置最低采样比例。

## Risk 6 — 额外人工监督造成不公平比较

应对：严格区分模型训练数据、生成时使用的参考资料和错误反馈的 Gold Labels。披露每一部分的来源和人工成本。

## Risk 7 — 方法难以区分葡语因素与数据领域因素

应对：明确将结论限制在 `pt_bosque` 场景，不将单数据集结果推广为所有非英语语言的共同规律。

# 16. Expected Contributions — 预期研究贡献

**Contribution 1 — Reproduction and Adaptation**

实现 GLiNER、GenLLM-inspired 合成数据生成及 R-GDA-inspired 质量反馈方法，并将其组合应用于葡萄牙语 NER。

**Contribution 2 — Error-Aware Generation**

实现基于类别识别错误的合成样本生成预算分配机制，验证其改善葡语实体识别的有效性。

**Contribution 3 — Empirical Evaluation**

通过 E0–E5 的统一实验，分析合成数据质量、错误感知生成和有限真实标注微调之间的性能差异。

这些是计划研究的贡献方向，而不是已经得到验证的实验结论。

# 17. Final Research Objective — 最终研究目标

本项目最终希望回答：

**Can constrained LLM-generated synthetic data, guided by GLiNER's recognition errors and enhanced through quality feedback, improve Portuguese Named Entity Recognition performance?**

中文：

**受约束 LLM 合成数据能否在错误感知与质量反馈的指导下，有效改善 GLiNER 在葡萄牙语命名实体识别任务中的性能？**

最终交付物是一套能够独立运行、训练、评估和比较六组方法的实验研究项目，而不只是一个实体识别 Demo。

---

# References — 核心参考资料

[1] Zaratiana et al. (2024). GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. NAACL 2024. https://aclanthology.org/2024.naacl-long.300/

[2] Dao et al. (2025). Overcoming Data Scarcity in Named Entity Recognition: Synthetic Data Generation with Large Language Models. BioNLP 2025. https://aclanthology.org/2025.bionlp-1.28/

[3] Kang et al. (2026). R-GDA: Reflective Guidance Data Augmentation with Multi-Agent Feedback for Domain-Specific Named Entity Recognition. Findings of EACL 2026. https://aclanthology.org/2026.findings-eacl.259/

[4] Mayhew et al. (2024). Universal NER: A Gold-Standard Multilingual Named Entity Recognition Benchmark. NAACL 2024. https://aclanthology.org/2024.naacl-long.243/

[5] GLiNER Official Repository. https://github.com/urchade/GLiNER

[6] GLiNER Multilingual Model. https://huggingface.co/urchade/gliner_multi-v2.1

[7] Universal NER Dataset. https://huggingface.co/datasets/universalner/universal_ner

