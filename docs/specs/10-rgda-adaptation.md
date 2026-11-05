# R-GDA 适配对照表（A9 / G5-b 闸门材料）

**状态：** 已填，四项可确认 → **具备宣称 "R-GDA 适配" 的条件**
**论文：** Kang, Namgoong, Pyeon, Jung. *R-GDA: Reflective Guidance Data Augmentation with Multi-Agent Feedback for Domain-Specific NER*. Findings of EACL 2026, pp. 4938–4953. DOI `10.18653/v1/2026.findings-eacl.259`
**核对方式：** 下载论文 PDF（834 KB / 16 页）用 `pypdf` 提取全文，逐节核对 §3.1–§3.3、Algorithm 1、Appendix A（Figure 7–9）

---

## 一、闸门四项（G5-b）

| # | 项目 | 论文原文依据 | 结论 |
|---|---|---|---|
| 1 | **GR 评分标准** | §3.1 + Figure 8：100 分制，5 项加权 —— Context Abstraction **30**、Keyword Detection **15**、Structure Relationship **15**、Entity Explanation **20**、Correct Relationship **20** | ✅ 已确认 |
| 2 | **AC 校验维度** | §3.2 + Figure 9：100 分制，4 项 —— Domain Alignment **30**、Semantic Consistency **30**、Entity Role Integrity **25**、Tag Format Validation **15** | ✅ 已确认 |
| 3 | **反馈轮数** | §3.3 + Algorithm 1：初值 score=0，`round ≤ 3`，每轮 EVALUATE → (score, feedback) → REFINE | ✅ 已确认（**最多 3 轮**） |
| 4 | **停止条件** | Algorithm 1：`if score < 90 and round < 3 then refine`；`until score ≥ 90 or round = 3` | ✅ 已确认（**阈值 90/100 或满 3 轮**） |

四项全部可从论文正文与附录直接读出，**不需要降级为 "R-GDA-inspired"**。

---

## 二、环节级对照

| 论文环节 | 原参数 / 做法 | 本项目参数 / 做法 | 实现位置 | 适配原因 |
|---|---|---|---|---|
| **Seed**（种子输入） | 从目标域训练集取样，供抽象参照 | 固定 20 条真实葡语新闻句（`data/curated/seeds_20.jsonl`），与 E2 完全共用 | `prompts/templates/data/seeds_20.json` | 与 E2 保持同一信息输入，才能把 E3−E2 解释为"方法差异" |
| **Guidance**（初始抽象） | 把句子拆成抽象组件（实体类型、领域信息） | 输出 JSON 五数组：`observations / generation_guidance / coverage_plan / annotation_risks / avoid_until_clarified` | `prompts/templates/prompts/e3_guidance.txt` → `run_e3.py` | 指令语言换成中文以对齐本次会话的标注口径；结构保留"观察 vs 规则"的区分 |
| **GR（Guidance Refiner）** | 100 分制 5 项评分 + 反馈，score<90 且 round<3 则 refine | **需改为 100 分制 5 项 + 最多 3 轮循环**（当前模板只做 1 轮无评分，见 §四偏差 D1） | `e3_refiner.txt` → `run_e3.py` | 领域从英文科学生物医学换为巴西葡语新闻，评分细则保留论文权重 |
| **GDA Composer**（生成） | 依 refined guidance 批量生成带标注样本 | `e3_generation` / `e3_negative` 双路径，每批 11 含实体 + 9 无实体 | `e3_generation.txt`、`e3_negative.txt` | 项目要求 45% 无实体句（`specs/05`），论文无此要求 |
| **AC（Augmentation Calibrator）** | 100 分制 4 项评分 + 反馈，score<90 且 round<3 则 refine | **需改为 100 分制 4 项 + 最多 3 轮**；每样本再由规则层判 keep/revise/reject | `e3_calibrator.txt`、`e3_calibrator_negative.txt` → `run_e3.py` | 保留论文评分维度，另加项目侧的格式硬校验（见 `convert_validate.py`） |
| **Iterative Refinement Loop** | Algorithm 1，score 初值 0，threshold 90，max 3 rounds | 同一算法，GR 作用于 guidance、AC 作用于 batch 输出 | `run_e3.py` | 直接照搬 |
| **下游评估** | SciERC / NCBI-disease / FIN，BIO/BILOU + MISC | `pt_bosque` PER/ORG/LOC + occurrence，GLiNER 微调，严格实体级 F1 | `src/train/train_arm.py`、`src/eval/evaluate.py` | 项目主数据集与基座模型不同 |

---

## 三、必须声明的适配偏差

| 编号 | 偏差 | 性质 | 处理 |
|---|---|---|---|
| **D1** | 当前 `e3_refiner` / `e3_calibrator` 模板**没有 0–100 打分，也没有 3 轮循环**（只做 1 轮结构化反馈） | **实质性偏差** | **必须在 `run_e3.py` 中实现打分 + 循环**，否则不得宣称"完整适配" |
| D2 | 语言：英文 → 巴西葡语 | 跨语言 | 保留并说明 |
| D3 | 领域：科学生物医学/金融 → 新闻 | 跨领域 | 保留并说明 |
| D4 | 标注体系：BIO/BILOU + MISC → PER/ORG/LOC + `occurrence` 定位 | 格式差异 | AC 的 "Tag Format Validation(15)" 改为校验 `occurrence` 与区间一致性 |
| D5 | 无实体句 45%：论文无此设定 | 项目新增约束 | 需额外一条 `e3_negative` 路径，成本另计 |
| D6 | 种子：论文用完整训练集抽象，本项目固定 20 条 | 信息量更少 | 与 E2 共用，保证可比 |
| D7 | 多 Agent：论文称 multi-agent，实现上可为单进程多角色 | 实现方式 | 保留角色与信息流，不要求多个自治进程 |

**若 D1 未修**：方法名须改为 `R-GDA-inspired`，报告中不得出现"完整适配"字样（依据 `docs/TODO.md` A9 闸门与项目书 §6.2）。

---

## 四、结论

- 闸门四项（GR 评分、AC 维度、轮数、停止条件）**全部可确认**，E3 具备按论文实现的基础；
- 但**现有模板只是 1 轮无评分反馈**，等于 D1 偏差。E3 开工前必须把 100 分制与 3 轮循环补上；
- 语言、领域、标签体系、种子规模四处偏差需在报告中逐条披露，结论边界限定在 `pt_bosque` + 葡语新闻。
