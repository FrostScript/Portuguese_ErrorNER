# 规范索引

**项目：** 葡萄牙语 NER（GLiNER + LLM 合成数据，E0–E3）
**用途：** 把散落在项目书、修订文档、日志里的**已定规范**拆成单文件，每个文件只说一件事，实现时逐条对照。
**更新规则：** 决策落地即改对应文件；文件内标 `状态：已定` 的条款才可作为实现依据，标 `待定` 的不得默认采用。

| 文件 | 主题 | 状态 |
|---|---|---|
| [01-dataset.md](01-dataset.md) | 数据集来源、规模、文本与实体口径 | 已定 |
| [02-split-and-leakage.md](02-split-and-leakage.md) | 数据划分与泄漏防护 | 部分待定（dev 二次切分未实施） |
| [03-labeling.md](03-labeling.md) | 标签集、映射、区间约定 | 已定 |
| [04-generation-constraints.md](04-generation-constraints.md) | 合成数据的句长、实体密度、模板化约束 | 已定（阈值待生成后实测校准） |
| [05-negative-sampling.md](05-negative-sampling.md) | 无实体句比例与生成方式 | 已定（2026-11-04） |
| [06-training-protocol.md](06-training-protocol.md) | 微调协议：步数、超参、API、产物 | 部分待定（S 待 pilot） |
| [07-evaluation-protocol.md](07-evaluation-protocol.md) | 评测指标、阈值搜索、错误分类、置信区间 | 已定 |
| [08-reproducibility-and-env.md](08-reproducibility-and-env.md) | 环境、路径、台账、编码与磁盘约定 | 已定 |
| [09-run-records.md](09-run-records.md) | 每次实验的运行记录格式（写入 `docs/runs/`） | 已定 |

## 与上游文档的关系

- `docs/葡萄牙语NER项目书_E0-E3.md` — 研究目标与实验组的定义来源
- `docs/Portuguese_NER_Experimental_Design_Revision_V2.0.md` — 修订理由与缺陷编号（D/N/Y 系列）
- `docs/TODO.md` — 未完成事项与待决策项
- `docs/logs/project_log.md` — 每次任务的实测数据与变更记录

**冲突处理：** 本目录与项目书正文冲突时，以本目录为准（它是后定的）；若冲突涉及研究目标本身，回到项目书逐条对齐后再改本目录。
