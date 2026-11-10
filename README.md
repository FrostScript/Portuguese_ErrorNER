# Portuguese_ErrorNER

葡萄牙语命名实体识别（NER）实验项目：以 GLiNER 为基座，比较**零样本**、**真实数据微调**与**LLM 合成数据微调**在 `pt_bosque` 上的效果。

> **维护约定：** 本文件随每次结构性改动自动同步（新增/删除文件、目录变化、约定变更、结论更新），不需逐次确认。文档分工见第九节。

**当前状态（2026-11-04）**

| 实验 | 状态 | 结果（dev，阈值 0.80） |
|---|---|---|
| E0 零样本 | ✅ 完成 | Micro-F1 **0.7497** |
| E1 真实数据微调（3 种子 × {100,200,500}） | ✅ 完成 | S=500 时 Micro-F1 **0.8487 ± 0.0045** |
| E2 普通合成数据微调（3 种子 × {100,200,500}） | ✅ 数据与训练完成 | S=200 时 **0.8030**；S=500 过拟合至 0.7667 |
| E3 R-GDA 合成数据微调 | ⚠️ 仅 smoke（20 条） | 待正式生成 300 条 |

---

## 一、五分钟上手

```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
$env:PYTHONIOENCODING="utf-8"           # 必须，否则打印葡语报错
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"

# 1) 看实验台账（所有 arm 的最终指标）
Import-Csv results\experiments.csv | Format-Table arm,seed,max_steps,micro_f1,org_f1

# 2) 开实验看板（浏览器打开 http://127.0.0.1:8765/）
& $py tools\dashboard\server.py --port 8765

# 3) 复现一次评测（读缓存，不加载模型）
& $py src\eval\evaluate.py --model models/gliner_multi-v2.1 --split dev `
    --cache D:/pt_ner_work/results/e0_dev_candidates.jsonl `
    --out D:/pt_ner_work/results/e0_dev --threshold 0.80 --arm E0
```

**第一次来先读哪里：** `docs/葡萄牙语NER项目书_E0-E3.md`（研究目标）→ `docs/specs/README.md`（规范索引）→ `docs/TODO.md`（当前进度）。要看代码结构则直接读 `docs/ARCHITECTURE.md`。

---

## 二、目录总览

```text
Portuguese_ErrorNER/
├── README.md            ← 本文件
├── configs/             路径与外部资源（唯一配置源）
├── data/                数据：官方切分 + 派生训练集
├── docs/                项目书、规范、日志、运行记录、TODO
├── models/              GLiNER 权重（不入库，1.1 GB）
├── prompts/             生成用 Prompt
├── results/             最终成果 + 实验台账
├── scripts/             终端用的运行包装器
├── src/                 全部代码
├── tools/               实验看板
└── logs/                运行日志（不入库）
```

外部目录（不在仓库内）：

```text
D:\pt_ner_work\
├── checkpoints/    训练产物（每份约 1.16 GB）
├── results/        评测原始输出（完整 JSON / md / 候选缓存）
├── data/synthetic/ 合成数据与候选池（待启用）
├── raw_responses/  LLM 原始响应（待启用）
├── hf_cache/       HuggingFace 缓存
└── tmp/            torch 编译临时目录（由 src/common/env.py 指定）
```

---

## 三、逐个文件夹详解

### 3.1 `configs/` — 配置

| 文件 | 作用 | 入库 |
|---|---|---|
| `paths.yaml` | 全部路径、HF 镜像、模型 ID、标签映射的**唯一配置源**。脚本里不应再硬编码路径 | ✅ |

### 3.2 `data/` — 数据

#### `data/pt_bosque/` — 官方切分（转换后）

| 文件 | 内容 | 规模 |
|---|---|---|
| `train.jsonl` | 训练集，234 篇 / 7,018 句 / 8,101 实体 | 3.6 MB |
| `dev.jsonl` | 开发集，40 篇 / 1,172 句 / 1,401 实体 | 0.6 MB |
| `test.jsonl` | 测试集，39 篇 / 1,167 句 / 1,215 实体 | 0.6 MB |
| `stats.json` | 统计快照（每类计数、无实体句比例、重复句数） | 1.8 KB |
| `split_manifest.json` / `.md` | 三集文档 ID 清单 + 泄漏审计（文档级交叠全为 0）+ 文件 sha256 | 22 KB / 7 KB |

单条记录格式：

```json
{"id": "batch-0000-0001", "doc_id": "dev:batch-0000", "split": "dev",
 "tokens": ["Pequenos", "são", "..."],
 "text": "Pequenos são ...",
 "entities": [{"start": 0, "end": 8, "text": "Pequenos", "label": "PER"}]}
```

> `doc_id` 带 split 前缀（`dev:batch-0000`），因为原始 iob2 里三个 split 的文档编号各自从 0 开始，不加前缀会让泄漏审计误报。

#### `data/raw_uner/` — 原始文件（不入库，可重下）

| 文件 | 内容 |
|---|---|
| `pt_bosque-ud-{train,dev,test}.iob2` | UNER 官方 IOB2 原始文件，共 4.6 MB |

由 `src/data/prepare_uner.py` 下载并转换。

#### `data/curated/` — 派生的训练与生成数据集

| 文件 | 内容 |
|---|---|
| `real_300.jsonl` | **E1 训练集**：300 句 = 135 无实体（45%）+ 165 含实体；376 实体（PER 134 / ORG 113 / LOC 129）；覆盖 167 篇文档 |
| `real_300_manifest.json` | 随机种子 42、ID 清单、源文件与输出 sha256、类别计数 |
| `real_300_stats.md` | 人读统计 |
| `seeds_20.jsonl` | **20 条生成种子**（E2/E3 的 Prompt 示例）：45 实体（PER 17 / ORG 18 / LOC 10）、19 篇文档、句长 20–30 |
| `seeds_20_manifest.json` / `_stats.md` | 选择约束、ID 清单、sha256 |

> `real_300` 与 `seeds_20` 都是**确定性产物**（固定种子），重跑脚本得到完全一致的文件。`seeds_20` 一旦用于生成即冻结。

### 3.3 `docs/` — 文档

#### 根目录

| 文件 | 作用 |
|---|---|
| `ARCHITECTURE.md` | **架构与代码组织说明**：六层划分、模块职责、四条主链路、四种数据契约格式、已知技术债（新增 / 重构前先读） |
| `葡萄牙语NER项目书_E0-E3.md` | **当前有效的研究方案**（研究问题、E0–E3 定义、实施阶段） |
| `Portuguese_NER_Project_Proposal_V1.0.md` | 初版项目书（E0–E5 方案，已被取代，留档） |
| `Portuguese_NER_Experimental_Design_Revision_V1.1.md` | 针对 V1.0 的修订（已标注"被 V2.0 取代"） |
| `Portuguese_NER_Experimental_Design_Revision_V2.0.md` | 针对 E0–E3 的修订：数据集核查、P0 缺陷、Y1–Y6 新增条款 |
| `TODO.md` | 未完成事项与待决策项（A 主流程 / B 阻塞 / C 决策 / D 实现 / E 已废弃） |

#### `docs/specs/` — 规范（实现依据）

每个文件只说一件事，标 `状态：已定` 的条款才能作为实现依据。

| 文件 | 主题 |
|---|---|
| `README.md` | 索引与冲突处理规则 |
| `01-dataset.md` | 数据来源、规模、文本与实体口径、已知重复句 |
| `02-split-and-leakage.md` | 划分政策（含 dev 不二次切分的决定）、泄漏防护、E1 抽样口径 |
| `03-labeling.md` | 标签集、两套写法（PER/ORG/LOC ↔ person/organization/location）、三种区间约定 |
| `04-generation-constraints.md` | 生成约束：句长 20–30、实体密度、模板化指标、已定生成参数（DeepSeek Flash / Prompt v1） |
| `05-negative-sampling.md` | 负样本规范：E2/E3 各 135 无实体 + 165 含实体，同批产出、只补不删 |
| `06-training-protocol.md` | 微调协议：S=500、batch 2、lr 2e-5、seed 13/23/33、API 与数据格式 |
| `07-evaluation-protocol.md` | 评测协议：严格匹配、阈值统一 0.80、指标定义、bootstrap CI、主表结构 |
| `08-reproducibility-and-env.md` | 环境、路径、台账、磁盘与写权限实测 |
| `09-run-records.md` | 每次实验的运行记录格式 |

#### `docs/runs/` — 单次实验记录

一次实验一个文件，含运行配置、完整命令、结果表、观察、后续动作。

| 文件 | 内容 |
|---|---|
| `README.md` | 记录规范 |
| `2026-11-04_E0_zeroshot_dev.md` | E0 零样本（含阈值扫描全表） |
| `2026-11-04_A4_pilot_S.md` | A4 pilot：S 从 100 扫到 500，冻结 S=500 |
| `2026-11-04_E1_S500_3seeds.md` | **E1 正式结果**（三种子） |
| `2026-11-04_E1_seed13_dev.md` | E1 首轮 S=300 的历史记录（已标注作废） |

#### `docs/logs/` — 项目日志

| 文件 | 内容 |
|---|---|
| `project_log.md` | 任务级日志，六段固定结构（做了什么 / 关键数据 / 变更文件 / 验证方式 / 遗留问题），已积累 15 条任务记录 |

### 3.4 `src/` — 代码

| 路径 | 作用 |
|---|---|
| `common/env.py` | **所有脚本第一行导入**：统一 `HF_ENDPOINT`、`HF_HOME`、`TMPDIR`（防 torch 把缓存落到仓库根）、`PYTHONIOENCODING` |
| `data/prepare_uner.py` | 下载 UNER iob2 → 转项目 JSONL + 统计 |
| `data/fetch_model.py` | 直链下载 GLiNER 权重与基座分词器（绕开被限流的 HF API） |
| `data/to_gliner.py` | 字符区间 → GLiNER 词级闭区间的通用转换 |
| `data/make_real_300.py` | 构造 E1 训练集（45% 无实体 + 分层保证类覆盖） |
| `data/make_seeds.py` | 选 20 条生成种子（确定性） |
| `data/make_split_manifest.py` | 三集文档清单 + 泄漏审计 |
| `eval/evaluate.py` | **主评测器**：候选缓存 → 阈值扫描 → 严格匹配 → per-class → 错误分类 → bootstrap CI |
| `eval/record.py` | 往 `results/experiments.csv` 追加一行（27 字段统一台账） |
| `generate/validator.py` | **第二阶段**项目级检查（句长 17–30 词、实体数 ≤5、抄种子与 8-gram、跨批重复、无实体句比例）；格式与偏移由 `prompts/templates/convert_validate.py` 负责 |
| `generate/client.py` | DeepSeek 客户端：读 key、调用 `/chat/completions`、429/5xx 指数退避、每次响应落盘 `D:\pt_ner_work\raw_responses\` |
| `generate/pipeline.py` | 通用生成流水线：渲染 → 调用 → 逐行解析（一行坏不废整批）→ 两级校验 → 只补不删 → 合并 |
| `generate/run_e2.py` | **E2 入口**：165 含实体 + 135 无实体，跑完自动做项目级检查与质量度量 |
| `generate/ledger.py` | 生成台账 `D:\pt_ner_work\results\gen.csv`（token、候选、接受、拒绝、费用） |
| `generate/metrics.py` | 质量度量：实体复用率、句式重复率、两类句长分布（Mann-Whitney）、种子 8-gram 重叠 |
| `generate/preflight.py` | API key 预检（只报来源/长度/格式，不打印明文） |
| `train/train_arm.py` | **通用微调入口**（E1/E2/E3 共用），产出 checkpoint + `train_config.json` |
| `train/smoke_finetune.py` | Python 3.14 微调可行性冒烟测试（A1 用，2 步） |
| `smoke_test_e0.py` | 环境冒烟：加载权重 + 20 句零样本推理 |

### 3.5 `prompts/` — 生成用 Prompt

| 路径 | 内容 |
|---|---|
| `templates/` | **当前使用的模板集**：8 个 stage（E2 两个 + E3 六个）+ `prepare_prompt.py`（占位符注入）+ `convert_validate.py`（格式与偏移校验）+ `data/{label_schema.json, seeds_20.json}` |
| `E2_E3_Prompt_模板版.zip` | 用户提供的原始压缩包（归档，不再使用） |
| `e2_plain_v1.md` | 最早的自制 Prompt v1（**已被 `templates/` 取代**，保留作设计演进记录） |

**8 个 stage 一览：**

| stage | 用途 | 每批 | 校验模式 |
|---|---|---:|---|
| `e2_generation` | E2 含实体 | 11 | `--expect entity` |
| `e2_negative` | E2 无实体 | 9 | `--expect none` |
| `e3_guidance` | E3 第 1 步：初始指导 | — | 产物供下一步 |
| `e3_refiner` | E3 第 2 步：审查优化指导 | — | 同上 |
| `e3_generation` | E3 含实体 | 11 | `--expect entity` |
| `e3_negative` | E3 无实体（沿用同一 refined guidance） | 9 | `--expect none` |
| `e3_calibrator` | E3 校准含实体批 | — | `--expect entity` |
| `e3_calibrator_negative` | E3 校准无实体批 | — | `--expect none` |

详见 `prompts/templates/README.md`。

### 3.6 `results/` — 成果

| 文件 | 内容 |
|---|---|
| `README.md` | 阈值约定 0.80 的依据与代价、台账字段说明、与其他目录的分工 |
| `E0_zeroshot_dev.md` | E0 成果说明（可引用） |
| `experiments.csv` | **实验总台账**：所有 arm 的 P/R/F1/三类 F1/TP/FP/FN/Type/Boundary/CI。由 `record.py` 自动追加，不手工编辑 |

### 3.7 `scripts/`

| 文件 | 内容 |
|---|---|
| `run.ps1` | 终端用运行包装器：tee stdout+stderr 到 `logs/`，并登记 `logs/runs.jsonl`。**沙箱内无法执行 .ps1**，那种情况用内联 tee（见 `tools/dashboard/README.md`） |

### 3.8 `tools/` — 实验看板

| 路径 | 内容 |
|---|---|
| `dashboard/server.py` | 标准库 HTTP 服务，绑 `127.0.0.1:8765`，只读运行产物，路径白名单 |
| `dashboard/static/` | 单页前端：运行状态 / 实验台账 / 运行记录 / 实时日志 / checkpoint 五区块 |
| `dashboard/README.md` | 启动方式、两种接入写法、三个环境坑 |

### 3.9 不入库的目录

| 目录 | 内容 | 为什么忽略 |
|---|---|---|
| `models/gliner_multi-v2.1/` | 权重 1,155.8 MB + 基座分词器 | 可由 `fetch_model.py` 重下 |
| `data/raw_uner/` | 原始 iob2 4.6 MB | 可由 `prepare_uner.py` 重下 |
| `logs/` | 运行日志与 `runs.jsonl` | 运行产物 |
| `.idea/` | IDE 配置 | — |

`.gitignore` 还兜底忽略 `tmp*/`、`torchinductor_*/`、`checkpoints/`。

---

## 四、数据流

```text
GitHub UNER iob2
      │  src/data/prepare_uner.py
      ▼
data/pt_bosque/{train,dev,test}.jsonl ──► data/pt_bosque/split_manifest.*（泄漏审计）
      │
      ├── src/data/make_real_300.py ──► data/curated/real_300.jsonl（E1 训练集）
      │                                        │
      │                                        └── src/data/make_seeds.py ──► seeds_20.jsonl
      │                                                                          │
      │                                    prompts/templates/（8 个 stage，注入种子与标签规范）
      │                                                                          │
      │                          ┌───────────────────────┴───────────────────────┐
      │                          ▼                                               ▼
      │              含实体批（11 条/批）                              无实体批（9 条/批）
      │              --expect entity                                     --expect none
      │                          └───────────────────────┬───────────────────────┘
      │                                                  ▼
      │                                 convert_validate.py（格式与偏移）
      │                                                  ▼
      │                                 src/generate/validator.py（项目规范）
      │                                                  ▼
      │                                   src/generate/metrics.py（质量度量）
      │                                                  ▼
      │                                        E2 / E3 各 300 条（165 + 135）
      ▼
src/train/train_arm.py（S=500）──► D:\pt_ner_work\checkpoints\<arm>\<seed>\final
                                          │
                                          ▼
                            src/eval/evaluate.py（阈值 0.80）
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
          results/experiments.csv                D:\...\results\*.json
                （汇总台账）                        （完整指标与缓存）
```

---

## 五、必须记住的约定

| 项 | 值 | 出处 |
|---|---|---|
| 评测阈值 | **0.80**，所有 arm 统一，不逐组搜 | `specs/07` §2 |
| 训练预算 | **500 步**，batch 2，lr 2e-5 | `specs/06` §2 |
| 随机种子 | 13 / 23 / 33（pilot 用 7） | `specs/06` |
| 标签两套写法 | 数据与报告用 `PER/ORG/LOC`；喂模型用 `person/organization/location` | `specs/03` §1 |
| 字符区间 | 左闭右开 `[start, end)`，且 `text[start:end] == 实体文本` | `specs/03` §3 |
| 负样本比例 | E2/E3 各 135 无实体 + 165 含实体，同批产出，只补不删 | `specs/05` |
| 每句词数 | **17–30 个空格切分的词**（≈ UD 20–35 token） | `specs/04` §2.1 |
| 每句实体数 | **1–5**（覆盖 E1 的 95%） | `specs/04` §2.2 |
| 生成模型 | DeepSeek Flash（固定，不中途更换） | `specs/04` §6.1 |
| 种子 | `seeds_20.jsonl` 已冻结，不得重抽 | `docs/TODO.md` A6 |

---

## 六、环境与已知坑

| 坑 | 处理 |
|---|---|
| `huggingface.co` 被 DNS 污染 | 必须 `HF_ENDPOINT=https://hf-mirror.com`（`src/common/env.py` 已默认设置） |
| 控制台 GBK 编码 | 必须 `$env:PYTHONIOENCODING="utf-8"`，否则打印葡语 `UnicodeEncodeError` |
| torch 把编译缓存落到仓库根 | `src/common/env.py` 已把 `TMPDIR/TEMP/TMP` 指向 `D:\pt_ner_work\tmp` |
| **训练报 `PermissionError [WinError 5]`** | 受限沙箱禁止 DataLoader 建命名管道，训练必须用完整权限；**评测在受限模式可跑** |
| 沙箱不能执行 `.ps1` | `ConstrainedLanguage` 限制，会话内改用内联 tee |
| `datasets` 5.0.1 不支持脚本型数据集 | UNER 改走 GitHub 原始 iob2 |
| 密钥泄露 | 见第八节 |
| **`deepseek-flash` 是推理模型** | thinking 会吃掉输出配额（high 时 6.8k–7.4k token，正文常为空）。**必须 `reasoning_effort=low`**，`client.py` 已默认；详见 `specs/04` §6.3 |

---

## 八、密钥与凭证

**原则：API key 不进仓库、不进日志、不进对话。本项目采用方式 B。**

| 方式 | 做法 | 采用 |
|---|---|---|
| A | 系统环境变量 `DEEPSEEK_API_KEY` | 备用（读取优先级最高） |
| **B** | **仓库外文件 `D:\pt_ner_work\.secrets\deepseek.key`** | ✅ **本项目采用** |
| C | 直接把 key 发给 AI | ✗ 不采用（会进入会话存档与模型上下文） |

**当前状态：** 目录与占位文件已建好，等待填入真实 key。

```
D:\pt_ner_work\.secrets\
├── deepseek.key    ← 单行，只放 key 本身（当前是 sk-REPLACE_ME）
└── README.md       ← 放置与验证说明
```

**验证方式**（不打印 key 明文，只报来源/长度/格式）：

```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\generate\preflight.py
# 加 --ping 会额外发一次最小请求验证额度与网络
```

退出码：`0` 通过 / `2` 未找到 / `3` 格式有问题（占位符、太短、非 sk- 开头、含空白）/ `4` 连通失败。

**硬性禁止：**

- 把 key 写进仓库内任何文件（`.env`、config、脚本）——一旦提交会永久留在 git 历史；
- 在终端打印完整 key ——每次命令输出都会 tee 到 `logs/`；
- 复用登录账号的主凭证——应使用独立、限额、可随时吊销的 key。

`.gitignore` 已兜底忽略 `.env`、`.secrets/`、`*.key`、`*.pem`、`credentials*.json`；但密钥文件本身放在工作区外，不依赖这条兜底。

---

## 九、文档维护约定

- **README 由 AI 维护**：每次结构性改动（新增/删除文件、目录变化、约定变更、结论更新）后自动同步本文件，无需逐次确认。
- 文档分工：`README.md`（全局导航）→ `docs/ARCHITECTURE.md`（结构与调用关系）→ `docs/specs/`（实现依据）→ `docs/runs/`（单次实验快照）→ `docs/logs/`（任务级变更摘要）→ `docs/TODO.md`（未完成与待决策）。

---

## 十、当前关键数字

| Arm | 阈值 | Micro-F1 | PER | ORG | LOC | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| E0 零样本 | 0.80 | 0.7497 | 0.8360 | 0.6561 | 0.7141 | [0.7148, 0.7814] |
| E1 真实数据（三种子均值） | 0.80 | **0.8487 ± 0.0045** | 0.9340 | 0.7383 | 0.8527 | [0.8179, 0.8728] |

**ORG 始终是最弱类别**（E0 0.656 → E1 0.738），是合成数据最该补的方向。
