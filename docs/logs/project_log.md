# 项目日志

**项目：** 葡萄牙语命名实体识别（GLiNER + LLM 合成数据，E0–E3）
**工作区：** `C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER`
**约定：** 每个任务完成后在文末追加一节，六段固定结构、按时间正序。最新内容永远在文件最后。

---

## 决策记录（跨任务，随时更新）

| 日期 | 决策 | 依据 |
|---|---|---|
| 2026-11-04 | 主数据集用 UNER `pt_bosque`，HAREM 仅作后续旁证 | `pt_bosque` 规格已知、划分官方、许可 CC-BY-SA-4.0；`Linguateca/harem` 验证集仅 8 篇、10 类标签、许可 unknown |
| 2026-11-04 | 暂不立项 E2+（成本匹配对照） | 主实验先跑通 E0–E3 |
| 2026-11-04 | 贡献定位按"受控比较 + 跨语言适配验证"表述，不主张新算法 | 项目书 E0–E3 版 §2 |
| 2026-11-04 | 磁盘分工：工作区放代码/文档/真实数据/模型权重；D 盘只放训练与生成产物 | 用户要求；C 盘仅 33.4 GB，checkpoint 需外置 |

---

## 任务 1 — 2026-11-04 文档整理与实验设计修订

### 做了什么
通读原项目书（791 行），整理为可执行结构；对实验设计做两轮评审，产出 V1.1（针对 E0–E5 方案）与 V2.0（针对 E0–E3 方案）两份修订文档；核查 R-GDA 论文真实性与三份候选数据集的规格。

### 关键数据
- R-GDA 属实：EACL 2026 Findings, pp. 4938–4953, DOI `10.18653/v1/2026.findings-eacl.259`；原实验场景为 SciERC / NCBI-disease / FIN（英文科学+生物医学）。
- 候选数据集规格：`pt_bosque` 7,018/1,172/1,167 句、3 类标签、CC-BY-SA-4.0；`Linguateca/harem` 121/8/128 **文档**、10 类（selective 5 类）、license unknown；`SegundoHAREM` 129 文档、无 train/dev/test、MIT（第三方再许可）。
- 修订条款：P0 缺陷 D1–D5、P1 修正 D6–D8、复查新增 N1–N7；V2.0 中 D3/D5 因删掉 E4/E5 失效，新增 Y1–Y6。

### 变更文件
- `docs/Portuguese_NER_Experimental_Design_Revision_V1.1.md`（新建，后标注"已被 V2.0 取代"）
- `docs/Portuguese_NER_Experimental_Design_Revision_V2.0.md`（新建）

### 验证方式
`Select-String -Path docs/*.md -Pattern "^#{1,2} "` 核对章节结构；数据集规格可通过 HF 卡片 API 与 arXiv 2311.09122v3 Table 1 复核。

### 遗留问题
V2.0 §8 台账中"标签集冻结到几类""E2+ 是否立项"仍待定；`葡萄牙语NER项目书_E0-E3.md` 的 §5.2 尚未写入句长/实体密度约束（N1）。

---

## 任务 2 — 2026-11-04 E0 前置：环境、数据、模型准备

### 做了什么
在 `Desktop\NLP Ass#1\.venv`（Python 3.14.7）中安装 `gliner 0.2.30` 与 `seqeval 1.2.2`；下载并转换 UNER `pt_bosque`；下载 GLiNER 多语言权重与基座分词器；跑通零样本推理。

### 关键数据
数据集实测（与 UNER 论文 Table 1 完全一致，字符区间转换零回退）：

| Split | 文档 | 句子 | 实体 | Token | 实体/句 | Token/句 | 无实体句占比 | PER / ORG / LOC |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | 235 | 7,018 | 8,101 | 171,776 | 1.154 | 24.48 | 44.5% | 2,914 / 2,762 / 2,425 |
| dev | 41 | 1,172 | 1,401 | 28,447 | 1.195 | 24.27 | 43.9% | 570 / 417 / 414 |
| test | 40 | 1,167 | 1,215 | 27,604 | 1.041 | 23.65 | 46.6% | 442 / 456 / 317 |

环境与模型：torch 2.14.0+cu130（cuda=True，RTX 5070 12,227 MiB）、transformers 5.17.0、gliner 0.2.30；权重 1,155.8 MB + 分词器 4.3 MB；加载耗时 16.5 s。

冒烟测试（dev 前 20 条含实体句，阈值 0.5，非正式评测）：gold=40 pred=45 TP=28 FP=17 FN=12 → P=0.622 R=0.700 F1=0.659，58–121 ms/句，区间还原失败 0。

### 变更文件
- `configs/paths.yaml`（新建，路径唯一配置源）
- `src/data/prepare_uner.py`（新建，iob2 → JSONL + 统计）
- `src/data/fetch_model.py`（新建，直链下载权重与分词器）
- `src/smoke_test_e0.py`（新建，E0 冒烟测试）
- `data/pt_bosque/{train,dev,test}.jsonl` + `stats.json`（新建）
- `data/raw_uner/*.iob2`（新建，原始文件 4.6 MB）
- `models/gliner_multi-v2.1/`（新建，1,106 MB）

### 验证方式
```powershell
$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8
$env:PYTHONIOENCODING="utf-8"; $env:HF_HOME="D:\pt_ner_work\hf_cache"
$py="C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
& $py src\data\prepare_uner.py    # 期望 7018/1172/1167 句，8101/1401/1215 实体
& $py src\smoke_test_e0.py        # 期望 F1≈0.66，区间还原失败 0，exit=0
```

### 遗留问题
1. `datasets` 5.0.1 已不支持脚本型数据集，`universalner/universal_ner` 必须走 GitHub 原始 `.iob2`（已改为直连下载）。
2. HF Hub API 被 hf-mirror 限流（429），改用 `resolve` 直链下载。
3. `torch.jit.script` 在 Python 3.14 上告警"may break"，零样本推理可用，**微调路径未验证**，E1 开工前必须先跑 1 步微调。
4. 跨集完全重复句：train&test 1 条、train&dev 2 条、train 内部 11 条，待 Phase 1 剔除。
5. 44% 无实体句意味着 E1 的 300 条抽查口径需先定（随机抽 vs 只抽含实体句），且合成数据需匹配无实体句比例，否则成为混杂变量。
6. 控制台 GBK 编码会导致打印葡语报 `UnicodeEncodeError`，必须设 `PYTHONIOENCODING=utf-8`。

---

## 任务 3 — 2026-11-04 目录重整与模型回迁

### 做了什么
按用户要求建立 `docs/` 用于存放项目书、`docs/logs/` 用于存放日志；把 4 份项目书移入 `docs/`；把 1,106 MB 模型权重与 4.6 MB 原始数据从 `D:\pt_ner_work` 搬回工作区；同步 4 处硬编码路径；重跑数据转换与冒烟测试验证未搬坏。

### 关键数据
- 搬移：`D:\pt_ner_work\models\gliner_multi-v2.1` → `models/gliner_multi-v2.1`（5 文件 / 1,106.4 MB，0.9 s）；`D:\pt_ner_work\raw_uner` → `data/raw_uner`（3 文件 / 4.6 MB，0.3 s）。搬移前后文件数与字节数一致。
- D 盘现状：`checkpoints`、`data/synthetic`、`raw_responses`、`results`、`hf_cache`、`models`（空目录，未删）全部 0 MB；D 盘仅保留训练与生成产物的位置。
- 搬移后回归验证：`prepare_uner.py` 输出 7018/1172/1167 句、8101/1401/1215 实体、回退 0、exit=0；`smoke_test_e0.py` 输出 F1=0.6588、区间还原失败 0、exit=0。

### 变更文件
- `docs/`（新建目录）、`docs/logs/`（新建目录）、`docs/logs/project_log.md`（本文件，新建）
- 移入 `docs/`：`Portuguese_NER_Project_Proposal_V1.0.md`、`葡萄牙语NER项目书_E0-E3.md`、`Portuguese_NER_Experimental_Design_Revision_V1.1.md`、`Portuguese_NER_Experimental_Design_Revision_V2.0.md`
- 修改：`configs/paths.yaml`（新增 `uner_raw`/`models`/`docs`/`logs`，`model.local_dir` 指向工作区）
- 修改：`src/data/prepare_uner.py`（`RAW_DIR` → 工作区）、`src/data/fetch_model.py`（`DEST` → 工作区）、`src/smoke_test_e0.py`（`MODEL_DIR` → 工作区）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
Get-ChildItem -Recurse -Depth 2 | Select-Object FullName   # 应看到 docs/、models/、data/、configs/、src/
Get-ChildItem D:\pt_ner_work -Directory                     # D 盘只剩产物目录，均为空
```

### 遗留问题
- `D:\pt_ner_work\models` 空目录保留未删除。
- 沙箱后端在本机不可用（`workspace-write` 启动失败，`windows-acl-run` 临时目录缺失），本轮起所有写操作都需 `danger-full-access`。**（已在任务 4 修复）**

---

## 任务 4 — 2026-11-04 修复受限沙箱 + 建立 TODO 文档

### 做了什么
定位并修复 DSH ACL 受限运行器启动失败（`workspace-write` 模式下连只读命令都无法执行）；新建 `docs/TODO.md`，把日志里散落的遗留问题、待决策项和待实现工作收拢成一份可勾选的清单。

### 关键数据
- 报错：`Runner failure: windows-acl-run: --temp is not an existing directory: C:\Users\liush\AppData\Local\Temp\dsh-gG8Pxe`
- 诊断结果：`%TEMP%` 本身存在、所有者 `DESKTOP-LEWIS\liush`、本账号 FullControl，**没有 deny 项，也不是权限问题**；纯粹是运行器要求的那个临时子目录不存在（同类目录 `dsh-acl-locks`、`dsh-spill-Mg3hX5` 等都还在，只有它丢了）。
- 修复：新建 `C:\Users\liush\AppData\Local\Temp\dsh-gG8Pxe` 一个空目录。
- 复测：默认 `workspace-write` 下执行 `Get-Location` / `Get-ChildItem` 成功，沙箱恢复正常；受限模式为 ConstrainedLanguage（`[System.Text.UTF8Encoding]::new()` 之类 .NET 静态调用会报 `CannotCreateTypeConstrainedLanguage`，属预期）。
- TODO 清单规模：A 阻塞项 3 条、B 待决策 6 条、C 待实现 19 条、D 已废弃 5 条。

### 变更文件
- `docs/TODO.md`（新建）
- `docs/logs/project_log.md`（追加本节）
- 工作区外：`C:\Users\liush\AppData\Local\Temp\dsh-gG8Pxe`（新建空目录）

### 验证方式
```powershell
Test-Path "C:\Users\liush\AppData\Local\Temp\dsh-gG8Pxe"   # True
Get-ChildItem "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER\docs"   # 应含 TODO.md 与 logs/
```
不需要 `danger-full-access`：默认受限模式下这两条命令即可执行。

### 遗留问题
1. 该临时目录是运行器按会话行为要求"预先存在"的，若再被临时目录清理删掉，会重现同样报错；处理办法是建回报错里给出的那个路径，若届时目录已存在仍失败，再用 `diagnose-windows-sandbox-acl` 脚本排查权限。
2. `docs/TODO.md` 中 A1（Python 3.14 微调验证）、A2（E0 全量评测脚本）仍是 E1 开工前的阻塞项。

---


## 任务 5 — 2026-11-04 A1：验证 Python 3.14 上的微调链路

### 做了什么
写 `src/train/smoke_finetune.py`，把 `train.jsonl` 的字符区间实体转成 GLiNER 词级训练格式，用 2 步微调验证「数据格式 → collator → Trainer → 反向传播 → 保存 → 重新推理」整条链路是否可用。

### 关键数据
- 版本组合：Python 3.14.7 / torch 2.14.0+cu130 / transformers 5.17.0 / gliner 0.2.30。
- 入口 API：`from gliner import GLiNER` 拿到的是元类，`from_pretrained` 后实例真实类型是 `UniEncoderSpanGLiNER`，训练方法是它的 `train_model(...)`（类对象上没有该方法，用 `inspect.signature(GLiNER.train_model)` 会 `AttributeError`）。
- 训练数据格式：`{"tokenized_text": [...], "ner": [[start_word, end_word, label], ...]}`，标签用 `person / organization / location`。
- 结果：**通过**。loss 1.665 → 2.161（2 步，lr 5e-5），grad_norm 80.99 → 129.6，train_runtime 49.51 s。
- 保存：`D:\pt_ner_work\checkpoints\smoke\final`，4 个文件，最大 1,155.9 MB。
- 微调后推理：「Vera Lúcia Barbosa」person 0.996 /「São Paulo」location 0.982 /「rio Tietê」location 0.969。
- 告警：`torch.jit.script is not supported in Python 3.14+`（仅 FutureWarning，未影响训练）；tokenizer 的 mistral regex 提示（DeBERTa SPM 分词器，实测分词正常）。

### 变更文件
- `src/train/smoke_finetune.py`（新建）
- `docs/TODO.md`（A1 勾掉，新增 C20 单步耗时测量）
- 工作区外：`D:\pt_ner_work\checkpoints\smoke\`（2 步冒烟产物，可删）

### 验证方式
```powershell
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\train\smoke_finetune.py
# 期望末行「结论：A1 通过」，exit=0
```

### 遗留问题
1. 2 步耗时 49.5 s 含首次 CUDA 预热，稳定单步耗时未知，正式训练前须测（TODO C20）；若真是 25 s/步，300 步单次训练约 2 小时。
2. 单份 checkpoint 1.16 GB，3 组 × 3 种子约 10.5 GB，D 盘 758 GB 够用但仍需在 C16 台账里记录。
3. 冒烟产物 `D:\pt_ner_work\checkpoints\smoke` 未删除。

## 任务 6 — 2026-11-04 建立规范目录 + 负样本比例定案

### 做了什么
按用户要求把"目前所有已知规范"拆成单文件放进新建的 `docs/specs/`（8 份规范 + 1 份索引），并更新 TODO：B2 决策落地、C 组条目全部挂上对应规范文件。

### 关键数据
- 规范文件：`01-dataset`（来源与规模）、`02-split-and-leakage`（划分与泄漏）、`03-labeling`（标签与区间）、`04-generation-constraints`（句长/密度/模板化）、`05-negative-sampling`（负样本）、`06-training-protocol`（微调）、`07-evaluation-protocol`（评测）、`08-reproducibility-and-env`（环境与台账）。
- 负样本定案（B2）：**E2/E3 各 300 条 = 135 条无实体句（45%）+ 165 条含实体句**（含实体句按均值 2.08 ≈ 343 个实体，与 E1 的 ~345 对齐）；每次生成批 20 条 = 9 + 11；要求生成器在同一次生成中同时产出两类，禁止"先写含实体句再删实体"与"后期按比例裁剪"；比例不足时只补不删；判定以 `entities` 是否为空为准，不采信生成器自述。
- 新增风险项：LLM 写无实体句容易写成模板化/过干净的句子，导致两类句子长度语体出现系统差异 → 生成后须做长度分布对比（已写入规范 04 §5）。
- TODO 状态：未完成 27 项、已完成/废弃 7 项。

### 变更文件
- `docs/specs/README.md`、`docs/specs/01-dataset.md` … `08-reproducibility-and-env.md`（9 个新建文件）
- `docs/TODO.md`（B2 勾掉并写明 135/165；B1 补注与规范联动；C 组条目挂规范；新增 C3 的 dev 二次切分、C18 改为回写正文）
- `docs/logs/project_log.md`（追加本节）

### 验证方式
```powershell
Get-ChildItem "docs\specs"            # 应为 9 个 md
Select-String -Path "docs\specs\05-negative-sampling.md" -Pattern "135|165|9 \+ 11"
Select-String -Path "docs\TODO.md" -Pattern "B2"
```

### 遗留问题
1. B1（E1 抽 300 条的随机/只抽含实体句口径）仍未定；若改选"只抽含实体句"，`specs/05` 必须同步重写为 0% 无实体句。
2. `specs/02` 的 dev 二次切分（872/300，按文档）尚未实施，E0 评测目前跑在完整 dev 上。
3. `specs/06` 的总步数 S 与超参仍待 pilot。
4. 规范与项目书正文存在重复表述，C18 待回写统一。

## 任务 7 — 2026-11-04 A2 评测器 + TODO 重排 + git init

### 做了什么
1. 写 `src/eval/evaluate.py`：候选一次前向（threshold=0.0）缓存 → 离线扫 17 档阈值 → 严格实体匹配 → per-class → 错误分类 → 按文档 bootstrap CI，输出 md + json。
2. 在完整 dev 上跑出 E0 基线。
3. 重排 TODO：E0–E3 主流程拆成 A1–A13 放进 A 栏，原 A/B/C 顺延为 B/C/D，新增旧编号对照表。
4. `git init`。

### 关键数据
- E0 零样本（完整 dev 1,172 句 / 41 文档）：**最优阈值 0.80 → Micro-F1 0.7497**（P 0.7680 / R 0.7323），Macro-F1 0.7354，95% CI [0.5679, 0.7952]。
- 分类别 F1：**PER 0.8360 / LOC 0.7141 / ORG 0.6561**（支持数 570 / 414 / 417）→ ORG 比 PER 低 18 点，是明确短板。
- 错误结构（阈值 0.80）：Type Error 45、Boundary Error 45、其余 FP 220、其余 FN 285。
- **阈值效应**：默认 0.5 时 Micro-F1 仅 0.6843，调到 0.80 得 0.7497，**纯阈值收益 +6.5 点**。
- 性能：候选收集 1,172 句 21.2 s（加载 5.4 s）；阈值扫描 + 错误分类 + 1000 次 bootstrap 秒级完成。
- git：`git init` 成功（git 2.55.0），仓库根 `.git` 已建，5 个未跟踪目录：configs / data / docs / models / src。

### 变更文件
- `src/eval/evaluate.py`（新建，约 300 行）
- `docs/TODO.md`（整体重排：A1–A13 主流程 / B 阻塞 / C 决策 / D 实现 / E 已废弃）
- `docs/logs/project_log.md`（追加本节）
- `.git/`（新建仓库，工作区外无变更）
- `D:\pt_ner_work\results\e0_dev.md`、`e0_dev.json`、`e0_dev_candidates.jsonl`（新建）

### 验证方式
```powershell
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\eval\evaluate.py `
    --model models/gliner_multi-v2.1 --split dev `
    --cache D:/pt_ner_work/results/e0_dev_candidates.jsonl --out D:/pt_ner_work/results/e0_dev
Get-Content D:\pt_ner_work\results\e0_dev.md -TotalCount 10
git -C "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER" status --short
```

### 遗留问题
1. **工作区根目录出现 7 个 `tmpXXXXXXXX` 目录 + `torchinductor_liush`**（10:45–10:46 生成，torch 编译缓存的临时目录落到了 CWD），git 读不动它们、每次命令都警告。未删除，待定处理方式；`.gitignore` 需加 `tmp*/` 与 `torchinductor_*/`。
2. 仓库**尚无 `.gitignore`**，`models/` 1.1 GB 与 `data/raw_uner/` 都是未跟踪状态，**切勿执行 `git add -A`**（TODO D17）。
3. E0 数字基于完整 dev，A1 的二次切分落地后需在 Dev Selection 上重跑，才是 G2 正式数字。
4. C1（E1 抽样口径）仍未定，卡住 A1。

## 任务 8 — 2026-11-04 清理编译缓存 + 建立实验运行记录制度

### 做了什么
1. 删除工作区根目录的 torch 编译缓存与临时目录（8 个）：`tmpXXXXXXXX` ×7、`torchinductor_liush` ×1。
2. 定位再生根因并修复：受限沙箱下 `%TEMP%` 不可写，torch/triton 把临时目录建到 CWD。新增 `src/common/env.py`，在 import torch 之前把 `TMPDIR/TEMP/TMP` 指到 `D:/pt_ner_work/tmp`，并统一 `HF_ENDPOINT` / `HF_HOME` / `PYTHONIOENCODING`；三个脚本改为 `from src.common import env`。
3. 建立实验运行记录制度：`docs/runs/` + 规范 `specs/09-run-records.md`，并补写首份 `2026-11-04_E0_zeroshot_dev.md`。

### 关键数据
- 清理：首次在受限模式下删除失败（7 个 tmp 目录"访问被拒绝"），改用完整权限后全部删除；`torchinductor_liush` 受限模式即可删。
- 修复验证：清空后重跑 `src/smoke_test_e0.py`，工作区根目录只剩 `.git .idea configs data docs models src`，**不再生成新的 tmp 目录**；`D:\pt_ner_work\tmp` 接管了 torch 缓存。
- 冒烟复测结果与之前一致：20 句 P 0.6222 / R 0.7000 / F1 0.6588，57 ms/句，区间还原失败 0，exit=0。
- 新建文件：`src/common/env.py`、`src/common/__init__.py`、`src/__init__.py`、`docs/runs/README.md`、`docs/runs/2026-11-04_E0_zeroshot_dev.md`、`docs/specs/09-run-records.md`。

### 变更文件
- 新建：`src/common/env.py`、`src/common/__init__.py`、`src/__init__.py`、`docs/runs/README.md`、`docs/runs/2026-11-04_E0_zeroshot_dev.md`、`docs/specs/09-run-records.md`
- 修改：`src/smoke_test_e0.py`、`src/eval/evaluate.py`、`src/train/smoke_finetune.py`（统一导入 env）
- 修改：`docs/specs/README.md`（索引加 09）、`docs/TODO.md`（新增 D20）
- 删除：工作区根目录 7 个 `tmp*` 目录 + `torchinductor_liush`

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
Get-ChildItem -Force | Select Name        # 不应再有 tmp* / torchinductor_*
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\smoke_test_e0.py
Get-ChildItem D:\pt_ner_work\tmp          # torch 缓存落在这里
```

### 遗留问题
1. 工作区出现 `.idea/`（IDE 目录），未处理；`.gitignore`（D17）应一并忽略。
2. 受限模式下仍无法删除部分目录（本次 7 个 tmp 全部需完整权限），影响后续清理操作。
3. `docs/runs/` 目前只有 E0 一份；后续每次实验必须同步补记录（D20）。

## 任务 9 — 2026-11-04 git 首次提交 + 阈值统一 0.80 + 成果目录 + E1 数据与代码

### 做了什么
1. 建 `.gitignore`（排除 models/、data/raw_uner/、.idea/、tmp*/、torchinductor_*/、checkpoints/），完成首次提交 `0312645`（32 文件，13,199 行插入）。
2. 阈值统一为 **0.80**：改 `evaluate.py` 支持 `--threshold / --arm / --record`，E0 在 0.80 上重算并写入台账。
3. 新建 `results/` 成果目录：`README.md`（阈值约定与台账说明）、`E0_zeroshot_dev.md`（成果）、`experiments.csv`（总台账）。
4. 新增 `src/eval/record.py`：统一往 `results/experiments.csv` 追加行（27 个字段）。
5. 按 C1 决策产出 E1 训练集：`src/data/make_real_300.py` → `data/curated/real_300.jsonl` + manifest + stats。
6. 新增 `src/data/to_gliner.py`（字符区间→词级区间通用转换）、`src/train/train_arm.py`（E1–E3 通用微调入口，**未运行**）。

### 关键数据
- E0 @ 0.80（修正后）：Micro-F1 **0.7497**，P 0.7680 / R 0.7323，Macro 0.7354，CI **[0.7148, 0.7814]**，TP/FP/FN = 1026/310/375。分类别：PER 0.8360 / LOC 0.7141 / ORG 0.6561。
- **修复 bug：doc_id 传递。** 原 `prepare_uner.py` 只在每文档首句写入 `newdoc id`，其余 1,132/1,172 条 dev 句子的 doc_id 为空串，导致按文档 bootstrap 实际只重采样了 2 类（空串大桶 + 40 个单句桶），把 CI 严重拉宽（初版 [0.5679, 0.7952]）。修复后按文档正确分组：train 234 / dev 40 / test 39 篇，每篇 ≤30 句，中位数 30。点估计不变（同一批预测），CI 收窄为 [0.7148, 0.7814]。
- 修复方式：改 `parse_iob2` 向前传递文档 ID → 重跑转换 → 用 `id → doc_id` 映射**就地修补候选缓存**（不重新推理）。
- E1 训练集 `real_300.jsonl`：300 句 / 167 文档 / 376 实体（PER 134、ORG 113、LOC 129）/ 7,861 token；135 无实体（45%）+ 165 含实体；含实体句均值 2.28；种子 42；去重剔除内部 11 条、跨集 3 条。
- 阈值曲线的平坦性：0.75/0.80/0.85 = 0.7442/0.7497/0.7427，极差 0.7 点，说明 0.80 不是尖峰。
- 用户新规则：**未收到「开始实验」指令不得运行任何训练/评测实验**，已写入 TODO 头部执行纪律。

### 变更文件
- 新建：`.gitignore`、`src/data/to_gliner.py`、`src/data/make_real_300.py`、`src/train/train_arm.py`、`src/eval/record.py`、`results/README.md`、`results/E0_zeroshot_dev.md`、`results/experiments.csv`、`data/curated/real_300.jsonl`、`real_300_manifest.json`、`real_300_stats.md`
- 修改：`src/data/prepare_uner.py`（doc_id 向前传递）、`src/eval/evaluate.py`（--threshold/--arm/--record、报告头、doc 计数修正）
- 修改：`docs/specs/02`（E1 抽样已定）、`docs/specs/05`（目标改为对齐 E1 实测 376 实体 / 1.2533）、`docs/specs/07`（阈值统一 0.80）、`docs/TODO.md`（A1 部分完成、A1-b、C1 完成、执行纪律、D6-b/D7-b）
- 重新生成：`data/pt_bosque/{train,dev,test}.jsonl`、`stats.json`（doc_id 修复）
- git：首次提交 `0312645`

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
git log --oneline -1
Import-Csv results\experiments.csv | Format-Table arm,split,threshold,micro_f1,org_f1,ci_low,ci_high
(Get-Content data\curated\real_300.jsonl).Count            # 300
Select-String -Path data\curated\real_300_stats.md -Pattern "135|376"
```

### 遗留问题
1. `experiments.csv` 的 `no_entity_ratio` 列目前为空（评测时未统计），待 D7-b 补。
2. dev 二次切分（872/300）仍未做，A1 未完全关闭；B3 的重跑要等它。
3. E1 训练代码 `src/train/train_arm.py` 已写好但**按要求未运行**，等「开始实验」指令。
4. 首次提交后新增的文件（本轮）尚未提交，工作区为脏。

## 任务 10 — 2026-11-04 排查"每次都要批准"的问题

### 做了什么
用户要求把 `D:\pt_ner_work` 加成工作区以省去反复授权。检查 DSH 的工作区模型并实测沙箱写权限边界。

### 关键数据
- DSH 的工作区是"一个会话 = 一个目录"：`~/.dsh/storages/workspace.json` 只有 workspaces 路径表（3 个工作区），**没有"额外可写根"或"多根工作区"配置项**；profile 层 `cordis.patch.yml` 也只配插件，与沙箱路径无关。
- **实测受限模式（workspace-write）写入边界：**
  | 路径 | 结果 |
  |---|---|
  | `D:\` 根、`D:\pt_ner_work\checkpoints`、`D:\pt_ner_work\results`、新建 `D:\pt_ner_work\_probe_dir` | 全部**成功**，无需批准 |
  | `C:\Users\liush\Desktop\`、`Documents\`、`AppData\Local\` | 全部**被拒** |
  | 工作区内 | 成功 |
- 结论：**D 盘本来就免批准**。此前的多次批准请求是我主动传了 `danger-full-access` 造成的过度提升，不是沙箱拒绝。
- 附带验证：在受限模式下通过 junction 写 D 盘同样成功；junction 已用 `rmdir` 删除（只删链接），`D:\pt_ner_work` 24 个文件完好。

### 变更文件
- 修改 `docs/specs/08-reproducibility-and-env.md`：新增 §6.1「磁盘写权限实测」表与"不要预先提升"的结论
- 修改 `docs/TODO.md`：环境事实表加"磁盘写权限"一行
- 创建后删除：工作区 `_dwork` junction（临时验证用）

### 验证方式
```powershell
# 受限模式下应成功
"ok" | Set-Content D:\pt_ner_work\results\_probe.txt; Remove-Item D:\pt_ner_work\results\_probe.txt
# 工作区外 C 盘应被拒
"x" | Set-Content C:\Users\liush\Desktop\_probe.txt
```

### 遗留问题
1. 若确实想要"零批准"（含 C 盘任意位置），只能把会话切到「完全权限」，这是 UI 设置，我无法代改。
2. 若希望所有产物都落在同一棵目录树里便于打包/备份，可考虑整体迁到 `D:\pt_ner_work\Portuguese_ErrorNER` 并把工作区切到那里（需在 GUI 重新添加工作区）。

## 任务 11 — 2026-11-04 关闭 A1：取消 dev 二次切分 + 文档清单落盘

### 做了什么
1. 按决策取消 dev 的 872/300 二次切分（理由见下），完整 dev 兼作 checkpoint 选择与指标报告。
2. 新增 `src/data/make_split_manifest.py`，落盘三集文档级清单与泄漏审计。
3. 顺带修掉一个会误报泄漏的命名 bug：三个 split 的文档编号各自从 `batch-0000` 起，跨集必然重名。

### 关键数据
- **取消二次切分的原因**：Dev Feedback（300 句）只服务于已删除的 Error-Aware（算类别 Recall）；Dev Selection 的"搜阈值"用途也因阈值统一 0.80 而消失；剩下的"选 checkpoint"用完整 dev 即可。继续切会把报告集从 1,172 句压到 300 句，CI 宽度由 0.067 扩到约 0.13，白丢统计功效。
- **文档 ID 命名空间**：`prepare_uner.py` 给 doc_id 加 split 前缀（`dev:batch-0000`），否则跨集审计全部误判为交叠。
- **审计结果**：文档级交叠 train&dev = 0、train&test = 0、dev&test = 0；句级完全重复全库 14 条（train 内部 11 + 跨集 3），均已排除在 E1 训练集外；dev/test 内部重复为 0。
- 三集规模：train 234 篇 / 7,018 句 / 8,101 实体；dev 40 篇 / 1,172 句 / 1,401 实体；test 39 篇 / 1,167 句 / 1,215 实体。
- 重跑验证：E0 @0.80 仍为 Micro-F1 0.7497（重命名 doc_id 不改变分组，CI 仍 [0.7148, 0.7814]）；real_300 重抽结果完全一致（376 实体 / 167 文档 / 45% 无实体）。
- G1 闸门：每类 mention ≥ 30 ✅、跨集文档交叠 0 ✅、训练集重复句 0 ✅ —— **G1 通过，A1 关闭**。

### 变更文件
- 新建：`src/data/make_split_manifest.py`、`data/pt_bosque/split_manifest.json`、`data/pt_bosque/split_manifest.md`
- 修改：`src/data/prepare_uner.py`（doc_id 加 split 前缀）
- 修改：`docs/specs/02-split-and-leakage.md`（§2 改为"已取消"并写清理由与披露要求）
- 修改：`docs/TODO.md`（A1 全部勾选并写明 G1 结果）
- 重新生成：`data/pt_bosque/{train,dev,test}.jsonl`、`stats.json`、`data/curated/real_300.*`
- 修补：`D:\pt_ner_work\results\e0_dev_candidates.jsonl` 的 doc_id（1,172 条，不重新推理）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
Select-String -Path data\pt_bosque\split_manifest.md -Pattern "交叠|0\*\*"
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\data\make_split_manifest.py
```

### 遗留问题
1. 句级重复句仍留在官方 `train.jsonl` 里（14 条），只是被排除在 E1 训练集外；若以后要用全量 train 训练，需要先做清洗。
2. dev 兼作模型选择与报告，limitation 里必须披露（已写入 specs/02）。
3. E1 训练代码仍未运行，等「开始实验」。

## 任务 12 — 2026-11-04 实验看板（不改实验代码）

### 做了什么
用户反馈"跑的时候看不到代码输出"。原因是两层：一是我的命令都套了 `Select-Object -Last N` 只回传尾部，二是 tqdm/警告走 stderr，命令结束即丢。按要求做了一个本地只读看板，**实验脚本一行未改**。

- `tools/dashboard/server.py`：标准库 HTTP 服务，绑 `127.0.0.1:8765`，接口 `/api/state`、`/api/log`、`/api/progress`、`/api/doc`，路径白名单限制在 `logs/`、`results/`、`docs/runs/`、`D:\pt_ner_work\`。
- `tools/dashboard/static/`：单页看板，2 秒轮询状态、1.5 秒追日志；五个区块（运行状态 / 实验台账 / 运行记录 / 实时日志 / checkpoint）。
- `scripts/run.ps1`：运行包装器，把 stdout+stderr tee 到 `logs/<时间戳>_<名字>.log` 并登记 `logs/runs.jsonl`。
- `.gitignore` 增加 `logs/`。

### 关键数据
- 地址：**http://127.0.0.1:8765/**，自检 HTTP 200，`/api/state` 能读到 1 行实验台账、运行记录与 checkpoint 列表。
- 端到端验证：跑一个 4 行渐进输出的假任务，日志实时写入，看板接口读回的文本干净（`tick 0..3`），`runs.jsonl` 记录 start/done 两条同 id 记录并被去重。
- **踩到三个环境坑，都已写进 `tools/dashboard/README.md`：**
  1. 沙箱是 Windows PowerShell 5.1，**`pwsh` 不在 PATH**；
  2. ConstrainedLanguage **禁止执行 .ps1 与 `[scriptblock]::Create`**（`DotSourceNotSupported`），所以包装器只能在用户自己的终端用，会话内必须内联 tee；
  3. **编码不能混**：`Tee-Object` 写 UTF-16LE+BOM，`Out-File -Encoding utf8` 写 UTF-8+BOM，混在同一文件里会变成 `t i c k` 这种夹空字节的乱码 → 日志头也必须用 Tee 写；服务端按 BOM 自动识别。

### 变更文件
- 新建：`tools/dashboard/server.py`、`tools/dashboard/static/{index.html,app.js,style.css}`、`tools/dashboard/README.md`、`scripts/run.ps1`
- 修改：`.gitignore`（`logs/`）
- 修改：`docs/logs/project_log.md`（本节）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
Invoke-WebRequest http://127.0.0.1:8765/ -UseBasicParsing | Select-Object StatusCode
Invoke-RestMethod http://127.0.0.1:8765/api/state | Select-Object now
# 浏览器打开 http://127.0.0.1:8765/
```

### 遗留问题
1. 看板服务作为托管后台任务运行（job id `pwsh-0f87fbce1b5b4b7d`），会话结束即停；重启命令见 README。
2. 训练脚本没有 `progress.jsonl`（要求不改实验代码），所以"跑到第几步"目前靠解析日志里的 HF Trainer 输出来看；若以后允许加一行写透，实时性更好。
3. `scripts/run.ps1` 在沙箱内不可执行，会话里只能用内联写法；两者行为等价。

## 任务 13 — 2026-11-04 跑通首个 E1（seed 13）

### 做了什么
1. 按协议训练 E1：`real_300.jsonl` → GLiNER 微调 300 步（seed 13）。
2. 在完整 dev 上按统一阈值 0.80 评测，写入台账 `results/experiments.csv`。
3. 写运行记录 `docs/runs/2026-11-04_E1_seed13_dev.md`。

### 关键数据
- **训练**：4.2 s 加载 + **81.6 s 训练 300 步**（3.69 步/秒，2 epoch），loss 2.519 收尾，checkpoint 约 1.16 GB。
- **结果（阈值 0.80，完整 dev 1,172 句）**：

  | 指标 | E0 | E1 seed13 | 变化 |
  |---|---:|---:|---:|
  | Precision | 0.7680 | 0.7686 | +0.001 |
  | Recall | 0.7323 | 0.9079 | **+0.176** |
  | Micro-F1 | 0.7497 | **0.8325** | **+0.083** |
  | Macro-F1 | 0.7354 | 0.8250 | +0.090 |
  | CI | [0.7148,0.7814] | [0.8044,0.8568] | 不重叠 |

- 分类别 F1：PER 0.8360→0.9125、ORG 0.6561→0.7361、LOC 0.7141→0.8264。
- 错误结构：Type 56、Boundary 29、其余 FP 298、其余 FN 44。
- **固定阈值的代价实测**：扫描最优在 0.90（0.8448），固定 0.80 是 0.8325，即让出 1.2 点——印证 specs/07 §2 的预判，需在报告中披露。
- **环境坑**：受限沙箱下训练报 `PermissionError [WinError 5]`（multiprocessing.Queue 建命名管道），必须用完整权限；评测在受限模式可跑。
- 性能修正：先前按冒烟 2 步 49.5 s 外推"300 步约 2 小时"完全错误，实测 81.6 s。3 个种子总计约 4 分钟，A4 的 pilot 成本也随之变得极低。

### 变更文件
- 新增：`docs/runs/2026-11-04_E1_seed13_dev.md`
- 修改：`docs/TODO.md`（A5 记录 seed 13 完成）
- 修改：`results/experiments.csv`（追加 E1 行）
- 产物：`D:\pt_ner_work\checkpoints\E1\seed13\`（final + train_config.json）、`D:\pt_ner_work\results\e1_s13_dev.{md,json}`、候选缓存
- 日志：`logs/20261104_115626_train_E1_seed13.log`、`logs/*_eval_E1_seed13.log`

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
Import-Csv results\experiments.csv | Format-Table arm,seed,threshold,micro_f1,org_f1,ci_low,ci_high
Get-Content D:\pt_ner_work\results\e1_s13_dev.md -TotalCount 8
```
或打开看板 http://127.0.0.1:8765/ 看台账与 checkpoint。

### 遗留问题
1. E1 还差 seed 23 / 33（各约 82 s + 评测 25 s）。
2. A4 的 pilot 定 S 仍未做；现在知道单次训练只要 80 秒，扫 4 个 S 值成本可忽略，建议尽快补齐。
3. 固定 0.80 对微调组偏低 1.2 点，结论口径需要在报告中统一说明。

## 任务 14 — 2026-11-04 A4 pilot 定 S + E1 三种子跑齐

### 做了什么
1. A4：用 pilot 种子 7 扫 S ∈ {100,200,300,500}，冻结训练步数。
2. 按冻结的 S 重训 E1 三个正式种子（13/23/33），逐个评测并写入台账。
3. 台账补 `max_steps` 列，重建 `results/experiments.csv`（删掉 S=300 的旧行）。

### 关键数据
- **Pilot 曲线（种子 7，dev @0.80）**：S=100 → 0.7617、200 → 0.7990、300 → 0.8209、500 → **0.8366**。单调上升无掉点，按判据取网格内最大者 → **S = 500**。收益递减：300→500 只 +1.6 点。
- 提升全部来自 Precision（0.6652→0.7908），Recall 微降（0.8908→0.8879）。
- **E1 三种子 @S=500**：

  | seed | P | R | Micro-F1 | CI |
  |---|---:|---:|---:|---|
  | 13 | 0.8161 | 0.8901 | 0.8515 | [0.8313,0.8710] |
  | 23 | 0.7977 | 0.8951 | 0.8436 | [0.8179,0.8677] |
  | 33 | 0.8142 | 0.8915 | 0.8511 | [0.8280,0.8728] |
  | 均值 | 0.8093 | 0.8922 | **0.8487 ± 0.0045** | — |

  分类别均值：PER 0.9340 / ORG 0.7383 / LOC 0.8527。对比 E0（0.7497）**+9.9 点**，CI 完全不重叠。
- 训练时长 113.6 / 120.2 / 116.6 s，极稳定。种子标准差仅 0.45 点（S=300 时 seed 间差达 1.2 点，步数增加后方差收窄）。
- **PER−ORG 差距略扩大**：E0 为 0.1799，E1 为 0.1957 —— 微调对 PER 帮助（+9.8）大于 ORG（+8.2），ORG 仍是最该被合成数据补的短板。
- 固定 0.80 的代价：三种子扫描最优均在 0.90，比 0.80 高 0.3–0.5 点。

### 变更文件
- 新增：`docs/runs/2026-11-04_A4_pilot_S.md`、`docs/runs/2026-11-04_E1_S500_3seeds.md`
- 修改：`docs/runs/2026-11-04_E1_seed13_dev.md`（标注已被取代）
- 修改：`docs/specs/06-training-protocol.md`（S=500 冻结 + pilot 证据 + 待回验说明）
- 修改：`docs/TODO.md`（A4 / A5 勾选）
- 修改：`src/eval/record.py`、`src/eval/evaluate.py`（台账新增 max_steps 列与 `--max-steps`）
- 重建：`results/experiments.csv`（E0 一行 + E1 三行，均带 max_steps）
- 产物：`D:\pt_ner_work\checkpoints\E1_s500\{seed13,seed23,seed33}\`、`D:\pt_ner_work\checkpoints\pilot_S{100,200,300,500}\`

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
Import-Csv results\experiments.csv | Format-Table arm,seed,max_steps,micro_f1,org_f1,ci_low,ci_high
Get-Content docs\runs\2026-11-04_A4_pilot_S.md -TotalCount 12
```
看板 http://127.0.0.1:8765/ 的台账区已能按行显示 max_steps。

### 遗留问题
1. A4 的判据只用了 E1（E2 尚未生成），E2 就绪后必须回验 S。
2. 旧 checkpoint `D:\pt_ner_work\checkpoints\E1\seed13`（S=300）与 4 个 pilot checkpoint 共约 5.8 GB 占用，未删除。
3. ORG 仍是短板（0.738），E2/E3 的生成配额应偏向 ORG。

## 任务 15 — 2026-11-04 清理 TODO + 固定 20 条种子 + Prompt v1 + 校验器

### 做了什么
1. 清理 TODO 里 7 个过期复选框（D1/D2/D3/D9/D10/D17/D20/B3），让状态反映真实进度。
2. A6：写 `src/data/make_seeds.py` 并跑出 **20 条固定种子**。
3. D12：写 `prompts/e2_plain_v1.md`（Prompt v1）。
4. D13：写 `src/generate/validator.py` 并跑 8 个自检用例。
5. C3/C4 落地：不另建实体种子清单；生成模型定为 **DeepSeek Flash**。

### 关键数据
- **种子（`data/curated/seeds_20.jsonl`，种子 42）**：20 条 / 45 实体（PER 17、ORG 18、LOC 10）/ 19 篇文档 / 504 token / 句长 20–30 / 实体 2.25 条每句。约束：必须含实体、每类 ≥3 次、每文档 ≤2 条。
- **发现一条我自己写错的闸门**：原 A6 闸门要求"实体串与 test gold 无重叠"，实测 45 个种子实体里有 5 个（china / câmara / domingos / pmdb / são paulo）与 test gold 重名。这是新闻语料里常见专名的必然重合，**不构成泄漏**。已把判据改为"种子来自 train 侧 + 句级重复为 0"，重叠率 11% 作为披露项。
- **校验器自检 8/8 通过**：合法无实体、合法含实体、要求无实体却给了、实体找不到、标签非法、句长太短、照抄种子、实体重复出现导致区间歧义。
- **Prompt 的关键设计**：不要求模型输出字符下标（LLM 数位置不可靠），只输出实体文本 + 类别，由校验器定位；定位失败或重复歧义即判无效。
- TODO 现状：未完成 21 项、已完成 29 项。

### 变更文件
- 新建：`src/data/make_seeds.py`、`data/curated/seeds_20.jsonl`、`seeds_20_manifest.json`、`seeds_20_stats.md`
- 新建：`prompts/e2_plain_v1.md`、`src/generate/validator.py`
- 修改：`docs/TODO.md`（A6 完成并修正闸门、A7 部分完成、C3/C4 定案、D 组状态清理、B3 关闭）
- 修改：`docs/specs/04-generation-constraints.md`（新增 §6.1 已定生成参数：DeepSeek Flash、prompt v1、温度 0.8、不输出下标）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\data\make_seeds.py      # 重跑应得到同样 20 条
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\generate\validator.py   # 8/8 通过
Get-Content data\curated\seeds_20_stats.md -TotalCount 20
```

### 遗留问题
1. 生成仍缺：DeepSeek Flash 客户端（D16-b，需 API key）、模板化度量剩余三项（D14）、成本台账（D16）。
2. `seeds_20` 一旦用于生成即冻结，不得重抽；若必须改，另建 `seeds_20_v2`。
3. 温度 0.8 是初始值，需试产（G4）确认后冻结。

## 任务 16 — 2026-11-04 采用模板版 Prompt 并解决与负样本规范的冲突

### 做了什么
1. 通读用户提供的 `prompts/E2_E3_Prompt_模板版.zip`（5 个模板 + 2 个数据文件 + 2 个脚本），确认可以应用。
2. 把模板集收进 `prompts/templates/`，做两处最小改动适配本项目：
   - 新增 `prompts/e2_negative.txt`（无实体变体）；
   - `convert_validate.py` 增加 `--expect {auto,entity,none}`。
3. 重写 `src/generate/validator.py`，改为**第二阶段的项目级规范检查**，与 `convert_validate.py` 职责分离。
4. 把 `label_schema.json` 的 4 条标注规则并入 `specs/03` §5.1，`occurrence` 语义写入 §6。
5. 更新 `specs/04` §6.1、`specs/05` §2（改为每批两次请求）。

### 关键数据
- **先纠正一个我自己的错误**：上一轮我说压缩包缺 `convert_validate.py`，实际是存在的（1,943 字节）——我当时用 `Select-Object -Last 12` 把成员列表的第一行截掉了。已核实文件完整。
- **核心冲突**：模板第 9 行要求 "Annotate every target entity, **with at least one per sample**"，与 `specs/05` 的 135 条无实体句（45%）直接冲突；且 `convert_validate.py` 里 `assert x['entities']` 也会拒绝空实体。
- **解法**：每批拆成两次请求 —— 含实体批 11 条（`e2_generation.txt`，`--expect entity`）、无实体批 9 条（`e2_negative.txt`，`--expect none`），15 批合计 165 + 135 = 300。批号分号段 `e2_001…` / `e2_101…`。
- **`occurrence` 设计优于我的旧版**：一句话里出现两次 "Petrobras" 时，可精确指向第 1/2 次非重叠出现；我的旧实现是"重复即判无效"，会误杀合法样本。已按新语义改。
- 端到端验证：正样本批 2/2 通过（`--expect entity`）、无实体批 2/2 通过（`--expect none`）、混批与错用模式均被整批拒绝。项目级检查器 6 个自检用例全过。
- 模板渲染验证：`prepare_prompt.py` 渲染 pos 7,843 字节 / neg 7,914 字节，**占位符全部替换**，无残留 `{...}`。
- 种子一致性：压缩包内 `seeds_20.json` 与项目 `data/curated/seeds_20.jsonl` 的 20 条文本 **20/20 完全一致**，无需对齐。

### 变更文件
- 新建：`prompts/templates/`（10 个文件）、`prompts/templates/prompts/e2_negative.txt`
- 修改：`prompts/templates/convert_validate.py`（`--expect`）、`prompts/templates/prepare_prompt.py`（stage 增加 `e2_negative`）、`prompts/templates/README.md`（改用本项目流程）
- 重写：`src/generate/validator.py`（改为第二阶段项目级检查，支持 JSONL 与 `--expect`）
- 修改：`prompts/e2_plain_v1.md`（标注已被模板集取代）
- 修改：`docs/specs/03-labeling.md`（新增 §5.1 标注工作规则、§6 occurrence）
- 修改：`docs/specs/04-generation-constraints.md` §6.1（模板集、两阶段校验、批号号段）
- 修改：`docs/specs/05-negative-sampling.md` §2（每批两次请求的修订说明）
- 修改：`docs/TODO.md`（A7 进度更新）
- 删除：`prompts/_extracted/`（临时解压目录）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER\prompts\templates"
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
& $py prepare_prompt.py e2_negative --batch e2_101 --n 9 --out req.txt      # 应无残留占位符
& $py convert_validate.py raw.jsonl checked.jsonl --expect none             # 无实体批次
cd ..\..
& $py src\generate\validator.py checked.jsonl --expect none                  # 项目级检查
```

### 遗留问题
1. 仍缺 DeepSeek Flash 客户端（D16-b，需 API key）、成本台账（D16）、模板化度量剩余三项（D14）。
2. `prepare_prompt.py` 不校验渲染结果（例如 `{N}` 未替换时不会报错），靠人工核对；已实测本次无残留。
3. `label_schema.json` 的规则是"依据种子整理的工作规则，尚非官方规范"，与 `pt_bosque` 的 UNER gold 可能存在口径差异，需在人工抽检时留意。

## 任务 17 — 2026-11-04 E2 与 E3 统一使用模板集（补齐 E3 无实体路径）

### 做了什么
确认 E2 与 E3 都走 `prompts/templates/` 同一套模板；E3 原模板同样要求"每条至少一个实体"，因此补齐三个 E3 变体并全部实测渲染。

### 关键数据
- 新增 3 个模板：`e3_negative.txt`（无实体生成，沿用同一份 refined guidance）、`e3_calibrator_negative.txt`（AC 校准无实体批）。加上原有 5 个与 E2 的 `e2_negative.txt`，共 **8 个 stage**。
- `prepare_prompt.py` 的 choices 已扩到 8 个 stage。
- **八个 stage 全部渲染通过**，占位符 0 残留：e2_generation 7,843 / e2_negative 7,916 / e3_guidance 7,952 / e3_refiner 8,104 / e3_generation 7,979 / e3_negative 8,079 / e3_calibrator 8,307 / e3_calibrator_negative 7,895 字节。
- **E3 校准输出端到端**：模拟 keep/revise/reject 三条 → `convert_validate --expect none` 通过 2 条、校准拒绝 1 条；项目级检查 accepted 2 / rejected 0 / no_entity_ratio 1.0。
- 反例仍正确：含实体批错用 `--expect none` → 整批拒绝"本批次要求无实体，但给出了实体"。
- E2/E3 的批次号段：`e2_001…` / `e2_101…`、`e3_001…` / `e3_101…`；两类各 15 批，最终各自 165 + 135 = 300 条。

### 变更文件
- 新建：`prompts/templates/prompts/e3_negative.txt`、`prompts/templates/prompts/e3_calibrator_negative.txt`
- 修改：`prompts/templates/prepare_prompt.py`（choices 扩到 8 个）
- 修改：`prompts/templates/README.md`（八个 stage 总表 + E2/E3 完整流程 + E3 双路径）
- 修改：`docs/specs/04-generation-constraints.md`（§6.1 增加"E2/E3 共用模板"，新增 §6.2 八 stage 与校验模式表）
- 修改：`docs/specs/05-negative-sampling.md` §2（表格扩为 E2/E3 四条请求）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER\prompts\templates"
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
& $py prepare_prompt.py e3_negative --refined refined_guidance.json --batch e3_101 --n 9 --out step3b.txt
Select-String -Path step3b.txt -Pattern "\{[A-Z_]+\}"        # 应无输出
& $py convert_validate.py e3_neg_calibrated.jsonl out.jsonl --expect none
```

### 遗留问题
1. E3 的 guidance / refiner 两步产物（JSON）尚无 schema 校验，目前靠人工看；建议在客户端里加一次结构检查。
2. 仍缺 DeepSeek Flash 客户端（D16-b）、成本台账（D16）、模板化度量剩余三项（D14）——都要 API key。
3. `prompts/` 下仍有两份模板副本（用户原始解压的 `E2_E3_Prompt_模板版/` 与 `templates/`），未清理。

## 任务 18 — 2026-11-04 删除重复模板目录 + README 同步 + 密钥约定

### 做了什么
1. 删除用户自己解压的 `prompts/E2_E3_Prompt_模板版/`（10 个文件），保留 `E2_E3_Prompt_模板版.zip` 作为原始归档。
2. 更新根目录 `README.md`：prompts 一节改写为 templates 的 8 个 stage；更新 validator 的两阶段分工；数据流补上生成与双路校验；新增「八、密钥与凭证」「九、文档维护约定」两节。
3. `.gitignore` 增加密钥兜底：`.env`、`.env.*`、`.secrets/`、`*.key`、`*.pem`、`credentials*.json`。
4. `specs/09-run-records.md` 新增 §5 文档维护约定（README 由 AI 自动同步）。

### 关键数据
- 删除前核对：目标路径 `.../prompts/E2_E3_Prompt_模板版`，确认位于 prompts 下、10 个文件，删除后 `prompts/` 只剩 zip + e2_plain_v1.md + templates/（11 文件）。
- `.gitignore` 兜底实测：`.env` / `.secrets/x` / `t.key` / `credentials.json` 四个探针全部被正确忽略。
- README 现有 10 个一级章节，新增密钥与维护约定两节。

### 变更文件
- 删除：`prompts/E2_E3_Prompt_模板版/`（10 个文件）
- 修改：`README.md`（§3.4 validator 描述、§3.5 prompts 重写、§四 数据流、新增 §八 §九、编号顺延）
- 修改：`.gitignore`（密钥兜底）
- 修改：`docs/specs/09-run-records.md`（§5 文档维护约定）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
Get-ChildItem prompts -Recurse -File | Measure-Object      # 13 个文件（zip + md + templates 11）
git check-ignore -v .env .secrets/x t.key                  # 全部命中
Select-String -Path README.md -Pattern "^## "
```

### 遗留问题
1. **API key 仍未提供**，生成链路（D16-b 客户端、D16 成本台账、D14 剩余度量）无法开工。推荐方式 A：用户自行设 `DEEPSEEK_API_KEY` 环境变量，我只读不打印。
2. 用户询问"直接提供 key 是否有安全问题"，已在 README §八给出三级方案与硬性禁止项。

## 任务 19 — 2026-11-04 建立方法 B 的密钥目录与预检脚本

### 做了什么
1. 在工作区外建立 `D:\pt_ner_work\.secrets\`，放入占位 `deepseek.key`（内容 `sk-REPLACE_ME`）与使用说明 `README.md`。
2. 新增 `src/generate/preflight.py`：读 key（环境变量优先，其次密钥文件），只报来源/长度/格式，**从不打印明文**；`--ping` 可发一次最小请求验证额度。
3. `configs/paths.yaml` 增加 `secrets` 与 `llm` 两节（密钥路径、模型、base_url、温度）。
4. README 第八节改写为"本项目采用方式 B"，并列出验证方式与退出码。

### 关键数据
- 密钥文件：`D:\pt_ner_work\.secrets\deepseek.key`，13 字节，当前为占位符。
- preflight 实测（占位符状态）：来源 `文件 D:\pt_ner_work\.secrets\deepseek.key`、长度 13、前缀判定通过、**正确报出"仍是占位符"与"长度不足"两条问题，退出码 3**。
- preflight 退出码约定：0 通过 / 2 未找到 / 3 格式有问题 / 4 连通失败。
- 踩坑记录：`write` 工具对该工作区外路径报 `EISDIR`（其"临时目录 + 改名"的写盘方式在该路径不适用），改用 shell `Set-Content` 写入成功。

### 变更文件
- 新建（工作区外）：`D:\pt_ner_work\.secrets\deepseek.key`、`D:\pt_ner_work\.secrets\README.md`
- 新建：`src/generate/preflight.py`
- 修改：`configs/paths.yaml`（secrets / llm 两节）
- 修改：`README.md` §八（改为方式 B，附验证方式与退出码）
- 修改：`docs/TODO.md` C4（记录密钥方式 B 与文件路径）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" src\generate\preflight.py
# 填入真实 key 后应返回 0；加 --ping 再验证额度和网络
```

### 遗留问题
1. 用户填入 key 后需跑一次 `preflight.py`（建议加 `--ping`）确认可用。
2. key 就绪后待做：DeepSeek Flash 客户端（D16-b）、成本台账（D16）、模板化度量剩余三项（D14）、E2 试产（G4 闸门）。
3. `configs/paths.yaml` 里新增的 `llm.base_url` 默认写的是 `https://api.deepseek.com`，首次实调时需确认路径与模型名 `deepseek-flash` 是否正确。

## 任务 20 — 2026-11-04 Prompt 全文检查与 6 项修正

### 做了什么
逐条审阅 8 个模板与渲染后的实际 Prompt，发现 4 个会直接影响产量或数据质量的问题（其中 2 个是代码 bug），按建议全部修正。

### 关键数据（问题 → 证据 → 修正）
1. **两把尺子量句长（代码 bug）**：specs 与种子的 20–30 是按 **UD token**（真实数据 24.48/句），而 `validator.py` 按**空格切词**算。20 条种子实测 UD 20–30 ↔ 空格 **15–28**，比值 **0.861**。修正：校验器改为 **17–30 空格词**；specs/04 新增 §2.1 说明换算。
2. **Prompt 没写任何数字**：`e2_generation.txt` 原文只有"参考种子长度但允许自然变化"，实体数一字未提，而校验器硬卡数值。修正：4 个生成模板加入硬约束段（17–30 词、1–5 实体）。
3. **种子本身违反约束**：20 条里 5 条词数 <20、**4 条实体数 >3（最多 5）**。进一步查 E1 真实分布：`1:70 / 2:44 / 3:26 / 4:11 / 5:7 / 6+:7`，**15% 的句子有 4 个以上实体**。修正：实体上界放宽到 **5**（覆盖 E1 的 95%），而不是改种子——卡 3 会让合成数据系统性偏离 E1。
4. **ID 唯一性无保证**：含实体模板只说"用 batch ID 作前缀"，示例是常量 `batch_001`，若逐行复用会让合并后整批失败。修正：明确 `{BATCH_ID}_001` 递增格式。
5. **种子注入带 start/end**：渲染后 7,338 字符里 6,007 是带偏移的种子 JSON，与第 9 行"Do not output character offsets"自相矛盾。修正：注入时剥掉偏移并改紧凑 JSON，**6,007 → 4,661 字符**，整份 Prompt 7,338 → 6,255。
6. **guidance/refiner 头部与正文冲突**：头部要"输出 JSONL 样本"，正文要"一个含 5 个数组的 JSON 对象"。修正：两模板头部改为"本步骤不生成样本"，calibrator 两模板也加了状态信封说明与 17–30 词检查项。

### 验证数据
- 8 个 stage 全部重渲染：**占位符 0 残留、无一处注入 start/end**。
- 校验器自检 6/6 通过（实体超限用例改为 6 个实体）。
- 词数边界实测：16 词拒绝 / 17 ok / 30 ok / 31 拒绝。
- 端到端回归：新 JSONL 格式 `e2_001_001` 递增 id → convert 2/2 通过 → 项目级检查 2 accepted / 0 rejected。

### 变更文件
- 修改：`prompts/templates/prepare_prompt.py`（剥 start/end、紧凑 JSON）
- 修改：`src/generate/validator.py`（17–30 词、实体上界 5、自检用例）
- 重写/修改：`prompts/templates/prompts/` 下 6 个模板（e2_generation、e2_negative、e3_generation、e3_negative、e3_guidance、e3_refiner、e3_calibrator、e3_calibrator_negative）
- 修改：`docs/specs/04-generation-constraints.md`（§2 硬约束表 + §2.1 口径换算 + §2.2 实体上界依据）
- 修改：`README.md`（§五 约定表新增词数与实体数两行）、`prompts/templates/README.md`（注入与硬约束说明）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER\prompts\templates"
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" prepare_prompt.py e2_generation --batch e2_001 --n 11 --out r.txt
Select-String -Path r.txt -Pattern "17 and 30 words|1 and 5 entities|e2_001_001|`"start`""
```

### 遗留问题
1. 种子未重选（按决策放宽了实体上界而非换种子）；若后续抽检发现 4–5 实体的样本质量差，可考虑 `seeds_20_v2`。
2. 仍缺 DeepSeek 客户端（D16-b）、成本台账（D16）、模板化度量三项（D14）。
3. `label_schema.json` 的规则仍是工作规范而非官方规范，抽检时需区分口径差异与生成错误。

## 任务 21 — 2026-11-04 确认种子注入情况 + 记录无实体批次风险

### 做了什么
用户询问 E2 Prompt 是否纳入 20 条真实数据。核实后确认**已全部注入**，同时发现一处不对称，按用户决定"不改模板、试产后再评估"记录为闸门检查项。

### 关键数据
- **20/20 全部注入**：渲染后 `e2_generation` 的 `Real Examples:` 行（4,660 字符）包含全部 20 条种子原文，逐条文本匹配命中 20/20。
- 注入字段：每条只有 `text` + `entities[{text, label}]`，**偏移已剥掉**（上一轮改造）。
- 注入方式：**每批都全量注入 20 条**，不是分批轮换 → 30 次调用 × 4,660 字符 ≈ 14 万字符（约 35k token）种子注入量，成本可接受。
- **发现的不对称**：`seeds_20` 的 20 条全部含实体（选出时要求必须含实体，以示范标注格式），因此 `e2_negative` / `e3_negative` 只能在"看着 20 个含实体例子"的前提下产出零实体句，缺真实范例锚定语体。
- 用户决策：**暂不改模板**，留 G4 试产人工评估。已把三项判据写进 `specs/04` §5.1 与 TODO 的 G4 条目：① 不是"挖空"残缺句；② 语体与含实体批一致；③ 不空泛。任一不过再补 3 条真实无实体范例。

### 变更文件
- 修改：`docs/specs/04-generation-constraints.md`（新增 §5.1 已知风险与三项判据）
- 修改：`docs/TODO.md`（G4 条目补"必须额外检查"三项）
- 修改：`docs/logs/project_log.md`（本节）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" -c "import json;p=open('D:/pt_ner_work/tmp/pos_v3.txt',encoding='utf-8').read();s=[json.loads(l) for l in open('data/curated/seeds_20.jsonl',encoding='utf-8')];print('命中', sum(1 for x in s if x['text'] in p), '/', len(s))"
git log --oneline -1
```

### 遗留问题
1. 无实体批次缺范例的风险未消除，只是延后到试产验证；若试产发现语体漂移，需加 `{NO_ENTITY_EXAMPLES}` 占位符并改两个 negative 模板。
2. 生成链路仍缺 DeepSeek 客户端（D16-b）、成本台账（D16）、模板化度量三项（D14）。

## 任务 22 — 2026-11-04 生成链路完成（客户端 + 流水线 + 台账 + 度量）

### 做了什么
把 D16-b / D16 / D14 三块一次写完，并新增流水线与 E2 入口；把两个模板脚本重构成可 import 的函数（命令行行为不变），避免在沙箱里起子进程（管道会 EPERM）。

| 新模块 | 职责 |
|---|---|
| `src/generate/client.py` | DeepSeek 客户端：key 从环境变量或密钥文件读、POST `/chat/completions`、429/5xx 指数退避、4xx 直接失败、每次响应落盘 `D:\pt_ner_work\raw_responses\`、累计 token |
| `src/generate/pipeline.py` | 单批流程（渲染→调用→逐行解析→两级校验）与"只补不删"的采集循环；**一行坏不废整批**（先丢解析失败行，再严格校验） |
| `src/generate/run_e2.py` | E2 入口：pos 11/批 → 165 条、neg 9/批 → 135 条；`--dry-run` / `--smoke` / `--budget-usd` / `--max-calls` |
| `src/generate/ledger.py` | 生成台账 `gen.csv`（20 列）+ 费用估算 + 汇总 |
| `src/generate/metrics.py` | 实体复用率、句式重复率（骨架去重 + 4-gram 共享率）、两类句长分布（Mann-Whitney）、与种子 8-gram 重叠 |
| `prompts/templates/prepare_prompt.py` | 重构为 `render()` + CLI（命令行用法不变） |
| `prompts/templates/convert_validate.py` | 重构为 `convert_validate()` + CLI（行为、报错文案不变） |

### 关键数据
- 7 个新/改模块 **全部 `py_compile` 通过**。
- `--dry-run` 渲染：含实体 6,224 字符 / 无实体 6,206 字符；**占位符 0 残留、无 start/end 注入、种子各 20/20 命中**。
- 度量自检通过：Mann-Whitney 正常执行（scipy 1.18.1 可用），能给出"长度分布是否有显著差异"的判词。
- 台账写入验证：20 列表头正确，`summarize()` 能算出 accept_rate 与 cost；测试行已清理，`gen.csv` 等正式运行时重建。
- 额度保护：`--max-calls`（默认 60）与 `--budget-usd`（默认 1.0）任一触顶即停；价格未配置时 cost 记 0 但 token 照实记录。

### 变更文件
- 新建：`src/generate/client.py`、`pipeline.py`、`run_e2.py`、`ledger.py`、`metrics.py`
- 重构：`prompts/templates/prepare_prompt.py`、`prompts/templates/convert_validate.py`（均保留原 CLI）
- 修改：`README.md`（src 模块表补 6 项、数据流补 metrics 环节）、`docs/TODO.md`（A7 的 D14/D16/D16-b/D16-c 勾选）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
& $py src\generate\run_e2.py --dry-run      # 只渲染，不花钱
& $py src\generate\metrics.py               # 度量自检
& $py src\generate\preflight.py --ping      # 验证 key 与额度
```

### 遗留问题
1. **尚未真实调用 API**：`--smoke`（各 1 批，约 2 次请求）还没跑，因此 `llm.base_url`、模型名 `deepseek-flash`、返回格式均未实测。
2. `--price-in/--price-out` 默认 0，费用列会显示 0；正式跑前需按账号价格填入（或事后按 token 数换算）。
3. E3 入口 `run_e3.py` 未写（四阶段 + 双路径），等 E2 跑通再补。
4. `--smoke` 产出的 20 条即 G4 试产样本，需人工按 `specs/04` §5.1 三项判据检查。

## 任务 23 — 2026-11-04 生成链路真实调用验证（发现并修复推理模型配额问题）

### 做了什么
跑 `--smoke`（各 1 批）验证链路，暴露出一个隐蔽问题：deepseek-flash 是**推理模型**，thinking 会吃掉输出配额。修复后重跑，链路完全跑通。

### 关键数据
**问题 1（严重）：推理占满配额，正文为空**
| 配置 | reasoning tokens | 正文 | 结果 |
|---|---:|---|---|
| 默认 high，max_tokens 4096 | 4,096 | **0 字符** | `finish_reason=length`，首次 smoke 5 次调用里 **3 次全废** |
| 默认 high，max_tokens 8192 | 6,861–7,418 | 约 800 token | 仍截断，只出 6–7 条 |
| **low，max_tokens 8192** | 1,575–5,805 | 完整 | `finish=stop`，11 条一次出完 |

修法：`client.py` 默认 `reasoning_effort=low`；新增 `TruncatedError` 检测"finish=length 且正文为空"，由流水线记为失败批次（不再静默归零）；台账新增 `reasoning_tokens` 列。

**问题 2：首次 smoke 的长度分布误报**
首轮 pos 只收到 6 条（截断的残批），Mann-Whitney p=0.0051 报"显著差异"。修复后完整批次重跑：pos 22.45 词（18–26）/ neg 22.11 词（20–25），**p=0.7008，无显著差异**，符合 specs/04 §5。此前那条"违反规范"的结论是残批造成的假阳性。

**最终验证结果（2 次调用）**
- e2_0001：prompt 1,875 / completion 6,863（reasoning 5,805）→ 11/11 通过，`finish=stop`
- e2_1001：prompt 1,881 / completion 2,036（reasoning 1,575）→ 9/9 通过，`finish=stop`
- 项目级检查：含实体 11 accepted / 0 rejected；无实体 9 accepted / 0 rejected，no_entity_ratio 1.0
- 质量度量：实体复用率 **0**（没抄种子里的名字）、句式重复率 **0**、种子 8-gram 重叠 **0**
- 类别分布：PER 6 / ORG 13 / LOC 8（ORG 占 48%，E1 是 30%，需在正式跑后复核）
- 脚本返回码：dry-run exit=0，smoke 全流程跑完（终端里的 exit 1 是 PowerShell 对 stderr 的 NativeCommandError 假象）

### 变更文件
- 修改：`src/generate/client.py`（max_tokens 8192、reasoning_effort 默认 low、TruncatedError、reasoning_tokens）
- 修改：`src/generate/pipeline.py`（失败批次记台账并继续；台账加 reasoning_tokens）
- 修改：`src/generate/run_e2.py`（`--max-tokens`、`--reasoning-effort`）
- 修改：`src/generate/ledger.py`（新增 reasoning_tokens 列）
- 修改：`docs/specs/04-generation-constraints.md`（§6.1 加两行、新增 §6.3 推理模型实测表）
- 修改：`README.md`（已知坑新增"deepseek-flash 是推理模型"）、`docs/TODO.md`（A7 记录试产结果 + G4 待人工验收 + 类别平衡观察）

### 验证方式
```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe"
& $py src\generate\run_e2.py --smoke          # 2 次调用，各 1 批
& $py src\generate\metrics.py data\synthetic\e2\e2_pos.jsonl --neg data\synthetic\e2\e2_neg.jsonl
Import-Csv D:\pt_ner_work\results\gen.csv | Format-Table stage,prompt_tokens,completion_tokens,reasoning_tokens,accepted
```

### 遗留问题
1. **G4 人工验收未做**：无实体批次的 9 条需人看是否"挖空/空泛"（specs/04 §5.1）。
2. 价格参数仍为 0，台账 cost 列为 0；需按账号单价填入或将 token 数事后换算。
3. ORG 偏多（48% vs E1 的 30%），正式跑 165 条后要复核类别分布，必要时在 Prompt 或配额上干预。
4. reasoning 长度波动大（1.5k–5.8k），同一配置下成本不稳定；正式跑前建议先跑 3 批估平均开销。
5. E3 入口 run_e3.py 未写。

## 任务 24 — 2026-11-05 E2 数据定稿（三轮迭代）

### 做了什么
按用户指令正式生成 E2 的 300 条合成数据，过程经历三轮迭代与两处判据/代码修正。

### 三轮结果
| 版本 | 含实体批 | 无实体批 | 问题 |
|---|---|---|---|
| v1 | 478 实体 / 2.90 每句 / 23.5 词 | 135 条 / 24.05 词 | 密度比 E1 高 +27%，ORG 43% 偏斜 |
| v2 | 412 / 2.50 / 22.0 词 | 同上 | 两类长度差 −1.7 词，**方向反了** |
| **v3 定稿** | **413 / 2.50 / 22.3 词** | **135 条 / 16.84 词** | **+5.49 词**，类别 124/146/143 接近 E1 |

### 两处关键修正
1. **判据错误（不是数据错误）**：原 specs/04 §5 要求"两类句长无显著差异"，但真实数据 E1 是 +13.9 词、全量 train 是 +9.7 词——无实体句本就是标题/短讯，短很多。改为"差幅与真实数据方向一致"（合格区间 +4~+18），并按类分设长度区间（含实体 20–32、无实体 10–22）。
2. **流水线陈旧产物 bug**：重跑时批号回到 001，旧 `.checked.jsonl` 仍在 → convert_validate 拒绝覆盖 → 每批失败，而 collect() 把旧文件当新产出读回，**表面成功数据未变**。修法：加 `run_tag` 时间戳隔离产物文件名。

### 最终质量
- 实体总数 413（E1 376，+9.8%）、每含实体句 2.50（E1 2.28）、类别 PER 124 / ORG 146 / LOC 143（E1 134/113/129）
- 句式重复率 0、种子 8-gram 重叠 0、实体复用率 0.2%
- 含实体句 22.3 词 vs 无实体 16.8 词，差幅 +5.49（真实 +9.7~+13.9，方向正确、幅度偏小）
- 训练文件 `e2_300.jsonl` 经 to_gliner 转换验证通过

### 变更文件
- 数据：`data/synthetic/e2/{e2_pos,e2_neg,e2_300}.jsonl` + 三个历史版本
- 修改：`src/generate/validator.py`（按类分设长度区间 + 自检夹具）、`src/generate/pipeline.py`（run_tag 隔离）、`src/generate/metrics.py`（长度判据改为差幅）、`src/generate/run_e2.py`（--only、合并 e2_300、对齐检查）
- 模板：`e2_generation.txt`（1–3 实体为主 + 三类均衡 + 24 词目标）、`e2_negative.txt` / `e3_negative.txt`（12–20 词，目标 15）
- 规范：`specs/04` §2 按类分设区间 + 新增 §2.0 说明，§5 判据改写
- 记录：`docs/runs/2026-11-04_E2_generation.md`（三轮完整记录）

### 遗留问题
1. 实体密度仍 +9.8%、含实体句偏短（22.3 vs 29.1 词），需在报告披露。
2. 三轮迭代累计约 75 次调用；价格未配置，费用按 token 事后换算。
3. E3 入口 run_e3.py 未写。

### 任务 24 补充 — 分词口径 bug（丢实体）

`to_gliner` 转换时**丢掉了 147/413 个实体**（35%）。原因：生成文本里标点紧贴单词（`Braga,` `público.`），而合并训练文件时用 `text.split()` 当 token → 实体结束位置落在 token 中间 → 词级区间映射失败、被静默丢弃。

后果若未发现：训练数据会有 1/3 实体没标注，等于教模型"这些不是实体"。

修法：新增 `src/generate/tokenize.py`，按「词 + 独立标点」切分（与 UD 口径一致），并给出字符偏移；`run_e2.py` 合并时改用它。
验证：重建后实体总数 413、边界不对齐 **0**；`to_gliner` 转换 `entities_dropped: 0 / entities_kept: 413`。

## 任务 25 — 2026-11-05 E2 训练 + 步数作为变量（网格扫描）

### 做了什么
1. E2 三种子 @S=500 训练评测。
2. 发现 E2 过拟合，补测 200 步确认 E1/E2 最优步数相反。
3. 按用户决定把**训练步数作为实验变量**，写 `src/train/sweep_steps.py` 跑 {100,200,500} × {13,23,33} 网格（E1/E2 各 9 组，共 18 组）。
4. 改 `specs/06`：S 从"固定值"改为"扫描变量"，并写清报告口径。

### 关键数据
**网格结果（dev @0.80，3 种子均值±std）**
| arm | S=100 | S=200 | S=500 |
|---|---|---|---|
| E1 真实 | 0.7559 ± 0.0130 | 0.7961 ± 0.0244 | **0.8487 ± 0.0045** |
| E2 合成 | 0.7708 ± 0.0148 | **0.8030 ± 0.0009** | 0.7667 ± 0.0094 |
| E2−E1 | **+0.0150** | **+0.0068** | **−0.0820** |

**核心发现：预算越小合成数据越占优，出现交叉。** S=100/200 时 E2 略高于 E1；S=500 时 E1 反超 8.2 点。E2 在 300 步后开始过拟合（训练 loss 0.58 vs E1 1.87）。

**分类别（各自最优）**
| 配置 | P | R | F1 | PER | ORG | LOC |
|---|---|---|---|---|---|---|
| E0 | 0.7680 | 0.7323 | 0.7497 | 0.8360 | 0.6561 | 0.7141 |
| E1 @500 | 0.8093 | 0.8922 | **0.8487** | 0.9340 | 0.7383 | 0.8527 |
| E2 @200 | 0.7756 | 0.8325 | **0.8030** | 0.8968 | 0.7047 | 0.7945 |

两组最优差 4.6 点，CI 不重叠（E1 [0.8313,0.8710] vs E2 [0.7772,0.8237]）。ORG 始终最弱。

### 变更文件
- 新建：`src/train/sweep_steps.py`、`docs/runs/2026-11-05_E2_training_grid.md`、`results/E1_E2_training_grid.md`
- 修改：`docs/specs/06-training-protocol.md`（§2 步数改变量 + 新增 §2.1 依据与口径）
- 数据：`results/experiments.csv` 现有 19 行（E0 1 + E1 9 + E2 9）
- checkpoint：`D:\pt_ner_work\checkpoints\{E1,E2}_sweep\seed*_S*`（18 份，约 21 GB）

### 遗留问题
1. 磁盘：18 份 sweep checkpoint + 先前若干份，D 盘占用需清理（旧的 E1/E2 @500 单份与 probe 可删）。
2. E3 仍未跑；`run_e3.py` 未写。
3. 主表口径已改，`results/E1_E2_training_grid.md` 是新的成果文件，旧 `E0_zeroshot_dev.md` 需补一句"E0 无种子维度"。

## 任务 26 — 2026-11-05 看板日志修复 + A9 对照表 + E3 schema 校验

### 做了什么
1. **修看板日志不更新**：定位到 `logs/` 自 11/4 12:24 后未再写入（改用后台任务后输出没 tee 到日志）。新增 `src/common/runlog.py`（Python 侧 tee + 登记 runs.jsonl），接入 `sweep_steps.py` / `run_e2.py`；看板新增「最近活动」面板显示各产物 mtime。
2. **A9 / D15**：下载 R-GDA 论文 PDF（834 KB / 16 页），pypdf 提取全文，核对 §3.1–§3.3、Algorithm 1、Appendix A（Fig 7–9），产出 `docs/specs/10-rgda-adaptation.md`。
3. **E3 schema 校验**：`src/generate/e3_schema.py`，覆盖 guidance / refiner / calibrator。
4. **TODO 重排**：勾选 A7/A8/A9 与 schema 校验，清理重复条目，A11 改为步数网格口径。

### 关键数据
- 看板 `recent` 面板直接暴露问题：「最新运行日志 1509 分钟前」，其余产物均在 5 分钟内。
- **R-GDA 闸门四项全部确认**：GR 100 分制 5 项（30/15/15/20/20）、AC 100 分制 4 项（30/30/25/15）、最多 **3 轮**、阈值 **90**。
- **D1 偏差**：现有 `e3_refiner` / `e3_calibrator` 模板无打分无循环 → `run_e3.py` 必须补，否则降级命名。
- schema 校验自测 6 例全过（含评分一致性与 Algorithm 1 循环判据 85@1→True / 85@3→False / 92@1→False）。
- TODO 现状：未完成 18 / 已完成 35。

### 变更文件
- 新建：`src/common/runlog.py`、`src/generate/e3_schema.py`、`docs/specs/10-rgda-adaptation.md`
- 修改：`tools/dashboard/{server.py,static/*}`、`src/train/sweep_steps.py`、`src/generate/run_e2.py`、`docs/TODO.md`

### 遗留问题
1. E3 模板需加 100 分制评分与分数输出；`run_e3.py` 需实现 3 轮循环。
2. G4 人工验收未闭环。
3. 磁盘：18 份 sweep checkpoint 未清理。

## 任务 27 — 2026-11-05 E3 编排与模板（补 D1 偏差）

### 做了什么
按 A9 对照表暴露的 D1 偏差，把论文的打分与循环真正实现出来：
1. `e3_refiner.txt`：加入 GR 五准则（30/15/15/20/20）+ 要求输出 score / criteria / feedback / issues_found / revised_guidance / batch_coverage_plan / annotation_checklist。
2. `e3_calibrator.txt` / `e3_calibrator_negative.txt`：加入 AC 四准则（30/30/25/15）+ 逐行 status/score/criteria/issues/sample 信封；无实体批的第 3 项改为"该空的地方确实空"。
3. 新增 `src/generate/run_e3.py`：四阶段 + 双路径 + 两个循环。

### 关键设计
- **GR 循环**：最多 3 轮，`score < 90` 且未满 3 轮就带着上一轮 revised_guidance 再评一次（Algorithm 1）；每轮 raw/score/台账都落盘，轨迹存 `*_gr_trace.json`。
- **AC 循环**：逐条打分，保留 keep/revise，平均分 < 90 且未满 3 轮就把保留样本再校准一轮；`reject` 直接丢弃。
- **终止条件**统一由 `e3_schema.should_continue_loop()` 判定，与论文一致。
- 每阶段都过 `e3_schema` 校验，不合规立即中止（不把坏数据带进下一步）。
- 额度双保险：`--budget-usd`（默认 3）与 `--max-calls`（默认 40）。

### 验证（未花钱）
- `run_e3.py` 编译通过。
- `--dry-run` 渲染四阶段 Prompt：5,632 / 6,475 / 6,474 / 6,380 字符，**7 个占位符残留 0 个**，种子线索正常注入。
- refiner 与 calibrator 的 Prompt 中 `score` / `criteria` / `feedback|issues` 三个字段均存在。

### 变更文件
- 新建：`src/generate/run_e3.py`
- 修改：`prompts/templates/prompts/e3_refiner.txt`、`e3_calibrator.txt`、`e3_calibrator_negative.txt`

### 遗留问题
1. **E3 尚未真跑**：`--smoke` 需花几次调用，等用户确认。
2. G4 人工验收仍未正式闭环（我已提供 4 条抽样，判定通过）。
3. 磁盘：checkpoints 约 65 GB，pilot/probe/smoke 可回收约 40 GB，待用户点头。

## 任务 28 — 2026-11-05 E3 smoke（链路跑通，暴露两个循环的行为问题）

### 做了什么
1. 重构成"生成为校准服务"：每批生成 → AC 循环（≤3 轮）→ 收下合格样本 → 不够再产下一批（原实现把整侧 165 条一次性送 AC，必然截断）。
2. 修三个 bug（见下），跑通 smoke：**8 次调用，11 + 9 = 20 条全部合格**。

### 关键数据
- 含实体 11 条 / 24 个实体 / **区间错误 0**；无实体 9 条。
- 两类长度 pos 20.45 vs neg 14.78，差幅 **+5.68**（合格区间 +4~+18）；句式重复率 0；种子 8-gram 重叠 0。
- 样本质量正常，含 `região de Coimbra` 这种跨词 LOC 边界。

### 发现 1：GR 分数越改越低
三次 smoke 的轨迹：#1 82→78→截断；#2 73→68→74；#3 **77→68→64**。**评分单调下降**，3 轮内从未到 90。
且当前实现用的是**最后一轮**（最低分）的指导，应该用**最高分那一轮**。
推测原因：① 改写后条目变多，Context Abstraction(30) 因冗长被扣；② 把上一轮完整评分对象（含自列 issues）喂回去反而更易被判低；③ 论文阈值 90 基于英文科学生物医学语料，葡语新闻不可直接搬。

### 发现 2：AC 打分虚高，零过滤
6 批 66 条：平均分 97.64–100，**keep 60 / revise 6 / reject 0**。
同一模型在 GR 上给 64–82 分，说明不是模型宽松，而是 AC 四项（领域对齐/语义一致/实体角色/标签格式）在本任务下几乎不可能不满足——生成的句子同源同模板。论文里 AC 主要抓"错误领域标签"（ARTICLE/STATUS），本项目只有 PER/ORG/LOC 且同模板产出，缺这类错误。
**结论：E3 与 E2 的差异主要来自 guidance 一步，AC 几乎不起过滤作用。**

### 三个 bug
1. **输出截断**：GR 第 3 轮 9.4k 字符 + reasoning > 8192 → 模板加"每数组 ≤5 条、每条 ≤20 词"，max_tokens 提到 16384，解析失败改为沿用上一轮而非中止。
2. **格式混用（严重）**：convert_validate 输出 `start/end`，校准器要 `occurrence`，导致 6 批 77 条全部校验失败。新增 `_to_model_format()`；`revise` 输出不完整时退回原样本。
3. **遮蔽标准库**：`src/generate/tokenize.py` 与 stdlib 同名 → scipy 导入即崩。改名 `pt_tokenize.py` 并更新引用。

### 变更文件
- 新建：`docs/runs/2026-11-05_E3_smoke.md`
- 修改：`src/generate/run_e3.py`（批级 AC + 三个 bug 修）、`src/generate/metrics.py`（无实体时 to_markdown 崩溃）、`src/generate/run_e2.py`（引用改名）、`prompts/templates/prompts/e3_refiner.txt`、`e3_calibrator{,_negative}.txt`
- 改名：`src/generate/tokenize.py` → `src/generate/pt_tokenize.py`

### 遗留问题/待决策
1. GR 应取最高分那一轮（当前取最后一轮）——建议改。
2. AC 过滤力弱：是否调严（阈值降到 85 / 强制列问题）或作为记录性检查并在报告说明。
3. 数据仍只有 20 条 smoke，正式 300 条未跑。
