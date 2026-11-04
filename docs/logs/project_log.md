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
