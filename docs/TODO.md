# TODO

**项目：** 葡萄牙语命名实体识别（GLiNER + LLM 合成数据，E0–E3）
**更新方式：** 任务推进或决策落地时直接改本文件，勾掉即完成；实测数据写进 `docs/logs/project_log.md`
**规范目录：** 已定规范拆成单文件放在 `docs/specs/`，本文件只记排期与待决策项
**执行纪律（2026-11-04）：** 未收到明确「开始实验」指令前，**不得运行任何训练或评测实验**。数据转换、抽样、下载等确定性脚本不受此限。
**统一阈值：** 0.80（E0–E3 全部适用，见 `specs/07` §2）
**最近更新：** 2026-11-04（重排：E0–E3 主流程进 A 栏，原 A/B/C 顺延为 B/C/D）

---

## A. E0–E3 主实验步骤（按依赖顺序，带闸门）

### 阶段一：数据与基线

- [x] ~~**A1｜数据冻结与切分**~~ — **已完成（2026-11-04）**
  - [x] 剔除重复句：train 内部 11 条、与 dev/test 重复 3 条（去重后 train 池 7,004 条）
  - [x] 生成 E1 训练集 `data/curated/real_300.jsonl`：300 句 = 135 无实体（45%）+ 165 含实体，376 实体（PER 134 / ORG 113 / LOC 129），覆盖 167 文档，种子 42，附 manifest 与 sha256
  - [x] 修复 `doc_id` 传递 bug（原来只有每个文档首句有 doc_id，其余为空串）
  - [x] 修复文档 ID 跨集重名（三个 split 都从 `batch-0000` 起编号）→ 加 split 前缀 `dev:batch-0000`
  - [x] **取消 dev 的 872/300 二次切分**（Dev Feedback 只服务已删除的 Error-Aware；阈值已固定 0.80），完整 dev 兼作选 checkpoint 与报告，偏差写入 limitations
  - [x] 三集文档清单落盘：`data/pt_bosque/split_manifest.{json,md}`（`src/data/make_split_manifest.py`）
  **闸门 G1 结果：** 每类 mention ≥ 30（PER 134 / ORG 113 / LOC 129）✅；文档级交叠 train&dev / train&test / dev&test 全为 **0** ✅；E1 训练集重复句 0 ✅（全库残留 14 条重复句已排除在训练集外）

- [x] ~~**A1-b｜阈值统一为 0.80**~~ — **已定（2026-11-04）**：E0–E3 全部用 0.80，不再逐组搜索；每次评测仍输出 17 档扫描表作敏感性分析。见 `specs/07` §2、`results/README.md`。

- [x] ~~**A2｜E0 零样本基线**~~ — **已完成（2026-11-04）**。全量 dev（1,172 句 / 41 文档），17 档阈值扫描，最优 **阈值 0.80 → Micro-F1 0.7497**（P 0.7680 / R 0.7323），Macro-F1 0.7354，95% CI [0.5679, 0.7952]。产物：`D:\pt_ner_work\results\e0_dev.{md,json}` + 候选缓存。

- [x] ~~**A3｜E0 错误画像**~~ — **已完成（2026-11-04）**，随 A2 一并产出：
  - 分类别 F1：**PER 0.8360 / LOC 0.7141 / ORG 0.6561**（ORG 最弱，比 PER 低 18 点）
  - 支持数：PER 570 / ORG 417 / LOC 414
  - 错误结构：Type Error 45、Boundary Error 45、其余 FP 220、其余 FN 285
  - **阈值效应实测**：默认 0.5 时 F1 仅 0.6843，调到 0.80 后 0.7497 —— **+6.5 点纯阈值收益**，证明 `specs/07` 的阈值搜索是必需项
  - 待办：dev 二次切分（A1）落地后需在 Dev Selection 上重跑，才是 G2 的正式数字

### 阶段二：真实数据基线

- [x] ~~**A4｜微调封装 + pilot 定步数 S**~~ — **已完成（2026-11-04）**。封装 `src/train/train_arm.py`；pilot（种子 7）扫 S ∈ {100,200,300,500}：0.7617 / 0.7990 / 0.8209 / **0.8366**，单调无掉点 → **S 冻结为 500**。见 `docs/runs/2026-11-04_A4_pilot_S.md`、`specs/06` §2。
  **待回验：** E2 数据就绪后需确认 E2 在 500 步不过拟合，否则下调 S 并重跑全部 arm。

- [x] ~~**A5｜E1 训练（300 条真实 × 3 种子）**~~ — **已完成（2026-11-04，S=500）**：seed 13/23/33 → **0.8515 / 0.8436 / 0.8511，均值 0.8487 ± 0.0045**；对比 E0（0.7497）**+9.9 点**，CI 不重叠。分类别均值 PER 0.9340 / ORG 0.7383 / LOC 0.8527。训练 113–120 s。见 `docs/runs/2026-11-04_E1_S500_3seeds.md`。
  **注意：** 训练必须用完整权限（受限沙箱 DataLoader 建命名管道报 WinError 5）；固定 0.80 比扫描最优 0.90 低约 0.4 点，报告中需披露。

### 阶段三：合成数据

- [x] ~~**A6｜20 条种子选取与披露**~~ — **已完成（2026-11-04）**：`src/data/make_seeds.py`（种子 42）→ `data/curated/seeds_20.jsonl`，20 条 / 45 个实体（PER 17 / ORG 18 / LOC 10）/ 19 篇文档 / 504 token / 句长 20–30。附 manifest 与 sha256。**种子从此固定，不得重抽。**
  **闸门修正：** 原写"实体串与 test gold 无重叠"不成立也不应作为判据 —— 实测 45 个种子实体中有 5 个（china / câmara / domingos / pmdb / são paulo）也出现在 test gold，这是新闻语料中常见专名的必然重合，不构成泄漏。真正的判据是：**种子句子全部来自 train 侧（已核实 True）、与 dev/test 句级重复为 0**。重叠率作为披露项报告（5/45 = 11%）。

- [ ] **A7｜生成管线搭建**（进行中 2026-11-04）
  - [x] D12 Prompt → **采用 `prompts/templates/` 模板集**（5 个原始模板 + 新增 `e2_negative.txt` 无实体变体）
  - [x] D13 两阶段校验：`prompts/templates/convert_validate.py`（格式与偏移，新增 `--expect auto/entity/none`）+ `src/generate/validator.py`（句长/密度/抄种子/跨批重复/负样本比例）
  - [x] `label_schema.json` 的 4 条标注规则并入 `specs/03` §5.1；`occurrence` 语义写入 §6
  - [x] D14 模板化度量：`src/generate/metrics.py` —— 实体复用率、句式重复率（骨架 + 4-gram）、两类句长分布（Mann-Whitney）、种子 8-gram 重叠
  - [x] D16 生成成本台账：`src/generate/ledger.py` → `D:\pt_ner_work\results\gen.csv`（token / 候选 / 接受 / 拒绝 / 费用）
  - [x] D16-b DeepSeek 客户端：`src/generate/client.py`（读 key、429/5xx 退避、原始响应落盘）
  - [x] D16-c 流水线：`src/generate/pipeline.py` + `src/generate/run_e2.py`（一行坏不废整批、只补不删、额度保护）
  - [x] 实跑试产（`--smoke`，各 1 批）：**已完成 2026-11-04**，2 次调用、11 + 9 条全部通过，长度分布 p=0.7008（无显著差异）、实体复用率 0、句式重复率 0、种子 8-gram 重叠 0
  - [ ] **G4 人工验收**：看 `data/synthetic/e2/e2_neg.jsonl` 的无实体句是否"挖空/空泛"，见 `specs/04` §5.1
  - [ ] 类别平衡观察：试产 27 个实体里 ORG 13 / LOC 8 / PER 6（ORG 占 48%，E1 是 30%），正式跑后需复核
  **闸门 G4：** 试产 30–50 条人工检查通过率达标；两类各自达标（含实体批 11 条、无实体批 9 条）；无实体句比例可精确控制到 45%。
  **G4 必须额外检查（已知风险）：** 无实体批次没有真实范例（20 条种子全含实体），试产时逐条确认：① 无实体句不是"挖空"的残缺句；② 语体与含实体批一致；③ 没有写得空泛。三项任一不过 → 从 `real_300` 的 135 条真实无实体句中取 3 条作为补充范例注入。详见 `specs/04` §5.1。

- [ ] **A8｜E2 生成 300 条（普通生成）** — 135 条无实体句 + 165 条含实体句，每批 20 条 = 9 + 11；不足只补不删。
  **闸门 G5-a：** 精确 135/165；句长与密度达标；两组句子长度分布无系统差异。

- [ ] **A9｜R-GDA 适配对照表** — 论文环节 / 原参数 / 项目参数 / 实现位置 / 适配原因，至少覆盖 GR 评分标准、AC 校验维度、反馈轮数、停止条件四项。实现见 D14。
  **闸门 G5-b：** 四项任一无法确认 → 方法名称降级为 "R-GDA-inspired"，报告不得写"完整适配"。

- [ ] **A10｜E3 生成 300 条（R-GDA）** — 同种子、同标签规范、同 135/165 配比；产出指导、反馈轨迹、终止原因、成本。

### 阶段四：训练与评测

- [ ] **A11｜E1/E2/E3 统一训练（各 3 种子）** — 同一 S、同一 batch、同一优化器；每组独立从同一预训练权重初始化。实现见 D6/D9。
  **闸门：** 9 份 checkpoint 齐全；台账完整；磁盘约 10.5 GB。

- [ ] **A12｜冻结方案并一次性测 test** — 阈值已固定 0.80（无需再选）；在 test（1,167 句 / 39 篇）上跑一次；bootstrap CI；隐藏方法来源的人工盲审。
  **闸门 G6：** 主表 + 分类别表 + CI + 成本对照 + 盲审 κ 齐全。

- [ ] **A13｜结果分析与交付** — 图 1–5、错误案例、报告正文、PPT；结论边界限定在 `pt_bosque`。

---

## B. 阻塞项与环境

- [x] ~~**B1｜Python 3.14 微调风险**~~ — **已通过（2026-11-04）**：`UniEncoderSpanGLiNER.train_model` 跑通 2 步，不需要降级 Python。
- [ ] **B2｜沙箱临时目录（已修复，留意复发）** — 受限运行器要求 `%TEMP%\dsh-<id>` 预存在，缺失则所有命令失败并提示 `--temp is not an existing directory`。复发时建回该目录；目录已在仍失败则跑 `diagnose-windows-sandbox-acl`。
- [x] ~~**B3｜E0 报告尚未在正式 dev 切分上重跑**~~ — **已随切分取消而失效**（2026-11-04）。dev 不再二次切分，E0 的 0.7497 就是正式数字，CI [0.7148, 0.7814]。

## C. 待你决策

- [x] ~~**C1｜E1 的 300 条抽样口径**~~ — **已定（2026-11-04）**：随机口径 + 45% 无实体句（135 + 165）。
- [ ] **C2｜项目书正文回写** — §1.3 标签冻结、§5.2 句长与密度、§7 负样本比例目前只在 `specs/` 里。
- [x] ~~**C3｜实体种子清单来源**~~ — **已定（2026-11-04）**：不另建实体种子清单。`seeds_20.jsonl` 里的 45 个实体本身就是 Prompt 的实体示例来源，来源可披露（UNER pt_bosque train）。
- [x] ~~**C4｜LLM 选型与预算**~~ — **已定（2026-11-04）**：**DeepSeek Flash**（固定版本，全程不换）。E2 约 15 批 × 2 次请求，预估总 token 数万级、成本 < 1 USD。
  **密钥方式：B**（仓库外文件）——目录与占位文件已建好：`D:\pt_ner_work\.secrets\deepseek.key`，等用户填入真实 key。验证用 `src/generate/preflight.py`（只报来源/长度/格式，不打印明文）。
- [ ] **C5｜修订条款是否回写项目书** — V2.0 的 D1–D8 / N1–N7 / Y1–Y6 是否并入正文。

## D. 实现与工程

### 数据
- [x] ~~D1｜剔除跨集重复句并记录清单~~ — 已在 A1 完成（内部 11 条 + 跨集 3 条），清单见 `split_manifest.md`
- [x] ~~D2｜`real_300.jsonl` 生成脚本~~ — 已完成：`src/data/make_real_300.py`
- [x] ~~D3｜dev 二次切分（872/300）~~ — **已取消**（2026-11-04 决策，见 `specs/02` §2）；文档清单已由 `src/data/make_split_manifest.py` 落盘
- [x] ~~D4｜20 条种子选取脚本 + ID 清单~~ — 已完成：`src/data/make_seeds.py` → `data/curated/seeds_20.*`
- [ ] D5｜风格参考池切分（若生成阶段需要葡语参考文本；当前 Prompt 只用 20 条种子，暂不需要）

### 评测
- [x] ~~D6｜严格实体匹配 + 阈值网格 + 错误分类 + bootstrap CI~~ — **已完成（2026-11-04）**：`src/eval/evaluate.py`。
- [x] ~~D6-b｜固定阈值 + 实验结果台账~~ — **已完成（2026-11-04）**：`evaluate.py --threshold 0.80 --record`，写入 `results/experiments.csv`（`src/eval/record.py`）。
- [ ] D7｜把 evaluate.py 对接 Dev Selection / test 两个 split，参数化 `--split` 并区分候选缓存路径
- [ ] D7-b｜把 `no_entity_ratio` 列填上（评测时从数据集统计，目前为空）

### 训练
- [x] ~~D8｜测稳定单步耗时~~ — 已完成：实测 4.4 步/秒（500 步 113 s），单次训练约 2 分钟
- [x] ~~D9｜微调封装~~ — 已完成：`src/train/train_arm.py`
- [x] ~~D10｜pilot 脚本（扫 S）~~ — 已完成（A4，种子 7 扫四档），方法见 `docs/runs/2026-11-04_A4_pilot_S.md`
- [ ] D11｜运行台账 CSV（训练侧字段；评测侧已由 `results/experiments.csv` + `record.py` 承担，训练侧目前落在 `train_config.json`）

### 生成
- [x] ~~D12｜Prompt v1~~ — 已完成：`prompts/e2_plain_v1.md`（模型只输出实体文本，不输出下标；每批 9 空 + 11 含实体）
- [x] ~~D13｜校验器~~ — 已完成：`src/generate/validator.py`（JSON/标签/定位/歧义/重叠/重复/抄袭/句长/密度/负样本判定），8 个自检用例全过
- [ ] D14｜模板化度量：n-gram 重叠与完全重复已有；**仍缺**实体复用率、句式重复率、两类句子长度分布对比
- [ ] D15｜R-GDA 适配对照表（G5 闸门材料，只卡 E3）
- [ ] D16｜生成成本台账：候选数 / 有效数 / 失败数 / 重试数 / token / 费用
- [ ] D16-b｜DeepSeek Flash 调用客户端（`src/generate/client.py`），需 API key

### 工程
- [x] ~~D17｜`.gitignore`~~ — 已完成（排除 models/、raw_uner/、.idea/、tmp*/、torchinductor_*/、logs/、checkpoints/）
- [ ] D18｜把 `configs/paths.yaml` 接进各脚本（目前仍是脚本内路径推导）
- [ ] D19｜人工盲审工具：抽样表 + Cohen's κ 计算
- [x] ~~D20｜每次实验写运行记录~~ — 已建立（`docs/runs/` 现有 E0、A4、E1 三种子、E1 seed13 历史共 4 份）；后续每次仍在补

## E. 已废弃（留档，不再做）

- [x] ~~E4/E5 Error-Aware 实验线~~ — E0–E3 版项目书删除该方法
- [x] ~~E2+ 成本匹配对照~~ — 用户决定暂不立项
- [x] ~~HAREM 作为主数据集~~ — 改用 `pt_bosque`，HAREM 仅作旁证
- [x] ~~用 `datasets` 库加载 `universalner/universal_ner`~~ — `datasets` 5.0.1 不支持脚本型数据集
- [x] ~~HF Hub API 下载模型~~ — 被镜像限流，改走 `resolve` 直链

---

## 附一：旧编号对照

| 旧 | 新 | 说明 |
|---|---|---|
| A1 Python 3.14 微调风险 | B1 | 已完成 |
| A2 E0 全量评测脚本 | A2 / D6 | 已完成 |
| A3 沙箱临时目录 | B2 | 仍待留意 |
| B1 E1 抽样口径 | C1 | 仍未定，卡 A1 |
| B2 负样本比例 | — | 已定案，见 `specs/05` |
| B3–B6 | C2–C5 | 顺延 |
| C1–C20 | D1–D19（含重排） | E0/E3 流程项已上移至 A 栏引用 |

## 附二：环境事实（避免重复踩坑）

| 项 | 值 |
|---|---|
| 解释器 | `C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe`（Python 3.14.7） |
| 已装关键包 | torch 2.14.0+cu130、transformers 5.17.0、gliner 0.2.30、seqeval 1.2.2 |
| HF 下载 | 必须 `HF_ENDPOINT=https://hf-mirror.com`；`huggingface.co` 被 DNS 污染不可达 |
| 控制台编码 | 必须设 `$env:PYTHONIOENCODING="utf-8"` |
| 磁盘分工 | 工作区放代码/文档/真实数据/模型权重；`D:\pt_ner_work\` 放 checkpoint、合成数据、结果 |
| 磁盘写权限 | **`D:\` 整盘在受限模式下已可写，无需批准、无需加成工作区**（2026-11-04 实测）；只有 C 盘工作区外需要批准。不要再预先用 `danger-full-access` |
| 受限模式限制 | `workspace-write` 下是 ConstrainedLanguage，`.NET` 静态调用会失败，改用 cmdlet |
