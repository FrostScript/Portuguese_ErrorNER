# Portuguese_ErrorNER 架构与代码组织说明

**文档定位：** 讲清楚**每个代码文件是干什么的**，以及它们在链路上的位置。不逐函数剖析实现。
**生成日期：** 2026-11-10
**代码基线：** `src/` 26 个 Python 文件 / 3,822 行；全仓 Python 合计 4,298 行
**与其他文档分工：** 研究目标看 `docs/葡萄牙语NER项目书_E0-E3.md`，实现口径看 `docs/specs/`，单次实验看 `docs/runs/`，全局导航看 `README.md`

---

## 一、项目在做什么

以 GLiNER（`urchade/gliner_multi-v2.1`）为基座，在葡语 `pt_bosque` 上比较四种条件下的实体识别效果。标签固定 `PER / ORG / LOC`，评测阈值全项目统一 0.80。

| Arm | 训练数据 | 数据怎么来的 | 数据文件 | 状态 |
|---|---|---|---|---|
| E0 | 无（零样本） | — | — | ✅ 完成 |
| E1 | 300 条真实句 | 从官方 train 里抽 | `data/curated/real_300.jsonl` | ✅ 完成 |
| E2 | 300 条合成句 | LLM 单轮生成 | `data/synthetic/e2/e2_300.jsonl` | ✅ 数据与 9 组训练完成 |
| E3 | 300 条合成句 | LLM + R-GDA（指导→精炼循环→生成→校准） | `data/synthetic/e3/` | ⚠ 仅 smoke 规模（20 条） |

四组共用同一套训练脚本、同一套评测器、同一张台账。这是整个代码组织的第一原则：**只有"数据"这一维是变量，其余环节全部复用**。所以你会看到 `train_arm.py` 有三个 arm 却只有一个文件。

---

## 二、代码怎么分层

六层，依赖严格单向，没有任何反向 import，也没有循环依赖。

```text
L5 展示层     tools/dashboard/          只读消费，不参与计算
        ▲     读 logs/ · results/ · docs/runs/ · checkpoints/
L4 评测层     src/eval/                 输入：checkpoint + dev.jsonl，输出：台账行
        ▲
L3 训练层     src/train/                输入：data/*.jsonl，输出：checkpoints/
        ▲
L2 生成层     src/generate/ + prompts/templates/*.py
        ▲     输入：seeds_20.jsonl + 外部 LLM API，输出：synthetic/*.jsonl
L1 数据层     src/data/                 输入：原始 iob2 / HF 镜像，输出：pt_bosque/ + curated/
        ▲
L0 运行环境   src/common/               每个入口脚本第一行 import
```

读代码时容易踩的三个反直觉点：

1. **`src/common/env.py` 必须第一个 import。** 它在 `import torch` 之前把镜像地址和缓存目录写进环境变量。顺序错了，torch 会把编译缓存灌进仓库根目录。
2. **`configs/paths.yaml` 不是运行时配置。** 全库没有任何代码读它（见第十节 P0-1）。实际路径是各脚本自己拼的。
3. **`prompts/templates/` 不只是一个放文本的目录。** 里面两个 `.py` 被当作顶层模块 import，属于 L2 的代码，不是数据。

---

## 三、目录结构

```text
Portuguese_ErrorNER/
├── README.md                    全局导航（随结构变更同步维护）
├── configs/paths.yaml           路径/模型/标签/LLM 参数声明（⚠ 当前无代码读取）
├── data/                        数据，全部入 git（除 raw_uner/）
│   ├── pt_bosque/               官方三集切分 + 泄漏审计清单
│   ├── curated/                 E1 训练集与生成种子
│   └── synthetic/{e2,e3}/       LLM 合成数据
├── docs/
│   ├── ARCHITECTURE.md          ← 本文件
│   ├── TODO.md                  排期/闸门/待决策
│   ├── 葡萄牙语NER项目书_E0-E3.md  当前有效研究方案
│   ├── specs/                   10 份单点规范
│   ├── runs/                    单次实验快照（7 份）
│   └── logs/project_log.md      任务级变更日志
├── models/                      GLiNER 权重约 1.1 GB（不入库，可重下）
├── prompts/
│   ├── templates/               8-stage 模板集 + 两个 Python 工具
│   ├── e2_plain_v1.md           早期自制 Prompt（已废弃，留档）
│   └── E2_E3_Prompt_模板版.zip   用户提供的原始压缩包（归档）
├── results/                     最终成果 + 实验总台账
├── scripts/run.ps1              Windows 终端包装器（macOS 下不可用）
├── src/                         全部 Python 代码
├── tools/dashboard/             本地实验看板
└── logs/                        运行日志与 runs.jsonl（不入库）
```

仓库外（历史设计，Windows 磁盘分工）：训练产物与 API key 刻意放在 `D:\pt_ner_work\`，与代码分离 —— checkpoint（每份约 1.16 GB）、完整评测 JSON、`gen.csv`、原始 LLM 响应、HF 缓存、torch 临时目录、`.secrets/deepseek.key`。

代码规模分布（`wc -l` 实测）：

| 模块 | 文件数 | 行数 | 占 `src/` |
|---|---:|---:|---:|
| `src/generate/` | 10 | 1,881 | 49% |
| `src/data/` | 6 | 916 | 24% |
| `src/eval/` | 2 | 436 | 11% |
| `src/train/` | 3 | 358 | 9% |
| `src/common/` | 3 | 140 | 4% |
| `src/` 根 | 2 | 91 | 2% |
| **合计** | **26** | **3,822** | **100%** |
| `tools/dashboard/server.py` | 1 | 305 | 不计入 src |
| `prompts/templates/*.py` | 2 | 171 | 不计入 src |

**生成层占了近一半代码**，说明这个项目的工程量主要花在"怎么把 LLM 输出变成能进训练集的数据"，而不是训练本身。所以读代码的优先级是：先 `pipeline.py`，再 `run_e3.py`，训练和评测反而是薄层。

---

## 四、每个代码文件的作用

下面按层列出全部 30 个代码文件（26 个在 `src/`，2 个在 `prompts/templates/`，1 个在 `tools/`，加前端）。"输入 → 输出"一栏就是它需要什么、产出什么，跨文件对接时看这列。

### 4.1 L0 运行环境 `src/common/`（2 个）

| 文件 | 行 | 作用 | 输入 → 输出 |
|---|---:|---|---|
| `env.py` | 43 | 给进程设环境变量：HF 镜像地址、HF 缓存目录、临时目录、UTF-8 编码 | 无 → 进程环境变量 |
| `runlog.py` | 96 | 运行日志同时写终端和文件，并向 `logs/runs.jsonl` 登记这次运行的开始与结束 | 无 → `logs/*.log` + `runs.jsonl` |

**`env.py` 是唯一一个"import 位置有意义"的文件。** 用 `setdefault` 而不是强制赋值，是为了让外部可以通过环境变量覆盖。临时目录指向仓库外，是为了防止 torch 的编译缓存在仓库里堆几百 MB。

**`runlog.py` 当初是为绕一个坑而写的。** 早期只有 `run.ps1` 做日志重定向，但受限沙箱模式禁止执行 `.ps1`，导致一段时间的运行完全没留下日志，看板日志面板一直停在旧文件上。放进 Python 后就与启动方式无关了。用法固定是入口脚本的 `__main__` 里包一层 `with`。

### 4.2 L1 数据层 `src/data/`（6 个）

六个都是**一次性的确定性转换器**：输入原始材料，输出带 sha256 的清单。不参与训练循环，重跑得到字节级一致的结果。

| 文件 | 行 | 作用 | 输入 → 输出 |
|---|---:|---|---|
| `prepare_uner.py` | 205 | 把官方 UNER 标注格式转成项目统一的 JSONL，是全项目数据的总源头 | `raw_uner/*.iob2` → `pt_bosque/{train,dev,test}.jsonl` + `stats.json` |
| `fetch_model.py` | 79 | 下载基座权重和分词器，绕开被限流的 Hub API，走直链 | HF 镜像 URL → `models/gliner_multi-v2.1/` |
| `to_gliner.py` | 75 | 格式适配器：把项目的字符区间标注换成 GLiNER 要的词级闭区间 | 项目 JSONL → GLiNER 训练样本 |
| `make_real_300.py` | 218 | 从 train 里抽 300 条做 E1 训练集（135 无实体 + 165 含实体），固定种子 42 | `train.jsonl` → `curated/real_300.jsonl` + manifest + `stats.md` |
| `make_seeds.py` | 178 | 从 300 条里选 20 条当生成种子，带约束（含实体、长度区间、每类至少出现几次） | `real_300.jsonl` → `curated/seeds_20.jsonl` |
| `make_split_manifest.py` | 161 | 泄漏审计：证明三个集之间没有文档交叠、没有句子重复，落 sha256 作为可复现证据 | 三个 split → `split_manifest.{json,md}` |

三个需要注意的点：

- **`prepare_uner.py` 修过两个会静默毁数据的 bug。** 一是文档 ID 只在每个文档的第一句出现，必须向后传递，否则绝大多数记录 `doc_id` 为空，按文档做的 bootstrap 和划分全部失真；二是三个 split 的文档编号各自从 `batch-0000` 起，不加 split 前缀会让审计误报"全量交叠"。
- **`fetch_model.py` 除了 GLiNER 还要额外抓文件。** GLiNER 仓库本身不含分词器，得再从 `microsoft/mdeberta-v3-base` 补 3 个文件，否则加载时报缺词表。
- **`to_gliner.py` 是 L1 与 L3 的唯一接口。** 字符位置到词位置的映射一旦对不上，它就丢弃该实体并计数（`entities_dropped`），不猜、不硬凑 —— 所以这个计数非零就说明上游有问题。
- **`make_seeds.py` 的产物一旦用于生成就冻结。** 种子变了，E2/E3 的历史数据就不可比。

### 4.3 L2 生成层 `src/generate/`（10 个）+ `prompts/templates/*.py`（2 个）

这是项目最复杂的部分。按角色分两类：**链路组件**（被入口调用）和 **入口**（自己带 `__main__`）。

**链路组件（8 个）**

| 文件 | 行 | 作用 | 输入 → 输出 |
|---|---:|---|---|
| `client.py` | 201 | 与 DeepSeek API 通信的唯一出口：重试、限流退避、原始响应落盘、token 与费用统计 | Prompt → 原始响应 text + usage |
| `pipeline.py` | 220 | 单批流水线：渲染 Prompt → 调 API → 解析 → 校验 → 落盘；外层循环凑够目标条数 | 种子 + 模板 → 合格的 JSONL + 台账行 |
| `validator.py` | 201 | 第二阶段校验，管"能不能进训练集"：句长区间、实体数上限、抄种子、跨批重复、负样本比例 | 候选 JSONL → 通过/拒绝 + 原因 |
| `e3_schema.py` | 166 | E3 三类中间产物（指导 / 精炼 / 校准）的结构校验，以及"要不要再来一轮"的判据 | 模型输出 JSON → 合法对象或报错 |
| `metrics.py` | 261 | 生成质量度量：实体复用率、句式骨架重复率、含实体与无实体句的长度差幅 | 最终 JSONL → `*_metrics.{json,md}` |
| `ledger.py` | 80 | 生成台账，一次调用记一行：批次、token、费用、通过条数、失败原因 | 调用结果 → `gen.csv`（21 列） |
| `preflight.py` | 112 | 跑之前先检查 API key 配好没有。只报来源/长度/前缀/空白字符，不打印明文 | 环境变量或 key 文件 → 退出码 |
| `pt_tokenize.py` | 42 | 分词，把标点切成独立 token，保证实体边界落在 token 边界上 | 文本 → tokens + 对齐后的实体 |

**入口（2 个）**

| 文件 | 行 | 作用 |
|---|---:|---|
| `run_e2.py` | 170 | E2 入口。跑含实体侧 165 条 + 无实体侧 135 条，合并成训练文件，再跑项目级检查和与 E1 的分布对齐检查 |
| `run_e3.py` | 428 | E3 入口，全仓最长文件。四阶段：生成指导 → 精炼循环（≤3 轮）→ 生成 + 按批校准 → 合并度量 |

**`prompts/templates/` 里的两个 Python 文件属于 L2，不是数据**

| 文件 | 行 | 作用 |
|---|---:|---|
| `prepare_prompt.py` | 82 | 把占位符替换成真实内容，拼出完整 Prompt |
| `convert_validate.py` | 89 | 第一阶段校验，只管格式与偏移：ID 与文本唯一、实体位置能定位回原文、实体不重叠 |

几个设计决策值得单独说明，因为它们解释了代码为什么长这样：

- **为什么要有两级校验。** `convert_validate.py` 管"格式对不对"（偏移能不能定位、字段齐不齐），`validator.py` 管"内容符不符合项目规范"（长度分布、重复率）。一级 fail-closed —— 任一行失败就整批不写出；二级逐条判。职责不重叠，所以不能合并。也正因为一级拒绝覆盖已有输出，`pipeline.py` 才必须在文件名里拼运行时间戳（`run_tag`），否则重跑会读到上一轮残留的旧文件。
- **为什么 `client.py` 只对 429 和 5xx 重试。** 其余 4xx 是请求本身有问题，重试只是白烧额度。另外单独区分了"推理把 `max_tokens` 吃光、正文为空"这种情况，因为它在响应里看起来是成功的。
- **为什么注入种子时要剥掉 `start/end`。** 模板里明确要求 "Do not output character offsets"，把偏移喂进去既自相矛盾，又把输入从 4,661 字符推到 6,007 字符。
- **为什么 `run_e3.py` 的校准按批而不是按侧。** 整侧 165 条一次性送进校准器必然爆 `max_tokens`，且评分质量会垮。
- **为什么 `pt_tokenize.py` 存在。** 合成文本里会出现 `Braga,` 这种标点紧贴单词的情况。若直接用空格切词，实体结束位置落在 token 中间，GLiNER 的词级映射会静默丢弃它 —— 实测 413 个实体里丢掉 147 个，且不报错。

### 4.4 L3 训练层 `src/train/`（3 个）

| 文件 | 行 | 作用 | 输入 → 输出 |
|---|---:|---|---|
| `train_arm.py` | 115 | **三个 arm 共用的微调入口。** 从同一个预训练权重独立初始化，不在上一组基础上续训 | `--arm --data --seed --max-steps` → `checkpoints/<arm>/seed<k>/` + `train_config.json` |
| `sweep_steps.py` | 104 | 步数 × 种子网格扫描。已完成的组合自动跳过，可反复运行补漏 | 网格参数 → 多条台账行 |
| `smoke_finetune.py` | 139 | 可行性冒烟，只跑 2 步，验证 Python 3.14 + transformers 5.17 + gliner 0.2.30 这条链路能反传并保存 | 少量样本 → 一个能加载的 checkpoint |

`train_config.json` 不只是配置副本，它记了数据统计、torch/CUDA 版本和完整 loss 曲线，是复现和看板的数据源。

**`sweep_steps.py` 存在的原因不是省事，是必要性**：E1 在 500 步仍未见顶，E2 在 500 步已明显过拟合。步数本身是实验变量，必须扫。

### 4.5 L4 评测层 `src/eval/`（2 个）

| 文件 | 行 | 作用 | 输入 → 输出 |
|---|---:|---|---|
| `evaluate.py` | 357 | 主评测器。一次前向把全部候选 span 和分数存下来，之后在缓存上离线扫 17 档阈值算 P/R/F1，再拆错误类型、按文档重采样算置信区间 | checkpoint + `dev.jsonl` → `<prefix>.{json,md}` |
| `record.py` | 79 | 实验总台账写入器，27 列统一 schema | 评测结果 → `results/experiments.csv` |

**评测器的核心是"候选缓存 + 离线扫阈值"。** 模型只前向一次，此后调阈值、换指标、重算 CI 都不再碰 GPU，秒级就能重出任意阈值下的完整报告。这是整个仓库里最值得抄的设计。错误分析分两类：边界对但类别错（Type Error）、类别对但边界错（Boundary Error）。置信区间按**文档**重采样，不是按句子 —— 同文档的句子不独立。

### 4.6 L5 展示层 `tools/dashboard/`（1 个 Python + 3 个前端文件）

| 文件 | 行 | 作用 |
|---|---:|---|
| `server.py` | 305 | 本地只读 HTTP 服务，只绑 `127.0.0.1:8765`。4 个接口分别给汇总状态、日志、进度、文档；可读路径有白名单，限制在 `logs/ results/ docs/runs/ checkpoints/` 四个目录 |
| `static/index.html` | 69 | 页面骨架，五个区块：运行状态 / 实验台账 / 运行记录 / 实时日志 / checkpoint |
| `static/app.js` | 202 | 前端逻辑：2 秒轮询状态、1.5 秒 tail 日志 |
| `static/style.css` | 118 | 样式 |

**设计约束：看板不修改任何实验脚本。** 它只读产物，实验侧一行代码都不为它让步。页面上还特意摊开了各产物的修改时间，用来识别"哪一环没写文件"。日志读取按 BOM 自动判断编码，是为了兼容 PowerShell 5.1 重定向输出的 UTF-16LE。

### 4.7 `src/` 根（2 个）

| 文件 | 行 | 作用 |
|---|---:|---|
| `__init__.py` | 1 | 让 `from src.xxx import ...` 可用 |
| `smoke_test_e0.py` | 90 | E0 环境冒烟：20 句零样本，确认权重能加载、GPU 可用、预测的字符区间与 gold 口径一致。输出 P/R/F1 但明确标注"仅供冒烟，非正式评测"，避免被误引用 |

### 4.8 非 Python 的代码相关文件

| 路径 | 作用 | 状态 |
|---|---|---|
| `prompts/templates/prompts/*.txt` | 8 个 stage 的 Prompt 模板 | 在用 |
| `prompts/templates/data/label_schema.json` | 注入 `{LABEL_SCHEMA}` 的标签定义 | 在用 |
| `prompts/templates/data/seeds_20.json` | 注入 `{SEED_EXAMPLES}` 的种子 | 在用 |
| `configs/paths.yaml` | 路径/模型/标签/LLM 参数声明 | ⚠ 无代码读取 |
| `scripts/run.ps1` | Windows 日志包装器 | 仅 Windows |
| `prompts/e2_plain_v1.md` | 早期自制 Prompt | 已废弃 |

---

## 五、数据怎么在文件之间流动

只画文件级别的流转，不展开函数。

### 5.1 数据准备链（一次性）

```text
raw_uner/*.iob2
  └─ prepare_uner.py ──► pt_bosque/{train,dev,test}.jsonl
        ├─ make_split_manifest.py ──► split_manifest.{json,md}   泄漏证据
        ├─ make_real_300.py ──────► curated/real_300.jsonl       E1 训练集
        └─                            └─ make_seeds.py ──► curated/seeds_20.jsonl
HF 镜像
  └─ fetch_model.py ──► models/gliner_multi-v2.1/
```

### 5.2 E2 生成链

```text
run_e2.py
  ├─ pipeline.py（含实体侧，165 条）──► e2_pos.jsonl
  ├─ pipeline.py（无实体侧，135 条）──► e2_neg.jsonl
  │     └─ 每批经过：prepare_prompt.py → client.py → convert_validate.py → ledger.py
  ├─ pt_tokenize.py ──► e2_300.jsonl（补 tokens / doc_id / split）
  ├─ validator.py   ──► 项目级检查报告
  └─ metrics.py     ──► e2_metrics.{json,md}
```

### 5.3 E3 生成链（R-GDA 四阶段）

```text
run_e3.py
  [1/4] 生成指导（1 次调用）──────────► guidance.json
        └─ e3_schema.py 校验
  [2/4] 精炼循环（≤3 轮）────────────► refined.json + gr_trace.json
        └─ 每轮用 e3_schema.py 判"分数<90 且轮次<3 才继续"
  [3/4] 生成含实体侧（165 条）──────► e3_pos.jsonl
  [4/4] 生成无实体侧（135 条）──────► e3_neg.jsonl
        └─ 每批：pipeline.py 生成 → 校准循环（≤3 轮）→ convert_validate.py 再过一遍
  ──► e3_300.jsonl ──► e3_metrics.{json,md}
```

两处容易读错的地方：**校准循环里 `keep` 一律用原样本**（模型复述容易丢 id 和位置），**`revise` 若缺字段就退回原样本**（宁可少改也不能丢数据）。校准完还要再过一遍 `convert_validate.py`，因为 revise 可能把偏移改坏。

### 5.4 训练-评测链

```text
train_arm.py --arm E1 --data curated/real_300.jsonl --seed 13 --max-steps 500
  ├─ to_gliner.py 转格式
  └─► checkpoints/E1/seed13/（含 train_config.json）
        │
        ▼
evaluate.py --model <ckpt>/final --split dev --threshold 0.80 --record
  ├─ 无缓存 → 前向一次，落候选缓存
  ├─ 有缓存 → 直接复用，秒级重出报告
  └─► <prefix>.{json,md} + record.py ──► results/experiments.csv
```

`sweep_steps.py` 把这条链自动化：先查台账哪些 `(arm, seed, max_steps)` 已完成，再逐组调 `train_arm.py` 和 `evaluate.py`，失败即跳过继续下一组。

---

## 六、四种数据格式，别混

同一批数据在链路里被转成四种形态。这是整个项目最容易出错的地方。

| # | 形态 | 出现在哪 | 实体怎么表示 | 标签写法 |
|---|---|---|---|---|
| 1 | 项目 JSONL | `data/**/*.jsonl` | `start/end` **字符**区间，左闭右开 | `PER/ORG/LOC` |
| 2 | 模型 I/O | Prompt 与 LLM 请求响应 | `occurrence`（句中第几次出现），无偏移 | `PER/ORG/LOC` |
| 3 | GLiNER 训练格式 | `to_gliner.py` 的输出 | `[start_word, end_word, label]` **词级**闭区间 | `person/organization/location` |
| 4 | 候选缓存 | `evaluate.py` 的 `cache.jsonl` | `[start, end, label, score]`，阈值为 0 的全量候选 | `PER/ORG/LOC`（已回映射） |

谁来转：

```text
形态 2 ──convert_validate.py──► 形态 1      occurrence → 字符区间
形态 1 ──pt_tokenize.py───────► 形态 1+tokens  标点独立成 token
形态 1 ──to_gliner.py─────────► 形态 3      字符 → 词级，对不上就丢弃
形态 1 ──evaluate.py──────────► 形态 4      推理产物
形态 1 ──run_e3.py────────────► 形态 2      校准器输入必须带 occurrence
```

单条形态 1 长这样：

```json
{"id": "batch-0000-0001", "doc_id": "dev:batch-0000", "split": "dev",
 "tokens": ["Pequenos", "são", "..."],
 "text": "Pequenos são ...",
 "entities": [{"start": 0, "end": 8, "text": "Pequenos", "label": "PER"}]}
```

---

## 七、两张台账，不要合并

| 台账 | 位置 | 列数 | 回答什么问题 | 写入者 |
|---|---|---:|---|---|
| 实验总台账 | `results/experiments.csv` | 27 | 这个 arm 效果多少 | `record.py` |
| 生成台账 | `D:/pt_ner_work/results/gen.csv` | 21 | 这 300 条花了多少钱、废了多少 | `ledger.py` |

两者列数不同、用途不重叠，都由代码自动追加，不手工编辑。

---

## 八、现状快照（2026-11-10 实测）

| 数据文件 | 条数 | 说明 |
|---|---:|---|
| `pt_bosque/train.jsonl` | 7,018 | 官方 train |
| `pt_bosque/dev.jsonl` | 1,172 | 官方 dev，兼作模型选择与报告 |
| `pt_bosque/test.jsonl` | 1,167 | 官方 test，尚未启用 |
| `curated/real_300.jsonl` | 300 | E1 训练集 |
| `curated/seeds_20.jsonl` | 20 | 已冻结 |
| `synthetic/e2/e2_300.jsonl` | 300 | E2 正式数据 |
| `synthetic/e2/e2_pos_v1/v2.jsonl` | 165 × 2 | 历史版本，非当前数据 |
| `synthetic/e3/e3_pos.jsonl` / `e3_neg.jsonl` | 11 / 9 | 仅 smoke 规模 |
| `synthetic/e3/e3_300.jsonl` | **20** | ⚠ 名字是 300，实际 20 条 |

台账结果（`experiments.csv`，19 行）：

| Arm | 步数 | Micro-F1（3 种子均值） | 备注 |
|---|---:|---:|---|
| E0 | — | 0.7497 | per 0.8360 / org 0.6561 / loc 0.7141，CI [0.7148, 0.7814] |
| E1 | 500 | **0.8487 ± 0.0045** | 相对 E0 +9.9 点，CI 不重叠 |
| E1 | 200 | 0.7961 | 未见顶，故冻结 500 |
| E1 | 100 | 0.7559 | — |
| E2 | 500 | 0.7667 | 过拟合（低于 E0） |
| E2 | 200 | **0.8030** | E2 的最优步数 |
| E2 | 100 | 0.7708 | — |

橙旗信号：**E2 的最优步数与 E1 不同**（200 vs 500）。跨 arm 比较必须同时给"同预算"和"各自最优"两列，否则结论会被步数选择带偏。

---

## 九、未完成链路

| 环节 | 缺什么 | 卡在哪 |
|---|---|---|
| E3 正式数据 | 300 条（现 20 条） | 需 API key 与预算；代码已具备条件 |
| E3 训练 | 9 组网格 | 依赖上一条 |
| test 评测 | 方案冻结后跑一次 | 依赖 E3 |
| 人工盲审 | 抽样表 + Cohen's κ | `tools/` 下尚无实现 |
| `paths.yaml` 接线 | 全库无引用 | TODO D18 |

---

## 十、已知问题

按"会不会已经造成错误结果"分三级。

### P0 —— 会造成静默错误或数据污染

**P0-1｜`configs/paths.yaml` 完全没有被读取。** 全库 grep 零命中。真实风险不是"配置没生效"这么温和，而是**读者会以为改 yaml 能改路径**，改完发现无效，进而误判某个 arm 跑在旧路径上。要么接线，要么在文件头写明"仅供人工参考"。

**P0-2｜22 处硬编码 `D:/pt_ner_work`，跨 13 个文件，迁移到 macOS 后会污染工作区。** 在 POSIX 上 `Path("D:/pt_ner_work")` 不是绝对路径，而是相对当前目录的字面量。`env.py` 里对临时目录的 `mkdir` 会**在仓库根下真的建出一个叫 `D:` 的目录树**。工作区已迁 macOS 但还没跑过脚本，所以尚未触发 —— 这是迁移期最该先处理的一项。

受影响文件：`common/env.py`、`generate/{client,ledger,metrics,preflight,run_e2,run_e3}.py`、`train/{train_arm,sweep_steps,smoke_finetune}.py`、`eval/evaluate.py`、`data/fetch_model.py`、`tools/dashboard/server.py`。

**P0-3｜`e3_300.jsonl` 名不副实，只有 20 条。** `run_e3.py` 无论 smoke 还是正式都写同一个文件名，smoke 跑完就留下一个 20 条的同名文件。有人直接拿它训练会在 20 条上得到看起来正常的指标。建议 smoke 产物换名，或在 `train_arm.py` 里加最小条数守卫。

### P1 —— 会导致误解或环境绑定

**P1-1｜文档与代码的句长口径不一致。** `README.md` 和 `specs/04` 写 17–30 词，`validator.py` 实际执行 20–32（含实体）/ 10–22（无实体）。代码是对的（按真实数据校准），文档过时。

**P1-2｜`secrets/` 没被 git 忽略，且已有文件被跟踪。** `.gitignore` 写的是 `.secrets/`（带点），仓库里实际是 `secrets/`（不带点）。实测 `git check-ignore` 返回未忽略，`git ls-files` 显示 `secrets/x` 已被跟踪。当前内容只是 2 字节，无害；但一旦放进真 key 就会直接进 git 历史。

**P1-3｜`run.ps1` 与解释器绝对路径绑定 Windows。** `sweep_steps.py` 里解释器路径直接写死成 Windows 绝对路径。迁移后网格扫描跑不起来，需改成当前解释器。

**P1-4｜`README.md` 状态表已漂移。** 仍标 E2/E3「待生成」，实际 E2 已完成。README 自称随结构变更同步，此处未同步。

**P1-5｜`experiments.csv` 的 `no_entity_ratio` 列恒为空。** 列已定义，但评测器从不传值。

### P2 —— 结构冗余，不影响正确性

**P2-1｜跨目录 import 靠 `sys.path` 顺序注入。** 三个文件先把工作区路径、再把模板目录塞进 `sys.path`，把 `prompts/templates/` 当顶层包用。能用，但 IDE 无法解析，且一旦别处出现同名模块会静默串味。

**P2-2｜路径推导样板重复 17 处。** 26 个文件里 17 个重复同样三行引导代码。收敛到 `src/__init__.py` 或统一用 `python -m src.xxx` 可去掉。

**P2-3｜`evaluate.py --record` 默认值与 docstring 不一致。** docstring 说默认写台账，实际不传就不写。行为安全，描述误导。

---

## 十一、建议的阅读顺序

**只想理解方法**：`docs/葡萄牙语NER项目书_E0-E3.md` → `docs/specs/07-evaluation-protocol.md` → `src/eval/evaluate.py`

**要改生成链路**：`prompts/templates/README.md`（8 个 stage 分工）→ `src/generate/pipeline.py` → `src/generate/run_e3.py` → `src/generate/validator.py`

**要做迁移或重构**：本文件第十节 → `src/common/env.py` → 各脚本顶部的路径样板

**只有半小时**：第二节分层图 + 第六节四种格式 + 第十节 P0 三条。这三部分覆盖了"架构长什么样""数据怎么变形""哪里会出错"。
