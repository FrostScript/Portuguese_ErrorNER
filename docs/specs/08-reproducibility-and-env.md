# 08 可复现与环境

**状态：** 已定
**相关：** 项目书 §10/§12、修订 V2.0 §9

## 1. 解释器与依赖

| 项 | 值 |
|---|---|
| 解释器 | `C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe` |
| Python | 3.14.7 |
| torch | 2.14.0+cu130（CUDA 可用，RTX 5070 / 12,227 MiB） |
| transformers | 5.17.0 |
| gliner | 0.2.30 |
| seqeval | 1.2.2 |
| 关键约束 | `gliner` 需 `transformers<5.23.0`、`python>=3.10`；当前组合**实测可微调**（见日志任务 5） |

## 2. 运行环境约定

| 项 | 必需设置 | 原因 |
|---|---|---|
| `HF_ENDPOINT` | `https://hf-mirror.com` | `huggingface.co` 被 DNS 污染不可达 |
| `PYTHONIOENCODING` | `utf-8` | 否则打印葡语报 `UnicodeEncodeError`（控制台 GBK） |
| `HF_HOME` | `D:\pt_ner_work\hf_cache` | 缓存不占工作区 |
| 模型本地目录 | `<工作区>\models\gliner_multi-v2.1` | 含权重 + 基座分词器（`microsoft/mdeberta-v3-base` 的 `spm.model` 等） |

**下载方式：** HF Hub API 会被镜像限流（429），模型文件一律走 `/{repo}/resolve/main/{file}` 直链（`src/data/fetch_model.py`）。

## 3. 路径与磁盘分工

| 位置 | 内容 |
|---|---|
| 工作区 `docs/` | 项目书、修订文档、规范、TODO、日志 |
| 工作区 `models/` | GLiNER 权重（1.1 GB） |
| 工作区 `data/` | `pt_bosque/*.jsonl`、`raw_uner/*.iob2` |
| 工作区 `configs/`、`src/` | 配置与代码 |
| `D:\pt_ner_work\checkpoints\` | 训练产物（单份约 1.16 GB） |
| `D:\pt_ner_work\data\synthetic\` | 合成数据与候选池 |
| `D:\pt_ner_work\raw_responses\` | LLM 原始响应 |
| `D:\pt_ner_work\results\` | 预测、指标、台账 |
| `D:\pt_ner_work\hf_cache\` | HF 缓存 |

**唯一配置源：** `configs/paths.yaml`。脚本内不得硬编码路径（过渡期允许 `Path(__file__).parents[n]` 推导）。

## 4. 运行台账

每次训练一行，写到 `D:\pt_ner_work\results\runs.csv`：

```
arm, seed, data_hash, max_steps, batch, lr, threshold,
checkpoint_path, dev_micro_f1, gen_cost_usd, wall_time, gpu_peak_mem
```

每次生成一行，写到 `D:\pt_ner_work\results\gen.csv`：

```
arm, prompt_version, llm_model, temperature, batch_size,
candidates, valid, failed, retries, tokens_used, cost_usd, no_entity_ratio
```

## 5. 必须记录并可复现的项

- GLiNER 模型 ID 与权重版本（含本地目录哈希）
- 数据版本与转换参数（`prepare_uner.py` 的输入文件哈希）
- 20 条种子与 E1 的 300 条样本 ID 清单
- dev 二次切分清单
- LLM 名称、版本、温度、采样参数、Prompt 版本
- 生成数据快照（不得依赖重新调用 LLM）
- 训练超参、随机种子、最优 checkpoint
- 评测脚本版本与阈值

## 6. 已知环境坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 所有命令失败，`--temp is not an existing directory` | 受限沙箱运行器要求 `%TEMP%\dsh-<id>` 预存在 | 建回该目录（见日志任务 4） |
| `CannotCreateTypeConstrainedLanguage` | `workspace-write` 下是 ConstrainedLanguage | 改用 cmdlet，不用 `[System.IO.*]::` 等 .NET 静态调用 |
| `torch.jit.script is not supported in Python 3.14+` | Python 3.14 限制 | 仅 FutureWarning，实测不影响微调 |
| tokenizer 的 `fix_mistral_regex` 提示 | DeBERTa SPM 分词器的误报 | 实测分词正常，忽略 |
| 工作区根出现 `tmp*/`、`torchinductor_*/` | 受限下 `%TEMP%` 不可写，torch 落到 CWD | `src/common/env.py` 把 `TMPDIR/TEMP/TMP` 指向 `D:/pt_ner_work/tmp`，已解决 |
| 写 C 盘工作区外被拒 | 沙箱只放行会话工作区 | 需要时切「完全权限」 |

### 6.1 磁盘写权限实测（2026-11-04）

| 路径 | 受限模式下可写 | 说明 |
|---|---|---|
| 工作区内（含 `data/`、`results/`） | ✅ | 默认放行 |
| `D:\` 整盘（含 `D:\pt_ner_work\*`、`D:\` 根） | ✅ | **已实测 6 次写入全部成功，无需批准** |
| `C:\Users\liush\Desktop\`、`Documents\`、`AppData\Local\` | ❌ | 被拒，需要批准或完全权限 |

**结论：`D:\pt_ner_work` 不需要单独加成工作区**，受限模式下直接可写。此前多次用 `danger-full-access` 属于**过度提升**，不要再用——先按受限模式执行，真被拒了才升级。

**不可行方案（已排除）：** DSH 是"一个会话 = 一个工作区目录"的模型（见 `~/.dsh/storages/workspace.json`，只有路径列表，没有"额外可写根"配置），无法把第二个盘加成同等工作区。用目录联接（junction）指向 D 盘也不必要——直接写就通。
