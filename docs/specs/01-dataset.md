# 01 数据集

**状态：** 已定
**相关：** 项目书 §1、修订 V2.0 §2.1 / §3

## 1. 数据源

| 项 | 值 |
|---|---|
| 数据集 | Universal NER — `pt_bosque` |
| 实际下载源 | `https://raw.githubusercontent.com/UniversalNER/UNER_Portuguese-Bosque/master/pt_bosque-ud-{train,dev,test}.iob2` |
| HF 镜像 | `universalner/universal_ner`（**不可直接用**：`datasets` 5.0.1 已不支持脚本型数据集） |
| 许可 | CC-BY-SA-4.0（原始语料为 UD Portuguese-Bosque） |
| 语言变体 | 巴西葡语为主，含欧洲葡语成分（CETENFolha + CETEMPúblico） |
| 原始格式 | 每 token 一行 `id \t token \t tag \t misc \t annotator`，句间空行，句前有 `# newdoc id` / `# sent_id` / `# text` |
| 原始文件位置 | `data/raw_uner/*.iob2`（4.6 MB） |
| 转换脚本 | `src/data/prepare_uner.py` |

## 2. 规模（实测，与 UNER 论文 Table 1 一致）

| Split | 文档 | 句子 | 实体 | Token | 实体/句 | Token/句 | 无实体句 | 无实体占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 235 | 7,018 | 8,101 | 171,776 | 1.154 | 24.48 | 3,122 | 44.5% |
| dev | 41 | 1,172 | 1,401 | 28,447 | 1.195 | 24.27 | 514 | 43.9% |
| test | 40 | 1,167 | 1,215 | 27,604 | 1.041 | 23.65 | 544 | 46.6% |

类别分布：

| Split | PER | ORG | LOC |
|---|---:|---:|---:|
| train | 2,914 | 2,762 | 2,425 |
| dev | 570 | 417 | 414 |
| test | 442 | 456 | 317 |

## 3. 文本与实体口径

1. **句子文本**取 `# text` 元数据；若缺失则回退 `" ".join(tokens)`（实测 train/dev/test 全部有 `# text`，回退 0 次）。
2. **字符区间**左闭右开 `[start, end)`，由 token 在文本中**按顺序定位**得到；`text[start:end]` 必须等于实体串。
3. **区间自洽性**已全量校验：三集均 0 例失败。
4. 转换后每条记录的字段：

```json
{"id": "batch-0000-0001", "doc_id": "batch-0000", "split": "dev",
 "tokens": ["Pequenos", "são", "..."], "text": "Pequenos são ...",
 "entities": [{"start": 0, "end": 8, "text": "Pequenos", "label": "PER"}]}
```

## 4. 已知数据问题

| 问题 | 数量 | 处理 |
|---|---|---|
| train 内部完全重复句 | 11 | Phase 1 剔除 |
| train & test 完全重复句 | 1（`"Paraíba 1"`） | Phase 1 从 train 侧剔除 |
| train & dev 完全重复句 | 2 | Phase 1 从 train 侧剔除 |
| 44.5% 训练句无实体 | — | 不是问题，是分布事实；见 `05-negative-sampling.md` |

## 5. 硬约束

- 只使用 `pt_bosque` 一个数据集做完 E0–E3；HAREM 只作后续欧洲葡语旁证，不进入主实验。
- 不得用机器翻译语料替代真实葡语测试集。
- 数据文件的哈希与转换参数写入台账（见 `08-reproducibility-and-env.md`）。
