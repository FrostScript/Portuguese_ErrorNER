# 三集划分清单

生成脚本 `src/data/make_split_manifest.py`　生成日期 2026-11-04

## 划分政策

- **dev 不做二次切分**：完整 dev 同时用于 checkpoint 选择与指标报告（2026-11-04 决策）
- 阈值全项目统一 **0.80**
- 取消 872/300 切分的原因：Dev Feedback 只服务于已删除的 Error-Aware；Dev Selection 的阈值搜索用途也已消失

## 规模

| Split | 句子 | 实体 | 文档 | 内部重复句 | sha256 |
|---|---:|---:|---:|---:|---|
| train | 7018 | 8101 | 234 | 11 | `b305f11b1ea32428…` |
| dev | 1172 | 1401 | 40 | 0 | `17784cf71b3dc057…` |
| test | 1167 | 1215 | 39 | 0 | `a51fcd3a4874f4fa…` |

## 文档级互斥（泄漏审计）

| 组合 | 交叠文档数 |
|---|---:|
| train&dev | **0** |
| train&test | **0** |
| dev&test | **0** |

## 句级完全重复

| 组合 | 重复句数 | 示例 |
|---|---:|---|
| train&dev | 2 | Moustafa El-Guindy, 51, professor-titular de a Unicamp:; The Washington Post |
| train&test | 1 | Paraíba 1 |
| dev&test | 0 | — |

## 重复句处理

- 全库重复句总数：**14**（train 内部 11 + 跨集 3）
- E1 训练集生成时已剔除（`data/curated/real_300.jsonl`，剔除内部 11 条、跨集 3 条）
- 合成数据由校验器保证重复率为 0
- dev / test 内部重复为 0

## 文档 ID 清单

### train（234 篇）

```text
train:batch-0000, train:batch-0001, train:batch-0002, train:batch-0003, train:batch-0004, train:batch-0005, train:batch-0006, train:batch-0007, train:batch-0008, train:batch-0009, train:batch-0010, train:batch-0011, train:batch-0012, train:batch-0013, train:batch-0014, train:batch-0015, train:batch-0016, train:batch-0017, train:batch-0018, train:batch-0019, train:batch-0020, train:batch-0021, train:batch-0022, train:batch-0023, train:batch-0024, train:batch-0025, train:batch-0026, train:batch-0027, train:batch-0028, train:batch-0029, train:batch-0030, train:batch-0031, train:batch-0032, train:batch-0033, train:batch-0034, train:batch-0035, train:batch-0036, train:batch-0037, train:batch-0038, train:batch-0039, train:batch-0040, train:batch-0041, train:batch-0042, train:batch-0043, train:batch-0044, train:batch-0045, train:batch-0046, train:batch-0047, train:batch-0048, train:batch-0049, train:batch-0050, train:batch-0051, train:batch-0052, train:batch-0053, train:batch-0054, train:batch-0055, train:batch-0056, train:batch-0057, train:batch-0058, train:batch-0059, train:batch-0060, train:batch-0061, train:batch-0062, train:batch-0063, train:batch-0064, train:batch-0065, train:batch-0066, train:batch-0067, train:batch-0068, train:batch-0069, train:batch-0070, train:batch-0071, train:batch-0072, train:batch-0073, train:batch-0074, train:batch-0075, train:batch-0076, train:batch-0077, train:batch-0078, train:batch-0079, train:batch-0080, train:batch-0081, train:batch-0082, train:batch-0083, train:batch-0084, train:batch-0085, train:batch-0086, train:batch-0087, train:batch-0088, train:batch-0089, train:batch-0090, train:batch-0091, train:batch-0092, train:batch-0093, train:batch-0094, train:batch-0095, train:batch-0096, train:batch-0097, train:batch-0098, train:batch-0099, train:batch-0100, train:batch-0101, train:batch-0102, train:batch-0103, train:batch-0104, train:batch-0105, train:batch-0106, train:batch-0107, train:batch-0108, train:batch-0109, train:batch-0110, train:batch-0111, train:batch-0112, train:batch-0113, train:batch-0114, train:batch-0115, train:batch-0116, train:batch-0117, train:batch-0118, train:batch-0119, train:batch-0120, train:batch-0121, train:batch-0122, train:batch-0123, train:batch-0124, train:batch-0125, train:batch-0126, train:batch-0127, train:batch-0128, train:batch-0129, train:batch-0130, train:batch-0131, train:batch-0132, train:batch-0133, train:batch-0134, train:batch-0135, train:batch-0136, train:batch-0137, train:batch-0138, train:batch-0139, train:batch-0140, train:batch-0141, train:batch-0142, train:batch-0143, train:batch-0144, train:batch-0145, train:batch-0146, train:batch-0147, train:batch-0148, train:batch-0149, train:batch-0150, train:batch-0151, train:batch-0152, train:batch-0153, train:batch-0154, train:batch-0155, train:batch-0156, train:batch-0157, train:batch-0158, train:batch-0159, train:batch-0160, train:batch-0161, train:batch-0162, train:batch-0163, train:batch-0164, train:batch-0165, train:batch-0166, train:batch-0167, train:batch-0168, train:batch-0169, train:batch-0170, train:batch-0171, train:batch-0172, train:batch-0173, train:batch-0174, train:batch-0175, train:batch-0176, train:batch-0177, train:batch-0178, train:batch-0179, train:batch-0180, train:batch-0181, train:batch-0182, train:batch-0183, train:batch-0184, train:batch-0185, train:batch-0186, train:batch-0187, train:batch-0188, train:batch-0189, train:batch-0190, train:batch-0191, train:batch-0192, train:batch-0193, train:batch-0194, train:batch-0195, train:batch-0196, train:batch-0197, train:batch-0198, train:batch-0199, train:batch-0200, train:batch-0201, train:batch-0202, train:batch-0203, train:batch-0204, train:batch-0205, train:batch-0206, train:batch-0207, train:batch-0208, train:batch-0209, train:batch-0210, train:batch-0211, train:batch-0212, train:batch-0213, train:batch-0214, train:batch-0215, train:batch-0216, train:batch-0217, train:batch-0218, train:batch-0219, train:batch-0220, train:batch-0221, train:batch-0222, train:batch-0223, train:batch-0224, train:batch-0225, train:batch-0226, train:batch-0227, train:batch-0228, train:batch-0229, train:batch-0230, train:batch-0231, train:batch-0232, train:batch-0233
```

### dev（40 篇）

```text
dev:batch-0000, dev:batch-0001, dev:batch-0002, dev:batch-0003, dev:batch-0004, dev:batch-0005, dev:batch-0006, dev:batch-0007, dev:batch-0008, dev:batch-0009, dev:batch-0010, dev:batch-0011, dev:batch-0012, dev:batch-0013, dev:batch-0014, dev:batch-0015, dev:batch-0016, dev:batch-0017, dev:batch-0018, dev:batch-0019, dev:batch-0020, dev:batch-0021, dev:batch-0022, dev:batch-0023, dev:batch-0024, dev:batch-0025, dev:batch-0026, dev:batch-0027, dev:batch-0028, dev:batch-0029, dev:batch-0030, dev:batch-0031, dev:batch-0032, dev:batch-0033, dev:batch-0034, dev:batch-0035, dev:batch-0036, dev:batch-0037, dev:batch-0038, dev:batch-0039
```

### test（39 篇）

```text
test:batch-0000, test:batch-0001, test:batch-0002, test:batch-0003, test:batch-0004, test:batch-0005, test:batch-0006, test:batch-0007, test:batch-0008, test:batch-0009, test:batch-0010, test:batch-0011, test:batch-0012, test:batch-0013, test:batch-0014, test:batch-0015, test:batch-0016, test:batch-0017, test:batch-0018, test:batch-0019, test:batch-0020, test:batch-0021, test:batch-0022, test:batch-0023, test:batch-0024, test:batch-0025, test:batch-0026, test:batch-0027, test:batch-0028, test:batch-0029, test:batch-0030, test:batch-0031, test:batch-0032, test:batch-0033, test:batch-0034, test:batch-0035, test:batch-0036, test:batch-0037, test:batch-0038
```

