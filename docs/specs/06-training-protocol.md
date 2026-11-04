# 06 训练协议

**状态：** 部分已定；步数 S 与超参最终值待 pilot（TODO C15）
**相关：** 项目书 §4、修订 V2.0 §4 D1-S、§5

## 1. 统一前提

- 所有可训练组（E1 / E2 / E3）都从**同一个预训练 checkpoint** 独立初始化，不在上一组基础上续训；
- 统一优化器、学习率政策、批大小、最大长度、训练预算、checkpoint 选择规则；
- 任何一组不得单独调参。

## 2. 训练预算（已定原则，数值待定）

| 项 | 值 | 状态 |
|---|---|---|
| 总优化步数 S | **500** | **已冻结（2026-11-04，A4 pilot）** |
| Batch size | 2 | 已定（实测 12 GB 显存充裕） |
| 学习率 | 2e-5（encoder 与其余部分同值） | 已定（未做 lr 扫描） |
| 精度 | FP32（`bf16=False`） | 已定；训练只要 116 s，暂无优化必要 |
| 随机种子 | 13 / 23 / 33 | 已定 |
| 早停 | 无（固定步数，训练后统一评测） | 已定 |

**S=500 的依据（A4 pilot，种子 7）：**

| S | epoch | Micro-F1（dev @0.80） |
|---:|---:|---:|
| 100 | 0.67 | 0.7617 |
| 200 | 1.33 | 0.7990 |
| 300 | 2.00 | 0.8209 |
| **500** | 3.33 | **0.8366** |

曲线单调上升、无掉点，按"取无连续两点下降的最大 S"取网格内最大者。收益递减（300→500 只有 +1.6 点）。

**待回验：** 判据原文要求 E1 与 E2 都不掉点，但 E2 数据尚未生成，本次仅用 E1 定 S。E2 就绪后必须回验一次；若 E2 在 500 步已过拟合，需下调 S 并重跑所有 arm。

**S 的确定方法（已定）：** 用 1 个种子做 pilot，扫 `S ∈ {100, 200, 300, 500}`，取"E1 与 E2 的 Dev Micro-F1 均未出现连续两个评估点下降"的最大 S，写入配置后冻结。

**注意：** 固定 S 时 E1（300 条真实）可能过拟合，需在 pilot 阶段观察 E1 的 dev 曲线；若 E1 在 S 之前就开始掉点，说明 S 定得过大，应下调而不是给 E1 单独减步数（否则预算又不对等）。

## 3. 调用方式（实测确认）

```python
from gliner import GLiNER

model = GLiNER.from_pretrained(str(MODEL_DIR), map_location="cuda")
# 实例真实类型为 UniEncoderSpanGLiNER；GLiNER 本体是元类，类上无 train_model
trainer = model.train_model(
    train_dataset=examples,      # list[dict]
    eval_dataset=None,
    output_dir=out_dir,
    max_steps=S,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    logging_steps=1,
    report_to=[],
)
model.save_pretrained(str(save_dir))
```

**坑：** `inspect.signature(GLiNER.train_model)` 会 `AttributeError`，因为 `train_model` 定义在具体子类上。

## 4. 训练数据格式

```python
[{"tokenized_text": ["Vera", "Lúcia", "Barbosa", "morreu", "em", "São", "Paulo", "."],
  "ner": [[0, 2, "person"], [5, 6, "location"]]}]
```

- 词级下标、**闭区间**（终点 token 计入）；
- 标签用小写 `person / organization / location`；
- 字符区间到词级下标的换算见 `03-labeling.md` §4。

## 5. 产物

| 产物 | 位置 | 说明 |
|---|---|---|
| checkpoint | `D:\pt_ner_work\checkpoints\<arm>\<seed>\` | 单份约 1.16 GB |
| 训练日志 | 同上 | loss 曲线、grad_norm、lr |
| 运行台账 | `D:\pt_ner_work\results\runs.csv` | 见 `08-reproducibility-and-env.md` |

## 6. 已知性能数据

- 2 步微调（batch 2）实测 `train_runtime 49.51 s`，含首次 CUDA 预热；**稳定单步耗时未测**（TODO C20）；
- 若稳定值接近 25 s/步，300 步单次约 2 小时，3 组 × 3 种子约 18 小时机时 —— 排期前必须先测准；
- 单份 checkpoint 1,155.9 MB；9 份约 10.5 GB。

## 7. 报告要求

每组必须报告：实际步数、实际 batch、lr、种子、最优 checkpoint 的 dev F1、训练耗时、GPU 峰值显存、真实样本访问次数。
