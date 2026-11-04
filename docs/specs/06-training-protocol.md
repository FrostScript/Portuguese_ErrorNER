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
| 总优化步数 S | 全部可训练组相同 | **待 pilot 确定** |
| Batch size | 2–4（按显存） | 已定范围 |
| 学习率 | 2e-5 为初始候选 | 待 pilot |
| 精度 | BF16（环境支持时） | 待确认 |
| 随机种子 | 3 个配对种子 | 已定 |
| 早停 | 统一规则，不得逐组不同 | 待 pilot |

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
