# 06 训练协议

**状态：** 训练步数已改为**实验变量**（2026-11-05 修订）；其余超参已定
**相关：** 项目书 §4、修订 V2.0 §4 D1-S、§5

## 1. 统一前提

- 所有可训练组（E1 / E2 / E3）都从**同一个预训练 checkpoint** 独立初始化，不在上一组基础上续训；
- 统一优化器、学习率政策、批大小、最大长度、checkpoint 选择规则；
- 训练步数作为自变量统一扫描，不按 arm 单独拍板（见 §2.1）。

## 2. 训练预算

| 项 | 值 | 状态 |
|---|---|---|
| 训练步数 S | **扫描 {100, 200, 500}** | **2026-11-05 修订**（原固定 500，见 §2.1） |
| Batch size | 2 | 已定（实测 12 GB 显存充裕） |
| 学习率 | 2e-5（encoder 与其余部分同值） | 已定（未做 lr 扫描） |
| 精度 | FP32（`bf16=False`） | 已定 |
| 随机种子 | 13 / 23 / 33 | 已定 |
| 早停 | 无（用步数网格代替） | 已定 |

### 2.1 为什么 S 从"固定值"改成"变量"（2026-11-05）

原方案要求所有 arm 统一 S=500（由 E1 单独 pilot 定）。E2 训练后该假设被推翻：

| | 训练 loss | dev F1 @S=500 |
|---|---:|---:|
| E1（真实） | 1.87 | 0.8515 |
| E2（合成） | **0.58** | 0.7578 |

E2 训练 loss 只有 E1 的三分之一、dev 反而低 9 点 —— 300 条合成数据在 500 步已被记住。补测 200 步：E2 → 0.7913，E1 → 0.8191，**两组最优步数相反**。

完整网格（`src/train/sweep_steps.py`，3 步数 × 3 种子，dev @0.80）：

| arm | S=100 | S=200 | S=500 |
|---|---:|---:|---:|
| E1 | 0.7559 | 0.7961 | **0.8487** |
| E2 | 0.7708 | **0.8030** | 0.7667 |

**现行口径：**

1. 训练步数作为自变量进结果表，每格 3 种子均值 ± 标准差；
2. 报"自身最优 S"时必须注明该 S 在 dev 上选出（与"dev 兼作选择与报告"的既有披露一致）；
3. 跨 arm 比较必须同时给**同预算**与**各自最优**两列，不得只挑有利的一列。

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
