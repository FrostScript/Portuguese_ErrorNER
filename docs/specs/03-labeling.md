# 03 标签与区间约定

**状态：** 已定
**相关：** 项目书 §1.3、修订 V1.1 §10（大小写不一致的修正）

## 1. 标签集

项目冻结为 **3 类**，不裁剪也不扩展：

| 项目标签 | UNER 原始标签 | 模型推理用词（prompt label） | 中文 |
|---|---|---|---|
| PER | `B-PER` / `I-PER` | `person` | 人物 |
| ORG | `B-ORG` / `I-ORG` | `organization` | 组织机构 |
| LOC | `B-LOC` / `I-LOC` | `location` | 地点 |

**两套写法不可混用：**

- 数据文件、指标表、报告正文 → 用大写 `PER / ORG / LOC`；
- 传入 GLiNER 的 `labels=[...]`、训练数据 `ner` 三元组里的标签 → 用小写 `person / organization / location`。

这条是硬约定：`gliner_multi-v2.1` 实测用小写词效果正常（`person` 0.996 / `location` 0.982），若某处写成 `PER` 会造成模型不认识该类别而不是报错。

## 2. 原始标签清洗

- `B-O` → `O`（UNER 官方脚本同样的处理）
- 含 `OTH` 的标签 → `O`（UNER 的 Other 类不进最终数据）
- 其余标签必须落在上表三种之内，出现其他值直接报错，不得静默丢弃

## 3. 区间约定

| 场景 | 约定 | 示例 |
|---|---|---|
| 项目 JSONL（`data/pt_bosque/*.jsonl`） | **字符**区间，左闭右开 `[start, end)` | `{"start": 0, "end": 6, "text": "Lisboa"}` |
| GLiNER 训练数据 | **词级**下标，闭区间 `[start_word, end_word, label]` | `[0, 0, "person"]` |
| GLiNER 推理输出 | 字符区间，左闭右开，与项目 JSONL 一致 | `{"start": 0, "end": 6, "text": "Lisboa", "score": 0.83}` |

**不变量：** 任意区间必须满足 `text[start:end] == entity["text"]`。三集全量校验已通过，0 例失败。

## 4. 字符区间 → 词级下标的换算

由 `src/train/smoke_finetune.py` 实现，规则：

1. 按 token 在 `text` 中顺序定位，得到每个 token 的起始字符偏移 `offsets[i]`；
2. 实体起点 = 满足 `offsets[i] == start` 的 token 下标；
3. 实体终点 = 满足 `offsets[i] + len(tokens[i]) == end` 的 token 下标；
4. 任一端找不到则丢弃该实体并计数（当前数据 0 例）。

## 5. 实体边界规则

- 多词实体保持完整跨度，不得拆成多个单 token 实体；
- 嵌套与重叠：本项目数据无嵌套，生成数据出现重叠时按"完全包含则保留最长、部分重叠则整条样本丢弃"处理（待生成阶段验证后可能调整）；
- 不为省事把 `de o` / `da` 这类 UD 拆分成分合并——文本与 token 一律按 UD 原样保留（`# text` 里本身就是分开写的）。

## 5.1 合成数据的标注工作规则（2026-11-04 并入）

来源：`prompts/templates/data/label_schema.json`，由 20 条种子的观察整理而成。

| # | 规则 |
|---|---|
| 1 | 完整标注全部目标实体；**普通职业、泛称、时间及数量不单独标注** |
| 2 | 连续非重叠标注；完整机构与地点**不再嵌套拆分** |
| 3 | 保留实体内部必要字符，**排除外部标点** |
| 4 | **作品名本身不设标签**；避免尚未确认的歧义场景 |

**限定：** 这四条是"依据当前种子整理的工作规则，尚非已核实的官方语料规范"（原文 status 字段）。使用时的风险是：`pt_bosque` 的 gold 按 UNER 官方指南标注，若生成数据的规则与 gold 不一致，会人为制造"标注体系不匹配"的噪声。因此：

- 规则 1–4 只用于**生成侧的一致性控制**，不改动已有 gold；
- 若后续人工抽检发现某些类别系统性不一致，须记录为"标注口径差异"，而不是当作生成器的错误。

## 6. occurrence 字段（合成数据专用）

`/prompts/templates/` 的模板不要求模型输出字符下标，而要求给 `occurrence`：

> one-based non-overlapping occurrence of the exact substring, counted left to right

含义：实体串在同一句里第 k 次**非重叠**出现，从 1 开始数。由 `prompts/templates/convert_validate.py` 按 `text.find(v, prev_end)` 迭代定位换算成 `[start, end)`。

例：`"Petrobras cresceu; Petrobras investiu."` 中两处 `Petrobras` 分别标 `occurrence: 1` 与 `occurrence: 2`。

**为什么这样做：** 一句话里出现同名实体时，只给文本无法唯一确定区间；直接判"重复即无效"会丢掉大量合法样本。用 occurrence 精确指向第几次出现，既保住样本又不引入歧义。
