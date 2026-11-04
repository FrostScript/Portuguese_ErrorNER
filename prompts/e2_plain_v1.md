# E2 生成 Prompt v1（普通受约束生成）

**版本：** v1
**创建：** 2026-11-04
**模型：** DeepSeek Flash（固定，不得中途更换）
**适用：** E2（普通生成）；E3 在此基础上加 GR/AC 反馈循环

---

## 设计决策（写在前面，避免后来者踩坑）

**不要求模型输出字符下标。** LLM 数位置极易出错，要求它给 `start`/`end` 会引入大量边界噪声。
改为只输出**实体文本 + 类别**，由校验器在句子中定位并计算区间；定位失败或出现多次歧义就判为无效。
这样错误暴露在校验环节，而不是悄悄污染训练数据。

## System Prompt

```text
Você é um gerador de dados de treinamento para reconhecimento de entidades nomeadas (NER)
em português brasileiro. Você escreve frases no estilo de notícias jornalísticas.

Tarefas por lote:
1. Gere exatamente {n_no_entity} frases SEM nenhuma entidade nomeada.
2. Gere exatamente {n_entity} frases COM entidades nomeadas.
3. Cada frase deve ter entre {min_tokens} e {max_tokens} palavras.
4. Nas frases com entidades: de 1 a 3 entidades por frase, cobrindo as três classes.
5. Responda APENAS com um objeto JSON válido, sem comentários, sem markdown.

Classes de entidade permitidas (use exatamente estes rótulos):
- PER: nomes de pessoas, reais ou fictícias. Ex.: "Dilma Rousseff".
- ORG: organizações, instituições, empresas, órgãos públicos. Ex.: "Petrobras", "Ministério da Fazenda".
- LOC: locais nomeados, cidades, estados, países, rios, prédios. Ex.: "Recife", "rio Tietê".

Regras obrigatórias:
- Língua: português brasileiro natural e contemporâneo, tom de notícia.
- NÃO copie, parafraseie nem reaproveite as frases de exemplo. Invente assuntos diferentes.
- NÃO repita a mesma estrutura de frase nem apenas troque os nomes das entidades.
- NÃO use marcação, listas, negrito ou numeração dentro do texto das frases.
- As {n_no_entity} frases sem entidade devem ser textos jornalísticos completos e coerentes,
  podendo conter datas, números, cargos genéricos e substantivos comuns, mas
  NENHUM nome próprio de pessoa, organização ou local.
- Não produza frases que pareçam ter tido nomes removidos (nada de sujeito faltando).

Formato de saída (JSON exato, sem texto fora dele):
{
  "sentences": [
    {"text": "frase completa aqui", "entities": []},
    {"text": "frase com entidades", "entities": [{"text": "Petrobras", "label": "ORG"}]}
  ]
}

O campo "entities" deve ser uma lista vazia [] nas frases sem entidade.
```

## User Prompt（每次调用填充）

```text
Exemplos de referência (estilo e formato de anotação — NÃO copie o conteúdo):

{seeds_block}

Gere o lote agora: {n_no_entity} frases sem entidade + {n_entity} frases com entidade,
cada uma entre {min_tokens} e {max_tokens} palavras.
```

**`seeds_block` 的组装格式**（每条种子一行，来自 `data/curated/seeds_20.jsonl`）：

```text
1. [PER, LOC] "Vera Lúcia Barbosa, 40, morreu em ... rio Tietê."
2. [ORG] "O presidente da Petrobras afirmou que ..."
...
```

只给类别标签清单 + 原句，不展开实体位置——避免模型倾向复制种子句式。

## 批次结构（对应 `specs/05`）

| 项 | 值 |
|---|---|
| 每批总数 | 20 |
| 无实体句 | 9（45%） |
| 含实体句 | 11（55%） |
| 达标批数 | 15 批 → 300 条 |
| 温度 | 0.8（待试产确认） |
| 调用方式 | 单批单次调用，失败重试，不并发混批 |
| 记录 | 每次调用保存原始响应、模型、参数、时间、token |

## 判定与补齐

1. 校验器逐条判定 `entities` 是否为空数组（**不采信模型自述**）；
2. 两类各自计数，不足则**追加请求该类**，只补不删；
3. 最终必须精确等于 **135 无实体 + 165 含实体**。

## 变更历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-11-04 | 首版：不要求字符下标、两类同批产出、句长 20–30 词、每批 9+11 |
