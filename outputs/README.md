# Outputs 输出目录

训练产物保存位置。

## 结构

```
outputs/
├── tokenizers/       # 训练好的 tokenizer
├── predictors/       # 训练好的 predictor
├── training_logs/    # 训练日志
├── prediction_results/ # 预测结果
└── INDEX.md          # 模型索引
```

## Tokenizers

| Tokenizer | 说明 |
|-----------|------|
| `ma60_tokenizer_v1` | MA60 tokenizer (mini 模型) |
| `ma60_tokenizer_base_v1` | MA60 tokenizer (small/base 模型) |
| `full_tokenizer_2k_v1` | 全量 tokenizer (2k) |

## Predictors

| 目录 | 说明 |
|------|------|
| `mode3_ma60_sliding/` | 滑动窗口模式 (lb60/pd1~pd5) |
| `mode4_ma60_weighted/` | 加权 Loss 模式 |
| `archived/` | 历史模型 (数据泄露问题) |

## 原版模型

原版 Kronos 模型位于 `pretrained/` 目录，不在此处。