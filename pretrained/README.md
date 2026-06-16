# Pretrained Models

Kronos 原版预训练模型，从 HuggingFace 下载。

## 模型

| 模型 | 参数量 | 上下文 | 用途 |
|------|--------|--------|------|
| Kronos-mini | 4.1M | 2048 | 微调基座 |
| Kronos-small | 24.7M | 512 | 微调基座 |
| Kronos-base | 102.3M | 512 | 微调基座 |

## Tokenizer

| Tokenizer | group_size | 适用模型 |
|-----------|------------|----------|
| Kronos-Tokenizer-2k | 5 | mini |
| Kronos-Tokenizer-base | 4 | small, base |

## 来源

HuggingFace: https://huggingface.co/NeoQuasar

## 使用

```python
from model import Kronos, KronosTokenizer

# 加载模型
model = Kronos.from_pretrained("pretrained/Kronos-mini")
tokenizer = KronosTokenizer.from_pretrained("pretrained/Kronos-Tokenizer-2k")
```

## 注意

- mini 模型使用 Tokenizer-2k (group_size=5)
- small/base 模型使用 Tokenizer-base (group_size=4)
- 两者不可混用