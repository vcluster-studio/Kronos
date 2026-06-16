# Tokenizer 训练

Tokenizer 训练与 Predictor 训练是两个独立阶段。

## 目录

| 目录 | 归一化方式 | 适用 Predictor 模式 |
|------|-----------|-------------------|
| `global/` | 全窗口归一化 | mode1_original |
| `ma60/` | MA60滑动归一化 | mode2/3/4 |
| `shared/` | - | - |

## 流程

1. 准备数据 (在 `finetune/data/` 目录)
2. 配置参数 (编辑 `config.py`)
3. 训练 tokenizer: `python finetune/tokenizer/{type}/train.py`
4. 训练结果保存在 `outputs/tokenizers/{name}/`
5. 使用训练好的 tokenizer 训练 predictor

## 输出

```
outputs/tokenizers/
├── global_v1/    # 全窗口 tokenizer
│   ├── config.json
│   ├── model.safetensors
│   └── README.md
├── ma60_v1/      # MA60 tokenizer
│   ├── config.json
│   ├── model.safetensors
│   └── README.md
```
