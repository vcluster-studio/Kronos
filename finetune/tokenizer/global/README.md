# 全窗口归一化 Tokenizer 训练

训练用于 mode1_original 的 tokenizer。

## 配置

编辑 `config.py` 设置参数。

## 训练

```bash
cd finetune/tokenizer/global
python train.py
```

## 输出

模型保存到 `outputs/tokenizers/global_v1/`。