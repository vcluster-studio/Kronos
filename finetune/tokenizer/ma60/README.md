# MA60 Tokenizer 训练

训练用于 mode2/3/4 的 MA60 滑动归一化 tokenizer。

## 配置

编辑 `config.py` 设置训练参数。

## 训练

```bash
cd finetune/tokenizer/ma60
python train.py
```

## 输出

模型保存到 `outputs/tokenizers/ma60_v1/`。

## 适用模式

| 模式 | 说明 |
|------|------|
| mode2_ma60_t0 | MA60 归一化 + t0 自回归 |
| mode3_ma60_sliding | MA60 归一化 + 滑动窗口 |
| mode4_ma60_weighted | MA60 归一化 + 滑动窗口 + Loss 加权 |