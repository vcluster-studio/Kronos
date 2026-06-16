# Mode 1: 原版 Kronos

全窗口归一化 + t0 开始自回归。

## 特性

| 参数 | 值 |
|------|-----|
| 归一化 | 全窗口 (全局 mean/std) |
| 自回归起点 | t0 (从第一个预测步开始) |
| lookback | 可变 (默认 400) |
| predict | 可变 (默认 10) |
| Tokenizer | Pretrained (HuggingFace) |
| Loss | 标准 CE loss |

## 依赖

- 数据: `finetune/data/global_norm/full_series/`
- Tokenizer: Pretrained `NeoQuasar/Kronos-Tokenizer-base`
- Predictor: Pretrained `NeoQuasar/Kronos-{mini,small,base}`

## 使用

```bash
# 训练 (需新建训练脚本)
python finetune/predictor/mode1_original/train.py --epochs 30

# 评估
python finetune/predictor/mode1_original/eval.py --model outputs/predictors/mode1_xxx/

# 推理
python finetune/predictor/mode1_original/inference.py --model outputs/predictors/mode1_xxx/
```

## 参考

原版示例: `examples/prediction_example.py`
