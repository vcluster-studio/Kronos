# Mode 2: MA60 + t0 自回归

MA60 滑动归一化 + t0 开始自回归。

## 特性

| 参数 | 值 |
|------|-----|
| 归一化 | MA60 滑动 (局部60步 mean/std) |
| 自回归起点 | t0 (从第一个预测步开始) |
| lookback | 可变 (默认 400) |
| predict | 可变 (默认 10) |
| Tokenizer | 自训练 MA60 tokenizer |
| Loss | 标准 CE loss |

## 依赖

- 数据: `finetune/data/ma60_norm/full_series/` (或 `windowed_lb400_pd10/`)
- Tokenizer: `outputs/tokenizers/ma60_v1/`
- Predictor: `pretrained/Kronos-mini` (微调)

## 使用

```bash
# 训练
python finetune/predictor/mode2_ma60_t0/train.py --epochs 50

# 评估
python finetune/predictor/mode2_ma60_t0/eval.py --model outputs/predictors/mode2_xxx/

# 推理
python finetune/predictor/mode2_ma60_t0/inference.py --model outputs/predictors/mode2_xxx/
```

## 来源

从 `finetune/train_predictor_ma60.py` 整理而来。
