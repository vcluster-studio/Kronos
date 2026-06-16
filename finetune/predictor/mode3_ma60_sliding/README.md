# Mode 3: MA60 + 滑动窗口自回归

MA60 滑动归一化 + 滑动窗口自回归。

## 特性

| 参数 | 值 |
|------|-----|
| 归一化 | MA60 滑动 (局部60步 mean/std) |
| 自回归起点 | t60 (输入 t0~t59，从 t60 开始生成) |
| lookback | 60 |
| predict | 1 或 5 |
| Tokenizer | 自训练 MA60 tokenizer |
| Loss | 标准 CE loss |

## 依赖

- 数据: `finetune/data/ma60_norm/windowed_lb60_pd1/` (或运行时从 full_series 切分)
- Tokenizer: `outputs/tokenizers/ma60_v1/`
- Predictor: `pretrained/Kronos-mini` (微调)

## 使用

```bash
# 训练
python finetune/predictor/mode3_ma60_sliding/train.py --epochs 30 --predict 5

# 评估
python finetune/predictor/mode3_ma60_sliding/eval.py --model outputs/predictors/mode3_xxx/

# 推理
python finetune/predictor/mode3_ma60_sliding/inference.py --model outputs/predictors/mode3_xxx/
```

## 来源

从 `finetune/train_single_step.py` 整理而来。
