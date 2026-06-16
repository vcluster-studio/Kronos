# Mode 4: MA60 + 滑动窗口 + Loss 加权

MA60 滑动归一化 + 滑动窗口自回归 + 样本权重 + 条件 delta loss。

## 特性

| 参数 | 值 |
|------|-----|
| 归一化 | MA60 滑动 (局部60步 mean/std) |
| 自回归起点 | t60 (输入 t0~t59，从 t60 开始生成) |
| lookback | 60 |
| predict | 5 |
| Tokenizer | 自训练 MA60 tokenizer |
| Loss | 加权 CE + Delta loss + Horizon 衰减 |

## Loss 配置

- **样本权重**: `w = clip(abs(actual_return) / 0.02, 0.2, 2.0)`
- **Delta loss**: 仅对 `|actual_return| > 2%` 的样本生效
- **Horizon 衰减**: 远距离预测权重衰减 `gamma^i`

## 依赖

- 数据: `finetune/data/ma60_norm/windowed_lb60_pd1/` (或运行时从 full_series 切分)
- Tokenizer: `outputs/tokenizers/ma60_v1/`
- Predictor: `pretrained/Kronos-mini` (微调)

## 使用

```bash
# 训练
python finetune/predictor/mode4_ma60_weighted/train.py --epochs 30 --lambda1 0.05

# 评估
python finetune/predictor/mode4_ma60_weighted/eval.py --model outputs/predictors/mode4_xxx/

# 推理
python finetune/predictor/mode4_ma60_weighted/inference.py --model outputs/predictors/mode4_xxx/
```

## 来源

从 `finetune/train_single_step_v3.py` 整理而来。
