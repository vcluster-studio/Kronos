# Predictor 训练/评估/推理

支持四种训练模式。每种模式有独立的 train.py / eval.py / inference.py。

## 模式对比

| 模式 | 归一化 | 自回归起点 | 窗口 | Loss |
|------|--------|-----------|------|------|
| mode1_original | 全窗口 | t0 | 可变 | CE |
| mode2_ma60_t0 | MA60 | t0 | 可变 | CE |
| mode3_ma60_sliding | MA60 | t60 | lb60/pd5 | CE |
| mode4_ma60_weighted | MA60 | t60 | lb60/pd5 | 加权CE+Delta |

## 目录

| 目录 | 内容 |
|------|------|
| `mode1_original/` | 纯原版训练脚本 |
| `mode2_ma60_t0/` | MA60 + t0自回归 |
| `mode3_ma60_sliding/` | MA60 + 滑动窗口 |
| `mode4_ma60_weighted/` | MA60 + 滑动窗口 + Loss加权 |
| `shared/` | 共享模块 (dataset, metrics, loss, trainer) |

## 使用

```bash
# 训练
python finetune/predictor/mode4_ma60_weighted/train.py --epochs 30

# 评估
python finetune/predictor/mode4_ma60_weighted/eval.py --model outputs/predictors/mode4_xxx/

# 推理
python finetune/predictor/mode4_ma60_weighted/inference.py --model outputs/predictors/mode4_xxx/
```

## 依赖

- 数据: `finetune/data/`
- Tokenizer: `outputs/tokenizers/`
- 共享模块: `finetune/predictor/shared/`
