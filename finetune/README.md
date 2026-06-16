# Finetune 模块

Kronos 模型微调训练管线，支持四种训练模式。

## 目录结构

```
finetune/
├── data/           # 数据 (按归一化方式组织)
├── tokenizer/      # Tokenizer 训练 (单独阶段)
├── predictor/      # Predictor 训练/评估/推理 (四模式)
└── outputs/        # 模型输出
```

## 四种训练模式

| 模式 | 归一化 | 自回归起点 | 窗口 | Loss |
|------|--------|-----------|------|------|
| mode1_original | 全窗口 | t0 | 可变 | CE |
| mode2_ma60_t0 | MA60 | t0 | 可变 | CE |
| mode3_ma60_sliding | MA60 | t60 | lb60/pd5 | CE |
| mode4_ma60_weighted | MA60 | t60 | lb60/pd5 | 加权CE+Delta |

## 快速入口

```bash
# Tokenizer 训练
python finetune/tokenizer/ma60/train.py

# Predictor 训练 (mode4)
python finetune/predictor/mode4_ma60_weighted/train.py --epochs 30

# 评估
python finetune/predictor/mode4_ma60_weighted/eval.py --model outputs/predictors/mode4_xxx/

# 推理
python finetune/predictor/mode4_ma60_weighted/inference.py --model outputs/predictors/mode4_xxx/
```

## 训练流程

1. **数据准备**: 在 `data/` 目录生成或放置数据
2. **Tokenizer 训练**: 先训练 tokenizer (单独阶段)
3. **Predictor 训练**: 使用 tokenizer 训练 predictor
4. **评估**: 在测试集上评估模型
5. **推理**: 使用模型进行预测

## 历史文件

旧的训练脚本和数据保留在 `archive/` 目录，仅用于参考。详见 [ARCHIVE.md](archive/README.md)。
