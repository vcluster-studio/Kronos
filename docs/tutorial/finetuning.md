# 模型微调指南

## 概述

Kronos 提供完整的微调管道，支持在自有数据上适配模型。本指南以中国A股市场为例。

## 微调流程

### 流程概览

1. **配置**: 设置路径和超参数
2. **数据预处理**: 从原始数据生成训练样本
3. **分词器微调**: 适配数据分布
4. **预测器微调**: 针对预测任务优化
5. **评估与回测**: 验证模型性能

---

## 新统一入口（推荐）

### 数据预处理

```bash
python finetune/predictor/preprocess.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode block \
    --validate
```

### 训练

```bash
# 单卡
python finetune/predictor/train.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode block \
    --model mini \
    --epochs 50 \
    --lr 0.01

# 多卡 DDP
torchrun --nproc_per_node=4 \
    finetune/predictor/train.py \
    --norm-mode sliding_ma60 \
    --model mini
```

### 评估

```bash
# 单卡
python finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --n-samples 1000

# 多卡 DDP
torchrun --nproc_per_node=4 \
    finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --model mini
```

### 回测

```bash
python finetune/predictor/backtest.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --n-samples 1000
```

### 关键改进

新入口提供以下改进：
- **去趋势 trajectory IC**: 正确口径，不受价格趋势污染
- **可懂指标三件套**: 方向胜率、振幅误差率、涨跌停命中率
- **Target-based split**: 按 target 区间分割，无数据泄露
- **Runtime 归一化**: pkl 存原始数据，运行时归一化

---

## 配置参考

### 数据配置（DataConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| norm_mode | sliding_ma60 | 归一化模式 |
| lookback | 400 | 回看窗口 |
| predict | 10 | 预测窗口 |
| split_mode | block | 分割模式 |
| clip | 5.0 | 数据裁剪阈值 |

### 训练配置（TrainConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model_type | mini | 模型类型 |
| epochs | 50 | 训练轮数 |
| batch_size | 16 | 批大小 |
| learning_rate | 0.01 | 学习率（从大值起步） |
| early_stopping_patience | 12 | 早停耐心值 |

### 归一化模式

| norm_mode | 含义 |
|-----------|------|
| full_window | 基于 lookback 窗口归一化 |
| sliding_ma20 | MA20 滑动归一化 |
| sliding_ma60 | MA60 滑动归一化 |
| sliding_ma120 | MA120 滑动归一化 |

---

## 旧入口（Legacy）

以下入口已标记为 legacy，建议迁移到新统一入口：

- `finetune/predictor/mode*/train_ddp.py`
- `finetune/predictor/shared/eval_ddp.py`
- `finetune/predictor/simple_backtest.py`
- `finetune/train_tokenizer.py`
- `finetune/train_predictor.py`

旧入口的问题：
- trajectory IC 用原始价格序列计算（口径错误）
- window-index splitting 导致数据泄露
- 度量口径不一致（train/eval 用不同函数）

---

## 生产环境注意事项

### 信号处理

模型输出的是原始预测信号。生产环境中需要：
- 投资组合优化
- 风险因子中性化
- 动态仓位管理

### 回测改进

高质量回测应考虑：
- 交易成本
- 滑点
- 市场冲击

### 数据处理

根据数据源调整数据加载和预处理逻辑。