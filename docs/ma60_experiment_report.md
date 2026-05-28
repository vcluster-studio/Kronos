# MA60 微调实验报告

## 一、实验背景

Kronos 原始预训练模型使用 full_window 归一化（整个 lookback 窗口的均值/标准差），在 A 股数据上表现不佳。我们提出 sliding MA60 归一化策略（每个时间点使用前 60 步滚动均值/标准差），并基于此进行微调。

### 归一化方式对比

| 方式 | 说明 | 优势 | 劣势 |
|------|------|------|------|
| full_window | 整个 lookback 窗口统一归一化 | 简单 | 未来信息泄露，不适应趋势变化 |
| sliding MA60 | 每点用前 60 步滚动统计量 | 适应趋势变化，可在线计算 | 短窗口统计量噪声大 |

## 二、模型配置

### 模型规格

| 模型 | 参数量 | 层数 | d_model | heads | group_size | max_context |
|------|--------|------|---------|-------|------------|-------------|
| Kronos-mini | 4.1M | 4 | 256 | 4 | 5 | 2048 |
| Kronos-small | 24.7M | 8 | 512 | 8 | 4 | 512 |
| Kronos-base | 102.3M | 12 | 832 | 16 | 4 | 512 |

### Tokenizer 匹配关系

| Tokenizer | group_size | 压缩比 | 适用模型 |
|-----------|------------|--------|----------|
| Kronos-Tokenizer-2k | 5 | 5:1 | mini |
| Kronos-Tokenizer-base | 4 | 4:1 | small, base |

### MA60 微调配置（mini 版本）

| 参数 | V1 | V2 |
|------|------|------|
| epochs | 30 | 30 |
| batch_size | 16 | 16 |
| learning_rate | 0.02 | 0.02 |
| weight_decay | 0.1 | 0.1 |
| lr_scheduler | vl_adaptive | vl_adaptive |
| direction_loss_weight | 0.3 | 0.3 |
| lookback | 400 | 400 |
| pred_len | 10 | 10 |
| ic_test_samples | 200 | 200 |

### 训练结果（mini）

| 模型 | 训练 IC | 测试 IC | 训练-测试差距 | 过拟合程度 |
|------|---------|---------|--------------|-----------|
| V1 (best_ic) | 0.2602 | 0.2289 | -12% | 轻微 |
| V2 (best_ic) | 0.2944 | 0.1652 | -44% | 严重 |
| V1 (best_loss) | - | 0.0836 | - | - |
| V2 (best_loss) | - | 0.1680 | - | - |

**结论：V2 过拟合严重，V1-IC best 是 mini 上的最优方案，但仍有 12% 的泛化损失。**

## 三、全市场对比测试

### 测试配置

- 测试数据：2555 只 A 股，411 时间步
- IC 计算点：P+3（第 3 天收益率）
- lookback=400, pred_len=10

### 3.1 Pretrained 模型 vs MA60 微调模型

| 模型 | 归一化 | IC | Rank IC | 方向准确率 | MSE |
|------|--------|------|---------|-----------|------|
| Kronos-mini | full_window | -0.0031 | 0.0078 | 52.02% | 0.004232 |
| Kronos-small | full_window | -0.0955 | -0.0257 | 54.56% | 0.062843 |
| Kronos-base | full_window | -0.1767 | -0.0963 | 45.87% | 0.026998 |
| **MA60-mini-IC** | **sliding MA60** | **0.2116** | **0.1716** | **60.20%** | **0.002327** |

### 3.2 MA60 内部对比（5 组模型）

| 模型 | IC | Rank IC | 方向准确率 | MSE |
|------|------|---------|-----------|------|
| Original (pretrained) | 0.0100 | -0.0048 | 46.46% | 0.011510 |
| **V1-IC (best)** | **0.2289** | **0.1861** | **59.49%** | **0.002214** |
| V2-IC | 0.1652 | 0.1535 | 55.54% | 0.002489 |
| V1-Loss | 0.0836 | 0.0761 | 55.23% | 0.002753 |
| V2-Loss | 0.1680 | 0.1314 | 54.83% | 0.002730 |

### 3.3 Small 模型 MA60 微调

#### 训练过程

**V1 训练**（LR=0.02, 从 pretrained 开始）：

| Epoch | IC | 备注 |
|-------|------|------|
| 1 | 0.0605 | |
| 2 | 0.1103 | |
| **3** | **0.1819** | **best IC (saved)** |
| 4 | 0.0620 | IC 崩塌 |
| 5-8 | -0.03~0.01 | 持续低迷 |

V1 训练在 Ep3 达到 IC=0.1819 后迅速崩塌，LR 过高导致过冲。

**V2 训练**（LR=0.008, 从 V1 best_ic_model 恢复）：

| Epoch | IC | VL-Adaptive | 备注 |
|-------|------|-------------|------|
| 1 | -0.0017 | KEEP | |
| 2 | -0.0140 | DECAY | |
| 3 | 0.0791 | KEEP | saved |
| 4 | 0.1254 | ROLLBACK | saved |
| 5 | 0.0085 | ROLLBACK | |
| 6 | -0.0443 | ROLLBACK | |
| **7** | **0.2065** | **ROLLBACK** | **best IC (saved)** |
| 8 | 0.1940 | ROLLBACK | |
| 9 | 0.1420 | ROLLBACK | |
| 10 | -0.0100 | ROLLBACK | |
| 11-16 | -0.02~0.10 | ROLLBACK | 持续低迷 |
| 17 | 0.0443 | - | **Early Stop** |

V2 IC 峰值 0.2065（Ep7）高于 V1 的 0.1819，但 Ep7 后 10 个 epoch 无改善，Early Stop。IC 波动极大（-0.04~0.21），VL-Adaptive 持续 ROLLBACK，说明学习率仍不稳定。

**结论：Small 模型 24.7M 参数在当前训练策略下未能稳定收敛，IC 波动远大于 mini 模型。**

#### 对应 Tokenizer

| Tokenizer | 路径 | group_size |
|-----------|------|------------|
| MA60-Tokenizer-base | `outputs/models/ma60_tokenizer_base_v1/checkpoints/best_model` | 4 |

Tokenizer 训练 17 epochs, best val loss=0.0120。

### 3.4 全市场对比测试（5 模型）

测试时间：2026-05-27

| 模型 | 参数量 | 归一化 | IC | Rank IC | 方向准确率 | MSE |
|------|--------|--------|------|---------|-----------|------|
| Kronos-mini | 4.1M | full_window | 0.0163 | 0.0098 | 53.03% | 0.004099 |
| Kronos-small | 24.7M | full_window | -0.1024 | -0.0222 | 54.29% | 0.063659 |
| Kronos-base | 102.3M | full_window | -0.1820 | -0.0795 | 45.71% | 0.026555 |
| **MA60-mini-IC** | **4.1M** | **sliding MA60** | **0.1719** | **0.1553** | **59.10%** | **0.002375** |
| **MA60-small** | **24.7M** | **sliding MA60** | **0.1646** | **0.1441** | **60.27%** | **0.002800** |

#### 分组方向准确率（按实际收益率分 5 组）

| 组别 | Kronos-mini | Kronos-small | Kronos-base | MA60-mini-IC | MA60-small |
|------|-------------|--------------|-------------|--------------|------------|
| 最差(跌) | 66.34% | 78.47% | 64.77% | 71.23% | 71.23% |
| 较差 | 60.86% | 69.86% | 46.18% | 64.58% | **76.13%** |
| 中等 | 58.90% | 62.82% | 43.84% | 66.54% | **74.17%** |
| 较好 | 45.99% | 42.07% | 45.99% | 45.60% | 41.29% |
| 最优(涨) | 33.07% | 18.20% | 27.79% | **47.55%** | 38.55% |

#### 可重复性验证

与首次测试对比（数据增量更新后窗口偏移）：

| 模型 | 首次 IC | 本次 IC | 差异 |
|------|---------|---------|------|
| Kronos-mini | -0.0031 | 0.0163 | +0.019 |
| Kronos-small | -0.0955 | -0.1024 | -0.007 |
| Kronos-base | -0.1767 | -0.1820 | -0.005 |
| MA60-mini-IC | 0.2116 | 0.1719 | -0.040 |

Pretrained 模型结果高度可重复（差异 < 0.02）。MA60-mini IC 下降 0.04，可能与增量数据更新导致的测试窗口偏移有关。

#### 关键发现

1. **MA60-small IC 略低于 MA60-mini**：0.1646 vs 0.1719，但方向准确率更高（60.27% vs 59.10%）
2. **MA60-small 在中间组表现突出**：较差组 76.13%、中等组 74.17%，远超 MA60-mini
3. **MA60-small 在最优组（涨）偏弱**：38.55%，低于 mini 的 47.55%，说明对上涨行情识别不足
4. **Small 模型训练不稳定**：IC 波动大（Ep7 峰值 0.2065 后持续走低），24.7M 参数未充分发挥优势

### 3.5 分组方向准确率（按实际收益率分 5 组，3.1 首次测试）

| 组别 | mini | small | base | MA60-mini-IC |
|------|------|-------|------|-------------|
| 最差(跌) | 66.34% | 80.23% | 63.80% | 68.10% |
| 较差 | 63.80% | 66.93% | 50.49% | 67.32% |
| 中等 | 54.21% | 63.60% | 43.64% | 65.36% |
| 较好 | 44.23% | 43.25% | 42.27% | 48.14% |
| 最优(涨) | 31.51% | 18.79% | 29.16% | **52.05%** |

## 四、关键发现

### 4.1 归一化策略 >> 模型参数量

- pretrained 模型参数量从 4.1M → 102.3M（25 倍），IC 反而从 -0.003 恶化到 -0.177
- MA60-mini 仅 4.1M 参数，IC 达到 0.2116，碾压所有 pretrained 模型
- **正确的归一化远比增加模型参数量重要**

### 4.2 Pretrained 模型在 A 股上失效

- 三个 pretrained 模型 IC 均为负或接近 0
- small/base 参数越大效果越差，可能过拟合了训练数据分布，无法适应 A 股的特征
- 方向准确率：base 仅 45.87%，甚至不如随机

### 4.3 Mini 模型已过拟合

- V2 训练 IC=0.29 → 测试 IC=0.17，泛化损失 44%
- V1 训练 IC=0.26 → 测试 IC=0.23，泛化损失 12%
- 4.1M 参数在 2555 股票数据上容量不足，学习到的是噪声而非规律

### 4.4 IC best > Loss best

- V1-IC (0.2289) >> V1-Loss (0.0836)
- 直接优化 loss 不等于提升预测能力，IC 指标更能反映实际效果

### 4.5 MA60-mini 在涨跌两端都有优势

- 最优组（涨最多）：MA60-mini 52.05% vs pretrained 18-31%
- 最差组（跌最多）：MA60-mini 68.10%，也能识别下跌
- Pretrained small 虽然在跌组有 80%，但涨组仅 19%，严重偏向预测跌

## 五、下一步方向

### 5.1 Small 模型训练已完成

- 训练结果：V2 best IC=0.2065（Ep7），但训练不稳定，Early Stop at Ep17
- 全市场测试 IC=0.1646，略低于 MA60-mini 的 0.1719
- 24.7M 参数未充分发挥优势，需要优化训练策略

### 5.2 Small 模型训练优化方向

- **训练不稳定**：IC 波动大（-0.04~0.21），需要更稳定的学习率策略
- **可能方案**：
  - 更小的初始 LR（0.003~0.005）+ 更长的 warmup
  - 渐进式解冻（先冻结底层，再逐步解冻）
  - 更大的 batch size（受限于显存）
  - Cosine Annealing 替代 OneCycleLR

### 5.3 训练脚本

```
# Step 1: MA60 base tokenizer
python -u finetune/train_tokenizer_ma60_base.py

# Step 2: Small predictor
python -u finetune/train_predictor_ma60_small.py
```

### 5.4 Small 训练参数调整

| 参数 | Mini 值 | Small 值 | 原因 |
|------|---------|---------|------|
| learning_rate | 0.02 | 0.02 | 归一化方式变了需要充分探索 |
| weight_decay | 0.1 | 0.15 | 更大模型需要更强正则化 |
| batch_size | 16 | 8 | 显存限制 |
| max_context | 2048 | 512 | small 模型限制 |
| lr_max | 0.03 | 0.03 | 允许充分探索 |

## 六、模型文件索引

### 最终模型

| 路径 | 说明 |
|------|------|
| `final_models/Kronos-Tokenizer-2k-MA60` | MA60 tokenizer (group_size=5, 适用于 mini) |
| `final_models/Kronos-mini-MA60` | MA60 predictor (mini, best IC) |
| `pretrained/Kronos-mini` | 原始 mini 模型 |
| `pretrained/Kronos-small` | 原始 small 模型 |
| `pretrained/Kronos-base` | 原始 base 模型 |

### 训练产出

| 路径 | 说明 |
|------|------|
| `outputs/models/ma60_tokenizer_v1/` | MA60 tokenizer (mini, group_size=5) |
| `outputs/models/ma60_predictor_v1/` | MA60 predictor V1 (mini) |
| `outputs/models/ma60_predictor_v2/` | MA60 predictor V2 (mini) |
| `outputs/models/ma60_tokenizer_base_v1/` | MA60 tokenizer (base, group_size=4) 已完成 |
| `outputs/models/ma60_predictor_small_v1/` | MA60 predictor V1 (small, IC=0.1819 at Ep3) |
| `outputs/models/ma60_predictor_small_v2/` | MA60 predictor V2 (small, IC=0.2065 at Ep7) |

### 推理产出

| 路径 | 说明 |
|------|------|
| `outputs/prediction_results/predictions_ma60_v1ic_2025.sql` | 2025 年推理结果 (1,144,455 条) |
| `outputs/prediction_results/predictions_ma60_v1ic_2026.sql` | 2026 年推理结果 (430,694 条) |

---

*文档更新时间：2026-05-27*
