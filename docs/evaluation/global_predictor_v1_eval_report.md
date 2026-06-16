# Global Predictor V1 评估报告

## 模型信息

- **模型名称**: global_predictor_v1
- **训练模式**: Mode1 (full_window normalization)
- **基座模型**: Kronos-mini (4.1M)
- **Tokenizer**: pretrained/Kronos-Tokenizer-base
- **训练时间**: 2026-06-10 07:13 - 17:02 (约9小时49分钟)

## 训练配置

```json
{
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.003,
  "lookback": 200,
  "predict": 10,
  "n_samples": 200,
  "lr_scheduler": "cosine",
  "warmup_epochs": 2,
  "norm_mode": "full_window",
  "direction_loss_weight": 0.3
}
```

## 训练过程

| Epoch | Train Loss | Val Loss | close Traj_IC | Best IC |
|-------|------------|----------|---------------|---------|
| 1 | 2.68 | 2.64 | -0.004 | -0.004 |
| 15 | 2.26 | 2.34 | -0.024 | 0.180 |
| 50 | 2.15 | 2.23 | 0.025 | 0.180 |

**关键节点**:
- Best Trajectory IC = 0.180 (在某个早期 epoch 达到)
- 最终 epoch 50 的 close Traj_IC = 0.025

## 测试集评估 (500 samples)

### Trajectory IC

| Feature | Traj_IC | std | pos% |
|---------|---------|-----|------|
| open | 0.0891 | 0.57 | 56.8% |
| high | 0.0692 | 0.58 | 54.0% |
| low | 0.0334 | 0.57 | 51.5% |
| close | 0.0011 | 0.54 | 50.1% |
| vol | -0.0041 | 0.38 | 50.1% |
| amt | -0.0017 | 0.39 | 52.4% |

### Per-step MAE (close)

| Step | close_MAE |
|------|-----------|
| +1 | 2.41 |
| +2 | 3.67 |
| +3 | 4.61 |
| +4 | 5.84 |
| +5 | 5.87 |
| +6 | 6.75 |
| +7 | 6.95 |
| +8 | 7.82 |
| +9 | 8.14 |
| +10 | 7.79 |

### Per-step DA (close)

| Step | close_DA |
|------|----------|
| +1 | 47% |
| +2 | 65% |
| +3 | 65% |
| +4 | **71%** |
| +5 | **73%** |
| +6 | 58% |
| +7 | 63% |
| +8 | 53% |
| +9 | 49% |
| +10 | 48% |

## 分析

### 发现

1. **测试集 close Trajectory IC ≈ 0**: 与训练集最佳 IC (0.18) 差距大
2. **open/high Trajectory IC 较好**: 约 0.06-0.09，可能价格开盘方向预测更稳定
3. **DA 在中间步骤最高**: +4/+5 步 DA 达 71-73%
4. **Step +1 DA 较低**: 47%，接近随机水平

### 可能原因

- **过拟合**: 训练集 IC 0.18 vs 测试集 IC 0.001
- **数据分布差异**: val/test 时间段不同，市场环境变化
- **lookback=200 过长**: 信息稀释，预测窗口利用率低
- **评估标准变更**: 本次使用 Trajectory IC（新方法），与旧 Return IC 不同

### 下一步建议

1. 尝试更短 lookback (60/90) 观察效果
2. 增加测试集样本量验证稳定性
3. 尝试 Mode3/Mode4 的 MA60 归一化方式对比

## 归档信息

- **归档路径**: `outputs/models/archived/global_predictor_v1_ep50/`
- **最佳模型**: `best_ic_model` (基于 Trajectory IC 保存)
- **评估日期**: 2026-06-11