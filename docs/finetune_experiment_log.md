# Kronos A 股微调实验记录

本文档记录 Kronos 模型在 A 股主板数据上的微调实验过程和结果。

---

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce MX330 (2GB) |
| CPU | 多核 |
| Python | 3.10 |
| PyTorch | CUDA 版本 |
| 模型 | Kronos-small (24.7M 参数) |
| 分词器 | Kronos-Tokenizer-base |

---

## 数据配置

| 项目 | 值 |
|------|-----|
| 数据源 | CSV 格式，主板股票（000/002/600/601/603 开头） |
| 训练时间范围 | 2018-01-02 ~ 2024-12-31 |
| 验证时间范围 | 2025-01-02 ~ 2025-06-30 |
| 测试时间范围 | 2025-07-01 ~ 2026-05-07 |
| 过滤规则 | 排除 ST、次新股、流动性差、停牌过多 |
| 训练股票数 | ~2800 只 |
| 特征列 | open, high, low, close, vol, amt |

---

## 实验版本

### V1: 第一次微调（当前最佳）

**分词器训练配置：**
```
epochs = 30 (实际训练到 Epoch 17，早停)
batch_size = 8
n_train_iter = 3000 × 8 = 24,000
learning_rate = 2e-4
```

**分词器训练结果：**
| Epoch | Val Loss |
|-------|----------|
| 1 | 0.0109 |
| 4 | 0.0101 |
| 11 | 0.0092 |
| 17 | 0.0092 (停止，已收敛) |

**预测器训练配置：**
```
epochs = 30 (实际训练到 Epoch 9，过拟合停止)
batch_size = 8
n_train_iter = 3000 × 8 = 24,000
learning_rate = 4e-5
```

**预测器训练结果：**
| Epoch | Val Loss |
|-------|----------|
| 1 | 3.2149 |
| 2 | 3.1853 (最佳) |
| 3+ | 过拟合上升 |

**测试结果（100 只股票）：**
| 指标 | 原始模型 | V1 微调后 | 改善 |
|------|---------|-----------|------|
| IC | -0.1087 | +0.0394 | +0.1481 |
| Rank IC | -0.0500 | -0.0565 | +0.0065 |
| Direction Acc | 43% | 41% | -2% |

---

### V2: 优化尝试（失败）

**修改配置：**
```
learning_rate = 1e-5 (降低 4 倍)
n_train_iter = 5000 × 8 = 40,000 (增加样本)
```

**结果：**
- Val Loss 持续下降到 Epoch 4 (3.1828)
- 但 IC 反而变差 (-0.1328)
- **结论：配置调整无效，V2 已删除**

---

## 结论

1. **当前最佳模型是 V1**
2. 预测器在 Epoch 2 就过拟合，需要早停机制
3. 降低学习率反而效果变差
4. Val Loss ≠ 预测能力（IC）

---

## 后续优化方向

| 方向 | 建议 |
|------|------|
| 早停机制 | 预测器在 Epoch 2-3 停止 |
| 学习率 | 保持 4e-5 或尝试 2e-5 |
| 样本量 | 可尝试增加到 50,000 |
| 分层学习率 | 底层更低，顶层更高 |

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `finetune/config.py` | 训练配置 |
| `finetune/train_tokenizer_single.py` | 分词器训练脚本 |
| `finetune/train_predictor_single.py` | 预测器训练脚本 |
| `finetune/test_original_model.py` | 测试原始预训练模型 |
| `finetune/compare_models.py` | 对比多个版本模型 |
| `outputs/models/a_share_tokenizer/` | V1 分词器 |
| `outputs/models/a_share_predictor/` | V1 预测器 |

---

## 使用方法

### 训练
```bash
cd C:\workbench\Kronos

# 数据预处理
python finetune/csv_data_preprocess.py

# 分词器训练
python -u finetune/train_tokenizer_single.py

# 预测器训练
python -u finetune/train_predictor_single.py
```

### 测试
```bash
# 测试原始模型
python finetune/test_original_model.py

# 对比所有版本
python finetune/compare_models.py
```

### 使用微调模型预测
```python
from model import KronosTokenizer, Kronos, KronosPredictor

# 加载微调后的 V1 模型
tokenizer = KronosTokenizer.from_pretrained("outputs/models/a_share_tokenizer/checkpoints/best_model")
model = Kronos.from_pretrained("outputs/models/a_share_predictor/checkpoints/best_model")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 进行预测
pred_df = predictor.predict(df, x_timestamp, y_timestamp, pred_len=10)
```

---

## 实验日期

- 2026-05-08: 完成数据预处理和 V1 训练
- 2026-05-09: 完成 V2 优化尝试，确认 V1 为最佳版本