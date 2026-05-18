# Kronos-mini 微调实验计划

本文档规划 mini 版本的完整微调流程，基于 V1/V2 的经验教训。

---

## 一、经验教训总结

| 版本 | 分词器 | 预测器 | 结果 | 教训 |
|------|--------|--------|------|------|
| V1 | ✅ 微调 | ✅ 微调 | IC=0.1198 | ✅ 正确流程 |
| V2 | ❌ 未微调 | ✅ 微调 | IC=-0.1328 | ❌ 分词器必须配套微调 |

**关键教训**：
- 分词器和预测器必须配套微调
- V2 的失败是因为使用了 V1 分词器 + 新预测器，编码失配

---

## 二、mini 版本优势

| 项目 | mini | small |
|------|------|-------|
| 参数量 | 4.1M | 24.7M |
| 训练速度 | **快 5x** | 基准 |
| 显存占用 | 低 | 高 |
| 用途 | **调参、验证方案** | 生产部署 |

**策略**：用 mini 快速验证最佳方案，再用 small 正式训练。

---

## 三、实验流程

### Step 0：建立基线（必须）

测试原始 mini 模型在 A 股上的表现：

```bash
python finetune/test_stratified.py
```

**分层测试覆盖**：
- 大盘股（成交额 > 10 亿）
- 中盘股（成交额 1-10 亿）
- 小盘股（成交额 < 1 亿）
- 高波动股（波动 > 20%）
- 低波动股（波动 < 10%）

---

### Step 1：分词器微调

```bash
python -u finetune/train_tokenizer_mini.py
```

**配置**：
```python
epochs = 15
batch_size = 16
n_train_iter = 32,000
learning_rate = 2e-4
early_stopping_patience = 3
```

---

### Step 2：预测器微调

```bash
python -u finetune/train_predictor_mini.py
```

**配置**：
```python
epochs = 15
batch_size = 16
n_train_iter = 32,000
learning_rate = 4e-5  # 保持 V1 成功值
early_stopping_patience = 3
```

**新增功能**：
- 每 epoch 测试 IC
- 同时保存 best_loss 和 best_ic 两个模型

---

### Step 3：分层测试评估

```bash
python finetune/test_stratified.py
```

对比各类别股票的改善情况。

---

## 四、时间预估

| Step | 预计时间 |
|------|---------|
| Step 0：基线测试 | 10 分钟 |
| Step 1：分词器训练 | 30 分钟 |
| Step 2：预测器训练 | 45 分钟 |
| Step 3：分层测试 | 10 分钟 |
| **总计** | **~2 小时** |

---

## 五、成功标准

| 指标 | 原始模型目标 | 微调后目标 |
|------|-------------|-----------|
| Overall IC | 负值 → | **正值 > 0.05** |
| 大盘股 IC | 负值 → | **正值 > 0.03** |
| Direction Acc | 43% → | **> 50%** |

---

## 六、文件清单

| 文件 | 用途 |
|------|------|
| `config_mini.py` | mini 专属配置 |
| `train_tokenizer_mini.py` | 分词器训练（早停） |
| `train_predictor_mini.py` | 预测器训练（早停+IC监控） |
| `test_stratified.py` | 分层测试脚本 |

---

## 七、执行命令汇总

```bash
cd C:\workbench\Kronos

# Step 0: 基线测试
python finetune/test_stratified.py

# Step 1: 分词器训练
python -u finetune/train_tokenizer_mini.py

# Step 2: 预测器训练
python -u finetune/train_predictor_mini.py

# Step 3: 分层测试
python finetune/test_stratified.py
```

---

## 八、后续步骤

如果 mini 验证成功：

1. 使用相同配置训练 small 版本
2. 增加训练样本量
3. 尝试分层学习率
4. 生产部署测试