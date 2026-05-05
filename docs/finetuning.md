# 模型微调指南

## 概述

Kronos 提供完整的微调管道，支持在自有数据上适配模型。本指南以中国A股市场为例，使用 Qlib 准备数据。

## 微调流程

### 流程概览

1. **配置**: 设置路径和超参数
2. **数据准备**: 使用 Qlib 处理和划分数据
3. **分词器微调**: 适配数据分布
4. **预测器微调**: 针对预测任务优化
5. **回测评估**: 验证模型性能

---

## 环境准备

### 安装 Qlib

```bash
pip install pyqlib
```

### 配置 Qlib 数据

按照 [Qlib 官方指南](https://github.com/microsoft/qlib) 下载并设置本地数据。

---

## 步骤 1: 配置实验

编辑 `finetune/config.py`：

### 关键配置项

```python
# 数据路径
qlib_data_path = "~/.qlib/qlib_data/cn_data"  # Qlib数据目录
dataset_path = "./data/processed_datasets"     # 处理后数据保存路径
save_path = "./outputs/models"                 # 模型保存路径

# 时间范围
train_time_range = ["2011-01-01", "2022-12-31"]
val_time_range = ["2022-09-01", "2024-06-30"]
test_time_range = ["2024-04-01", "2025-06-05"]

# 模型参数
lookback_window = 90    # 回看窗口
predict_window = 10     # 预测窗口
max_context = 512       # 最大上下文

# 训练参数
epochs = 30
batch_size = 50
tokenizer_learning_rate = 2e-4
predictor_learning_rate = 4e-5

# 预训练模型路径
pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
pretrained_predictor_path = "NeoQuasar/Kronos-small"
```

---

## 步骤 2: 数据预处理

```bash
python finetune/qlib_data_preprocess.py
```

这将生成：
- `train_data.pkl`
- `val_data.pkl`
- `test_data.pkl`

---

## 步骤 3: 微调分词器

```bash
# 使用 2 个 GPU
torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py
```

分词器微调将使量化器适配目标市场的数据分布。

---

## 步骤 4: 微调预测器

```bash
# 使用 2 个 GPU
torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py
```

最佳模型将保存到配置的路径。

---

## 步骤 5: 回测评估

```bash
python finetune/qlib_test.py --device cuda:0
```

回测将输出：
- 策略表现分析
- 累计收益曲线图

---

## CSV 格式微调

对于非 Qlib 数据，可使用 `finetune_csv/` 目录的脚本：

```bash
python finetune_csv/train_sequential.py
```

### CSV 数据格式

CSV 文件应包含以下列：
- `timestamps`: 时间戳
- `open`, `high`, `low`, `close`: OHLC数据
- `volume`, `amount`: 成交量和成交额（可选）

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

根据数据源调整 `QlibDataset` 的数据加载和预处理逻辑。

---

## 微调配置参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 30 | 训练轮数 |
| batch_size | 50 | 批大小 |
| accumulation_steps | 1 | 梯度累积步数 |
| clip | 5.0 | 数据裁剪阈值 |
| tokenizer_learning_rate | 2e-4 | 分词器学习率 |
| predictor_learning_rate | 4e-5 | 预测器学习率 |
| adam_beta1 | 0.9 | Adam beta1 |
| adam_beta2 | 0.95 | Adam beta2 |
| adam_weight_decay | 0.1 | 权重衰减 |