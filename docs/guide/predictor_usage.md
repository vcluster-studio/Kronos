# Kronos Predictor 使用指南

从预处理到训练评估的完整流程参考。

---

## 1. 数据预处理

### 1.1 原始数据准备

**训练数据** (`kline_daily_raw.pkl`):
- 覆盖时间：2018-01-02 ~ 2026-05-18
- 用途：训练/验证/测试的输入源 + backtest 的 context

**回测数据** (`backtest_raw.pkl`):
- 覆盖时间：2026-05-19 ~ 2026-06-17
- 用途：backtest 的 target（与训练数据时间隔离）

```bash
# 若需重新生成 raw 数据（使用项目 SQL）
python data/generate_raw.py --output finetune/data/raw/kline_daily_raw.pkl --min-length 500

python data/generate_backtest_raw.py --output finetune/data/raw/backtest_raw.pkl --start 2026-05-19
```

### 1.2 预处理命令

```bash
# sliding_ma60（推荐默认）
python finetune/predictor/preprocess.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode block \
    --validate

# sliding_ma120（更长历史窗口）
python finetune/predictor/preprocess.py \
    --norm-mode sliding_ma120 \
    --lookback 400 \
    --predict 10 \
    --split-mode block

# full_window（全窗口归一化）
python finetune/predictor/preprocess.py \
    --norm-mode full_window \
    --lookback 400 \
    --predict 10 \
    --split-mode block
```

### 1.3 输出说明

预处理生成以下文件：

```
finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}/{split_mode}/
├── train.pkl      # 训练集（完整归一化数据 + 窗口索引）
├── val.pkl        # 验证集
├── test.pkl       # 测试集
├── meta.pkl       # 元数据（fingerprint、样本数、泄露检查结果）
```

**数据格式**（每个 pkl 内部）：
```python
{
    symbol: {
        'normalized': np.ndarray (T, 6),   # 归一化后的完整序列
        'means': np.ndarray (T, 6),       # 滚动均值
        'stds': np.ndarray (T, 6),        # 滚动标准差
        'original': np.ndarray (T, 6),    # 原始 OHLCV
        'index': DatetimeIndex,           # 时间戳
        'windows': np.ndarray,            # 该 split 的窗口起点索引
    }
}
```

---

## 2. Tokenizer 训练

每个 norm_mode 需单独训练 tokenizer（归一化后分布不同）。

```bash
# mini vocab_size=2048
python finetune/tokenizer/train.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --sample-ratio 0.1

# small vocab_size=4096
python finetune/tokenizer/train.py \
    --norm-mode sliding_ma60 \
    --model small \
    --sample-ratio 0.1

# base vocab_size=8192
python finetune/tokenizer/train.py \
    --norm-mode sliding_ma60 \
    --model base \
    --sample-ratio 0.1
```

**输出路径**：
```
outputs/tokenizers/{norm_mode}/{model_type}/
├── vocab.json
├── tokenizer_config.json
├── merges.txt
└── meta.json       # 记录 fingerprint、vocab_size 等
```

---

## 3. Predictor 训练

### 3.1 单卡训练

```bash
python finetune/predictor/train.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode block \
    --model mini \
    --epochs 50 \
    --lr 0.01 \
    --weight-decay 0.01 \
    --batch-size 32
```

### 3.2 多卡训练（DDP）

```bash
# 4 卡训练
torchrun --nproc_per_node=4 finetune/predictor/train.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode block \
    --model mini \
    --epochs 50 \
    --lr 0.01

# 8 卡训练
torchrun --nproc_per_node=8 finetune/predictor/train.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --epochs 50
```

### 3.3 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--norm-mode` | sliding_ma60 | 归一化模式 |
| `--lookback` | 400 | 回看窗口长度 |
| `--predict` | 10 | 预测步数 |
| `--split-mode` | block | 分割模式（block/time） |
| `--model` | mini | 模型类型（mini/small/base） |
| `--epochs` | 50 | 最大训练轮数 |
| `--lr` | 0.01 | 学习率（从大值起步） |
| `--lr-min` | 1e-6 | 最小学习率（Cosine 末端） |
| `--weight-decay` | 0.01 | 权重衰减 |
| `--batch-size` | 32 | 批大小 |
| `--early-stopping-patience` | 12 | 早停耐心值 |
| `--seed` | 42 | 随机种子 |

### 3.4 输出监控

训练过程输出：
```
=== Epoch 1/50 ===
LR: 0.010000

  Trajectory IC (close, detrended): 0.1234
  DA_score: 0.5678, Excess DA: +5.2%, Combined: 0.4567
  Train: 2.3456, Time: 12:34

  [IC] New best: 0.1234
  [COMBINED] 0.4567
```

**关键指标**：
- **IC**: trajectory IC（去趋势序列上的 Spearman 相关系数）
- **DA_score**: 综合 DA 评分（对数步权重加权）
- **Excess DA**: DA - naive DA，正值表示模型优于朴素预测
- **Combined**: IC 和 DA 的综合评分（权重 0.6/0.4）

### 3.5 Checkpoint 选择

训练保存多个 checkpoint：
- `latest_model`: 最近一个 epoch
- `best_ic_model`: IC 最高
- `best_combined_model`: Combined 最高（推荐）
- `best_model`: Val loss 最低

---

## 4. 模型评估

### 4.1 测试集评估

```bash
python finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode block \
    --model mini \
    --checkpoint best_combined_model \
    --n-samples -1   # 全量评估
```

### 4.2 评估输出示例

```
======================================================================
Evaluation Results (Detrended Trajectory IC)
======================================================================

[Direction Accuracy - close]
Step         DA      std      p50    naive   excess
+1        58.3%     7.2%    57.8%    52.1%     6.2%
+2        55.6%     8.1%    54.2%    52.3%     3.3%
+3        53.2%     9.0%    51.5%    52.0%     1.2%
...

[Amplitude Error Rate]
Mean: 0.92, Std: 0.18
Perfect (0.9-1.1): 35.2%
Usable (0.7-1.3): 72.1%

[Limit Hit Rate]
Hit Rate: 12.5% (random ~1-3%, signal >10%)
Predicted: 120, Actual: 350

[Trajectory IC - Appendix (Detrended)]
Feature      mean      std      p25      p50      p75      n
close      0.1456   0.0892   0.0923   0.1434   0.2012   5000
...
======================================================================
```

**解读**：
- **naive**: 多数方向比例（市场 bias 的 baseline）
- **excess**: DA - naive，>0 表示真 alpha
- **p25/p75**: IC 分布的离散程度，判断数字可信度

---

## 5. 回测评估

```bash
python finetune/predictor/backtest.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --model mini \
    --checkpoint best_combined_model
```

回测使用：
- context 来自 `kline_daily_raw.pkl`（末尾 lookback 根）
- target 来自 `backtest_raw.pkl`（时间隔离）

---

## 6. 完整流程一键脚本

```bash
#!/bin/bash
# complete_pipeline.sh

NORM_MODE="sliding_ma60"
LOOKBACK=400
PREDICT=10
MODEL="mini"
EPOCHS=50

echo "=== Step 1: Preprocess ==="
python finetune/predictor/preprocess.py \
    --norm-mode $NORM_MODE \
    --lookback $LOOKBACK \
    --predict $PREDICT \
    --split-mode block \
    --validate

echo "=== Step 2: Train Tokenizer ==="
python finetune/tokenizer/train.py \
    --norm-mode $NORM_MODE \
    --model $MODEL \
    --sample-ratio 0.1

echo "=== Step 3: Train Predictor ==="
python finetune/predictor/train.py \
    --norm-mode $NORM_MODE \
    --lookback $LOOKBACK \
    --predict $PREDICT \
    --split-mode block \
    --model $MODEL \
    --epochs $EPOCHS \
    --lr 0.01

echo "=== Step 4: Evaluate ==="
python finetune/predictor/eval.py \
    --norm-mode $NORM_MODE \
    --lookback $LOOKBACK \
    --predict $PREDICT \
    --split-mode block \
    --model $MODEL \
    --checkpoint best_combined_model

echo "=== Complete ==="
```

---

## 7. 目录结构总览

```
finetune/
├── data/
│   ├── raw/
│   │   ├── kline_daily_raw.pkl   # 训练原始数据
│   │   └── backtest_raw.pkl      # 回测原始数据
│   └── processed/
│       └── {norm_mode}/lb{lookback}_pd{predict}/
│           ├── {split_mode}/
│           │   ├── train.pkl
│           │   ├── val.pkl
│           │   ├── test.pkl
│           │   └── meta.pkl
│           └── backtest/
│               ├── samples.pkl
│               └── meta.pkl
├── predictor/
│   ├── preprocess.py             # 数据预处理
│   ├── train.py                  # 训练入口
│   ├── eval.py                   # 评估入口
│   ├── backtest.py               # 回测入口
│   └── core/
│       ├── config.py             # 配置定义
│       ├── paths.py              # 路径管理
│       ├── dataset.py            # 数据集类
│       ├── metrics.py            # 度量计算
│       ├── normalization.py      # 归一化器
│       ├── splitting.py          # 数据分割
│       └ schema.py               # 数据结构
│       └ utils.py                # 工具函数
└── tokenizer/
    └ train.py                    # Tokenizer 训练

outputs/
├── tokenizers/
│   └── {norm_mode}/{model_type}/
└── models/
    └ └── {norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{model_type}/
        ├── checkpoints/
        │   ├── latest_model/
        │   ├── best_ic_model/
        │   ├── best_combined_model/
        └── training_info.json
```

---

## 8. 常见问题

### Q1: naive DA 为什么是 50%+？

naive DA = max(上涨比例, 下跌比例)，反映市场 bias。
- 若市场涨多跌少（60%涨），naive DA = 60%
- 模型 DA > naive DA 才是真 alpha

### Q2: 为什么 IC 用去趋势序列？

原始价格序列强自相关，即使预测持平也会因趋势同向拿到虚高 IC。
去趋势后才能测量「轨迹形状」的预测能力。

### Q3: block_split vs time_split？

- **block_split**: 按 target 时间块随机分配，防止相邻样本跨 split
- **time_split**: 按时间边界分割，适合模拟真实交易时序

### Q4: 如何选择 checkpoint？

- **best_combined_model**: 推荐，平衡 IC 和 DA
- **best_ic_model**: 若只关心排序能力
- **best_model**: Val loss 最低，但可能与 IC 不一致

---

## 9. 参数速查表

| 参数 | sliding_ma60 | sliding_ma120 | full_window |
|------|--------------|---------------|-------------|
| 历史需求 | 60 步 | 120 步 | 0 步 |
| 归一化方式 | 滚动 mean/std | 滚动 mean/std | 窗口 mean/std |
| 适用场景 | 通用 | 长期趋势 | 窗口独立 |

| 参数 | mini | small | base |
|------|------|-------|------|
| vocab_size | 2048 | 4096 | 8192 |
| 参数量 | 4.1M | 24.7M | 102.3M |
| 训练速度 | 快 | 中 | 慢 |
| 推荐用途 | 快速实验 | 生产 | 高精度 |

---

*文档更新：2026-06-19*
*对应提交：P1-P5 fixes (5f4cb7a)*