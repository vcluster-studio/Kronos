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

# time split（按序列内时间位置切，需传 --train-end/--val-end 整数位置）
# 位置 = 交易日序号（从数据起始日期 2018-01-02 开始计数）
# A 股约 244 交易日/年，交易日累计表：
#   年份    累计位置    说明
#   2018    0~244      数据起始年
#   2019    244~488    
#   2020    488~732    
#   2021    732~976    
#   2022    976~1220   
#   2023    1220~1464  
#   2024    1464~1708  
#   2025    1708~1952  
#   2026    1952~2050   数据截止 2026-05-18（约100天）

# 旧版本约定时间边界（train_end: 2023-06-30, val_end: 2024-12-31）
# 2023-06-30 ≈ 1220 + 122（半年） = 1342
# 2024-12-31 = 1708（年末）
python finetune/predictor/preprocess.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode time \
    --train-end 1342 \
    --val-end 1708 \
    --validate
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

## 2. Tokenizer 微调

KronosTokenizer 是预训练的 VQ-VAE quantizer，**按 norm_mode 微调**（非从零训练）。

### 2.1 设计要点

- 架构由 model_type 决定：mini→Kronos-Tokenizer-2k，small/base→Kronos-Tokenizer-base
- vocab_size 由预训练架构固定，不是微调参数
- 不同 norm_mode 数据分布不同，必须各自微调
- 微调损失：`recon_loss + bsq_loss`

### 2.2 微调命令

```bash
# 按 norm_mode 微调 tokenizer
python finetune/tokenizer/train.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.001 \
    --n-train-iter 2000

# sliding_ma120 微调
python finetune/tokenizer/train.py \
    --norm-mode sliding_ma120 \
    --model mini \
    --epochs 30
```

**采样机制（关键）**：步数驱动放回采样，非比例抽样。每 epoch 采样步数 = `--n-train-iter` × `--batch-size`（默认 2000×16 = 32000 步）。val 同理 `--n-val-iter`（默认 400×16 = 6400 步）。跨多 epoch 覆盖全量数据。

### 2.3 输出路径

```
outputs/tokenizers/{norm_mode}/{model_type}/
├── model.safetensors       # 微调后权重（根目录副本，供 validate/eval/predictor 加载）
├── config.json             # 架构配置（来自预训练）
├── meta.json               # norm_mode、pretrained_base、fingerprint、best_val_loss
├── history.json            # 训练历史（train/val loss per epoch）
└── checkpoints/
    ├── best_model/         # val 重建损失最低
    └── final_model/        # 最后一个 epoch
```

> **路径说明**：微调同时存 `checkpoints/best_model`、`checkpoints/final_model`，以及根目录副本（`model.safetensors` = best 权重）。validate/eval/predictor 从根目录加载（`from_pretrained(output_path)`），故根目录副本是加载入口。

### 2.4 微调效果检验

微调完成后，验证 tokenizer 是否比预训练版本更好地适应目标分布：

```bash
python finetune/tokenizer/validate.py \
    --norm-mode sliding_ma60 \
    --model mini
```

**输出示例**：
```
[Reconstruction Loss on Val Data]
Pretrained (pretrained/Kronos-Tokenizer-2k):
  Mean: 0.001234
  Std:  0.000456

Fine-tuned (sliding_ma60):
  Mean: 0.000987
  Std:  0.000321

[Comparison]
  Improvement: 0.000247 (20.01%)

[Validation Criteria]
  [1] Fine-tuned loss < Pretrained loss: True
      PASS: 0.000987 < 0.001234
  [2] Early stopping triggered: True
      OK: Stopped at epoch 18/30

[RESULT] VALIDATION PASSED
```

**验收标准**：
- 微调后 val 重建损失 < 预训练在同数据重建损失（必要）
- 重建损失收敛（early stopping 触发）（必要）

### 2.5 模型-tokenizer 配对

| 模型 | 预训练 Tokenizer | 架构参数 |
|------|-----------------|---------|
| mini | Kronos-Tokenizer-2k | group_size=5, context=2048 |
| small | Kronos-Tokenizer-base | group_size=4, context=512 |
| base | Kronos-Tokenizer-base | group_size=4, context=512 |

**注意**：tokenizer 与 predictor 必须同 norm_mode，否则编码失真。

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
    --batch-size 16
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
| `--norm-mode` | sliding_ma60 | 归一化模式（full_window/sliding_ma20/sliding_ma60/sliding_ma120） |
| `--lookback` | 400 | 回看窗口长度 |
| `--predict` | 10 | 预测步数 |
| `--split-mode` | block | 分割模式（block/time） |
| `--model` | mini | 模型类型（mini/small/base） |
| `--epochs` | 50 | 最大训练轮数 |
| `--lr` | 0.01 | 学习率（从大值起步，Cosine 衰减至 TrainConfig.lr_min=1e-5） |
| `--weight-decay` | 0.01 | 权重衰减 |
| `--batch-size` | 16 | 批大小 |
| `--seed` | 42 | 随机种子（DDP 各 rank 实际用 seed+rank） |
| `--resume` | None | 断点续训，传 checkpoint 路径（加载 latest + optimizer/scheduler/best 状态） |
| `--output-folder` | None | 自定义输出目录名（默认按 norm_mode/lb/pd/split/model 自动生成） |
| `--use-block` | False | 兼容旧 block_lb400_pd10 数据（legacy） |

> **注**：`early_stopping_patience`(12)、`early_stopping_grace_period`(8)、`warmup_epochs`(2)、`lr_min`(1e-5)、`ic_patience_reset`(True)、`combined_ic_weight`(0.6)/`combined_da_weight`(0.4) 在 `TrainConfig` 中定义，**未通过 CLI 暴露**，需改代码调整。详见 `core/config.py`。

### 3.4 输出监控

训练过程每个 epoch 输出（各关键指标并列显示 current | best @ep，best 产生时额外算可懂指标）：
```
  Epoch 12/50  LR: 0.003000
    IC:        current 0.1823  | best 0.2105 @ep8
    DA_score:  current 0.5410  | best 0.5520 @ep10
    Excess DA: current +4.1%   | best +6.2% @ep10
    Combined:  current 0.4521  | best 0.4780 @ep8
    Val_loss:  current 2.31    | best 2.28 @ep7
    Train: 2.35, Time: 03:12
```

产生 best（val_loss/ic/combined 任一刷新）时，额外计算并记录可懂指标三件套（振幅误差率、涨跌停命中率）到 `training_info.json` 的该 epoch 记录与 `best` 块；平凡 epoch 不算（省时）。

**关键指标**：
- **IC**: trajectory IC（去趋势序列上的相关系数，close 主指标）
- **DA_score**: 综合 DA 评分（对数步权重 × 特征权重加权）
- **Excess DA**: DA − naive DA（naive=多数方向比例），正值表示模型优于朴素预测
- **Combined**: IC 和 DA 的综合评分（权重 0.6/0.4）
- **Val_loss**: val 集重建损失（真实前向计算，非 train loss）

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
# 单卡
python finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode block \
    --model mini \
    --checkpoint best_combined_model \
    --n-samples -1   # 全量评估

# 多卡 DDP（卡数由 torchrun 控制，单卡即 nproc_per_node=1）
torchrun --nproc_per_node=4 finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --checkpoint best_combined_model
```

**评估支持 DDP**：各 rank 按 `idx % world_size` 分片处理样本，IC/DA 用 `aggregate_ic`/`aggregate_da` 聚合（all_gather）。`--n-samples` 抽样时所有 rank 同 seed 抽同样本再分片，保证覆盖正确。

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
    --split-mode block \
    --model mini \
    --checkpoint best_combined_model \
    --n-samples 100
```

**参数说明**:
- `--model`:模型类型（mini/small/base）或直接传 checkpoint 目录完整路径。传 model_type 时按 `outputs/models/{norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{model_type}/checkpoints/{checkpoint}/` 定位。
- `--checkpoint`:checkpoint 名称（best_combined_model/best_ic_model/best_model/latest_model，默认 best_combined_model），与 eval 一致。
- `--split-mode`:定位 checkpoint 的分割模式（须与训练时一致）。
- `--n-samples`:回测股票数（-1 全量，默认 100）。
- `--signal-center`/`--signal-steepness`:因子打分 sigmoid 参数（默认 0.084/21）。

**回测数据**:
- context 来自 `kline_daily_raw.pkl`（末尾 lookback 根）
- target 来自 `backtest_raw.pkl`（时间隔离）

**回测启动时自动校验**（PR4）：对首个样本重算 context 段归一化（仅用 context 及前 N 步历史，不含 target），与 preprocess 存的对比。不一致则报错（说明归一化用了 target 数据 = 泄露）。

**加载校验**：checkpoint 与模型架构对不上（missing/unexpected keys）直接报错停止，不静默用错权重。

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

echo "=== Step 2: Fine-tune Tokenizer ==="
python finetune/tokenizer/train.py \
    --norm-mode $NORM_MODE \
    --model $MODEL \
    --epochs 30

echo "=== Step 2.5: Validate Tokenizer ==="
python finetune/tokenizer/validate.py \
    --norm-mode $NORM_MODE \
    --model $MODEL

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
├── tokenizer/
│   ├── train.py                  # tokenizer 微调入口
│   └── validate.py               # 微调效果检验
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
│       ├── schema.py             # 数据结构
│       └── utils.py              # 工具函数

outputs/
├── tokenizers/
│   └── {norm_mode}/{model_type}/    # 按 norm_mode 微调后的 tokenizer
│       ├── model.safetensors
│       ├── config.json
│       └── meta.json
└── models/
    └── {norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{model_type}/
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
- **time_split**: 按**每只股票序列内的样本位置序号**切分（`--train-end`/`--val-end` 传整数位置，非日期）。位置≈交易日序号（A 股约 244/年）。如 `--train-end 1340 --val-end 1710` ≈ train 至 2023-06、val 至 2024-12。各股票独立切，长股票切三份，短股票可能全 train。适合模拟真实交易时序

### Q4: 如何选择 checkpoint？

- **best_combined_model**: 推荐，平衡 IC 和 DA
- **best_ic_model**: 若只关心排序能力
- **best_model**: Val loss 最低，但可能与 IC 不一致

### Q5: 加载 checkpoint 报错 "Checkpoint 与模型不匹配"？

eval/backtest 加载 checkpoint 时，若权重与模型架构对不上（missing/unexpected keys 非空），**直接报错停止**，不静默用错权重。常见原因：
- checkpoint 是用别的 model_type 训练的（如 mini checkpoint 加载到 small 模型）
- checkpoint 路径指错（加载了预训练权重而非微调权重）
- 训练时 norm_mode 与评估时不一致（tokenizer 不匹配）

检查 `--model`/`--checkpoint`/`--norm-mode` 是否与训练时一致。

---

## 9. 参数速查表

| 参数 | sliding_ma60 | sliding_ma120 | full_window |
|------|--------------|---------------|-------------|
| 历史需求 | 60 步 | 120 步 | 0 步 |
| 归一化方式 | 滚动 mean/std | 滚动 mean/std | 窗口 mean/std |
| 适用场景 | 通用 | 长期趋势 | 窗口独立 |

| 参数 | mini | small | base |
|------|------|-------|------|
| 预训练 tokenizer | Kronos-Tokenizer-2k | Kronos-Tokenizer-base | Kronos-Tokenizer-base |
| group_size | 5 | 4 | 4 |
| context | 2048 | 512 | 512 |
| 参数量 | 4.1M | 24.7M | 102.3M |
| 训练速度 | 快 | 中 | 慢 |
| 推荐用途 | 快速实验 | 生产 | 高精度 |

> **注意**：vocab_size 由预训练架构固定（2k/base），**不是微调参数**，也不存在 small=4096 的映射。small 与 base 共用 Kronos-Tokenizer-base。详见方案 §1.5.1。

---

*文档更新：2026-06-20*
*对应提交：P1-P5 fixes (5f4cb7a) + M4 fix + 指南对齐*

**2026-06-20 修正**：
- §9 参数速查表：删除杜撰的 vocab_size 2048/4096/8192 映射（small 共用 base tokenizer，非 4096），改为预训练 tokenizer 架构映射
- §3.3 训练参数：`--batch-size` 默认值 32→16（对齐代码）；删除未通过 CLI 暴露的 `--lr-min`/`--early-stopping-patience`（改列 TrainConfig 内参数说明）；新增 `--resume`/`--output-folder`/`--use-block`
- §3.4 输出监控：示例改为 I10 实际并列格式（`current | best @ep` 五指标），补充"best 时算可懂指标三件套"说明
- §3.1 单卡训练：`--batch-size 32`→16（与 §3.3 默认一致）

**2026-06-20 二次对齐（代码审查后）**：
- §2.2 微调命令：补 `--n-train-iter`（步数驱动采样核心参数）+ 采样机制说明
- §2.3 输出路径：补 `checkpoints/best_model`、`final_model` 子目录 + 根目录副本说明（TK5 修复：根目录是加载入口）+ `history.json`
- §4.1 评估命令：补多卡 DDP 示例（`torchrun`）+ DDP 分片/聚合说明（I3）
- §5 回测命令：补 `--checkpoint` 参数（§6.11 修复：backtest 现支持 --checkpoint 定位 checkpoints/{name}，与 eval 一致）；补 `--model`/`--split-mode`/`--n-samples`/`--signal-*` 说明；补 PR4 自动校验 + 加载校验说明
- §8 新增 Q5（加载 checkpoint 报错原因，2-1 修复：对不上直接 raise）