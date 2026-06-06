# 数据集文档

## 命名规范

```
ma60_daily_lb{lookback}_pd{predict}_s{stride}
```

- `ma60` — MA60 滑动归一化
- `daily` — 日 K 线
- `lb` — lookback（历史步数）
- `pd` — predict（预测步数）
- `s` — stride（窗口步长）

## 数据集结构

每个数据集目录包含：

| 文件 | 内容 |
|------|------|
| `train_data.pkl` | 训练窗口列表 |
| `val_data.pkl` | 验证窗口列表 |
| `test_data.pkl` | 测试窗口列表 |
| `meta.pkl` | 元信息（参数、样本数） |

每个窗口是一个 dict：

| 键 | 形状 | 说明 |
|----|------|------|
| `symbol` | str | 股票代码 |
| `start` | int | 窗口起始索引 |
| `normalized` | (window, 6) | MA60 归一化数据 |
| `original` | (window, 6) | 原始价格 |
| `means` | (window, 6) | MA60 滚动均值 |
| `stds` | (window, 6) | MA60 滚动标准差 |
| `index` | (window,) | 时间戳 |

特征顺序：`[open, high, low, close, vol, amt]`

## 切分策略

- 每只股票按时间排序，前 70% 窗口 → train，中 15% → val，后 15% → test
- 时间外推：val/test 的预测目标严格在 train 之后，杜绝未来信息泄漏
- 非重叠（stride = window）：同一数据点不会同时出现在输入和预测目标中

## 当前数据集

### `ma60_daily_lb60_pd1_s61`（最新）

| 参数 | 值 |
|------|-----|
| lookback | 60 |
| predict | 1 |
| window | 61 |
| stride | 61 |
| 窗口重叠 | 无 |
| train 样本 | 92,374 |
| val 样本 | 20,160 |
| test 样本 | 21,009 |
| 股票数 | 4,797 |

生成命令：
```bash
python -c "
import pickle, numpy as np, os
LOOKBACK = 60; PREDICT = 1; WINDOW = LOOKBACK + PREDICT; STRIDE = 61
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
with open('finetune/data/kline_daily_ma60.pkl', 'rb') as f:
    raw = pickle.load(f)
# ... 见 preprocess_windowed.py 的逻辑
"
```

### `processed_datasets_ma60_windowed_v3`（旧版）

| 参数 | 值 |
|------|-----|
| lookback | 400 |
| predict | 10 |
| window | 411（含 +1 泄漏） |
| stride | 10 |
| 窗口重叠 | 有（stride < window） |
| 数据泄漏 | 有（window_size 多 +1） |

**注意**：v3 存在已知的数据泄漏问题，已不再用于新实验。保留仅供对比参考。

## 废弃数据集

以下为早期 Qlib 管线或中间实验产物，已不再使用：

- `processed_datasets` / `_clean` / `_new` — Qlib 管线
- `processed_datasets_small` / `_mid` / `_mid_small` — 市值分组实验
- `processed_datasets_ma60` / `_small_ma60` / `_mid_ma60` — 早期 MA60 实验
- `processed_datasets_ma60_windowed` / `_v2` — 旧版本窗口化数据
- `processed_datasets_ma60_windowed_v3_small` — small 模型的 v3 变体

## 源数据

`kline_daily_ma60.pkl` — 日 K MA60 归一化全量数据（4,797 只股票）

由 `finetune/preprocess_kline_daily_ma60.py` 从 SQL 生成，包含 normalized / original / means / stds / index。
