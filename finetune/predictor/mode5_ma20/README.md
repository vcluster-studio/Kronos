# Mode5: MA20 归一化预测器

使用 MA20 滑动归一化数据训练 predictor。

## 特点

- **MA20 归一化**: 每个时间点使用前 20 步的局部均值/标准差
- **更高波动率**: 比 MA60 保留更多波动信息
- **更适合 A 股**: A 股波动率低，MA20 能保留更多有效信号

## 训练

```bash
# 先生成 MA20 数据
python -u finetune/data/ma20_norm/windowed_lb400_pd10/generate.py

# 训练
python -u finetune/predictor/mode5_ma20/train.py
```

## 配置

| 参数 | 默认值 |
|------|--------|
| MA Window | 20 |
| Lookback | 400 |
| Predict | 10 |
| Epochs | 30 |
| Batch Size | 16 |
| Learning Rate | 0.003 |

## 对比实验

| 模式 | 归一化方式 | 窗口长度 | 特点 |
|------|-----------|---------|------|
| mode1 | full_window | 200 | 全窗口归一化 |
| mode2 | MA60 | 400 | 3个月滑动基准 |
| mode3 | MA60 sliding | 60 | 短窗口滑动 |
| mode4 | MA60 weighted | 60 | 加权损失 |
| **mode5** | **MA20** | **400** | **1个月滑动基准，高波动率** |