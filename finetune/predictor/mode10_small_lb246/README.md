# Mode10: Kronos-small + MA60 lb246_pd10

## 适配分析

small 模型 max_context=512 是 s1+s2 的总限制：
- s1 最多 256 tokens
- s2 最多 256 tokens
- 对应最多 **256 个 K线点**

## 配置

| Mode | Model | max_context | lb+pd | 实际K线点 |
|------|-------|-------------|-------|-----------|
| Mode8 | small | 512 | 400+10=410 | **截断** |
| Mode9 | small | 512 | 60+10=70 | 70 |
| **Mode10** | small | 512 | **246+10=256** | **256 (刚好填满)** |

## 训练参数

```python
epochs = 80
batch_size = 64 x 4 = 256
lr = 0.002
weight_decay = 0.02
lookback = 246
predict = 10
```

## 启动

```bash
sh run_ddp.sh
```

或手动：
```bash
torchrun --nproc_per_node=4 train.py --batch-size 64 --epochs 80
```

## 预期

lb246 是 small 模型的极限配置，比 lb400 无截断、比 lb60 有更多历史信息。预期：
- IC 稳定性优于 Mode9
- 可能达到或超过 Mode8 的 0.14