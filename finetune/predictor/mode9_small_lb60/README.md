# Mode9: Kronos-small + MA60 lb60_pd10

## 目的

验证假设：small 模型（max_context=512）对短周期更敏感。

## 对比实验

| Mode | Model | Context | Lookback | 目标 IC |
|------|-------|---------|----------|---------|
| Mode2 | mini (4M) | 2048 | 400 | 0.19+ |
| Mode8 | small (25M) | 512 | 400 | >0.16 |
| **Mode9** | small (25M) | 512 | **60** | **>0.19?** |

## 假设

small/base 的 max_context=512，对于 lb400 序列（470 tokens）几乎填满。
- 在短周期（lb60，序列130 tokens）上，可能表现更好
- 测试模型对序列长度的敏感性

## 数据

使用 `windowed_lb60_pd10` 数据（已存在）。

## 启动

```bash
sh run_ddp.sh
```

或手动：
```bash
torchrun --nproc_per_node=3 train.py --batch-size 64 --epochs 80
```

## 训练参数

```python
epochs = 80
batch_size = 64 x 3 = 192
lr = 0.002
weight_decay = 0.02
lookback = 60
predict = 10
```