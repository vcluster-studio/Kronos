# Mode8: Kronos-small + MA60 lb400_pd10

## 配置对比

| 项目 | Mode2 (mini) | Mode7 (base) | Mode8 (small) |
|------|--------------|--------------|---------------|
| 参数量 | 4.1M | 102.3M | **24.7M** |
| LR | 0.003 | 0.003 | **0.002** |
| Weight Decay | 0.01 | 0.01 | **0.02** |
| Batch | 16 | 128 | **192** |
| 目标 IC | 0.19 | TBD | **>0.19** |

## 训练策略

针对中型模型优化：
- **较低 LR**：0.002（mini用0.003，base需要更低）
- **中等正则化**：weight_decay=0.02
- **更大 batch**：64 x 3 = 192
- **更长训练**：80 epochs，grace period=10

## 启动

```bash
sh run_ddp.sh
```

或手动：
```bash
torchrun --nproc_per_node=3 train.py --batch-size 64 --epochs 80
```

## 预期

根据之前 MA60-small 实验（IC=0.1646），配合优化后的训练策略，预期：
- **Test IC > 0.19**（超越 Mode2）
- 比 base (102M) 更稳定
- 比 mini (4M) 更强