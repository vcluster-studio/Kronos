# Mode7: Kronos-base + MA60 lb400_pd10 + DDP

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Kronos-base (102.3M) |
| Tokenizer | MA60 Tokenizer base_v1 (兼容 Kronos-Tokenizer-base) |
| Lookback | 400 |
| Predict | 10 |
| 单卡 Batch | 32 |
| 总 Batch | 128 (4卡) |
| Learning Rate | 0.003 |
| Mixed Precision | fp16 |

## Tokenizer 兼容性

**关键说明**: Kronos-base 需要 Kronos-Tokenizer-base vocabulary。

MA60 Tokenizer (`ma60_tokenizer_base_v1`) 基于 Kronos-Tokenizer-base 的词汇表进行微调，
**完全兼容** Kronos-base 模型。

验证方式:
```python
from model.kronos import KronosTokenizer, Kronos

# MA60 tokenizer (base vocabulary)
tokenizer = KronosTokenizer.from_pretrained("outputs/tokenizers/ma60_tokenizer_base_v1/checkpoints/best_model")

# Kronos-base 模型
model = Kronos.from_pretrained("pretrained/Kronos-base")

# 编码测试
import numpy as np
x = np.random.randn(100, 6).astype(np.float32)
tokens = tokenizer.encode(x)  # 正常工作
```

## 启动方式

### 多卡 DDP (推荐)
```bash
bash run_ddp.sh
```

或手动启动:
```bash
torchrun --nproc_per_node=4 finetune/predictor/mode7_base_lb400/train.py
```

### 单卡测试
```bash
python finetune/predictor/mode7_base_lb400/train.py
```

## 数据

使用 `finetune/data/ma60_norm/windowed_lb400_pd10/` 数据，与 Mode2 相同。

## 评估

```bash
python finetune/predictor/mode7_base_lb400/eval.py
```

## 输出

模型保存在 `outputs/models/mode7_base_lb400_ddp/checkpoints/`:
- `best_model/` - 最佳 validation loss
- `best_ic_model/` - 最佳 IC
- `latest_model/` - 最新模型
- `final_model/` - 最终模型

## 与 Mode2 对比

| 项目 | Mode2 | Mode7 |
|------|-------|-------|
| 模型 | Kronos-mini (4.1M) | Kronos-base (102.3M) |
| 参数量 | 4.1M | 102.3M |
| 单卡 Batch | 16 | 32 |
| 训练卡数 | 1 | 4 |
| 总 Batch | 16 | 128 |
| 预期效果 | IC=0.19 | 更高 (待验证) |