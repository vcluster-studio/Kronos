# Predictor 共享模块

四种模式共享的代码模块。

## 模块

| 文件 | 功能 |
|------|------|
| `trainer.py` | 基础 Trainer 类、Callbacks (SaveBestModel, EvalCallback) |
| `dataset.py` | 基础 Dataset 类 (KronosWindowDataset) |
| `metrics.py` | 指标计算 (IC, DA, MAE, Bucket 评估) |
| `loss.py` | Loss 实现 (CE, Delta loss, Horizon 衰减, Sample weight) |
| `decode.py` | 解码工具 (soft_decode, bit_mask precompute) |

## 使用

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.dataset import KronosWindowDataset
from shared.loss import weighted_ce_loss, delta_loss
from shared.metrics import compute_ic, compute_da
from shared.trainer import SaveBestModelCallback
```
