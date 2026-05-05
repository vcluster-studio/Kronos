# Kronos 项目索引

本文档为 Claude Code 提供项目上下文索引。

## 项目概述

Kronos 是全球首个开源金融K线基础模型，用于预测金融市场的OHLCV数据。

- **类型**: 深度学习 / 金融预测
- **语言**: Python
- **框架**: PyTorch
- **许可证**: MIT

## 核心模块

### model/
核心模型实现，包含三个主要类：
- `KronosTokenizer` - K线数据分词器
- `Kronos` - Decoder-only Transformer 模型
- `KronosPredictor` - 高级预测接口

**导入方式**:
```python
from model import Kronos, KronosTokenizer, KronosPredictor
```

### examples/
预测示例脚本：
- `prediction_example.py` - 基本预测示例
- `prediction_batch_example.py` - 批量预测示例
- `prediction_wo_vol_example.py` - 无成交量预测

### finetune/
Qlib 微调管道：
- `config.py` - 配置管理
- `train_tokenizer.py` - 分词器微调
- `train_predictor.py` - 预测器微调

### webui/
Flask Web界面，运行命令：`python webui/run.py`

## 快速使用

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

pred_df = predictor.predict(df, x_timestamp, y_timestamp, pred_len=120)
```

## 数据格式要求

DataFrame 必须包含：
- `open`, `high`, `low`, `close` (必需)
- `volume`, `amount` (可选)

时间戳需要转换为 datetime 类型。

## 可用模型

| 模型 | 参数量 | 上下文 |
|------|--------|--------|
| Kronos-mini | 4.1M | 2048 |
| Kronos-small | 24.7M | 512 |
| Kronos-base | 102.3M | 512 |

## 文档索引

详细文档位于 `docs/` 目录：
- [MEMORY.md](docs/MEMORY.md) - 文档索引
- [overview.md](docs/overview.md) - 项目概述
- [installation.md](docs/installation.md) - 安装指南
- [quick-start.md](docs/quick-start.md) - 快速开始
- [api-reference.md](docs/api-reference.md) - API参考
- [finetuning.md](docs/finetuning.md) - 微调指南
- [model-zoo.md](docs/model-zoo.md) - 模型仓库

## 资源链接

- 论文: https://arxiv.org/abs/2508.02739
- HuggingFace: https://huggingface.co/NeoQuasar
- Demo: https://shiyu-coder.github.io/Kronos-demo/
