# Kronos 文档索引

本文档提供项目文档的快速索引。

## 项目概述

Kronos 是全球首个开源金融K线基础模型，用于预测金融市场的OHLCV数据。

- **类型**: 深度学习 / 金融预测
- **语言**: Python
- **框架**: PyTorch
- **许可证**: MIT

## 文档目录

### guide/ — 使用指南

| 文档 | 内容 |
|------|------|
| [installation.md](guide/installation.md) | 安装指南、依赖说明 |
| [quick-start.md](guide/quick-start.md) | 快速开始、预测示例 |
| [kronos-quant-usage.md](guide/kronos-quant-usage.md) | 量化使用手册、信号提取、接入指南 |
| [full-a-share-finetune.md](guide/full-a-share-finetune.md) | 全 A 股微调训练方案、数据规范、训练策略 |

### reference/ — 参考文档

| 文档 | 内容 |
|------|------|
| [api-reference.md](reference/api-reference.md) | API参考、类和方法说明 |
| [model-zoo.md](reference/model-zoo.md) | 模型仓库、可用模型列表 |

### tutorial/ — 教程

| 文档 | 内容 |
|------|------|
| [finetuning.md](tutorial/finetuning.md) | 微调指南、Qlib管道 |

### 根目录

| 文档 | 内容 |
|------|------|
| [overview.md](overview.md) | 项目概述、核心创新、项目结构 |

## 核心模块

### model/
- `KronosTokenizer` - K线数据分词器
- `Kronos` - Decoder-only Transformer
- `KronosPredictor` - 高级预测接口

### examples/
预测示例脚本

### finetune/
Qlib 微调管道

### webui/
Flask Web界面

## 快速使用

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

## 资源链接

- 论文: https://arxiv.org/abs/2508.02739
- HuggingFace: https://huggingface.co/NeoQuasar
- Demo: https://shiyu-coder.github.io/Kronos-demo/
