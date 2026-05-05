# Kronos 项目概述

## 简介

**Kronos** 是全球首个开源的金融K线（蜡烛图）基础模型，训练数据覆盖超过45个全球交易所。与通用时间序列预测模型不同，Kronos 专门为处理金融数据的独特高噪声特性而设计。

## 核心创新

Kronos 采用两阶段框架：

1. **专用分词器 (Tokenizer)**：将连续多维K线数据（OHLCV）量化为**层级离散token**
2. **大型自回归Transformer**：在这些token上进行预训练，可服务于多种量化任务

## 项目结构

```
Kronos/
├── model/                  # 核心模型实现
│   ├── __init__.py        # 导出: KronosTokenizer, Kronos, KronosPredictor
│   ├── kronos.py          # 主模型类定义
│   └── module.py           # Transformer模块和量化组件
├── examples/              # 使用示例和预测脚本
│   ├── prediction_example.py
│   ├── prediction_batch_example.py
│   └── ...
├── finetune/              # Qlib微调管道
│   ├── config.py          # 配置文件
│   ├── train_tokenizer.py
│   ├── train_predictor.py
│   └── qlib_data_preprocess.py
├── finetune_csv/          # CSV格式微调
├── webui/                 # Web界面 (Flask)
│   ├── app.py
│   └── run.py
├── tests/                 # 单元测试
├── requirements.txt       # 依赖
└── README.md
```

## 核心类

### KronosTokenizer
K-line数据分词器，使用混合量化方法：
- 编码器-解码器Transformer架构
- BSQuantizer（二进制球面量化器）进行数据压缩
- 支持半量化模式（half quantization）

### Kronos
主模型类，decoder-only Transformer：
- 层级嵌入（HierarchicalEmbedding）
- 时间嵌入（TemporalEmbedding）
- 双头输出（DualHead）预测s1和s2 token

### KronosPredictor
高级预测接口：
- 自动处理数据预处理和后处理
- 支持单序列和多序列批量预测
- 自动设备检测（CUDA/MPS/CPU）

## 资源链接

- **论文**: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739)
- **Hugging Face**: [NeoQuasar](https://huggingface.co/NeoQuasar)
- **在线演示**: [Kronos Demo](https://shiyu-coder.github.io/Kronos-demo/)
- **许可证**: MIT

## 新闻

- **2025.11.10**: Kronos 被 AAAI 2026 接收
- **2025.08.17**: 发布微调脚本
- **2025.08.02**: 论文发布于 arXiv