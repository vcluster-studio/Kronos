# Kronos 微调模块

本目录包含 Kronos 模型的微调训练和评估脚本。

## 目录结构

```
finetune/
├── README.md                   # 本文档
├── train_tokenizer.py          # Tokenizer训练（内置配置）
├── train_predictor.py          # Predictor训练（内置配置）
├── unified_test.py             # 统一测试脚本（固定随机种子）
├── validate_tokenizer.py       # Tokenizer重建误差验证
├── dataset.py                  # 数据集类定义
├── csv_data_preprocess.py      # CSV数据预处理
├── data/                       # 数据目录
│   ├── processed_datasets/     # 全A股
│   ├── processed_datasets_mid/ # 中盘股
│   ├── processed_datasets_small/ # 小盘股
│   └── processed_datasets_mid_small/ # 中小盘混合
└── utils/                      # 工具函数
```

## 快速开始

### 1. Tokenizer 微调

```bash
# 训练 mid 数据集的 tokenizer
python -u finetune/train_tokenizer.py --dataset mid > logs/tokenizer_mid.log 2>&1 &
tail -f logs/tokenizer_mid.log

# 训练 full 数据集的 tokenizer
python -u finetune/train_tokenizer.py --dataset full > logs/tokenizer_full.log 2>&1 &
```

### 2. Predictor 微调

```bash
# 训练 mini 模型，使用 mid 数据集
python -u finetune/train_predictor.py --model mini --dataset mid > logs/predictor_mid.log 2>&1 &
tail -f logs/predictor_mid.log

# 训练 small 模型，使用 full 数据集
python -u finetune/train_predictor.py --model small --dataset full > logs/predictor_small.log 2>&1 &
```

### 3. 评估模型

```bash
# 统一测试（所有模型组合）
python -u finetune/unified_test.py

# Tokenizer重建误差验证
python -u finetune/validate_tokenizer.py
```

## 参数说明

### Tokenizer训练 (--dataset)

| 参数 | 数据集 | 说明 |
|------|--------|------|
| mid | 中盘股 | 默认 |
| full | 全A股 | large+mid+small |
| small | 小盘股 | |
| mid_small | 中小盘 | |

### Predictor训练

**--model** 模型尺寸：
| 参数 | 模型 | 参数量 |
|------|------|--------|
| mini | Kronos-mini | 4.1M |
| small | Kronos-small | 24.7M |
| base | Kronos-base | 102.3M |

**--dataset** 数据集：
| 参数 | 数据集 | 说明 |
|------|--------|------|
| mid | 中盘股 | 默认 |
| full | 全A股 | |
| small | 小盘股 | |
| mid_small | 中小盘 | |

**--tokenizer** Tokenizer类型：
| 参数 | 说明 |
|------|------|
| finetuned | 使用训练后的 tokenizer（默认） |
| pretrained | 使用原始 tokenizer |

## 新增配置

在脚本顶部的配置变量中添加：

```python
# train_tokenizer.py / train_predictor.py
DATASET_CONFIGS = {
    'new_dataset': {
        'name': '新数据集',
        'path': 'finetune/data/new_dataset',
        'save_folder': 'new_tokenizer_v1',
    },
}

# train_predictor.py
MODEL_CONFIGS = {
    'new_model': {
        'name': 'Kronos-new',
        'pretrained_path': 'path/to/model',
        'params': 'XXM',
    },
}

# unified_test.py
MODEL_CONFIGS = {
    'new_model-trained': {
        'name': 'new (训练后)',
        'tokenizer': 'path/to/tokenizer',
        'predictor': 'path/to/predictor',
    },
}
```

## 输出路径

- 训练模型: `outputs/models/`
- 最终模型: `final_models/`
- 训练日志: `logs/`（自行创建）

## 文档

完整经验总结见 `final_models/TRAINING_EXPERIENCE.md`。