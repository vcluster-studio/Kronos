# Kronos Docker 使用指南

## 构建镜像

```bash
cd docker
docker build -t kronos:latest .
```

## 运行方式

### 1. 直接运行

```bash
# 单GPU训练
docker run --gpus 1 \
    -v /path/to/finetune/data:/app/data \
    -v /path/to/outputs:/app/outputs \
    -v /path/to/pretrained:/app/pretrained \
    kronos:latest train --model mini --epochs 50

# 多GPU DDP训练 (4卡)
docker run --gpus all \
    -v ./finetune/data:/app/data \
    -v ./outputs:/app/outputs \
    -v ./pretrained:/app/pretrained \
    kronos:latest train-ddp --model mini --epochs 50

# 评估
docker run --gpus all \
    -v ./outputs/models:/app/models \
    -v ./outputs/tokenizers:/app/tokenizers \
    kronos:latest eval --quick 1000

# 进入shell
docker run --gpus all -it \
    -v ./finetune/data:/app/data \
    -v ./outputs:/app/outputs \
    kronos:latest shell
```

### 2. 使用docker-compose

```bash
# 单GPU训练
docker-compose up kronos-train

# 多GPU训练
docker-compose up kronos-train-ddp

# 评估
docker-compose up kronos-eval

# 进入shell
docker-compose run kronos-shell
```

## 挂载卷说明

| 容器路径 | 内容 | 外部路径建议 |
|---------|------|-------------|
| `/app/data` | 训练数据 | `finetune/data` |
| `/app/models` | 模型权重 | `outputs/models` |
| `/app/tokenizers` | Tokenizer | `outputs/tokenizers` |
| `/app/pretrained` | 预训练模型 | `pretrained` |
| `/app/outputs` | 输出目录 | `outputs` |

## 环境变量

```bash
# 设置可见GPU
-e CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置数据路径
-e DATA_DIR=/app/data
-e MODEL_DIR=/app/models
```

## 完整示例

```bash
# 从项目根目录运行
docker run --gpus all \
    -v $(pwd)/finetune/data:/app/data \
    -v $(pwd)/outputs/models:/app/models \
    -v $(pwd)/outputs/tokenizers:/app/tokenizers \
    -v $(pwd)/pretrained:/app/pretrained \
    -v $(pwd)/outputs:/app/outputs \
    kronos:latest eval --test
```