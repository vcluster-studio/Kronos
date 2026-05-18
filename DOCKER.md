# Kronos Docker 部署指南

## 前置准备

### 1. 宿主机下载预训练权重

权重通过宿主机挂载，**不在容器内下载**：

```bash
mkdir -p /root/kronos/pretrained

# 从 HuggingFace 下载
cd /root/kronos/pretrained
git clone https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k
git clone https://huggingface.co/NeoQuasar/Kronos-mini
```

目录结构：
```
/root/kronos/pretrained/
├── Kronos-Tokenizer-2k/    # 2k 分词器（mini 模型用）
└── Kronos-mini/            # mini 预测器
```

### 2. 准备 CSV 数据

```bash
mkdir -p /root/kronos/csv/exported_kline_data/stocks
# 将 K 线 CSV 文件放入 stocks 目录
```

## 构建镜像

```bash
docker build -t kronos-train .
```

## 运行训练

### 单任务

```bash
# 训练 Tokenizer
docker run --gpus all \
    -v /root/kronos/pretrained:/app/pretrained:ro \
    -v /root/kronos/csv:/app/finetune_csv \
    -v /root/kronos/output:/app/output \
    -v /root/kronos/logs:/app/logs \
    kronos-train bash entrypoint.sh --task tokenizer --dataset mid

# 训练 Predictor
docker run --gpus all \
    -v /root/kronos/pretrained:/app/pretrained:ro \
    -v /root/kronos/csv:/app/finetune_csv \
    -v /root/kronos/output:/app/output \
    -v /root/kronos/logs:/app/logs \
    kronos-train bash entrypoint.sh --task predictor --dataset mid
```

### 完整流程

```bash
docker run --gpus all \
    -v /root/kronos/pretrained:/app/pretrained:ro \
    -v /root/kronos/csv:/app/finetune_csv \
    -v /root/kronos/output:/app/output \
    -v /root/kronos/logs:/app/logs \
    kronos-train bash entrypoint.sh --task pipeline --dataset mid
```

### Docker Compose

```bash
# 设置宿主机路径
export PRETRAINED_DIR=/root/kronos/pretrained
export DATA_DIR=/root/kronos/csv
export OUTPUT_DIR=/root/kronos/output
export LOG_DIR=/root/kronos/logs

# 训练 Tokenizer
docker compose run --gpus all tokenizer

# 完整流程
docker compose run --gpus all pipeline
```

## 可用任务

| 任务 | 说明 |
|------|------|
| `--task tokenizer` | 训练 Tokenizer |
| `--task predictor` | 训练 Predictor |
| `--task preprocess` | 仅数据预处理 |
| `--task evaluate` | 运行评估 |
| `--task pipeline` | 完整流程：预处理→Tokenizer→Predictor→评估 |
| `--task shell` | 进入交互式 Shell |

## 可用数据集

| 数据集 | 说明 | 参数 |
|--------|------|------|
| Mid | 中盘股 | `--dataset mid` |
| Small | 小盘股 | `--dataset small` |
| Mid+Small | 中+小盘 | `--dataset mid_small` |
| Full | 全 A 股 | `--dataset full` |

## 可选参数

```bash
--epochs 30              # Tokenizer 训练轮数
--batch-size 16          # 批次大小
--lr 0.0002              # 学习率
--predictor-epochs 20    # Predictor 训练轮数
--predictor-lr 0.01      # Predictor 学习率
```

## Volume 挂载

| 容器路径 | 宿主机内容 | 读写 | 是否必须 |
|----------|-----------|------|----------|
| `/app/pretrained` | 预训练权重 | **ro** | 训练/评估必须 |
| `/app/finetune_csv` | CSV 原始数据 | rw | 预处理必须 |
| `/app/output` | 训练输出 | rw | 建议 |
| `/app/logs` | 训练日志 | rw | 建议 |

> **注意**: `/app/pretrained` 挂载为只读 (`:ro`)，容器不会修改预训练权重。
> 如果权重缺失，容器启动时会报错并提示下载方式。

## 云 GPU 平台

### AutoDL

```bash
# 1. 选择 PyTorch + CUDA 12.1 镜像
# 2. 上传项目到 /root/Kronos
# 3. 下载权重到数据盘
cd /root/autodl-tmp
git clone https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k
git clone https://huggingface.co/NeoQuasar/Kronos-mini

# 4. 构建并运行
cd /root/Kronos
docker build -t kronos-train .
docker run --gpus all \
    -v /root/autodl-tmp:/app/pretrained:ro \
    -v /root/autodl-tmp/csv:/app/finetune_csv \
    -v /root/autodl-tmp/output:/app/output \
    kronos-train bash entrypoint.sh --task pipeline --dataset mid --epochs 30
```

## 故障排查

```bash
# 验证 GPU
docker run --gpus all kronos-train python -c "import torch; print(torch.cuda.is_available())"

# 验证权重挂载
docker run --rm -v /root/kronos/pretrained:/app/pretrained:ro kronos-train \
    ls -la /app/pretrained/

# 进入 Shell 调试
docker run --gpus all -it \
    -v /root/kronos/pretrained:/app/pretrained:ro \
    -v /root/kronos/csv:/app/finetune_csv \
    kronos-train bash entrypoint.sh --task shell
```
