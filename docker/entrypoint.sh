#!/bin/bash

# Kronos Docker 入口脚本
# 支持训练、评估、推理

set -e

# 默认路径配置
DATA_DIR=${DATA_DIR:-/app/data}
MODEL_DIR=${MODEL_DIR:-/app/models}
TOKENIZER_DIR=${TOKENIZER_DIR:-/app/tokenizers}
PRETRAINED_DIR=${PRETRAINED_DIR:-/app/pretrained}
OUTPUT_DIR=${OUTPUT_DIR:-/app/outputs}

# 显示帮助
show_help() {
    echo "Kronos Docker 容器"
    echo ""
    echo "使用方式:"
    echo "  docker run kronos:latest <command> [options]"
    echo ""
    echo "命令:"
    echo "  train       单GPU训练"
    echo "  train-ddp   多GPU DDP训练"
    echo "  eval        单GPU评估"
    echo "  eval-ddp    多GPU DDP评估"
    echo "  shell       进入shell"
    echo ""
    echo "通用参数:"
    echo "  --model NAME        模型类型: mini/small/base (默认mini)"
    echo "  --lookback N        回看窗口大小 (默认400)"
    echo "  --epochs N          训练轮数 (默认50)"
    echo "  --batch-size N      批大小 (默认16)"
    echo "  --lr FLOAT          学习率 (默认0.003)"
    echo "  --weight-decay F    权重衰减 (默认0.01)"
    echo "  --train-samples N   每epoch训练样本数 (-1为全量)"
    echo "  --n-samples N       验证+IC评估样本数 (-1为全量)"
    echo "  --norm-mode MODE    归一化模式: ma60/full_window (默认ma60)"
    echo "  --use-block         使用block分层数据集"
    echo "  --resume PATH       恢复训练的checkpoint路径"
    echo "  --save-folder NAME  输出文件夹名"
    echo ""
    echo "Eval专用参数:"
    echo "  --models NAME       预定义模型名 (如 latest_mini_lb400)"
    echo "  --model-path PATH   直接指定checkpoint路径"
    echo "  --test-data PATH    测试数据路径"
    echo "  --checkpoint NAME   checkpoint类型 (如 best_combined_model)"
    echo "  --output PATH       输出JSON文件路径"
    echo "  --seed N            随机种子 (默认42)"
    echo ""
    echo "示例:"
    echo "  # 单GPU训练"
    echo "  docker run kronos:latest train --use-block --epochs 50"
    echo ""
    echo "  # 多GPU DDP训练"
    echo "  docker run kronos:latest train-ddp --use-block --model small --epochs 50"
    echo ""
    echo "  # 单GPU评估"
    echo "  docker run kronos:latest eval --models latest_mini_lb400 --n-samples 500"
    echo ""
    echo "  # 多GPU评估"
    echo "  docker run kronos:latest eval-ddp --model-path outputs/models/xxx/checkpoints/best_ic_model"
}

case "$1" in
    --help|-h|help)
        show_help
        exit 0
        ;;

    train)
        shift
        python /app/finetune/predictor/mode2_ma60_t0/train.py \
            $@
        ;;

    train-ddp)
        shift
        NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
        torchrun --nproc_per_node=${NUM_GPUS} \
            /app/finetune/predictor/mode2_ma60_t0/train_ddp.py \
            $@
        ;;

    eval)
        shift
        python /app/finetune/predictor/eval_all.py \
            $@
        ;;

    eval-ddp)
        shift
        NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
        torchrun --nproc_per_node=${NUM_GPUS} \
            /app/finetune/predictor/shared/eval_ddp.py \
            $@
        ;;

    shell)
        exec /bin/bash
        ;;

    *)
        echo "未知命令: $1"
        show_help
        exit 1
        ;;
esac