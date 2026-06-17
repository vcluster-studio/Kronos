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
    echo "  eval        评估模型"
    echo "  inference   推理预测"
    echo "  shell       进入shell"
    echo ""
    echo "挂载卷:"
    echo "  -v /path/to/data:/app/data"
    echo "  -v /path/to/models:/app/models"
    echo "  -v /path/to/tokenizers:/app/tokenizers"
    echo "  -v /path/to/pretrained:/app/pretrained"
    echo "  -v /path/to/outputs:/app/outputs"
    echo ""
    echo "示例:"
    echo "  # 单GPU训练"
    echo "  docker run -v ./finetune/data:/app/data -v ./outputs:/app/output kronos:latest train --model mini"
    echo ""
    echo "  # 多GPU评估"
    echo "  docker run --gpus all -v ./outputs/models:/app/models kronos:latest eval --quick 1000"
}

case "$1" in
    --help|-h|help)
        show_help
        exit 0
        ;;

    train)
        shift
        python /app/finetune/predictor/mode2_ma60_t0/train.py \
            --tokenizer ${TOKENIZER_DIR}/ma60_tokenizer_base_v1/checkpoints/best_model \
            --data_path ${DATA_DIR}/ma60_norm/windowed_lb400_pd10 \
            --save_dir ${OUTPUT_DIR}/models \
            $@
        ;;

    train-ddp)
        shift
        # 获取GPU数量
        NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
        torchrun --nproc_per_node=${NUM_GPUS} \
            /app/finetune/predictor/mode2_ma60_t0/train_ddp.py \
            --tokenizer ${TOKENIZER_DIR}/ma60_tokenizer_base_v1/checkpoints/best_model \
            --data_path ${DATA_DIR}/ma60_norm/windowed_lb400_pd10 \
            --save_dir ${OUTPUT_DIR}/models \
            $@
        ;;

    eval)
        shift
        python /app/finetune/predictor/shared/eval_all_models.py \
            --model_dir ${MODEL_DIR} \
            --output ${OUTPUT_DIR}/eval_results.json \
            $@
        ;;

    eval-ddp)
        shift
        NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
        torchrun --nproc_per_node=${NUM_GPUS} \
            /app/finetune/predictor/shared/eval_ddp.py \
            --model_dir ${MODEL_DIR} \
            --output ${OUTPUT_DIR}/eval_results.json \
            $@
        ;;

    inference)
        shift
        python /app/finetune/predictor/mode2_ma60_t0/inference.py \
            --model ${MODEL_DIR}/$2 \
            --tokenizer ${TOKENIZER_DIR}/ma60_tokenizer_base_v1/checkpoints/best_model \
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