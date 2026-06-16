#!/bin/bash
# Mode7 4卡训练 - Kronos-base + MA60 lb400

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 \
    finetune/predictor/mode2_ma60_t0/train_ddp.py \
    --model base \
    --lookback 400 \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.003 \
    --weight-decay 0.01 \
    --save-folder mode7_base_lb400

echo "Training completed!"