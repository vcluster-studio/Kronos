#!/bin/bash
# Mode8 4卡训练 - Kronos-small + MA60 lb400

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 \
    finetune/predictor/mode2_ma60_t0/train_ddp.py \
    --model small \
    --lookback 400 \
    --epochs 80 \
    --batch-size 128 \
    --lr 0.002 \
    --weight-decay 0.01 \
    --save-folder mode8_small_lb400

echo "Training completed!"