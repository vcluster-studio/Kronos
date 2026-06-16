#!/bin/bash
# Mode1 4卡训练 - Kronos-mini + Full Window 归一化
# 数据: finetune/data/global_norm/full_series/ (时间切分: train 2018-2023, val 2023-2024, test 2024-2026)
# Tokenizer: pretrained Kronos-Tokenizer-2k
# 归一化: 训练时动态 full_window 归一化（与 pretrained 原始方式一致）

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 \
    finetune/predictor/mode2_ma60_t0/train_ddp.py \
    --model mini \
    --norm-mode full_window \
    --lookback 200 \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.003 \
    --weight-decay 0.01 \
    --save-folder mode1_full_window_ddp

echo "Training completed!"
