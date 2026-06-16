#!/bin/bash
# 多GPU全量评估脚本

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 \
    finetune/predictor/shared/eval_ddp.py \
    --models mode2_mini_lb400 mode8_small_lb400 mode10_small_lb246 mode1_global \
    --output outputs/full_test_results.json

echo "Evaluation completed!"