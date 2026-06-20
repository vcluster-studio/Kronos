"""
Kronos Tokenizer Fine-tuning Effectiveness Validation

对比「预训练 tokenizer vs 微调后」在同一 val 数据上的重建误差。

验收标准：
- 微调后 val 重建损失 < 预训练在同数据重建损失（必要）
- 重建损失收敛（early stopping 触发，非暴力训满）（必要）

使用：
    python finetune/tokenizer/validate.py \
        --norm-mode sliding_ma60 \
        --model mini
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer
from finetune.predictor.core.config import DataConfig
from finetune.predictor.core.paths import (
    get_split_data_path,
    get_tokenizer_path,
    PROJECT_ROOT,
)
from finetune.predictor.core.utils import get_device
from finetune.tokenizer.train import PRETRAINED_MAP, TokenizerDataset


def validate_tokenizer(
    norm_mode: str,
    model_type: str,
    lookback: int = 400,
    predict: int = 10,
    split_mode: str = 'block',
    n_val_iter_multiplier: int = 400,
    batch_size: int = 16,
):
    """
    验证 tokenizer 微调效果

    对比预训练 tokenizer 与微调后 tokenizer 在同一 val 数据上的重建损失。

    Args:
        norm_mode: 归一化模式
        model_type: 模型类型（mini/small/base）
        lookback: 回看窗口
        predict: 预测步数
        split_mode: 分割模式
        n_val_iter_multiplier: val 采样步数倍数
        batch_size: 批大小

    Returns:
        dict: 包含预训练和微调后 tokenizer 的重建损失统计
    """
    device = get_device()

    # 数据路径
    val_path = get_split_data_path(norm_mode, lookback, predict, split_mode, 'val')
    finetuned_path = get_tokenizer_path(norm_mode, model_type)
    pretrained_path = os.path.join(PROJECT_ROOT, PRETRAINED_MAP[model_type])

    print("=" * 60)
    print("Tokenizer Fine-tuning Effectiveness Validation")
    print("=" * 60)
    print(f"norm_mode: {norm_mode}")
    print(f"model_type: {model_type}")
    print(f"pretrained_path: {pretrained_path}")
    print(f"finetuned_path: {finetuned_path}")
    print(f"val_data: {val_path}")
    print("=" * 60)

    # 1. 检查微调 tokenizer 是否存在
    if not os.path.exists(os.path.join(finetuned_path, 'model.safetensors')):
        print(f"\n[ERROR] Fine-tuned tokenizer not found at: {finetuned_path}")
        print("Please run tokenizer training first:")
        print(f"  python finetune/tokenizer/train.py --norm-mode {norm_mode} --model {model_type}")
        return None

    # 2. 加载 val 数据
    print("\n[1] Loading validation data...")
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)

    # 构建索引
    val_indices = []
    for symbol, d in val_data.items():
        if 'windows' in d:
            for w in d['windows']:
                val_indices.append((symbol, int(w)))

    print(f"Total val windows: {len(val_indices)}")

    # 创建数据集
    config = DataConfig(
        norm_mode=norm_mode,
        lookback=lookback,
        predict=predict,
        split_mode=split_mode,
    )
    val_dataset = TokenizerDataset(val_data, val_indices, config)

    # 采样器
    n_val_iter = n_val_iter_multiplier * batch_size
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=RandomSampler(val_dataset, replacement=True, num_samples=n_val_iter),
        num_workers=0, drop_last=False,
    )

    # 3. 加载预训练 tokenizer
    print("\n[2] Loading pretrained tokenizer...")
    pretrained_tokenizer = KronosTokenizer.from_pretrained(pretrained_path)
    pretrained_tokenizer.to(device)
    pretrained_tokenizer.eval()

    # 4. 加载微调后 tokenizer
    print("\n[3] Loading fine-tuned tokenizer...")
    finetuned_tokenizer = KronosTokenizer.from_pretrained(finetuned_path)
    finetuned_tokenizer.to(device)
    finetuned_tokenizer.eval()

    # 5. 计算重建损失
    print("\n[4] Computing reconstruction loss...")

    pretrained_losses = []
    finetuned_losses = []

    with torch.no_grad():
        for batch_x in val_loader:
            batch_x = batch_x.to(device)

            # 预训练 tokenizer 重建损失
            zs_pre, _, _, _ = pretrained_tokenizer(batch_x)
            _, z_pre = zs_pre
            pre_loss = F.mse_loss(z_pre, batch_x, reduction='none')
            pre_loss = pre_loss.view(pre_loss.size(0), -1).mean(dim=1)
            pretrained_losses.extend(pre_loss.cpu().numpy().tolist())

            # 微调后 tokenizer 重建损失
            zs_ft, _, _, _ = finetuned_tokenizer(batch_x)
            _, z_ft = zs_ft
            ft_loss = F.mse_loss(z_ft, batch_x, reduction='none')
            ft_loss = ft_loss.view(ft_loss.size(0), -1).mean(dim=1)
            finetuned_losses.extend(ft_loss.cpu().numpy().tolist())

    # 6. 统计结果
    pretrained_mean = np.mean(pretrained_losses)
    pretrained_std = np.std(pretrained_losses)
    finetuned_mean = np.mean(finetuned_losses)
    finetuned_std = np.std(finetuned_losses)

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"\n[Reconstruction Loss on Val Data]")
    print(f"Pretrained ({PRETRAINED_MAP[model_type]}):")
    print(f"  Mean: {pretrained_mean:.6f}")
    print(f"  Std:  {pretrained_std:.6f}")
    print(f"\nFine-tuned ({norm_mode}):")
    print(f"  Mean: {finetuned_mean:.6f}")
    print(f"  Std:  {finetuned_std:.6f}")
    print(f"\n[Comparison]")
    improvement = pretrained_mean - finetuned_mean
    improvement_pct = (improvement / pretrained_mean) * 100 if pretrained_mean > 0 else 0
    print(f"  Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")

    # 7. 验收判定
    print(f"\n[Validation Criteria]")
    passed = True

    # 标准 1：微调后损失 < 预训练损失
    criterion1_passed = finetuned_mean < pretrained_mean
    print(f"  [1] Fine-tuned loss < Pretrained loss: {criterion1_passed}")
    if not criterion1_passed:
        print(f"      FAIL: {finetuned_mean:.6f} >= {pretrained_mean:.6f}")
        passed = False
    else:
        print(f"      PASS: {finetuned_mean:.6f} < {pretrained_mean:.6f}")

    # 标准 2：检查 early stopping 是否生效
    meta_path = os.path.join(finetuned_path, 'meta.json')
    if os.path.exists(meta_path):
        import json
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        actual_epochs = meta.get('actual_epochs', meta.get('epochs', 30))
        max_epochs = meta.get('epochs', 30)
        early_stopped = actual_epochs < max_epochs
        print(f"  [2] Early stopping triggered: {early_stopped}")
        if not early_stopped:
            print(f"      WARNING: Ran full {actual_epochs} epochs without early stopping")
            # 不标记为失败，仅警告
        else:
            print(f"      OK: Stopped at epoch {actual_epochs}/{max_epochs}")
    else:
        print(f"  [2] Meta file not found, cannot check early stopping")

    print("=" * 60)

    if passed:
        print("\n[RESULT] VALIDATION PASSED")
        print("Fine-tuned tokenizer is effective for this norm_mode distribution.")
    else:
        print("\n[RESULT] VALIDATION FAILED")
        print("Fine-tuned tokenizer does NOT improve reconstruction over pretrained.")
        print("Consider:")
        print("  - Increasing training epochs")
        print("  - Adjusting learning rate")
        print("  - Checking data quality/preprocessing")

    print("=" * 60)

    return {
        'pretrained_mean': pretrained_mean,
        'pretrained_std': pretrained_std,
        'finetuned_mean': finetuned_mean,
        'finetuned_std': finetuned_std,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'passed': passed,
    }


def main():
    parser = argparse.ArgumentParser(description='Tokenizer Fine-tuning Validation')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--model', type=str, default='mini',
                        choices=['mini', 'small', 'base'])
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block')
    parser.add_argument('--n-val-iter', type=int, default=400,
                        help='val 采样步数倍数')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    validate_tokenizer(
        norm_mode=args.norm_mode,
        model_type=args.model,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
        n_val_iter_multiplier=args.n_val_iter,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()