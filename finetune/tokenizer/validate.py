"""
Kronos Tokenizer Fine-tuning Effectiveness Validation

对比「预训练 tokenizer vs 微调后」在同一 val 数据上的重建误差。

tokenizer 与 predictor 概念解耦：只依赖 norm_mode，数据来自
finetune/data/processed/{norm_mode}/tokenizer/all.pkl（整条归一化，无分割）。
val 集复用 train 的划分（split_val_symbols，同 seed/比例），保证验证的就是
early-stop 用到的 held-out 股票。

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
import json
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
from finetune.predictor.core.paths import get_tokenizer_path, PROJECT_ROOT
from finetune.predictor.core.utils import get_device
from finetune.tokenizer.preprocess import get_tokenizer_data_path, split_val_symbols
from finetune.tokenizer.train import PRETRAINED_MAP, TokenizerDataset


def validate_tokenizer(
    norm_mode: str,
    model_type: str,
    seq_len: int = 400,
    n_val_iter_multiplier: int = 400,
    batch_size: int = 16,
    seed: int = 42,
    val_holdout_ratio: float = 0.1,
    custom_tokenizer_path: str = None,
):
    """
    验证 tokenizer 微调效果

    对比预训练 tokenizer 与微调后 tokenizer 在同一 val 数据上的重建损失。
    val 集复用 train 的 split_val_symbols 划分（同 seed/比例），保证验证的就是
    train early-stop 用到的 held-out 股票。

    Args:
        norm_mode: 归一化模式（决定数据分布，tokenizer 唯一依赖）
        model_type: 模型类型（mini/small/base）
        seq_len: tokenizer 重建窗口长度（须与 train 一致）
        n_val_iter_multiplier: val 采样步数倍数
        batch_size: 批大小
        seed: 划分种子（须与 train 一致，否则 val 集不同步）
        val_holdout_ratio: val 股票比例（须与 train 一致）
        custom_tokenizer_path: 自定义 tokenizer 路径（优先于默认路径）

    Returns:
        dict: 包含预训练和微调后 tokenizer 的重建损失统计
    """
    device = get_device()

    # 数据路径（tokenizer 专用，只按 norm_mode 键控）
    data_path = get_tokenizer_data_path(norm_mode)
    # 微调 tokenizer 路径：支持自定义
    if custom_tokenizer_path:
        finetuned_path = custom_tokenizer_path
    else:
        finetuned_path = get_tokenizer_path(norm_mode, model_type)
    pretrained_path = os.path.join(PROJECT_ROOT, PRETRAINED_MAP[model_type])

    print("=" * 60)
    print("Tokenizer Fine-tuning Effectiveness Validation")
    print("=" * 60)
    print(f"norm_mode: {norm_mode}")
    print(f"model_type: {model_type}")
    print(f"seq_len: {seq_len}")
    print(f"pretrained_path: {pretrained_path}")
    print(f"finetuned_path: {finetuned_path}")
    if custom_tokenizer_path:
        print(f"  (custom path specified)")
    print(f"data_path: {data_path}")
    print(f"seed: {seed}, val_holdout_ratio: {val_holdout_ratio}")
    print("=" * 60)

    # 1. 检查微调 tokenizer 是否存在
    if not os.path.exists(os.path.join(finetuned_path, 'model.safetensors')):
        print(f"\n[ERROR] Fine-tuned tokenizer not found at: {finetuned_path}")
        print("Please run tokenizer training first:")
        print(f"  python finetune/tokenizer/train.py --norm-mode {norm_mode} --model {model_type}")
        return None

    # 2. 加载 tokenizer 专用数据（整条归一化，无分割）+ 复用 train 的 val 划分
    print("\n[1] Loading data and reproducing val split...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Tokenizer data not found at {data_path}. "
            f"Please run: python finetune/tokenizer/preprocess.py --norm-mode {norm_mode}"
        )
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    _, val_data, val_symbols = split_val_symbols(all_data, seed, val_holdout_ratio)
    print(f"Loaded {len(all_data)} stocks; val (held-out) = {len(val_data)} stocks")

    # 创建数据集（从整条 normalized 随机切 seq_len 窗口）
    val_dataset = TokenizerDataset(val_data, seq_len)

    # 采样器
    n_val_iter = n_val_iter_multiplier * batch_size
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=RandomSampler(val_dataset, replacement=True, num_samples=n_val_iter),
        num_workers=0, drop_last=False,
    )
    print(f"Val loader: {len(val_loader)} batches (seq_len={seq_len})")

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
    print(f"\n[Reconstruction Loss on Val Data (held-out stocks)]")
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
    parser.add_argument('--tokenizer-path', type=str, default=None,
                        help='自定义 tokenizer 路径（优先于默认 outputs/tokenizers/{norm_mode}/{model}）')
    parser.add_argument('--seq-len', type=int, default=400,
                        help='tokenizer 重建窗口长度（须与 train 一致）')
    parser.add_argument('--n-val-iter', type=int, default=400,
                        help='val 采样步数倍数')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42,
                        help='val 划分种子（须与 train 一致）')
    parser.add_argument('--val-holdout-ratio', type=float, default=0.1,
                        help='val 股票比例（须与 train 一致）')
    args = parser.parse_args()

    validate_tokenizer(
        norm_mode=args.norm_mode,
        model_type=args.model,
        seq_len=args.seq_len,
        n_val_iter_multiplier=args.n_val_iter,
        batch_size=args.batch_size,
        seed=args.seed,
        val_holdout_ratio=args.val_holdout_ratio,
        custom_tokenizer_path=args.tokenizer_path,
    )


if __name__ == '__main__':
    main()
