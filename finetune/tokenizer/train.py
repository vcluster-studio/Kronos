"""
Kronos Tokenizer Fine-tuning Entry

微调 tokenizer（非从零训练）

关键：
- 步数驱动放回采样（非比例抽样）
- 架构由 model_type 决定
- 按 norm_mode 微调权重
- val 重建损失 + early stopping

使用：
    python finetune/tokenizer/train.py \
        --norm-mode sliding_ma60 \
        --model mini \
        --epochs 30
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
from datetime import datetime
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
    ensure_dir,
    PROJECT_ROOT,
)
from finetune.predictor.core.dataset import KronosDataset
from finetune.predictor.core.schema import compute_fingerprint
from finetune.predictor.core.utils import safe_save_json, get_device, set_seed


# 预训练 tokenizer 映射（架构由 model_type 决定）
PRETRAINED_MAP = {
    'mini': 'pretrained/Kronos-Tokenizer-2k',
    'small': 'pretrained/Kronos-Tokenizer-base',
    'base': 'pretrained/Kronos-Tokenizer-base',
}


def finetune_tokenizer(
    norm_mode: str,
    model_type: str,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    n_train_iter_multiplier: int = 2000,  # 每 epoch 采样步数 = multiplier × batch_size
    n_val_iter_multiplier: int = 400,      # val 步数
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-5,
    early_stopping_grace_period: int = 3,
    seed: int = 42,
    lookback: int = 400,
    predict: int = 10,
    split_mode: str = 'block',
):
    """
    微调 tokenizer（非从零训练）

    步骤：
    1. 按 model_type 选预训练 tokenizer
    2. 加载 train/val 数据
    3. 步数驱动放回采样（关键：非比例抽样）
    4. 微调循环：recon_loss + bsq_loss
    5. 每 epoch 用 val 算重建损失，early stopping
    6. 保存到 outputs/tokenizers/{norm_mode}/{model_type}/

    Args:
        norm_mode: 归一化模式（决定数据分布）
        model_type: 模型类型（决定架构）
        epochs: 最大微调轮数
        batch_size: 批大小
        learning_rate: 学习率
        n_train_iter_multiplier: 每 epoch 采样步数倍数（默认 2000 × 16 = 32000 步）
        n_val_iter_multiplier: val 采样步数倍数
        early_stopping_patience: 早停耐心值
        early_stopping_min_delta: 最小改进阈值
        early_stopping_grace_period: 起始宽容期
        seed: 随机种子
        lookback: 回看窗口
        predict: 预测步数
        split_mode: 分割模式
    """
    set_seed(seed)
    device = get_device()

    # 数据路径
    train_path = get_split_data_path(norm_mode, lookback, predict, split_mode, 'train')
    val_path = get_split_data_path(norm_mode, lookback, predict, split_mode, 'val')

    pretrained_path = os.path.join(PROJECT_ROOT, PRETRAINED_MAP[model_type])
    output_path = get_tokenizer_path(norm_mode, model_type)

    print("=" * 60)
    print("Kronos Tokenizer Fine-tuning")
    print("=" * 60)
    print(f"norm_mode: {norm_mode}")
    print(f"model_type: {model_type}")
    print(f"pretrained: {PRETRAINED_MAP[model_type]}")
    print(f"train_path: {train_path}")
    print(f"val_path: {val_path}")
    print(f"output_path: {output_path}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"n_train_iter: {n_train_iter_multiplier * batch_size}")
    print(f"n_val_iter: {n_val_iter_multiplier * batch_size}")
    print("=" * 60)

    # 1. 加载预训练 tokenizer（架构由 model_type 决定）
    print("\n[1] Loading pretrained tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(pretrained_path)
    tokenizer.to(device)
    print(f"Tokenizer loaded from: {pretrained_path}")

    # 2. 加载 train/val 数据
    print("\n[2] Loading data...")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    print(f"Loaded train: {len(train_data)} stocks, val: {len(val_data)} stocks")

    # 构建索引
    train_indices = []
    for symbol, d in train_data.items():
        if 'windows' in d:
            for w in d['windows']:
                train_indices.append((symbol, int(w)))

    val_indices = []
    for symbol, d in val_data.items():
        if 'windows' in d:
            for w in d['windows']:
                val_indices.append((symbol, int(w)))

    print(f"Total windows: train={len(train_indices)}, val={len(val_indices)}")

    # 创建数据集
    config = DataConfig(
        norm_mode=norm_mode,
        lookback=lookback,
        predict=predict,
        split_mode=split_mode,
    )

    train_dataset = TokenizerDataset(train_data, train_indices, config)
    val_dataset = TokenizerDataset(val_data, val_indices, config)

    # 3. 步数驱动放回采样（关键：非比例抽样）
    n_train_iter = n_train_iter_multiplier * batch_size
    n_val_iter = n_val_iter_multiplier * batch_size

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=n_train_iter),
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=RandomSampler(val_dataset, replacement=True, num_samples=n_val_iter),
        num_workers=0, drop_last=False,
    )

    print(f"Train loader: {len(train_loader)} batches/epoch")
    print(f"Val loader: {len(val_loader)} batches")

    # 4. 微调循环
    print("\n[3] Fine-tuning...")
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.03, div_factor=10,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    ensure_dir(output_path)
    checkpoint_path = os.path.join(output_path, 'checkpoints')
    ensure_dir(checkpoint_path)

    for epoch_idx in range(epochs):
        tokenizer.train()
        epoch_losses = []

        for batch_x in train_loader:
            batch_x = batch_x.to(device)

            # Forward
            zs, bsq_loss, _, _ = tokenizer(batch_x)
            z_pre, z = zs

            # 计算重建损失
            recon_loss = F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x)
            loss = (recon_loss + bsq_loss) / 2

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        current_lr = optimizer.param_groups[0]['lr']

        # 5. val 重建损失
        tokenizer.eval()
        val_loss_sum, val_count = 0.0, 0

        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                zs, _, _, _ = tokenizer(batch_x)
                _, z = zs
                vl = F.mse_loss(z, batch_x)
                val_loss_sum += vl.item() * batch_x.size(0)
                val_count += batch_x.size(0)

        avg_val_loss = val_loss_sum / val_count

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch_idx+1}/{epochs}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f}, lr={current_lr:.6f}")

        # Early stopping（跳过 grace period）
        if epoch_idx >= early_stopping_grace_period:
            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                tokenizer.save_pretrained(os.path.join(checkpoint_path, 'best_model'))
                print(f"  [BEST] val_loss={best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                    break

    # 6. 保存最终（TK5 修复：同时保存到 output_path 根目录，方便 validate/eval 加载）
    print("\n[4] Saving tokenizer...")
    tokenizer.save_pretrained(os.path.join(checkpoint_path, 'final_model'))
    # 同时保存到 output_path 根目录（validate/eval 期望的路径）
    tokenizer.save_pretrained(output_path)

    # 记录元数据
    meta = {
        'norm_mode': norm_mode,
        'model_type': model_type,
        'pretrained_base': PRETRAINED_MAP[model_type],
        'data_fingerprint': compute_fingerprint(train_path),
        'epochs': epochs,
        'actual_epochs': epoch_idx + 1,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_train_iter': n_train_iter,
        'n_val_iter': n_val_iter,
        'best_val_loss': float(best_val_loss),
        'early_stopping_patience': early_stopping_patience,
        'created_at': datetime.now().isoformat(),
    }
    safe_save_json(meta, os.path.join(output_path, 'meta.json'))

    # 保存训练历史
    safe_save_json(history, os.path.join(output_path, 'history.json'))

    print(f"\nTokenizer saved to: {output_path}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print("=" * 60)

    return tokenizer, best_val_loss


class TokenizerDataset:
    """Tokenizer 微调用的数据集"""

    def __init__(self, data, indices, config):
        self.data = data
        self.indices = indices
        self.config = config

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        symbol, window_start = self.indices[idx]
        d = self.data[symbol]

        if 'normalized' in d:
            x_norm = d['normalized'][window_start:window_start + self.config.lookback]
        else:
            x_raw = d['original'][window_start:window_start + self.config.lookback]
            x_mean = np.mean(x_raw, axis=0)
            x_std = np.std(x_raw, axis=0) + 1e-5
            x_norm = np.clip((x_raw - x_mean) / x_std, -self.config.clip, self.config.clip)

        return torch.from_numpy(x_norm.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description='Kronos Tokenizer Fine-tuning')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--model', type=str, default='mini',
                        choices=['mini', 'small', 'base'],
                        help='mini→Kronos-Tokenizer-2k, small/base→Kronos-Tokenizer-base')
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-train-iter', type=int, default=2000,
                        help='每 epoch 采样步数倍数（实际步数 = n_train_iter × batch_size）')
    parser.add_argument('--n-val-iter', type=int, default=400,
                        help='val 采样步数倍数')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--grace-period', type=int, default=3,
                        help='Early stopping grace period')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    finetune_tokenizer(
        norm_mode=args.norm_mode,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_train_iter_multiplier=args.n_train_iter,
        n_val_iter_multiplier=args.n_val_iter,
        early_stopping_patience=args.patience,
        early_stopping_grace_period=args.grace_period,
        seed=args.seed,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
    )


if __name__ == '__main__':
    main()