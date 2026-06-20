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
import time
import argparse
import pickle
import json
import numpy as np
from datetime import datetime
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer
from finetune.predictor.core.paths import (
    get_tokenizer_path,
    ensure_dir,
    PROJECT_ROOT,
)
from finetune.predictor.core.schema import compute_fingerprint
from finetune.predictor.core.utils import safe_save_json, get_device, set_seed, get_rank_info, cleanup_ddp


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
    seq_len: int = 400,                    # tokenizer 重建窗口长度（与 predictor lookback 无关）
    val_holdout_ratio: float = 0.1,        # 随机抽此比例股票作 val
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    use_ddp: bool = False,
    device=None,
):
    """
    微调 tokenizer（非从零训练）

    tokenizer 与 predictor 概念解耦：只依赖 norm_mode（归一化分布），不依赖
    train/val/test 分割、lookback/predict、block/time。做无监督重建（重建输入
    自身），无泄露概念，故用全量数据、整条归一化（由 tokenizer/preprocess.py 生成）。

    步骤：
    1. 按 model_type 选预训练 tokenizer
    2. 加载 tokenizer 专用数据（all.pkl，整条归一化）
    3. 随机抽 val_holdout_ratio 比例股票作 val（仅 early-stop 信号，非防泄露）
    4. 步数驱动放回采样：从整条 normalized 随机切 seq_len 窗口
    5. 微调循环：recon_loss + bsq_loss（MSE，与旧代码一致）
    6. 每 epoch 用 val 算重建损失，early stopping
    7. 保存到 outputs/tokenizers/{norm_mode}/{model_type}/

    DDP：各 rank 用不同 seed 采样不同样本；val_loss 跨 rank all_reduce 保证
    early-stop 决定一致（防死锁）；checkpoint 仅 rank0 保存。

    Args:
        norm_mode: 归一化模式（决定数据分布，tokenizer 唯一依赖）
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
        seq_len: tokenizer 重建窗口长度（喂给 KronosTokenizer 的 seq_len，默认 400）
        val_holdout_ratio: 随机抽此比例股票作 val
        rank: DDP rank（0 = 主进程）
        local_rank: DDP 本地 rank（用于指定 GPU）
        world_size: DDP 进程数
        use_ddp: 是否启用 DDP
        device: 计算设备
    """
    is_main = (rank == 0)
    set_seed(seed + rank)  # 各 rank seed 不同，RandomSampler 采不同样本
    if device is None:
        device = get_device(local_rank=0 if use_ddp else None)

    # 数据路径（tokenizer 专用，只按 norm_mode 键控）
    from finetune.tokenizer.preprocess import get_tokenizer_data_path, split_val_symbols
    data_path = get_tokenizer_data_path(norm_mode)

    pretrained_path = os.path.join(PROJECT_ROOT, PRETRAINED_MAP[model_type])
    output_path = get_tokenizer_path(norm_mode, model_type)

    if is_main:
        print("=" * 60)
        print("Kronos Tokenizer Fine-tuning")
        print("=" * 60)
        print(f"norm_mode: {norm_mode}")
        print(f"model_type: {model_type}")
        print(f"pretrained: {PRETRAINED_MAP[model_type]}")
        print(f"data_path: {data_path}")
        print(f"output_path: {output_path}")
        print(f"seq_len: {seq_len}")
        print(f"val_holdout_ratio: {val_holdout_ratio}")
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"world_size: {world_size}")
        print(f"n_train_iter: {n_train_iter_multiplier * batch_size}")
        print(f"n_val_iter: {n_val_iter_multiplier * batch_size}")
        print("=" * 60)

    # 1. 加载预训练 tokenizer（架构由 model_type 决定）
    if is_main:
        print("\n[1] Loading pretrained tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(pretrained_path)
    tokenizer.to(device)
    if is_main:
        print(f"Tokenizer loaded from: {pretrained_path}")

    # 2. 加载 tokenizer 专用数据（整条归一化，无分割）
    if is_main:
        print("\n[2] Loading data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Tokenizer data not found at {data_path}. "
            f"Please run: python finetune/tokenizer/preprocess.py --norm-mode {norm_mode}"
        )
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    if is_main:
        print(f"Loaded {len(all_data)} stocks (full-series normalized)")

    # 随机抽 val_holdout_ratio 比例股票作 val（仅 early-stop 信号，非防泄露）
    # 用 rank0 的 seed 划分，保证各 rank 划分一致（否则 val 集不同步）
    # 复用 preprocess.split_val_symbols，保证 train/validate 用同一 val 集
    train_data, val_data, _ = split_val_symbols(all_data, seed, val_holdout_ratio)
    if is_main:
        print(f"Split by symbol: train={len(train_data)} stocks, val={len(val_data)} stocks (holdout {val_holdout_ratio})")

    # 创建数据集（从整条 normalized 随机切 seq_len 窗口）
    train_dataset = TokenizerDataset(train_data, seq_len)
    val_dataset = TokenizerDataset(val_data, seq_len)

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

    if is_main:
        print(f"Train loader: {len(train_loader)} batches/epoch")
        print(f"Val loader: {len(val_loader)} batches")

    # DDP 包装（forward 必须走包装对象触发梯度 all-reduce）
    if use_ddp:
        tokenizer = DDP(tokenizer, device_ids=[local_rank])

    # 4. 微调循环
    if is_main:
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
        epoch_start = time.time()
        tokenizer.train()
        epoch_losses = []
        total_batches = len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        if is_main:
            print(f"\n=== Epoch {epoch_idx + 1}/{epochs} ===")
            print(f"LR: {current_lr:.6f}")

        for batch_idx, batch_x in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)

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

            # batch 级进度日志（每 50 batch + 首个 batch），仅 rank0
            if is_main and (batch_idx % 50 == 0 or batch_idx == 0):
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                elapsed = time.time() - epoch_start
                print(f"  Batch {batch_idx + 1}/{total_batches} - loss: {loss.item():.4f}, avg: {avg_loss:.4f}, lr: {current_lr:.6f} [{elapsed:.0f}s]", flush=True)

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

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

        # DDP：跨 rank 聚合 val_loss，保证各 rank 拿到同一值 → early-stop 决定一致（防死锁）
        if use_ddp:
            sum_tensor = torch.tensor([val_loss_sum, val_count], device=device, dtype=torch.float64)
            dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
            val_loss_sum = sum_tensor[0].item()
            val_count = sum_tensor[1].item()

        avg_val_loss = val_loss_sum / val_count

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        epoch_elapsed = time.time() - epoch_start
        if is_main:
            print(f"Epoch {epoch_idx + 1}/{epochs}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f}, lr={current_lr:.6f} [{epoch_elapsed:.0f}s]", flush=True)

        # Early stopping（跳过 grace period）
        if epoch_idx >= early_stopping_grace_period:
            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if is_main:
                    unwrapped = tokenizer.module if use_ddp else tokenizer
                    unwrapped.save_pretrained(os.path.join(checkpoint_path, 'best_model'))
                    print(f"  [BEST] val_loss={best_val_loss:.6f}", flush=True)
                if use_ddp:
                    dist.barrier()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if is_main:
                        print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs", flush=True)
                    break

    # 6. 保存最终（TK5 修复：同时保存到 output_path 根目录，方便 validate/eval 加载）
    # 仅 rank0 保存，DDP 包装对象无 save_pretrained，需解包
    if is_main:
        print("\n[4] Saving tokenizer...")
        unwrapped = tokenizer.module if use_ddp else tokenizer
        unwrapped.save_pretrained(os.path.join(checkpoint_path, 'final_model'))
        # 同时保存到 output_path 根目录（validate/eval 期望的路径）
        unwrapped.save_pretrained(output_path)

    # 记录元数据
    meta = {
        'norm_mode': norm_mode,
        'model_type': model_type,
        'pretrained_base': PRETRAINED_MAP[model_type],
        'data_fingerprint': compute_fingerprint(all_data),
        'epochs': epochs,
        'actual_epochs': epoch_idx + 1,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_train_iter': n_train_iter,
        'n_val_iter': n_val_iter,
        'best_val_loss': float(best_val_loss),
        'early_stopping_patience': early_stopping_patience,
        'world_size': world_size,
        'use_ddp': use_ddp,
        'created_at': datetime.now().isoformat(),
    }
    if is_main:
        safe_save_json(meta, os.path.join(output_path, 'meta.json'))

    # 保存训练历史
    if is_main:
        safe_save_json(history, os.path.join(output_path, 'history.json'))
        print(f"\nTokenizer saved to: {output_path}")
        print(f"Best val loss: {best_val_loss:.6f}")
        print(f"Actual epochs: {epoch_idx + 1}/{epochs}")
        print("=" * 60)

    # 所有 rank 同步后再清理，防 rank0 退出后其他 rank 还在通信
    if use_ddp:
        dist.barrier()
    cleanup_ddp()

    return tokenizer, best_val_loss


class TokenizerDataset:
    """Tokenizer 微调用的数据集：预建所有 (symbol, start) 窗口索引，RandomSampler 随机采样"""

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        # 预建所有合法窗口索引：(symbol, start)，start ∈ [0, T - seq_len]
        self.indices = []
        for symbol, d in data.items():
            T = len(d['normalized'])
            for start in range(0, max(0, T - seq_len + 1)):
                self.indices.append((symbol, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        symbol, start = self.indices[idx]
        d = self.data[symbol]
        x_norm = d['normalized'][start:start + self.seq_len]
        return torch.from_numpy(x_norm.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description='Kronos Tokenizer Fine-tuning')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--model', type=str, default='mini',
                        choices=['mini', 'small', 'base'],
                        help='mini→Kronos-Tokenizer-2k, small/base→Kronos-Tokenizer-base')
    parser.add_argument('--seq-len', type=int, default=400,
                        help='tokenizer 重建窗口长度（与 predictor lookback 无关）')
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
    parser.add_argument('--val-holdout-ratio', type=float, default=0.1,
                        help='val 股票比例（须与 validate 一致，否则 val 集不同步）')
    args = parser.parse_args()

    # DDP setup（torchrun 自动注入 RANK/LOCAL_RANK/WORLD_SIZE；单卡 python 直接跑则 use_ddp=False）
    rank, local_rank, world_size, use_ddp = get_rank_info()
    device = get_device(local_rank)

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
        seq_len=args.seq_len,
        val_holdout_ratio=args.val_holdout_ratio,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        use_ddp=use_ddp,
        device=device,
    )


if __name__ == '__main__':
    main()