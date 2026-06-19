"""
Kronos Tokenizer Fine-tuning Entry

微调 tokenizer（非从零训练）

关键：
- 加载预训练 tokenizer（架构由 model_type 决定）
- 按 norm_mode 归一化的数据微调权重
- 保存到 outputs/tokenizers/{norm_mode}/{model_type}/

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
    get_raw_path,
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
    data_path: str = None,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 0.1,
    seed: int = 42,
    lookback: int = 400,
    predict: int = 10,
    split_mode: str = 'block',
):
    """
    微调 tokenizer（非从零训练）

    步骤：
    1. 按 model_type 选预训练 tokenizer
    2. 加载数据（已按 norm_mode 预处理）
    3. 微调循环：recon_loss + bsq_loss
    4. 保存到 outputs/tokenizers/{norm_mode}/{model_type}/

    Args:
        norm_mode: 归一化模式（决定数据分布）
        model_type: 模型类型（决定架构）
        data_path: 训练数据路径（默认使用预处理后的 train.pkl）
        epochs: 微调轮数
        batch_size: 批大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        seed: 随机种子
        lookback: 回看窗口
        predict: 预测步数
        split_mode: 分割模式
    """
    set_seed(seed)
    device = get_device()

    # 数据路径
    if data_path is None:
        data_path = get_split_data_path(
            norm_mode, lookback, predict, split_mode, 'train'
        )

    pretrained_path = os.path.join(PROJECT_ROOT, PRETRAINED_MAP[model_type])
    output_path = get_tokenizer_path(norm_mode, model_type)

    print("=" * 60)
    print("Kronos Tokenizer Fine-tuning")
    print("=" * 60)
    print(f"norm_mode: {norm_mode}")
    print(f"model_type: {model_type}")
    print(f"pretrained: {PRETRAINED_MAP[model_type]}")
    print(f"data_path: {data_path}")
    print(f"output_path: {output_path}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print("=" * 60)

    # 1. 加载预训练 tokenizer（架构由 model_type 决定）
    print("\n[1] Loading pretrained tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(pretrained_path)
    tokenizer.to(device)
    tokenizer.train()  # 设置为训练模式
    print(f"Tokenizer loaded from: {pretrained_path}")

    # 2. 加载数据（已按 norm_mode 预处理）
    print("\n[2] Loading data...")
    with open(data_path, 'rb') as f:
        train_data = pickle.load(f)
    print(f"Loaded {len(train_data)} stocks")

    # 构建索引
    indices = []
    for symbol, d in train_data.items():
        if 'windows' in d:
            for w in d['windows']:
                indices.append((symbol, int(w)))

    print(f"Total windows: {len(indices)}")

    # 创建数据集
    config = DataConfig(
        norm_mode=norm_mode,
        lookback=lookback,
        predict=predict,
        split_mode=split_mode,
    )

    # 3. 微调循环
    print("\n[3] Fine-tuning...")
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 使用简单的数据加载方式
    sampler = RandomSampler(indices)
    loader = DataLoader(
        [(indices[i], train_data) for i in range(len(indices))],
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: prepare_batch(batch, config, device),
        num_workers=0,
    )

    for epoch_idx in range(epochs):
        epoch_losses = []
        recon_losses = []
        bsq_losses = []

        for batch_x in loader:
            # Forward
            zs, bsq_loss, _, _ = tokenizer(batch_x)
            z_pre, z = zs

            # 计算重建损失
            recon_loss = F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x)

            # 总损失
            loss = (recon_loss + bsq_loss) / 2

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            bsq_losses.append(bsq_loss.item())

        avg_loss = np.mean(epoch_losses)
        avg_recon = np.mean(recon_losses)
        avg_bsq = np.mean(bsq_losses)

        print(f"Epoch {epoch_idx+1}/{epochs}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, bsq={avg_bsq:.4f}")

    # 4. 保存微调后的 tokenizer
    print("\n[4] Saving tokenizer...")
    ensure_dir(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Tokenizer saved to: {output_path}")

    # 5. 记录元数据
    meta = {
        'norm_mode': norm_mode,
        'model_type': model_type,
        'pretrained_base': PRETRAINED_MAP[model_type],
        'data_fingerprint': compute_fingerprint(data_path),
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'created_at': datetime.now().isoformat(),
    }
    safe_save_json(meta, os.path.join(output_path, 'meta.json'))

    print("=" * 60)
    print("Fine-tuning complete!")
    print("=" * 60)

    return tokenizer


def prepare_batch(batch, config, device):
    """
    准备批次数据

    Args:
        batch: [(index, data_dict)]
        config: DataConfig
        device: torch.device

    Returns:
        batch_x: (B, lookback, 6) tensor
    """
    batch_x = []
    for (symbol, window_start), data_dict in batch:
        d = data_dict[symbol]
        if 'normalized' in d:
            x_norm = d['normalized'][window_start:window_start + config.lookback]
        else:
            x_raw = d['original'][window_start:window_start + config.lookback]
            x_mean = np.mean(x_raw, axis=0)
            x_std = np.std(x_raw, axis=0) + 1e-5
            x_norm = np.clip((x_raw - x_mean) / x_std, -config.clip, config.clip)

        batch_x.append(x_norm.astype(np.float32))

    return torch.from_numpy(np.stack(batch_x)).to(device)


def main():
    parser = argparse.ArgumentParser(description='Kronos Tokenizer Fine-tuning')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--model', type=str, default='mini',
                        choices=['mini', 'small', 'base'],
                        help='mini→Kronos-Tokenizer-2k, small/base→Kronos-Tokenizer-base')
    parser.add_argument('--data-path', type=str, default=None,
                        help='训练数据路径（默认使用预处理后的 train.pkl）')
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    finetune_tokenizer(
        norm_mode=args.norm_mode,
        model_type=args.model,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
    )


if __name__ == '__main__':
    main()