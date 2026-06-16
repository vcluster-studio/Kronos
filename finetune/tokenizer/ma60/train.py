"""
MA60 Tokenizer Training Script - for small/base models (group_size=4)

基于 Kronos-Tokenizer-base (group_size=4) 微调，使用 MA60 归一化数据。
供 small/base predictor 使用。

Usage:
    python -u finetune/train_tokenizer_ma60_base.py
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer

# ============================================================================
# 配置
# ============================================================================

# 基础 tokenizer（group_size=4，适用于 small/base）
TOKENIZER_BASE = 'pretrained/Kronos-Tokenizer-base'

# 预归一化 MA60 数据
DATA_PATHS = {
    'train': 'finetune/data/processed_datasets_ma60/train_data.pkl',
    'val': 'finetune/data/processed_datasets_ma60/val_data.pkl',
    'test': 'finetune/data/processed_datasets_ma60/test_data.pkl',
}

# 训练参数
TRAINING_PARAMS = {
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 0.001,
    'weight_decay': 0.1,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'seed': 100,
    'lookback': 400,
    'predict': 10,
    'clip': 5.0,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.0001,
    'early_stopping_grace_period': 5,
}

# 保存目录
SAVE_FOLDER = 'ma60_tokenizer_base_v1'


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


class MA60TokenizerDataset(Dataset):
    """MA60 预归一化数据集，用于 tokenizer 训练"""

    def __init__(self, data_type='train', config=None):
        self.config = config
        self.data_type = data_type
        self.py_rng = np.random.RandomState(config.seed)

        data_path = DATA_PATHS[data_type]
        print(f"[{data_type.upper()}] Loading: {data_path}")
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.symbols = list(self.raw_data.keys())
        self.window = config.lookback + config.predict + 1

        # 预计算索引
        print(f"[{data_type.upper()}] Pre-computing indices...")
        self.indices = []
        for symbol in self.symbols:
            data = self.raw_data[symbol]
            seq_len = len(data['normalized'])
            if seq_len >= self.window:
                for i in range(seq_len - self.window + 1):
                    self.indices.append((symbol, i))

        # 样本数
        n_iter = config.n_train_iter if data_type == 'train' else config.n_val_iter
        self.n_samples = min(n_iter, len(self.indices))
        print(f"[{data_type.upper()}] {len(self.indices)} windows, using {self.n_samples} samples/epoch")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rand_idx = self.py_rng.randint(0, len(self.indices))
        symbol, start_idx = self.indices[rand_idx]
        end_idx = start_idx + self.window

        data = self.raw_data[symbol]
        x_norm = data['normalized'][start_idx:end_idx].astype(np.float32)

        timestamps = data['index'][start_idx:end_idx]
        x_stamp = np.stack([
            timestamps.minute.values,
            timestamps.hour.values,
            timestamps.weekday.values,
            timestamps.day.values,
            timestamps.month.values,
        ], axis=1).astype(np.float32)

        return torch.from_numpy(x_norm), torch.from_numpy(x_stamp), torch.tensor(0.0)


def train_tokenizer(model, device, config, save_dir):
    """训练 MA60 tokenizer (base, group_size=4)"""
    start_time = time.time()

    train_dataset = MA60TokenizerDataset('train', config=config)
    val_dataset = MA60TokenizerDataset('val', config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               sampler=RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset)),
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                              sampler=SequentialSampler(val_dataset),
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.epochs,
        pct_start=0.03,
        div_factor=10
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch_idx in range(config.epochs):
        epoch_start = time.time()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_losses = []

        for i, (batch_x, _, _) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)

            zs, bsq_loss, _, _ = model(batch_x)
            z_pre, z = zs

            recon_loss = F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x)
            loss = (recon_loss + bsq_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

            if (i + 1) % 50 == 0 or i == 0:
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (i + 1) / len(train_loader) * 100
                print(f"  Batch {i+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")

        history['lr'].append(current_lr)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_x, _, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                zs, _, _, _ = model(batch_x)
                _, z = zs
                val_loss = F.mse_loss(z, batch_x)
                val_loss_sum += val_loss.item() * batch_x.size(0)
                val_count += batch_x.size(0)

        avg_val_loss = val_loss_sum / val_count
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"\n=== Epoch {epoch_idx+1}/{config.epochs} ===")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"LR: {current_lr:.6f}")
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}")

        # Save latest
        latest_path = f"{save_dir}/checkpoints/latest_model"
        model.save_pretrained(latest_path)

        # Check improvement
        improved = False
        if avg_val_loss <= best_val_loss - config.early_stopping_min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            improved = True
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"[VAL LOSS SAVED] {best_val_loss:.4f}")

        if not improved:
            patience_counter += 1

        # Early stopping
        if epoch_idx >= config.early_stopping_grace_period:
            if patience_counter >= config.early_stopping_patience:
                print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                final_path = f"{save_dir}/checkpoints/final_model"
                model.save_pretrained(final_path)
                break

        print(flush=True)

    # Final save
    final_path = f"{save_dir}/checkpoints/final_model"
    model.save_pretrained(final_path)

    return {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch_idx + 1,
        'history': history,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("MA60 Tokenizer Training (base, group_size=4)")
    print("=" * 60)
    print(f"Base Tokenizer: {TOKENIZER_BASE}")
    print(f"Data: {DATA_PATHS['train']}")
    print(f"Device: {device}")
    print(f"Save: outputs/models/{SAVE_FOLDER}")
    print("=" * 60)

    set_seed(TRAINING_PARAMS['seed'])

    save_dir = os.path.join(project_root, "outputs/models", SAVE_FOLDER)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # 加载 base tokenizer
    tokenizer_path = os.path.join(project_root, TOKENIZER_BASE)
    print(f"\nLoading base tokenizer from: {tokenizer_path}")
    model = KronosTokenizer.from_pretrained(tokenizer_path)
    model.to(device)
    print(f"Model size: {get_model_size(model):.2f}M parameters")

    # 配置
    class Config:
        pass

    config = Config()
    config.seed = TRAINING_PARAMS['seed']
    config.lookback = TRAINING_PARAMS['lookback']
    config.predict = TRAINING_PARAMS['predict']
    config.clip = TRAINING_PARAMS['clip']
    config.batch_size = TRAINING_PARAMS['batch_size']
    config.epochs = TRAINING_PARAMS['epochs']
    config.learning_rate = TRAINING_PARAMS['learning_rate']
    config.weight_decay = TRAINING_PARAMS['weight_decay']
    config.adam_beta1 = TRAINING_PARAMS['adam_beta1']
    config.adam_beta2 = TRAINING_PARAMS['adam_beta2']
    config.n_train_iter = 2000 * config.batch_size
    config.n_val_iter = 400 * config.batch_size
    config.early_stopping_patience = TRAINING_PARAMS['early_stopping_patience']
    config.early_stopping_min_delta = TRAINING_PARAMS['early_stopping_min_delta']
    config.early_stopping_grace_period = TRAINING_PARAMS['early_stopping_grace_period']

    # 训练
    result = train_tokenizer(model, device, config, save_dir)

    # 保存结果
    summary = {
        'tokenizer': TOKENIZER_BASE,
        'data_path': DATA_PATHS['train'],
        'group_size': 4,
        'config': TRAINING_PARAMS,
        'result': {
            'best_val_loss': result['best_val_loss'],
            'epochs_trained': result['epochs_trained'],
        }
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print("\n" + "=" * 60)
    print("Tokenizer Training completed!")
    print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"Epochs: {result['epochs_trained']}")
    print(f"Saved to: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
