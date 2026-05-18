"""
Kronos Tokenizer Training Script - with early stopping

Usage:
    python -u finetune/train_tokenizer.py --dataset mid
    python -u finetune/train_tokenizer.py --dataset full

Tokenizer matching (IMPORTANT):
    - mini: Kronos-Tokenizer-2k (group_size=5, 2048 context)
    - small/base: Kronos-Tokenizer-base (group_size=4, 512 context)

This script trains tokenizers based on Kronos-Tokenizer-2k (for mini model).
Output saved to outputs/models/{dataset}_tokenizer_2k_v1/

Config managed via static variables. Add new datasets by updating DATASET_CONFIGS.
"""

import os
import sys
import json
import time
import argparse
from time import gmtime, strftime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

print("Starting tokenizer training...", flush=True)

import torch
print("Torch imported", flush=True)

from model.kronos import KronosTokenizer
from finetune.dataset import QlibDataset

# ============================================================================
# 数据集配置（新增数据集只需更新此变量）
# ============================================================================

DATASET_CONFIGS = {
    'full': {
        'name': 'Full A-share',
        'path': 'finetune/data/processed_datasets',
        'save_folder': 'full_tokenizer_2k_v1',
        'desc': 'large+mid+small',
    },
    'mid': {
        'name': 'Mid-cap',
        'path': 'finetune/data/processed_datasets_mid',
        'save_folder': 'mid_tokenizer_2k_v1',
        'desc': 'mid-cap stocks',
    },
    'small': {
        'name': 'Small-cap',
        'path': 'finetune/data/processed_datasets_small',
        'save_folder': 'small_tokenizer_2k_v1',
        'desc': 'small-cap stocks',
    },
    'mid_small': {
        'name': 'Mid+Small',
        'path': 'finetune/data/processed_datasets_mid_small',
        'save_folder': 'mid_small_tokenizer_2k_v1',
        'desc': 'mid+small mixed',
    },
}

# Tokenizer 配置
TOKENIZER_CONFIG = {
    'pretrained_path': 'pretrained/Kronos-Tokenizer-2k',  # 用于 mini 模型
}

# 训练参数默认值
TRAINING_PARAMS = {
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 0.001,
    'weight_decay': 0.1,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'seed': 100,
    'n_train_iter_multiplier': 2000,
    'n_val_iter_multiplier': 400,
    'lookback': 400,
    'predict': 10,
    'clip': 5.0,
    'norm_mode': 'full_window',  # 归一化方式：full_window, sliding_ma20, sliding_ma60
    'feature_list': ['open', 'high', 'low', 'close', 'vol', 'amt'],
    'time_feature_list': ['minute', 'hour', 'weekday', 'day', 'month'],
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.0001,
}


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_dataloader(config, data_type):
    dataset = QlibDataset(data_type, config=config)

    if data_type == 'train':
        sampler = RandomSampler(dataset, replacement=True, num_samples=config.n_train_iter)
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=(data_type == 'train')
    )
    return loader, dataset


def train_model(model, device, config, params, save_dir):
    """训练模型，支持动态早停和周期性学习率"""
    start_time = time.time()
    print(f"BATCHSIZE: {params['batch_size']}", flush=True)
    print(f"LR: {params['learning_rate']}", flush=True)
    print(f"OneCycleLR", flush=True)

    train_loader, train_dataset = create_dataloader(config, 'train')
    val_loader, val_dataset = create_dataloader(config, 'val')

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
    print(f"Steps/epoch: {len(train_loader)}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
        betas=(params['adam_beta1'], params['adam_beta2'])
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params['learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=params['epochs'],
        pct_start=0.03,
        div_factor=10
    )

    best_val_loss = float('inf')
    patience_counter = 0

    val_loss_history = []
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch_idx in range(params['epochs']):
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

            # Step scheduler (for OneCycle and Cosine)
            scheduler.step()

            epoch_losses.append(loss.item())

            if (i + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"[E{epoch_idx+1}/{config.epochs} S{i+1}/{len(train_loader)}] LR {lr:.6f} Loss {loss.item():.4f}", flush=True)

        # Record LR
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
        val_loss_history.append(avg_val_loss)

        # Plateau scheduler step
        if config.lr_scheduler == 'plateau':
            scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"\n=== Epoch {epoch_idx+1}/{config.epochs} ===", flush=True)
        print(f"Train Loss: {avg_train_loss:.4f}", flush=True)
        print(f"Val Loss:   {avg_val_loss:.4f}", flush=True)
        print(f"LR: {current_lr:.6f}", flush=True)
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}", flush=True)

        # === 每轮结束都保存最新模型 ===
        latest_path = f"{save_dir}/checkpoints/latest_model"
        model.save_pretrained(latest_path)

        # === 动态早停判断 ===
        improved = False

        if avg_val_loss <= best_val_loss - config.early_stopping_min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            improved = True
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"[VAL LOSS SAVED] {best_val_loss:.4f}", flush=True)

        if not improved:
            patience_counter += 1

        # Grace period（前几轮不早停）
        if epoch_idx < config.early_stopping_grace_period:
            print(f"[GRACE] Epoch {epoch_idx+1} < {config.early_stopping_grace_period}", flush=True)
        else:
            # 窗口趋势判断：看最近 window 轮是否有改善
            window = config.early_stopping_window
            if len(val_loss_history) >= window:
                window_min = min(val_loss_history[-window:])

                # 窗口内有改善 → 继续
                if window_min < best_val_loss:
                    print(f"[WINDOW TREND] Recent improvement detected", flush=True)
                elif patience_counter >= config.early_stopping_patience:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs (patience={config.early_stopping_patience})", flush=True)
                    # 早停时也保存最终模型
                    final_path = f"{save_dir}/checkpoints/final_model"
                    model.save_pretrained(final_path)
                    print(f"[FINAL SAVED] Val Loss: {avg_val_loss:.4f}", flush=True)
                    break

            print(f"[PATIENCE] {patience_counter}/{config.early_stopping_patience}", flush=True)

        print(flush=True)

    # 训练正常结束时保存最终模型
    final_path = f"{save_dir}/checkpoints/final_model"
    model.save_pretrained(final_path)
    print(f"[FINAL SAVED] Val Loss: {avg_val_loss:.4f}", flush=True)

    return {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch_idx + 1,
        'history': history,
        'final_val_loss': avg_val_loss
    }


def main():
    from finetune.dataset import QlibDataset

    parser = argparse.ArgumentParser(description='Kronos Tokenizer Training')
    parser.add_argument('--dataset', type=str, default='mid',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning_rate')
    parser.add_argument('--norm-mode', type=str, default='full_window',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60'],
                        help='Normalization mode for dataset')
    parser.add_argument('--save-folder', type=str, default=None,
                        help='Override save folder name')
    args = parser.parse_args()

    # 从配置获取参数
    dataset_config = DATASET_CONFIGS[args.dataset]
    params = dict(TRAINING_PARAMS)

    # 命令行覆盖
    if args.epochs:
        params['epochs'] = args.epochs
    if args.batch_size:
        params['batch_size'] = args.batch_size
    if args.lr:
        params['learning_rate'] = args.lr
    params['norm_mode'] = args.norm_mode

    # 保存目录
    if args.save_folder:
        save_folder = args.save_folder
    else:
        save_folder = DATASET_CONFIGS[args.dataset]['save_folder']
    save_dir = os.path.join(project_root, "outputs", "models", save_folder)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}", flush=True)
    print(f"Kronos Tokenizer Training", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Dataset: {dataset_config['name']} ({dataset_config['desc']})", flush=True)
    print(f"Norm Mode: {params['norm_mode']}", flush=True)
    print(f"Lookback: {params['lookback']}", flush=True)
    print(f"Device: {device}", flush=True)

    set_seed(params['seed'])

    # 创建配置对象（供 dataset 和 train_model 使用）
    class ConfigAdapter:
        pass

    config = ConfigAdapter()
    config.dataset_path = os.path.join(project_root, dataset_config['path'])
    config.lookback_window = params['lookback']
    config.predict_window = params['predict']
    config.feature_list = params['feature_list']
    config.time_feature_list = params['time_feature_list']
    config.batch_size = params['batch_size']
    config.n_train_iter = params['n_train_iter_multiplier'] * params['batch_size']
    config.n_val_iter = params['n_val_iter_multiplier'] * params['batch_size']
    config.seed = params['seed']
    config.clip = params['clip']
    config.norm_mode = params['norm_mode']  # 归一化方式

    # 训练参数
    config.epochs = params['epochs']
    config.tokenizer_learning_rate = params['learning_rate']
    config.adam_weight_decay = params['weight_decay']
    config.adam_beta1 = params['adam_beta1']
    config.adam_beta2 = params['adam_beta2']
    config.lr_scheduler = 'onecycle'
    config.log_interval = 50
    config.early_stopping_patience = params['early_stopping_patience']
    config.early_stopping_min_delta = params['early_stopping_min_delta']
    config.early_stopping_grace_period = 5
    config.early_stopping_window = 5

    # 加载 Tokenizer
    tokenizer_path = os.path.join(project_root, TOKENIZER_CONFIG['pretrained_path'])
    print(f"\nLoading tokenizer from: {tokenizer_path}", flush=True)
    model = KronosTokenizer.from_pretrained(tokenizer_path)
    model.to(device)

    print(f"Model size: {get_model_size(model):.2f}M parameters", flush=True)

    # Train
    result = train_model(model, device, config, params, save_dir)

    # Save summary
    summary = {
        'model': 'tokenizer',
        'dataset': args.dataset,
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'params': {
            'epochs': params['epochs'],
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'patience': params['early_stopping_patience'],
        },
        'result': result
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print(f"\n{'='*60}", flush=True)
    print(f"Training completed!", flush=True)
    print(f"Dataset: {dataset_config['name']}", flush=True)
    print(f"Best Val Loss: {result['best_val_loss']:.4f}", flush=True)
    print(f"Epochs trained: {result['epochs_trained']}", flush=True)
    print(f"Saved to: {save_dir}", flush=True)
    print(f"\n更新 unified_test.py DATASET_CONFIGS:", flush=True)
    print(f"'{args.dataset}': {{", flush=True)
    print(f"    'tokenizer_finetuned': 'outputs/models/{dataset_config['save_folder']}/checkpoints/best_model',", flush=True)
    print(f"}}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
