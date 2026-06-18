"""
MA20 Predictor Training Script - Mode5

使用 MA20 滑动归一化数据训练 predictor。
MA20 窗口更短，保留更多波动信息，适合 A 股低波动市场。

关键修改：
- 使用 MA20 归一化数据（数据波动率更高）
- 复用 MA60 tokenizer（快速验证，后续可训练 MA20 tokenizer）

Usage:
    python -u finetune/predictor/mode5_ma20/train.py
"""

import os
import sys
import json
import time
import argparse
from time import gmtime, strftime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# ============================================================================
# 配置
# ============================================================================

# MA60 tokenizer 路径（暂时复用，后续可训练 MA20 tokenizer）
TOKENIZER_MA60 = 'outputs/tokenizers/ma60_tokenizer_base_v1/checkpoints/best_model'

# MA20 预归一化数据路径
DATA_PATHS = {
    'train': 'finetune/data/ma20_norm/windowed_lb400_pd10/train_data.pkl',
    'val': 'finetune/data/ma20_norm/windowed_lb400_pd10/val_data.pkl',
    'test': 'finetune/data/ma20_norm/windowed_lb400_pd10/test_data.pkl',
}

# Predictor 预训练路径
PREDICTOR_PRETRAINED = 'pretrained/Kronos-mini'

# 训练参数
TRAINING_PARAMS = {
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 0.003,
    'weight_decay': 0.01,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'seed': 100,
    'lookback': 400,
    'predict': 10,
    'clip': 5.0,
    'max_context': 2048,
    'early_stopping_patience': 12,
    'early_stopping_grace_period': 8,
    'ic_test_samples': 500,
    'ic_point': 3,
    'direction_loss_weight': 0.1,
    'lr_scheduler': 'cosine',
    'lr_min': 1e-5,
    'warmup_epochs': 2,
    'freeze_layers': 0,
    'freeze_embedding': False,
    'ma_window': 20,  # MA20 窗口长度
}


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


class MA20Dataset(Dataset):
    """使用预归一化的 MA20 数据"""

    def __init__(self, data_type='train', config=None):
        self.config = config
        self.data_type = data_type
        self.py_rng = np.random.RandomState(config.seed)

        data_path = DATA_PATHS[data_type]
        print(f"[{data_type.upper()}] Loading: {data_path}")
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.symbols = list(self.raw_data.keys())
        self.window = config.lookback + config.predict

        print(f"[{data_type.upper()}] Pre-computing indices...")
        self.indices = []

        sample_data = self.raw_data[self.symbols[0]]
        has_windows = 'windows' in sample_data

        if has_windows:
            for symbol in self.symbols:
                data = self.raw_data[symbol]
                for start_idx in data['windows']:
                    self.indices.append((symbol, int(start_idx)))
            print(f"[{data_type.upper()}] Windowed format: using pre-assigned {len(self.indices)} windows")
        else:
            for symbol in self.symbols:
                data = self.raw_data[symbol]
                seq_len = len(data['normalized'])
                if seq_len >= self.window:
                    for i in range(seq_len - self.window + 1):
                        self.indices.append((symbol, i))

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

        original_close = data['original'][start_idx:end_idx, 3]
        baseline_close = original_close[self.config.lookback - 1]
        pred_close_end = original_close[self.config.lookback + self.config.predict - 1]
        direction = pred_close_end > baseline_close

        return torch.from_numpy(x_norm), torch.from_numpy(x_stamp), torch.tensor(direction, dtype=torch.float32)


def quick_trajectory_ic_test(model, tokenizer, device, val_data, n_samples=500,
                              lookback=400, pred_len=10, clip=5.0, rng=None):
    """Trajectory IC 测试"""
    from finetune.predictor.shared.eval import FEATURE_NAMES

    model.eval()

    sample_symbol = list(val_data.keys())[0]
    has_windows = 'windows' in val_data[sample_symbol]

    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(pred_len)]

    if has_windows:
        all_windows = []
        for sym in val_data:
            for start in val_data[sym]['windows']:
                all_windows.append((sym, int(start)))

        if rng is not None:
            sample_indices = rng.choice(len(all_windows), size=min(n_samples, len(all_windows)), replace=False)
        else:
            sample_indices = np.arange(min(n_samples, len(all_windows)))

        for idx in sample_indices:
            symbol, start_idx = all_windows[idx]
            data = val_data[symbol]
            end_idx = start_idx + lookback + pred_len

            try:
                x_norm_full = data['normalized'][start_idx:end_idx].astype(np.float32)
                means_full = data['means'][start_idx:end_idx]
                stds_full = data['stds'][start_idx:end_idx]
                timestamps_full = data['index'][start_idx:end_idx]
                orig_full = data['original'][start_idx:end_idx]

                x_norm = x_norm_full[:lookback]
                x_ts = timestamps_full[:lookback]
                y_ts = timestamps_full[lookback:]

                x_stamp = np.stack([
                    x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values, x_ts.day.values, x_ts.month.values
                ], axis=1).astype(np.float32)

                y_stamp = np.stack([
                    y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values, y_ts.day.values, y_ts.month.values
                ], axis=1).astype(np.float32)

                baseline = orig_full[lookback - 1]

                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                    x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                    y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model,
                        x_tensor, x_stamp_tensor, y_stamp_tensor,
                        max_context=2048, pred_len=pred_len,
                        clip=clip, T=1.0, top_k=0, top_p=0.9,
                        sample_count=1, verbose=False
                    )

                    pred_norm = preds[0, lookback:lookback + pred_len, :]
                    pred_raw = pred_norm * stds_full[lookback:lookback + pred_len] + means_full[lookback:lookback + pred_len]
                    actual = orig_full[lookback:lookback + pred_len]

                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_traj = pred_raw[:, fi]
                    actual_traj = actual[:, fi]

                    if len(pred_traj) >= 3:
                        # 检查轨迹方差，避免除零警告
                        pred_std = np.std(pred_traj)
                        actual_std = np.std(actual_traj)
                        if pred_std > 1e-8 and actual_std > 1e-8:
                            traj_ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
                            if np.isfinite(traj_ic):
                                trajectory_ics[fn].append(traj_ic)

                            traj_ric, _ = spearmanr(pred_traj, actual_traj)
                            if np.isfinite(traj_ric):
                                trajectory_rics[fn].append(traj_ric)

                for step_idx in range(pred_len):
                    for fi, fn in enumerate(FEATURE_NAMES):
                        pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                        actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                        da_by_step[step_idx][fn].append(pred_dir == actual_dir)

            except Exception as e:
                if len(trajectory_ics['close']) == 0:
                    print(f"[Trajectory IC TEST] First error: {e}")
                continue
    else:
        all_symbols = list(val_data.keys())
        if rng is not None:
            symbols = rng.choice(all_symbols, size=min(n_samples, len(all_symbols)), replace=False).tolist()
        else:
            symbols = all_symbols[:n_samples]

        for symbol in symbols:
            data = val_data[symbol]
            seq_len = len(data['normalized'])

            if seq_len < lookback + pred_len:
                continue

            try:
                end_idx = seq_len
                start_idx = end_idx - (lookback + pred_len)

                full_len = lookback + pred_len
                x_norm_full = data['normalized'][end_idx - full_len:end_idx].astype(np.float32)
                means_full = data['means'][end_idx - full_len:end_idx]
                stds_full = data['stds'][end_idx - full_len:end_idx]
                timestamps_full = data['index'][end_idx - full_len:end_idx]
                orig_full = data['original'][end_idx - full_len:end_idx]

                x_norm = x_norm_full[:lookback]
                x_ts = timestamps_full[:lookback]
                y_ts = timestamps_full[lookback:]

                x_stamp = np.stack([
                    x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values, x_ts.day.values, x_ts.month.values
                ], axis=1).astype(np.float32)

                y_stamp = np.stack([
                    y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values, y_ts.day.values, y_ts.month.values
                ], axis=1).astype(np.float32)

                baseline = orig_full[lookback - 1]

                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_norm[:lookback]).unsqueeze(0).to(device)
                    x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                    y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model,
                        x_tensor, x_stamp_tensor, y_stamp_tensor,
                        max_context=2048, pred_len=pred_len,
                        clip=clip, T=1.0, top_k=0, top_p=0.9,
                        sample_count=1, verbose=False
                    )

                    pred_norm = preds[0, lookback:lookback + pred_len, :]
                    pred_raw = pred_norm * stds_full[lookback:lookback + pred_len] + means_full[lookback:lookback + pred_len]
                    actual = orig_full[lookback:lookback + pred_len]

                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_traj = pred_raw[:, fi]
                    actual_traj = actual[:, fi]

                    if len(pred_traj) >= 3:
                        # 检查轨迹方差，避免除零警告
                        pred_std = np.std(pred_traj)
                        actual_std = np.std(actual_traj)
                        if pred_std > 1e-8 and actual_std > 1e-8:
                            traj_ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
                            if np.isfinite(traj_ic):
                                trajectory_ics[fn].append(traj_ic)

                            traj_ric, _ = spearmanr(pred_traj, actual_traj)
                            if np.isfinite(traj_ric):
                                trajectory_rics[fn].append(traj_ric)

                for step_idx in range(pred_len):
                    for fi, fn in enumerate(FEATURE_NAMES):
                        pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                        actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                        da_by_step[step_idx][fn].append(pred_dir == actual_dir)

            except Exception as e:
                if len(trajectory_ics['close']) == 0:
                    print(f"[Trajectory IC TEST] First error: {e}")
                continue

    result = {'n_samples': len(trajectory_ics['close'])}

    for fn in FEATURE_NAMES:
        tics = trajectory_ics[fn]
        trics = trajectory_rics[fn]
        result[f'{fn}_trajectory_ic'] = float(np.mean(tics)) if tics else 0.0
        result[f'{fn}_trajectory_rank_ic'] = float(np.mean(trics)) if trics else 0.0

    for step_idx in range(pred_len):
        for fn in FEATURE_NAMES:
            da_list = da_by_step[step_idx][fn]
            result[f'{fn}_da_step{step_idx+1}'] = float(np.mean(da_list)) if da_list else 0.0

    return result


def freeze_model_layers(model, freeze_layers=2, freeze_embedding=True):
    """冻结模型前 N 层"""
    if freeze_embedding:
        for param in model.module.embedding.parameters():
            param.requires_grad = False
        for param in model.module.time_emb.parameters():
            param.requires_grad = False
        print(f"[FREEZE] Embedding + TemporalEmb frozen")

    for i in range(min(freeze_layers, len(model.module.transformer))):
        for param in model.module.transformer[i].parameters():
            param.requires_grad = False
        print(f"[FREEZE] Transformer layer {i} frozen")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FREEZE] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)")


def train_model(model, tokenizer, device, config, save_dir, val_data=None):
    """训练 MA20 predictor"""
    start_time = time.time()

    print(f"MA Window: {config.ma_window}")
    print(f"BATCHSIZE: {config.batch_size}")
    print(f"LR: {config.learning_rate}")
    print(f"LR Scheduler: {config.lr_scheduler}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"Lookback: {config.lookback}")
    print(f"Predict: {config.predict}")

    if config.freeze_layers > 0 or config.freeze_embedding:
        freeze_model_layers(model, config.freeze_layers, config.freeze_embedding)

    d_model = model.module.d_model
    close_direction_head = torch.nn.Linear(d_model, 1).to(device)

    train_dataset = MA20Dataset('train', config=config)
    val_dataset = MA20Dataset('val', config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               sampler=RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset)),
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                              sampler=SequentialSampler(val_dataset),
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    all_params = trainable_params + list(close_direction_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.learning_rate,
                                   weight_decay=config.weight_decay,
                                   betas=(config.adam_beta1, config.adam_beta2))

    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.lr_min
    )

    best_val_loss = float('inf')
    best_ic = -999
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'ic': [], 'lr': []}

    for epoch_idx in range(config.epochs):
        epoch_start = time.time()
        model.train()

        train_dataset.py_rng.seed(config.seed + epoch_idx * 10000)

        epoch_losses = []
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n=== Epoch {epoch_idx+1}/{config.epochs} ===")
        print(f"LR: {current_lr:.6f}")

        for i, (batch_x, batch_stamp, batch_direction) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_stamp = batch_stamp.to(device, non_blocking=True)
            batch_direction = batch_direction.to(device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
                token_seq_0, token_seq_1, batch_stamp
            )

            recon_loss, ce_s1, ce_s2 = model.module.head.compute_loss(
                s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
            )

            pred_hidden = hidden[:, -1, :]
            direction_logits = close_direction_head(pred_hidden).squeeze(-1)
            direction_pred = torch.sigmoid(direction_logits)
            direction_loss = F.binary_cross_entropy(direction_pred, batch_direction)

            total_loss = recon_loss + config.direction_loss_weight * direction_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
            optimizer.step()

            scheduler.step()

            epoch_losses.append(total_loss.item())

            if (i + 1) % 50 == 0 or i == 0:
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (i + 1) / len(train_loader) * 100
                print(f"  Batch {i+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {total_loss.item():.4f}, Avg: {avg_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_stamp, batch_direction in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_stamp = batch_stamp.to(device, non_blocking=True)
                batch_direction = batch_direction.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
                    token_seq_0, token_seq_1, batch_stamp
                )

                recon_loss, _, _ = model.module.head.compute_loss(
                    s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
                )

                pred_hidden = hidden[:, -1, :]
                direction_logits = close_direction_head(pred_hidden).squeeze(-1)
                direction_pred = torch.sigmoid(direction_logits)
                direction_loss = F.binary_cross_entropy(direction_pred, batch_direction)

                val_loss = recon_loss + config.direction_loss_weight * direction_loss

                val_loss_sum += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        ic_rng = np.random.RandomState(config.seed + epoch_idx * 9999)
        traj_result = quick_trajectory_ic_test(model.module, tokenizer, device, val_data,
                                                n_samples=config.ic_test_samples,
                                                rng=ic_rng)

        current_ic = 0
        if traj_result:
            current_ic = traj_result.get('close_trajectory_ic', 0)
            history['ic'].append(current_ic)

            # 各特征的 Trajectory IC 汇总
            print(f"\n  {'Feature':<8} {'Traj_IC':>8} {'RankIC':>8}")
            for fn in ['open', 'high', 'low', 'close', 'vol', 'amt']:
                tic = traj_result.get(f'{fn}_trajectory_ic', 0)
                ric = traj_result.get(f'{fn}_trajectory_rank_ic', 0)
                print(f"  {fn:<8} {tic:>8.4f} {ric:>8.4f}")

            # 各步各特征 DA 表格
            print(f"\n  {'Step':<6} {'open_DA':>7} {'high_DA':>7} {'low_DA':>7} {'close_DA':>7} {'vol_DA':>7} {'amt_DA':>7}")
            for step_idx in range(config.predict):
                suffix = f'_step{step_idx+1}'
                print(f"  +{step_idx+1:<5} "
                      f"{traj_result.get(f'open_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'high_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'low_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'close_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'vol_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'amt_da{suffix}', 0):>7.0%}")

        ic_window = 3
        if len(history['ic']) >= ic_window:
            ic_smoothed = np.mean(history['ic'][-ic_window:])
        else:
            ic_smoothed = current_ic

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        print(f"Trajectory IC (close): {current_ic:.4f}, IC_smoothed: {ic_smoothed:.4f} (best: {best_ic:.4f})")
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}")
        print(f"[LR] {current_lr:.6f}")

        latest_path = f"{save_dir}/checkpoints/latest_model"
        model.module.save_pretrained(latest_path)

        improved = False

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            improved = True
            save_path = f"{save_dir}/checkpoints/best_model"
            model.module.save_pretrained(save_path)
            print(f"[VAL LOSS SAVED] {best_val_loss:.4f}")

        if ic_smoothed > best_ic:
            best_ic = ic_smoothed
            patience_counter = 0
            improved = True
            ic_save_path = f"{save_dir}/checkpoints/best_ic_model"
            model.module.save_pretrained(ic_save_path)
            print(f"[IC SAVED] {best_ic:.4f}")

        if not improved:
            patience_counter += 1

        if epoch_idx >= config.early_stopping_grace_period:
            if patience_counter >= config.early_stopping_patience:
                print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                final_path = f"{save_dir}/checkpoints/final_model"
                model.module.save_pretrained(final_path)
                break

        print(flush=True)

    final_path = f"{save_dir}/checkpoints/final_model"
    model.module.save_pretrained(final_path)

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'epochs_trained': epoch_idx + 1,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='MA20 Predictor Training (Mode5)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-folder', type=str, default='mode5_lb400_pd10')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*60)
    print("MA20 Predictor Training (Mode5)")
    print("="*60)
    print(f"MA Window: {TRAINING_PARAMS['ma_window']}")
    print(f"Tokenizer: {TOKENIZER_MA60}")
    print(f"Data: {DATA_PATHS['train']}")
    print(f"Device: {device}")
    print("="*60)

    set_seed(TRAINING_PARAMS['seed'])

    save_dir = os.path.join(project_root, "outputs/models", args.save_folder)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, TOKENIZER_MA60))
    tokenizer.eval().to(device)
    print(f"Tokenizer loaded from: {TOKENIZER_MA60}")

    if args.resume:
        predictor_path = args.resume
        print(f"Predictor loaded from (resume): {predictor_path}")
    else:
        predictor_path = os.path.join(project_root, PREDICTOR_PRETRAINED)
        print(f"Predictor loaded from: {PREDICTOR_PRETRAINED}")
    model = Kronos.from_pretrained(predictor_path)
    model.to(device)
    print(f"Model size: {get_model_size(model):.2f}M")

    class ModelWrapper:
        def __init__(self, model):
            self.module = model
        def train(self):
            self.module.train()
        def eval(self):
            self.module.eval()
        def parameters(self):
            return self.module.parameters()
        def __getattr__(self, name):
            return getattr(self.module, name)

    model = ModelWrapper(model)

    class Config:
        pass

    config = Config()
    config.seed = TRAINING_PARAMS['seed']
    config.lookback = TRAINING_PARAMS['lookback']
    config.predict = TRAINING_PARAMS['predict']
    config.clip = TRAINING_PARAMS['clip']
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = TRAINING_PARAMS['weight_decay']
    config.adam_beta1 = TRAINING_PARAMS['adam_beta1']
    config.adam_beta2 = TRAINING_PARAMS['adam_beta2']
    config.n_train_iter = 2000 * config.batch_size
    config.n_val_iter = 400 * config.batch_size
    config.early_stopping_patience = TRAINING_PARAMS['early_stopping_patience']
    config.early_stopping_grace_period = TRAINING_PARAMS['early_stopping_grace_period']
    config.ic_test_samples = args.n_samples
    config.ic_point = TRAINING_PARAMS['ic_point']
    config.direction_loss_weight = TRAINING_PARAMS['direction_loss_weight']
    config.lr_scheduler = TRAINING_PARAMS['lr_scheduler']
    config.lr_min = TRAINING_PARAMS['lr_min']
    config.warmup_epochs = TRAINING_PARAMS['warmup_epochs']
    config.freeze_layers = TRAINING_PARAMS['freeze_layers']
    config.freeze_embedding = TRAINING_PARAMS['freeze_embedding']
    config.ma_window = TRAINING_PARAMS['ma_window']

    val_path = DATA_PATHS['val']
    print(f"Loading val data for IC evaluation: {val_path}")
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    print(f"Val: {len(val_data)} stocks")

    result = train_model(model, tokenizer, device, config, save_dir, val_data)

    summary = {
        'tokenizer': TOKENIZER_MA60,
        'data_path': DATA_PATHS['train'],
        'ma_window': TRAINING_PARAMS['ma_window'],
        'config': TRAINING_PARAMS,
        'result': {
            'best_val_loss': result['best_val_loss'],
            'best_ic': result['best_ic'],
            'epochs_trained': result['epochs_trained'],
        }
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print("\n" + "="*60)
    print("Mode5 MA20 Training completed!")
    print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"Best IC: {result['best_ic']:.4f}")
    print(f"Epochs: {result['epochs_trained']}")
    print(f"Saved to: {save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()