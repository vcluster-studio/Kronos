"""
MA60 Predictor Training Script - 使用预归一化数据

训练与 MA60 tokenizer 匹配的 predictor。
使用预计算好的 MA60 归一化数据，提高效率。

Usage:
    python -u finetune/train_predictor_ma60.py

数据格式：
    processed_datasets_ma60/{train,val,test}_data.pkl
    每个股票: {'normalized': (T,6), 'means': (T,6), 'stds': (T,6), 'original': (T,6), 'index': DatetimeIndex}
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
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# ============================================================================
# 配置
# ============================================================================

# MA60 tokenizer 路径
TOKENIZER_MA60 = 'outputs/models/ma60_tokenizer_v1/checkpoints/best_model'

# 预归一化数据路径
DATA_PATHS = {
    'train': 'finetune/data/processed_datasets_ma60/train_data.pkl',
    'val': 'finetune/data/processed_datasets_ma60/val_data.pkl',
    'test': 'finetune/data/processed_datasets_ma60/test_data.pkl',
}

# Predictor 预训练路径
PREDICTOR_PRETRAINED = 'pretrained/Kronos-mini'

# 训练参数
TRAINING_PARAMS = {
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 0.02,
    'weight_decay': 0.1,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'seed': 100,
    'lookback': 400,
    'predict': 10,
    'clip': 5.0,
    'max_context': 2048,
    # 早停
    'early_stopping_patience': 10,
    'early_stopping_grace_period': 5,
    # IC 测试
    'ic_test_samples': 500,
    'ic_point': 3,
    # 方向损失
    'direction_loss_weight': 0.3,
    # VL-Adaptive LR
    'lr_scheduler': 'vl_adaptive',
    'lr_max': 0.03,
    'lr_min': 1e-4,
    'lr_decay_factor': 0.7,
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


class MA60Dataset(Dataset):
    """
    使用预归一化的 MA60 数据

    数据格式:
        {symbol: {'normalized': (T,6), 'means': (T,6), 'stds': (T,6), 'original': (T,6), 'index': DatetimeIndex}}
    """

    def __init__(self, data_type='train', config=None):
        self.config = config
        self.data_type = data_type
        self.py_rng = np.random.RandomState(config.seed)

        # 加载预归一化数据
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
        # 随机采样
        rand_idx = self.py_rng.randint(0, len(self.indices))
        symbol, start_idx = self.indices[rand_idx]
        end_idx = start_idx + self.window

        data = self.raw_data[symbol]

        # 直接使用预归一化的数据
        x_norm = data['normalized'][start_idx:end_idx].astype(np.float32)

        # 时间戳
        timestamps = data['index'][start_idx:end_idx]
        x_stamp = np.stack([
            timestamps.minute.values,
            timestamps.hour.values,
            timestamps.weekday.values,
            timestamps.day.values,
            timestamps.month.values,
        ], axis=1).astype(np.float32)

        # 方向标签（用原始数据计算）
        original_close = data['original'][start_idx:end_idx, 3]  # close 列
        baseline_close = original_close[self.config.lookback - 1]
        pred_close_end = original_close[self.config.lookback + self.config.predict]
        direction = pred_close_end > baseline_close

        return torch.from_numpy(x_norm), torch.from_numpy(x_stamp), torch.tensor(direction, dtype=torch.float32)


def quick_ic_test_ma60(model, tokenizer, device, val_data, n_samples=500,
                       lookback=400, pred_len=10, ic_point=3, clip=5.0, rng=None):
    """
    使用预归一化 MA60 数据的 IC 测试（val_data，随机采样）

    val_data: 预归一化格式 {symbol: {'normalized', 'means', 'stds', 'original', 'index'}}
    rng: numpy RandomState，用于可复现的随机采样
    """
    model.eval()

    predictions = []
    actuals = []

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
            # 取最后 lookback + pred_len 数据
            end_idx = seq_len
            start_idx = end_idx - (lookback + pred_len)

            # 取 lookback + pred_len 长度的数据
            full_len = lookback + pred_len
            x_norm_full = data['normalized'][end_idx - full_len:end_idx].astype(np.float32)
            means_full = data['means'][end_idx - full_len:end_idx]
            stds_full = data['stds'][end_idx - full_len:end_idx]

            timestamps_full = data['index'][end_idx - full_len:end_idx]

            # 分割为 x (lookback) 和 y (pred_len)
            x_norm = x_norm_full[:lookback]
            x_ts = timestamps_full[:lookback]
            y_ts = timestamps_full[lookback:]

            # means/stds 用于反归一化预测部分
            means = means_full
            stds = stds_full

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values, x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values, y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            # 原始 close（用于计算实际收益率）
            original_close = data['original'][:, 3]
            baseline_close = original_close[end_idx - pred_len - 1]

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

                # 预测 close（取 pred_len 部分）
                # auto_regressive_inference 返回 numpy array，不需要 .cpu()
                pred_close_norm = preds[0, -pred_len:, 3]

                # 反归一化（使用预计算的 means/stds）
                pred_close_raw = pred_close_norm * stds[lookback:, 3] + means[lookback:, 3]

                # 预测收益率
                pred_return = (pred_close_raw[ic_point - 1] - baseline_close) / baseline_close

            # 实际收益率
            actual_close = original_close[end_idx - pred_len:]
            actual_return = (actual_close[ic_point - 1] - baseline_close) / baseline_close

            predictions.append(pred_return)
            actuals.append(actual_return)

        except Exception as e:
            # 打印第一个错误，便于调试
            if len(predictions) == 0:
                print(f"[IC TEST] First error: {e}")
            continue

    if len(predictions) > 5:
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        ic = np.corrcoef(predictions, actuals)[0, 1]
        rank_ic, _ = spearmanr(predictions, actuals)
        direction_acc = np.mean((predictions > 0) == (actuals > 0))

        return {'ic': ic, 'rank_ic': rank_ic, 'direction_acc': direction_acc, 'n_samples': len(predictions)}

    return None


def train_model(model, tokenizer, device, config, save_dir, val_data=None):
    """训练 MA60 predictor"""
    start_time = time.time()

    print(f"BATCHSIZE: {config.batch_size}")
    print(f"LR: {config.learning_rate}")
    print(f"Lookback: {config.lookback}")
    print(f"Predict: {config.predict}")

    # 数据集
    train_dataset = MA60Dataset('train', config=config)
    val_dataset = MA60Dataset('val', config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               sampler=RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset)),
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                              sampler=SequentialSampler(val_dataset),
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 方向预测 head
    import torch.nn as nn
    d_model = model.module.d_model
    close_direction_head = nn.Linear(d_model, 1).to(device)

    # 优化器
    all_params = list(model.parameters()) + list(close_direction_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.learning_rate,
                                   weight_decay=config.weight_decay,
                                   betas=(config.adam_beta1, config.adam_beta2))

    best_val_loss = float('inf')
    best_ic = -999
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'ic': [], 'lr': []}

    # VL-Adaptive LR 状态
    good_lr_pool = []
    lr_before_decay = None

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

            # Tokenize
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward
            s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
                token_in[0], token_in[1], batch_stamp[:, :-1, :]
            )
            recon_loss, s1_loss, s2_loss = model.module.head.compute_loss(
                s1_logits, s2_logits, token_out[0], token_out[1]
            )

            # 方向损失
            pred_hidden = hidden[:, -1, :]
            direction_logits = close_direction_head(pred_hidden).squeeze(-1)
            direction_pred = torch.sigmoid(direction_logits)
            direction_loss = F.binary_cross_entropy(direction_pred, batch_direction)

            total_loss = recon_loss + config.direction_loss_weight * direction_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())

            if (i + 1) % 50 == 0 or i == 0:
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (i + 1) / len(train_loader) * 100
                print(f"  Batch {i+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {total_loss.item():.4f}, Avg: {avg_loss:.4f}")

        history['lr'].append(current_lr)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_stamp, batch_direction in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_stamp = batch_stamp.to(device, non_blocking=True)
                batch_direction = batch_direction.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
                    token_in[0], token_in[1], batch_stamp[:, :-1, :]
                )
                recon_loss, _, _ = model.module.head.compute_loss(
                    s1_logits, s2_logits, token_out[0], token_out[1]
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

        # IC test (使用 val_data，随机采样)
        ic_rng = np.random.RandomState(config.seed + epoch_idx * 9999)
        ic_result = quick_ic_test_ma60(model.module, tokenizer, device, val_data,
                                        n_samples=config.ic_test_samples, ic_point=config.ic_point,
                                        rng=ic_rng)
        current_ic = 0
        if ic_result:
            current_ic = ic_result['ic']
            history['ic'].append(current_ic)

        # IC 滑动均值（用于决策，减少噪声）
        ic_window = 3
        if len(history['ic']) >= ic_window:
            ic_smoothed = np.mean(history['ic'][-ic_window:])
        else:
            ic_smoothed = current_ic

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        print(f"IC: {current_ic:.4f}, IC_smoothed: {ic_smoothed:.4f} (best: {best_ic:.4f})")
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}")

        # VL-Adaptive LR
        if config.lr_scheduler == 'vl_adaptive':
            tl_history = history['train_loss']
            vl_history = history['val_loss']

            def calc_stability(loss_history, window=3):
                if len(loss_history) < window:
                    return True, None
                recent = loss_history[-window:]
                std = np.std(recent)
                mean = np.mean(recent)
                is_stable = std < mean * 0.05
                is_declining = recent[-1] < recent[0]
                return is_stable, is_declining

            tl_stable, tl_declining = calc_stability(tl_history)
            vl_stable, vl_declining = calc_stability(vl_history)

            vl_worsening = avg_val_loss > best_val_loss
            new_lr = current_lr
            lr_action = ""

            if vl_worsening:
                if good_lr_pool:
                    best_good_lr = min(good_lr_pool, key=lambda x: x[1])[0]
                    new_lr = best_good_lr
                    lr_action = f"ROLLBACK to {best_good_lr:.6f}"
                else:
                    new_lr = max(current_lr * config.lr_decay_factor, config.lr_min)
                    lr_action = f"DECAY (VL worsening)"
                lr_before_decay = current_lr
            elif tl_stable and tl_declining and vl_stable:
                lr_action = "KEEP (stable)"
            elif tl_stable and vl_declining and current_lr < config.lr_max:
                # 探索更高 LR
                candidate = current_lr * 1.1
                if candidate <= config.lr_max:
                    new_lr = candidate
                    lr_action = f"EXPLORE {candidate:.6f}"
            else:
                lr_action = "KEEP"

            # 记录好 LR
            if tl_stable and tl_declining and vl_stable and vl_declining:
                good_lr_pool.append((current_lr, avg_val_loss, avg_train_loss))
                good_lr_pool = sorted(good_lr_pool, key=lambda x: x[1])[:5]
                lr_action += " [GOOD LR]"

            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"[VL-ADAPTIVE] {lr_action}")

        # Save latest
        latest_path = f"{save_dir}/checkpoints/latest_model"
        model.module.save_pretrained(latest_path)

        # Check improvement
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

        # Early stopping
        if epoch_idx >= config.early_stopping_grace_period:
            if patience_counter >= config.early_stopping_patience:
                print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                final_path = f"{save_dir}/checkpoints/final_model"
                model.module.save_pretrained(final_path)
                break

        print(flush=True)

    # Final save
    final_path = f"{save_dir}/checkpoints/final_model"
    model.module.save_pretrained(final_path)

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'epochs_trained': epoch_idx + 1,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='MA60 Predictor Training')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (e.g. outputs/models/ma60_predictor_v1/checkpoints/best_ic_model)')
    parser.add_argument('--save-folder', type=str, default='ma60_predictor_v1',
                        help='Save folder name (default: ma60_predictor_v1)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*60)
    print("MA60 Predictor Training")
    print("="*60)
    print(f"Tokenizer: {TOKENIZER_MA60}")
    print(f"Data: {DATA_PATHS['train']}")
    print(f"Device: {device}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("="*60)

    set_seed(TRAINING_PARAMS['seed'])

    # 保存目录
    save_dir = os.path.join(project_root, "outputs/models", args.save_folder)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # 加载 tokenizer
    tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, TOKENIZER_MA60))
    tokenizer.eval().to(device)
    print(f"Tokenizer loaded from: {TOKENIZER_MA60}")

    # 加载 predictor (支持 resume)
    if args.resume:
        predictor_path = args.resume
        print(f"Predictor loaded from (resume): {predictor_path}")
    else:
        predictor_path = os.path.join(project_root, PREDICTOR_PRETRAINED)
        print(f"Predictor loaded from: {PREDICTOR_PRETRAINED}")
    model = Kronos.from_pretrained(predictor_path)
    model.to(device)
    print(f"Model size: {get_model_size(model):.2f}M")

    # Wrap model
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

    # 配置
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
    config.lr_max = TRAINING_PARAMS['lr_max']
    config.lr_min = TRAINING_PARAMS['lr_min']
    config.lr_decay_factor = TRAINING_PARAMS['lr_decay_factor']

    # 加载验证数据（IC 评估用）
    val_path = DATA_PATHS['val']
    print(f"Loading val data for IC evaluation: {val_path}")
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    print(f"Val: {len(val_data)} stocks")

    # 训练
    result = train_model(model, tokenizer, device, config, save_dir, val_data)

    # 保存结果
    summary = {
        'tokenizer': TOKENIZER_MA60,
        'data_path': DATA_PATHS['train'],
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
    print("Training completed!")
    print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"Best IC: {result['best_ic']:.4f}")
    print(f"Epochs: {result['epochs_trained']}")
    print(f"Saved to: {save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()