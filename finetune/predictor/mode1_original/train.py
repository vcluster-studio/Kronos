"""
Kronos Predictor Training Script - with early stopping and IC monitoring

Usage:
    python -u finetune/train_predictor.py --model mini --dataset mid
    python -u finetune/train_predictor.py --model small --dataset full

Tokenizer matching (IMPORTANT):
    - mini: Kronos-Tokenizer-2k (group_size=5, 2048 context)
    - small/base: Kronos-Tokenizer-base (group_size=4, 512 context)

Config managed via static variables. Add new models/datasets by updating these.
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
import numpy as np
from scipy.stats import spearmanr

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

# Feature names (consistent across all evaluation functions)
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']

# ============================================================================
# 训练配置清单（新增模型/数据集只需更新此变量）
# ============================================================================

# Model configs - includes tokenizer matching info
MODEL_CONFIGS = {
    'mini': {
        'name': 'Kronos-mini',
        'pretrained_path': 'pretrained/Kronos-mini',
        'tokenizer_pretrained': 'pretrained/Kronos-Tokenizer-2k',  # mini uses 2k tokenizer
        'params': '4.1M',
        'max_context': 2048,
    },
    'small': {
        'name': 'Kronos-small',
        'pretrained_path': 'pretrained/Kronos-small',
        'tokenizer_pretrained': 'pretrained/Kronos-Tokenizer-base',  # small uses base tokenizer
        'params': '24.7M',
        'max_context': 512,
    },
    'base': {
        'name': 'Kronos-base',
        'pretrained_path': 'pretrained/Kronos-base',
        'tokenizer_pretrained': 'pretrained/Kronos-Tokenizer-base',  # base uses base tokenizer
        'params': '102.3M',
        'max_context': 512,
    },
}

# Dataset configs
DATASET_CONFIGS = {
    'global': {
        'name': 'Global Norm',
        'path': 'finetune/data/global_norm/full_series',
        'tokenizer_finetuned': None,  # 使用 pretrained tokenizer
        'desc': '原始数据，训练时动态 full_window 归一化',
    },
    'full': {
        'name': 'Full A-share',
        'path': 'finetune/data/processed_datasets',
        'tokenizer_finetuned': 'outputs/models/full_tokenizer_2k_v1/checkpoints/best_model',
        'desc': 'large+mid+small',
    },
    'mid': {
        'name': 'Mid-cap',
        'path': 'finetune/data/processed_datasets_mid',
        'tokenizer_finetuned': 'outputs/models/mid_tokenizer_2k_v1/checkpoints/best_model',
        'desc': 'mid-cap stocks',
    },
    'small': {
        'name': 'Small-cap',
        'path': 'finetune/data/processed_datasets_small',
        'tokenizer_finetuned': 'outputs/models/small_tokenizer_2k_v1/checkpoints/best_model',
        'desc': 'small-cap stocks',
    },
    'mid_small': {
        'name': 'Mid+Small',
        'path': 'finetune/data/processed_datasets_mid_small',
        'tokenizer_finetuned': 'outputs/models/mid_small_tokenizer_2k_v1/checkpoints/best_model',
        'desc': 'mid+small mixed',
    },
}

# Tokenizer 配置
TOKENIZER_CONFIG = {
    'pretrained_path': 'pretrained/Kronos-Tokenizer-base',
}

# 训练参数默认值
TRAINING_PARAMS = {
    'epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.003,  # 起始 LR（cosine annealing）
    'weight_decay': 0.01,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'seed': 100,
    'n_train_iter_multiplier': 2000,  # n_train_iter = multiplier * batch_size
    'n_val_iter_multiplier': 400,     # n_val_iter = multiplier * batch_size
    'lookback': 200,  # mini 模型使用较短 lookback
    'predict': 10,
    'clip': 5.0,
    # 归一化模式
    'norm_mode': 'full_window',  # 'full_window' 或 'sliding_ma60'
    # 早停参数
    'early_stopping_patience': 12,
    'early_stopping_min_delta': 0.0001,
    'early_stopping_grace_period': 8,
    'early_stopping_check_ic': True,
    'early_stopping_window': 5,
    # Cosine Annealing LR
    'lr_scheduler': 'cosine',
    'lr_min': 1e-5,
    'warmup_epochs': 2,
    # 冻结层
    'freeze_layers': 0,
    'freeze_embedding': False,
    # IC 计算目标点（Point+N，N=3表示预测窗口第3个点）
    'ic_point': 3,  # 默认计算 Point+3 的 IC（实际最有价值）
    # 方向损失（从 hidden state 直接预测，梯度可传）
    'direction_loss_weight': 0.3,  # 方向损失权重
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


def quick_ic_test(model, tokenizer, device, val_data, n_samples=500, lookback=90, pred_len=10, rng=None):
    """
    快速评估（训练过程中使用）

    使用 auto_regressive_inference 进行真正的未来预测

    评估指标（6个特征各自计算）：
    1. Trajectory IC: 同一股票内，预测轨迹 vs 实际轨迹的相关性
    2. MAE: 预测绝对误差
    3. Direction Acc: 方向准确率

    Returns:
        dict: 各特征的 trajectory_ic, mae, direction_acc（单步 + 整体）
    """
    from model.kronos import auto_regressive_inference
    model.eval()

    # 按特征收集：trajectory_ics[feature] = [所有股票的轨迹IC]
    trajectory_ics = {f: [] for f in FEATURE_NAMES}

    # 按特征按步收集：mae_by_step[step][feature] = [误差列表]
    mae_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(pred_len)]

    # 按特征按步收集：direction_by_step[step][feature] = [方向正确列表]
    direction_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(pred_len)]

    all_symbols = list(val_data.keys())
    if rng is not None:
        symbols = rng.choice(all_symbols, size=min(n_samples, len(all_symbols)), replace=False).tolist()
    else:
        symbols = all_symbols[:n_samples]

    n = 0
    for symbol in symbols:
        df = val_data[symbol]
        if len(df) < lookback + pred_len:
            continue

        try:
            import pandas as pd
            # datetime 可能是 index 或 column
            if 'datetime' in df.columns:
                dates = df['datetime'].values[-(lookback + pred_len):]
            else:
                dates = df.index[-(lookback + pred_len):]
            dates = pd.to_datetime(dates)
            x_timestamp = dates[:lookback]
            y_timestamp = dates[lookback:]

            time_df_x = pd.DataFrame()
            time_df_x['minute'] = x_timestamp.minute
            time_df_x['hour'] = x_timestamp.hour
            time_df_x['weekday'] = x_timestamp.weekday
            time_df_x['day'] = x_timestamp.day
            time_df_x['month'] = x_timestamp.month

            time_df_y = pd.DataFrame()
            time_df_y['minute'] = y_timestamp.minute
            time_df_y['hour'] = y_timestamp.hour
            time_df_y['weekday'] = y_timestamp.weekday
            time_df_y['day'] = y_timestamp.day
            time_df_y['month'] = y_timestamp.month

            # Prepare data
            values = df[FEATURE_NAMES].values.astype(np.float32)
            x = values[:lookback]  # lookback 部分
            y = values[lookback:lookback+pred_len]  # ground truth

            # 全窗口归一化（pretrained原始方式）
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0) + 1e-5
            x_norm = (x - x_mean) / x_std
            x_norm = np.clip(x_norm, -5.0, 5.0)

            # baseline（lookback 最后一个值，用于计算方向）
            baseline = values[lookback - 1]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(time_df_x.values.astype(np.float32)).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(time_df_y.values.astype(np.float32)).unsqueeze(0).to(device)

                # auto_regressive inference
                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=512, pred_len=pred_len,
                    clip=5, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                # preds shape: (1, lookback+pred_len, 6) -> 取预测部分
                pred_norm = preds[0, lookback:lookback+pred_len, :]  # (pred_len, 6)

                # 按特征收集轨迹
                pred_traj_by_feature = {f: [] for f in FEATURE_NAMES}
                actual_traj_by_feature = {f: [] for f in FEATURE_NAMES}

                for step_idx in range(pred_len):
                    pred_raw = pred_norm[step_idx] * x_std + x_mean
                    actual = values[lookback + step_idx]

                    # 收集各特征轨迹
                    for fi, fn in enumerate(FEATURE_NAMES):
                        pred_traj_by_feature[fn].append(pred_raw[fi])
                        actual_traj_by_feature[fn].append(actual[fi])

                        # MAE
                        abs_err = abs(pred_raw[fi] - actual[fi])
                        mae_by_step[step_idx][fn].append(abs_err)

                        # Direction Acc（相对于 baseline）
                        pred_dir = (pred_raw[fi] - baseline[fi]) > 0
                        actual_dir = (actual[fi] - baseline[fi]) > 0
                        direction_by_step[step_idx][fn].append(pred_dir == actual_dir)

                # 计算各特征的 Trajectory IC（同一股票内）
                for fn in FEATURE_NAMES:
                    pred_traj = np.array(pred_traj_by_feature[fn])
                    actual_traj = np.array(actual_traj_by_feature[fn])

                    if len(pred_traj) >= 3:
                        # 检查轨迹方差，避免除零警告
                        pred_std = np.std(pred_traj)
                        actual_std = np.std(actual_traj)
                        if pred_std > 1e-8 and actual_std > 1e-8:
                            traj_ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
                            if np.isfinite(traj_ic):
                                trajectory_ics[fn].append(traj_ic)

            n += 1

        except Exception as e:
            continue

    # 计算聚合指标
    result = {'n_samples': n}

    # 各特征的 Trajectory IC（整体）
    for fn in FEATURE_NAMES:
        traj_ics = trajectory_ics[fn]
        result[f'{fn}_trajectory_ic'] = float(np.mean(traj_ics)) if traj_ics else 0.0
        result[f'{fn}_trajectory_ic_std'] = float(np.std(traj_ics)) if len(traj_ics) >= 2 else 0.0
        result[f'{fn}_trajectory_ic_pos_pct'] = float(sum(t > 0 for t in traj_ics) / len(traj_ics)) if traj_ics else 0.0

    # 各步各特征的 MAE 和 DA
    for step_idx in range(pred_len):
        suffix = f'_step{step_idx+1}'
        for fn in FEATURE_NAMES:
            # MAE
            mae_list = mae_by_step[step_idx][fn]
            result[f'{fn}_mae{suffix}'] = float(np.mean(mae_list)) if mae_list else 0.0

            # Direction Acc
            da_list = direction_by_step[step_idx][fn]
            result[f'{fn}_da{suffix}'] = float(np.mean(da_list)) if da_list else 0.0

    # 整体指标（用于早停，使用 close）
    result['trajectory_ic'] = result.get('close_trajectory_ic', 0)
    result['ic'] = result.get('close_trajectory_ic', 0)
    result['direction_acc'] = result.get('close_da_step10', 0) if pred_len >= 10 else result.get(f'close_da_step{pred_len}', 0)

    return result


def create_dataloader(dataset, config, data_type):
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
    return loader


def freeze_model_layers(model, freeze_layers=2, freeze_embedding=True):
    """冻结模型前 N 层 transformer 和 embedding"""
    # 冻结 embedding
    if freeze_embedding:
        for param in model.module.embedding.parameters():
            param.requires_grad = False
        for param in model.module.time_emb.parameters():
            param.requires_grad = False
        print(f"[FREEZE] Embedding + TemporalEmb frozen", flush=True)

    # 冻结前 N 层 transformer
    for i in range(min(freeze_layers, len(model.module.transformer))):
        for param in model.module.transformer[i].parameters():
            param.requires_grad = False
        print(f"[FREEZE] Transformer layer {i} frozen", flush=True)

    # 统计可训练参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FREEZE] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)", flush=True)


def train_model(model, tokenizer, device, config, save_dir, val_data=None):
    """训练模型，使用 Cosine Annealing + Warmup + Layer Freezing"""
    start_time = time.time()
    print(f"BATCHSIZE: {config.batch_size}", flush=True)
    print(f"LR: {config.predictor_learning_rate}", flush=True)
    print(f"LR Scheduler: {config.lr_scheduler}", flush=True)
    print(f"Weight Decay: {config.adam_weight_decay}", flush=True)
    print(f"Warmup: {config.warmup_epochs} epochs", flush=True)
    print(f"IC Test Samples: {config.ic_test_samples}", flush=True)

    # 冻结层
    if config.freeze_layers > 0 or config.freeze_embedding:
        freeze_model_layers(model, config.freeze_layers, config.freeze_embedding)

    # Import dataset
    from finetune.predictor.shared.dataset import QlibDataset
    import torch.nn as nn  # 用于 close_direction_head

    train_dataset = QlibDataset('train', config=config)
    val_dataset = QlibDataset('val', config=config)

    train_loader = create_dataloader(train_dataset, config, 'train')
    val_loader = create_dataloader(val_dataset, config, 'val')

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
    print(f"Steps/epoch: {len(train_loader)}", flush=True)

    # === Close 方向预测 Head（直接从 hidden state 预测，梯度可传）===
    d_model = model.module.d_model
    close_direction_head = nn.Linear(d_model, 1).to(device)  # 输出方向概率
    print(f"Close direction head: Linear({d_model}, 1)", flush=True)

    # 优化器（只优化可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    all_params = trainable_params + list(close_direction_head.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=config.predictor_learning_rate,
        weight_decay=config.adam_weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    # Cosine Annealing LR（无 warmup）
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.lr_min
    )
    print(f"CosineAnnealing: total={total_steps} steps", flush=True)

    best_val_loss = float('inf')
    best_ic = -999
    patience_counter = 0

    # 动态早停：记录历史用于趋势判断
    val_loss_history = []
    ic_history = []

    history = {
        'train_loss': [],
        'val_loss': [],
        'ic': [],
        'lr': []
    }

    for epoch_idx in range(config.epochs):
        epoch_start = time.time()
        model.train()

        # Set dataset epoch
        train_dataset.py_rng.seed(config.seed + epoch_idx * 10000)

        epoch_losses = []
        current_lr = optimizer.param_groups[0]['lr']

        # 显示本轮训练使用的 LR
        print(f"\n=== Epoch {epoch_idx+1}/{config.epochs} ===", flush=True)
        print(f"Training with LR: {current_lr:.6f}", flush=True)

        for i, (batch_x, batch_stamp, batch_direction) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_stamp = batch_stamp.to(device, non_blocking=True)
            batch_direction = batch_direction.to(device, non_blocking=True)

            # Tokenize
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward（获取 logits 和 hidden state）
            s1_logits, s2_logits, hidden = model.module.forward_with_hidden(token_seq_0, token_seq_1, batch_stamp)
            recon_loss, s1_loss, s2_loss = model.module.head.compute_loss(
                s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
            )

            # === 方向损失（直接从 hidden state 预测，梯度可传）===
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

            # 每 50 个 batch 输出进度
            if (i + 1) % 50 == 0 or i == 0:
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (i + 1) / len(train_loader) * 100
                print(f"  Batch {i+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {total_loss.item():.4f}, Avg: {avg_loss:.4f}", flush=True)

        current_lr = optimizer.param_groups[0]['lr']
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
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                s1_logits, s2_logits, hidden = model.module.forward_with_hidden(token_seq_0, token_seq_1, batch_stamp)
                recon_loss, _, _ = model.module.head.compute_loss(
                    s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
                )

                # 方向损失（与训练一致）
                pred_hidden = hidden[:, -1, :]
                direction_logits = close_direction_head(pred_hidden).squeeze(-1)
                direction_pred = torch.sigmoid(direction_logits)
                direction_loss = F.binary_cross_entropy(direction_pred, batch_direction)

                # 组合损失（与训练一致）
                val_loss = recon_loss + config.direction_loss_weight * direction_loss

                val_loss_sum += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        val_loss_history.append(avg_val_loss)

        # Quick IC test（在 LR 调整前执行，使用 val_data 随机采样）
        ic_rng = np.random.RandomState(config.seed + epoch_idx * 9999)
        ic_result = quick_ic_test(model.module, tokenizer, device, val_data,
                                   n_samples=config.ic_test_samples,
                                   lookback=config.lookback_window,
                                   pred_len=config.predict_window,
                                   rng=ic_rng)
        current_ic = 0
        current_da = 0
        if ic_result:
            current_ic = ic_result.get('trajectory_ic', 0)
            current_da = ic_result.get('direction_acc', 0)
            history['ic'].append(current_ic)
            ic_history.append(current_ic)

        # IC 滑动均值（用于决策，减少噪声）
        ic_window = 3
        if len(history['ic']) >= ic_window:
            ic_smoothed = np.mean(history['ic'][-ic_window:])
        else:
            ic_smoothed = current_ic

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        # Epoch 结果
        print(f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}", flush=True)

        # 各特征的 Trajectory IC 汇总
        print(f"\n  {'Feature':<8} {'Traj_IC':>8} {'std':>8} {'pos%':>6}", flush=True)
        for fn in FEATURE_NAMES:
            tic = ic_result.get(f'{fn}_trajectory_ic', 0) if ic_result else 0
            tic_std = ic_result.get(f'{fn}_trajectory_ic_std', 0) if ic_result else 0
            tic_pos = ic_result.get(f'{fn}_trajectory_ic_pos_pct', 0) if ic_result else 0
            print(f"  {fn:<8} {tic:>8.4f} {tic_std:>8.4f} {tic_pos:>6.1%}", flush=True)

        # 各步各特征完整表格
        if ic_result and 'n_samples' in ic_result:
            pred_len = config.predict_window
            print(f"\n  {'Step':<6} {'open_MAE':>8} {'high_MAE':>8} {'low_MAE':>8} {'close_MAE':>8} {'vol_MAE':>10} {'amt_MAE':>10}", flush=True)
            for step_idx in range(pred_len):
                suffix = f'_step{step_idx+1}'
                print(f"  +{step_idx+1:<5} "
                      f"{ic_result.get(f'open_mae{suffix}', 0):>8.2f} "
                      f"{ic_result.get(f'high_mae{suffix}', 0):>8.2f} "
                      f"{ic_result.get(f'low_mae{suffix}', 0):>8.2f} "
                      f"{ic_result.get(f'close_mae{suffix}', 0):>8.2f} "
                      f"{ic_result.get(f'vol_mae{suffix}', 0):>10.0f} "
                      f"{ic_result.get(f'amt_mae{suffix}', 0):>10.0f}", flush=True)

            print(f"\n  {'Step':<6} {'open_DA':>7} {'high_DA':>7} {'low_DA':>7} {'close_DA':>7} {'vol_DA':>7} {'amt_DA':>7}", flush=True)
            for step_idx in range(pred_len):
                suffix = f'_step{step_idx+1}'
                print(f"  +{step_idx+1:<5} "
                      f"{ic_result.get(f'open_da{suffix}', 0):>7.0%} "
                      f"{ic_result.get(f'high_da{suffix}', 0):>7.0%} "
                      f"{ic_result.get(f'low_da{suffix}', 0):>7.0%} "
                      f"{ic_result.get(f'close_da{suffix}', 0):>7.0%} "
                      f"{ic_result.get(f'vol_da{suffix}', 0):>7.0%} "
                      f"{ic_result.get(f'amt_da{suffix}', 0):>7.0%}", flush=True)

        print(f"IC_smoothed: {ic_smoothed:.4f} (best: {best_ic:.4f})", flush=True)
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}", flush=True)
        print(f"[LR] {current_lr:.6f}", flush=True)

        # === 每轮结束都保存最新模型 ===
        latest_path = f"{save_dir}/checkpoints/latest_model"
        model.module.save_pretrained(latest_path)

        # === 动态早停判断 ===
        improved = False

        # 1. Val Loss 改善（只要提升就保存，无阈值）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            improved = True
            save_path = f"{save_dir}/checkpoints/best_model"
            model.module.save_pretrained(save_path)
            print(f"[VAL LOSS SAVED] {best_val_loss:.4f}", flush=True)

        # 2. IC 改善（使用滑动均值，只要提升就保存，无阈值）
        if config.early_stopping_check_ic and ic_smoothed > best_ic:
            best_ic = ic_smoothed
            patience_counter = 0
            improved = True
            ic_save_path = f"{save_dir}/checkpoints/best_ic_model"
            model.module.save_pretrained(ic_save_path)
            print(f"[IC SAVED] {best_ic:.4f}", flush=True)

        if not improved:
            patience_counter += 1

        # === 结合 LR 状态的动态早停策略 ===
        in_warmup = epoch_idx < config.warmup_epochs
        lr_high = current_lr > 0.001
        is_exploring = in_warmup or lr_high
        state_desc = f"WARMUP phase" if in_warmup else f"COSINE phase (LR={current_lr:.6f})"

        # Grace period（前几轮不早停）
        if epoch_idx < config.early_stopping_grace_period:
            print(f"[GRACE] Epoch {epoch_idx+1} < {config.early_stopping_grace_period}", flush=True)
        else:
            # 动态 patience：探索期宽松，精细期严格
            if is_exploring:
                dynamic_patience = config.early_stopping_patience * 2
            else:
                dynamic_patience = config.early_stopping_patience

            print(f"[STATE] {state_desc}, Dynamic patience: {dynamic_patience}", flush=True)

            # 窗口趋势判断：看最近 window 轮是否有改善
            window = config.early_stopping_window
            if len(val_loss_history) >= window:
                window_min = min(val_loss_history[-window:])
                window_best_ic = max(ic_history[-window:]) if len(ic_history) >= window else 0

                # 窗口内有改善 → 继续
                if window_min < best_val_loss or window_best_ic > best_ic:
                    print(f"[WINDOW TREND] Recent improvement detected", flush=True)
                elif patience_counter >= dynamic_patience:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs (dynamic_patience={dynamic_patience})", flush=True)
                    final_path = f"{save_dir}/checkpoints/final_model"
                    model.module.save_pretrained(final_path)
                    print(f"[FINAL SAVED] Val Loss: {avg_val_loss:.4f}, IC: {current_ic:.4f}", flush=True)
                    break

            print(f"[PATIENCE] {patience_counter}/{dynamic_patience}", flush=True)

        print(flush=True)

    # 训练正常结束时保存最终模型
    final_path = f"{save_dir}/checkpoints/final_model"
    model.module.save_pretrained(final_path)
    print(f"[FINAL SAVED] Val Loss: {avg_val_loss:.4f}, IC: {current_ic:.4f}", flush=True)

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'epochs_trained': epoch_idx + 1,
        'history': history,
        'final_val_loss': avg_val_loss,
        'final_ic': current_ic
    }


def main():
    from finetune.predictor.shared.dataset import QlibDataset
    from model.kronos import KronosTokenizer, Kronos

    # 命令行参数
    parser = argparse.ArgumentParser(description='Kronos Predictor Training')
    parser.add_argument('--model', type=str, default='mini',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model size to use')
    parser.add_argument('--dataset', type=str, default='mid',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to train on')
    parser.add_argument('--tokenizer', type=str, default='finetuned',
                        choices=['pretrained', 'finetuned'],
                        help='Tokenizer type: pretrained or finetuned')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning_rate')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples for IC test during training (default: 200)')
    parser.add_argument('--save-folder', type=str, default=None,
                        help='Override save folder name (default: {dataset}_predictor_v1)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (e.g. outputs/models/full_predictor_v4/checkpoints/best_model)')
    parser.add_argument('--norm-mode', type=str, default='full_window',
                        choices=['full_window', 'sliding_ma60'],
                        help='Normalization mode: full_window (pretrained) or sliding_ma60')
    args = parser.parse_args()

    # 从配置获取参数
    model_config = MODEL_CONFIGS[args.model]
    dataset_config = DATASET_CONFIGS[args.dataset]

    # 构建运行配置
    run_config = dict(TRAINING_PARAMS)
    run_config['model_name'] = model_config['name']
    run_config['model_params'] = model_config['params']
    run_config['dataset_name'] = dataset_config['name']
    run_config['dataset_desc'] = dataset_config['desc']

    # 命令行覆盖
    if args.epochs:
        run_config['epochs'] = args.epochs
    if args.batch_size:
        run_config['batch_size'] = args.batch_size
    if args.lr:
        run_config['learning_rate'] = args.lr
    run_config['ic_test_samples'] = args.n_samples
    run_config['norm_mode'] = args.norm_mode  # 归一化模式

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}", flush=True)
    print(f"Kronos Predictor Training", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Model: {model_config['name']} ({model_config['params']})", flush=True)
    print(f"Dataset: {dataset_config['name']} ({dataset_config['desc']})", flush=True)
    print(f"Tokenizer: {args.tokenizer}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"\n待验证模型配置（参考 unified_test.py MODEL_CONFIGS）:", flush=True)
    print(f"训练后模型key建议: {args.model}-{args.dataset}trained", flush=True)
    print(f"{'='*60}", flush=True)

    set_seed(run_config['seed'])

    # 保存目录
    if args.save_folder:
        save_folder = args.save_folder
    else:
        save_folder = f"{args.dataset}_predictor_v1"
    save_dir = os.path.join(project_root, "outputs", "models", save_folder)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # 加载 Tokenizer
    if args.tokenizer == 'finetuned' and os.path.exists(os.path.join(project_root, dataset_config['tokenizer_finetuned'])):
        tokenizer_path = os.path.join(project_root, dataset_config['tokenizer_finetuned'])
        print(f"\nLoading finetuned tokenizer: {tokenizer_path}", flush=True)
    else:
        tokenizer_path = os.path.join(project_root, TOKENIZER_CONFIG['pretrained_path'])
        print(f"\nLoading pretrained tokenizer: {tokenizer_path}", flush=True)
        if args.tokenizer == 'finetuned':
            print(f"Warning: Finetuned tokenizer not found, using pretrained", flush=True)

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    # 加载 Predictor
    if args.resume and os.path.exists(os.path.join(project_root, args.resume)):
        predictor_path = os.path.join(project_root, args.resume)
        print(f"Loading predictor from checkpoint: {predictor_path}", flush=True)
    else:
        predictor_path = os.path.join(project_root, model_config['pretrained_path'])
        print(f"Loading predictor: {predictor_path}", flush=True)
        if args.resume:
            print(f"Warning: Resume path not found, using pretrained", flush=True)
    model = Kronos.from_pretrained(predictor_path)
    model.to(device)

    # Wrap for DDP-like access
    class ModelWrapper:
        def __init__(self, model):
            self.module = model
            self.device = model.device if hasattr(model, 'device') else device

        def train(self):
            self.module.train()

        def eval(self):
            self.module.eval()

        def parameters(self):
            return self.module.parameters()

        def __call__(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.module, name)

    model = ModelWrapper(model)
    print(f"Model size: {get_model_size(model.module):.2f}M", flush=True)

    # 数据集路径
    dataset_path = os.path.join(project_root, dataset_config['path'])

    # 创建配置对象（合并数据集配置和训练参数）
    class ConfigAdapter:
        pass

    config = ConfigAdapter()

    # 数据集相关
    config.dataset_path = dataset_path
    config.lookback_window = run_config['lookback']
    config.predict_window = run_config['predict']
    config.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
    config.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
    config.batch_size = run_config['batch_size']
    config.n_train_iter = run_config['n_train_iter_multiplier'] * run_config['batch_size']
    config.n_val_iter = run_config['n_val_iter_multiplier'] * run_config['batch_size']
    config.clip = run_config['clip']
    config.seed = run_config['seed']
    config.norm_mode = run_config['norm_mode']  # 归一化模式

    # 训练参数
    config.epochs = run_config['epochs']
    config.predictor_learning_rate = run_config['learning_rate']
    config.adam_weight_decay = run_config['weight_decay']
    config.adam_beta1 = run_config['adam_beta1']
    config.adam_beta2 = run_config['adam_beta2']

    # 早停参数
    config.early_stopping_patience = run_config['early_stopping_patience']
    config.early_stopping_min_delta = run_config['early_stopping_min_delta']
    config.early_stopping_grace_period = run_config['early_stopping_grace_period']
    config.early_stopping_check_ic = run_config['early_stopping_check_ic']
    config.early_stopping_window = run_config['early_stopping_window']

    # Cosine Annealing + Warmup 参数
    config.lr_scheduler = run_config['lr_scheduler']
    config.lr_min = run_config['lr_min']
    config.warmup_epochs = run_config['warmup_epochs']
    config.freeze_layers = run_config['freeze_layers']
    config.freeze_embedding = run_config['freeze_embedding']
    config.ic_improve_threshold = 0.05
    config.ic_test_samples = run_config['ic_test_samples']
    config.ic_point = run_config['ic_point']

    # 方向损失参数（从 hidden state 直接预测）
    config.direction_loss_weight = run_config['direction_loss_weight']

    # 加载验证数据（IC 评估用）
    import pickle
    val_path = os.path.join(dataset_path, "val_data.pkl")
    if os.path.exists(val_path):
        print(f"Loading val data for IC evaluation...", flush=True)
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
        print(f"Val data: {len(val_data)} stocks", flush=True)
    else:
        val_data = None
        print("Warning: No val data for IC evaluation", flush=True)

    # Train
    result = train_model(model, tokenizer, device, config, save_dir, val_data)

    # Save summary
    summary = {
        'model': f"{args.model}_predictor",
        'dataset': args.dataset,
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'config': run_config,
        'result': {
            'best_val_loss': result['best_val_loss'],
            'best_ic': result['best_ic'],
            'epochs_trained': result['epochs_trained'],
        }
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print(f"\n{'='*60}", flush=True)
    print(f"Training completed!", flush=True)
    print(f"Best Val Loss: {result['best_val_loss']:.4f}", flush=True)
    print(f"Best IC: {result['best_ic']:.4f}", flush=True)
    print(f"Epochs: {result['epochs_trained']}", flush=True)
    print(f"Saved to: {save_dir}", flush=True)
    print(f"\n更新 unified_test.py MODEL_CONFIGS:", flush=True)
    print(f"'{args.model}-{args.dataset}trained': {{", flush=True)
    print(f"    'name': '{model_config['name']} ({dataset_config['name']} 训练后)',", flush=True)
    print(f"    'tokenizer': '{dataset_config['tokenizer_finetuned']}',", flush=True)
    print(f"    'predictor': 'outputs/models/{save_folder}/checkpoints/best_ic_model',", flush=True)
    print(f"    'params': '{model_config['params']}',", flush=True)
    print(f"}}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()