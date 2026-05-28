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
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

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

# Dataset configs - 使用 MA20 tokenizer
DATASET_CONFIGS = {
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
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 0.02,  # 起始 LR
    'weight_decay': 0.1,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'seed': 100,
    'n_train_iter_multiplier': 2000,  # n_train_iter = multiplier * batch_size
    'n_val_iter_multiplier': 400,     # n_val_iter = multiplier * batch_size
    'lookback': 400,  # 2k tokenizer 需要更长上下文
    'predict': 10,
    'clip': 5.0,
    # 归一化模式
    'norm_mode': 'full_window',  # 'full_window' 或 'sliding_ma60'
    # 早停参数
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.0001,
    'early_stopping_grace_period': 5,
    'early_stopping_check_ic': True,
    'early_stopping_window': 5,
    # VL-IC Adaptive 学习率（带上下限额和缓冲）
    'lr_scheduler': 'vl_adaptive',
    'lr_max': 0.03,           # 上限：不超过起始 LR 的 1.5 倍
    'lr_min': 1e-4,           # 下限：不低于 0.0001
    'lr_decay_factor': 0.7,   # 衰减保留 70%
    'lr_boost_factor': 1.2,   # 提升保留 120%（温和提升）
    'lr_boost_patience': 3,   # 连续 N epoch 改善才 boost（缓冲）
    'lr_decay_patience': 3,   # 连续 N epoch 无改善才 decay（缓冲）
    'ic_improve_threshold': 0.05,
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


def quick_ic_test(model, tokenizer, device, val_data, n_samples=500, lookback=90, pred_len=10, ic_point=3, rng=None):
    """
    快速 IC 测试（在训练过程中，使用 val_data，随机采样）

    使用 auto_regressive_inference 进行真正的未来预测

    计算方式与 unified_test.py v9 一致：
    - 基准值：预测窗口前的最后一个 close（实际买入价）
    - 预测收益率：相对于基准值的变化
    - 实际收益率：相对于基准值的变化
    - ic_point: 计算哪个点的IC（默认3，即Point+3）
    - rng: numpy RandomState，用于可复现的随机采样

    Returns:
        dict: { 'ic': float, 'rank_ic': float, 'direction_acc': float }
    """
    from model.kronos import auto_regressive_inference
    model.eval()

    predictions = []
    actuals = []

    all_symbols = list(val_data.keys())
    if rng is not None:
        symbols = rng.choice(all_symbols, size=min(n_samples, len(all_symbols)), replace=False).tolist()
    else:
        symbols = all_symbols[:n_samples]

    for symbol in symbols:
        df = val_data[symbol]
        if len(df) < lookback + pred_len:
            continue

        try:
            # Prepare timestamps
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
            feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
            values = df[feature_cols].values.astype(np.float32)
            x = values[-(lookback + pred_len):-pred_len]

            # === 全窗口归一化（pretrained原始方式）===
            # 使用整个lookback窗口的mean/std
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0) + 1e-5

            x_norm = (x - x_mean) / x_std
            x_norm = np.clip(x_norm, -5.0, 5.0)

            # 基准值：lookback 窗口最后一个 close（原始价格）
            baseline_close = values[-pred_len - 1, 3]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(time_df_x.values.astype(np.float32)).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(time_df_y.values.astype(np.float32)).unsqueeze(0).to(device)

                # 使用 auto_regressive_inference 进行预测
                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=512, pred_len=pred_len,
                    clip=5, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                # 只取预测窗口（最后 pred_len 个）
                pred_close_norm = preds[0, -pred_len:, 3]

                # Denormalize
                close_mean = x_mean[3]
                close_std = x_std[3]
                pred_close_raw = pred_close_norm * close_std + close_mean

                # 预测收益率：指定点 ic_point 的 close 相对于基准的变化
                # ic_point=3 表示预测窗口第3个点（Point+3）
                pred_return = (pred_close_raw[ic_point - 1] - baseline_close) / baseline_close

            # 实际收益率（同样取 ic_point 点）
            actual_close_window = df['close'].values[-pred_len:]
            actual_return = (actual_close_window[ic_point - 1] - baseline_close) / baseline_close

            predictions.append(pred_return)
            actuals.append(actual_return)

        except Exception as e:
            continue

    if len(predictions) > 5:
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        ic = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        rank_ic, _ = spearmanr(predictions, actuals) if len(predictions) > 1 else (0, 0)
        direction_acc = np.mean((predictions > 0) == (actuals > 0))

        return {
            'ic': ic,
            'rank_ic': rank_ic,
            'direction_acc': direction_acc,
            'n_samples': len(predictions)
        }

    return None


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


def train_model(model, tokenizer, device, config, save_dir, val_data=None):
    """训练模型，支持动态早停和周期性学习率"""
    start_time = time.time()
    print(f"BATCHSIZE: {config.batch_size}", flush=True)
    print(f"LR: {config.predictor_learning_rate}", flush=True)
    print(f"LR Scheduler: {config.lr_scheduler}", flush=True)
    print(f"IC Test Samples: {config.ic_test_samples}", flush=True)

    # Import dataset
    from finetune.dataset import QlibDataset
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

    # 合并参数到优化器
    all_params = list(model.parameters()) + list(close_direction_head.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=config.predictor_learning_rate,
        weight_decay=config.adam_weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    # === 学习率调度器 ===
    if config.lr_scheduler == 'vl_adaptive':
        print(f"VL-Adaptive: LR will adjust based on Val Loss improvement", flush=True)
        print(f"  - Max LR: {config.lr_max}, Min LR: {config.lr_min}", flush=True)
        print(f"  - IC improve threshold: {config.ic_improve_threshold}", flush=True)
        print(f"  - Decay factor: {config.lr_decay_factor}, Boost factor: {config.lr_boost_factor}", flush=True)
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.predictor_learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=config.epochs,
            pct_start=0.03,
            div_factor=10
        )
        print(f"OneCycleLR", flush=True)

    best_val_loss = float('inf')
    best_ic = -999
    patience_counter = 0

    # 动态早停：记录历史用于趋势判断
    val_loss_history = []
    ic_history = []

    # 滑动窗口 LR 调整：用近期趋势而非单点判断
    ic_window = []  # 最近 N 个 IC 值
    ic_window_size = 5  # 窗口大小
    ic_consecutive_improve = 0  # 连续改善计数
    ic_consecutive_decay = 0    # 连续恶化计数

    # === LR 智能调整策略 ===
    good_lr_pool = []           # [(lr, vl, tl), ...] 确认好的LR
    lr_before_decay = None      # decay前的LR（用于回退）
    explore_lr_candidates = []  # 探索候选LR列表

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

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward（获取 logits 和 hidden state）
            s1_logits, s2_logits, hidden = model.module.forward_with_hidden(token_in[0], token_in[1], batch_stamp[:, :-1, :])
            recon_loss, s1_loss, s2_loss = model.module.head.compute_loss(
                s1_logits, s2_logits, token_out[0], token_out[1]
            )

            # === 方向损失（直接从 hidden state 预测，梯度可传）===
            # 取预测窗口最后一个位置的 hidden state
            pred_hidden = hidden[:, -1, :]  # [batch, d_model] - 最后一个时间步
            direction_logits = close_direction_head(pred_hidden).squeeze(-1)  # [batch]
            direction_pred = torch.sigmoid(direction_logits)  # 方向概率

            # 方向损失（BCE）
            direction_loss = F.binary_cross_entropy(direction_pred, batch_direction)

            # 组合损失
            total_loss = recon_loss + config.direction_loss_weight * direction_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            # Step scheduler (for OneCycle and CosineAnnealingWarmRestarts - per batch)
            if config.lr_scheduler in ['cosine_warmup', 'onecycle']:
                scheduler.step()

            epoch_losses.append(total_loss.item())

            # 每 50 个 batch 输出进度
            if (i + 1) % 50 == 0 or i == 0:
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (i + 1) / len(train_loader) * 100
                print(f"  Batch {i+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {total_loss.item():.4f}, DirLoss: {direction_loss.item():.4f}, Avg: {avg_loss:.4f}", flush=True)

        # Record LR
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

                s1_logits, s2_logits, hidden = model.module.forward_with_hidden(token_in[0], token_in[1], batch_stamp[:, :-1, :])
                recon_loss, _, _ = model.module.head.compute_loss(
                    s1_logits, s2_logits, token_out[0], token_out[1]
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
                                   n_samples=config.ic_test_samples, ic_point=config.ic_point,
                                   rng=ic_rng)
        current_ic = 0
        if ic_result:
            current_ic = ic_result['ic']
            history['ic'].append(current_ic)
            ic_history.append(current_ic)

        # IC 滑动均值（用于决策，减少噪声）
        ic_window = 3
        if len(history['ic']) >= ic_window:
            ic_smoothed = np.mean(history['ic'][-ic_window:])
        else:
            ic_smoothed = current_ic

        # 计算 IC 改善幅度
        if best_ic > -999:
            ic_improve_ratio = (ic_smoothed - best_ic) / max(abs(best_ic), 0.01)
        else:
            ic_improve_ratio = 0

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        # Epoch 结果
        print(f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}", flush=True)
        print(f"IC: {current_ic:.4f}, IC_smoothed: {ic_smoothed:.4f} (best: {best_ic:.4f}, improve: {ic_improve_ratio:.2%})", flush=True)
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}", flush=True)

        # Plateau scheduler step (per epoch)
        if config.lr_scheduler == 'plateau':
            scheduler.step(avg_val_loss)
        # Warmup+Cosine scheduler step (per epoch)
        elif config.lr_scheduler == 'warmup_cosine':
            scheduler.step()
        # VL-Adaptive: 滑动窗口趋势判断（避免单点波动）
        elif config.lr_scheduler == 'vl_adaptive':
            # === 智能 LR 调整策略（综合 TL/VL 稳定性）===

            # 0. 更新 IC window（用于监控显示）
            ic_window.append(current_ic)
            if len(ic_window) > ic_window_size:
                ic_window.pop(0)
            avg_ic_window = sum(ic_window) / len(ic_window) if ic_window else 0

            # IC 连续计数（用于监控显示）
            if len(ic_history) >= 2 and current_ic > ic_history[-2]:
                ic_consecutive_improve += 1
                ic_consecutive_decay = 0
            elif len(ic_history) >= 2 and current_ic < ic_history[-2]:
                ic_consecutive_decay += 1
                ic_consecutive_improve = 0
            else:
                ic_consecutive_improve = 0
                ic_consecutive_decay = 0

            # 1. 更新历史记录
            tl_history = history['train_loss']
            vl_history = history['val_loss']

            # 2. 计算 TL/VL 稳定性（最近3轮波动）
            def calc_stability(loss_history, window=3):
                if len(loss_history) < window:
                    return True, None  # 数据不足，默认稳定
                recent = loss_history[-window:]
                std = np.std(recent)
                mean = np.mean(recent)
                is_stable = std < mean * 0.05  # 波动小于5%认为稳定
                is_declining = recent[-1] < recent[0]  # 下降趋势
                return is_stable, is_declining

            tl_stable, tl_declining = calc_stability(tl_history)
            vl_stable, vl_declining = calc_stability(vl_history)

            # 3. TL/VL 综合状态判断
            tl_worsening = len(tl_history) >= 2 and tl_history[-1] > tl_history[-2] * 1.02  # TL上升超过2%
            tl_volatility = len(tl_history) >= 3 and np.std(tl_history[-3:]) > np.mean(tl_history[-3:]) * 0.08  # TL波动超过8%

            vl_worsening = avg_val_loss > best_val_loss  # VL恶化（直接判断，无阈值）
            vl_improving = avg_val_loss < best_val_loss  # VL改善

            # 4. 决策逻辑
            new_lr = current_lr
            lr_action = ""

            # === 下降通道 ===
            # VL恶化优先判断（最高优先级）
            if vl_worsening:
                # VL恶化 → 需要调整
                if good_lr_pool:
                    best_good_lr = min(good_lr_pool, key=lambda x: x[1])[0]
                    new_lr = best_good_lr
                    lr_action = f"ROLLBACK (VL worsening, back to good LR={best_good_lr:.6f})"
                else:
                    new_lr = max(current_lr * config.lr_decay_factor, config.lr_min)
                    lr_action = f"DECAY (VL worsening: {avg_val_loss:.4f} > {best_val_loss:.4f})"
                lr_before_decay = current_lr

            elif tl_worsening or tl_volatility:
                # TL不稳定 → 需要调整（太激进）
                if good_lr_pool:
                    # 回退到上一个好LR
                    best_good_lr = min(good_lr_pool, key=lambda x: x[1])[0]  # 取VL最低的好LR
                    new_lr = best_good_lr
                    lr_action = f"ROLLBACK (TL unstable, back to good LR={best_good_lr:.6f})"
                else:
                    # 没有好LR记录，用decay
                    new_lr = max(current_lr * config.lr_decay_factor, config.lr_min)
                    lr_action = f"DECAY (TL unstable, no good LR history)"
                lr_before_decay = current_lr  # 记录decay前的LR

            # === 保持稳定 ===
            elif tl_stable and tl_declining and vl_stable:
                # TL/VL都稳定下降 → 最佳状态，保持
                lr_action = f"KEEP (TL/VL stable and declining)"

            elif tl_stable and tl_declining and not vl_stable:
                # TL稳定下降，VL有波动 → 内部稳定，可容忍VL波动
                lr_action = f"KEEP (TL stable, VL fluctuation tolerable)"

            # === 回升通道 ===
            elif lr_before_decay and tl_declining and vl_improving:
                # decay后TL/VL都在改善 → 可能decay过头，尝试回升
                if current_lr < lr_before_decay:
                    # 回升到decay前的LR（探索）
                    new_lr = lr_before_decay
                    lr_action = f"RECOVER (decay was too aggressive, back to {lr_before_decay:.6f})"
                    lr_before_decay = None  # 清除标记

            elif tl_stable and vl_declining and current_lr < config.lr_max:
                # TL稳定、VL下降，且LR未到上限 → 可小幅度探索
                if len(explore_lr_candidates) == 0:
                    # 生成探索候选（当前LR × 1.1, 1.15, 1.2）
                    explore_lr_candidates = [
                        current_lr * 1.1,
                        current_lr * 1.15,
                        current_lr * 1.2,
                    ]
                if explore_lr_candidates:
                    candidate = explore_lr_candidates.pop(0)
                    if candidate <= config.lr_max:
                        new_lr = candidate
                        lr_action = f"EXPLORE (try higher LR={candidate:.6f})"

            # === 默认保持 ===
            if not lr_action:
                lr_action = f"KEEP (TL={avg_train_loss:.4f}, VL={avg_val_loss:.4f})"

            # 5. 记录好LR（当TL/VL都稳定下降且VL创新低时）
            if tl_stable and tl_declining and vl_stable and vl_declining:
                good_lr_pool.append((current_lr, avg_val_loss, avg_train_loss))
                # 去重并按VL排序，保留最近5个
                good_lr_pool = sorted(good_lr_pool, key=lambda x: x[1])[:5]
                lr_action += " [GOOD LR RECORDED]"

            # 更新 optimizer LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            current_lr = new_lr

            print(f"[VL-ADAPTIVE] {lr_action}", flush=True)
            print(f"  IC window avg: {avg_ic_window:.4f}, consecutive: ↑{ic_consecutive_improve} ↓{ic_consecutive_decay}", flush=True)
            print(f"Next epoch LR: {new_lr:.6f}", flush=True)

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
        # 判断是否处于"爬山过程"（LR 较高，正在探索）
        if config.lr_scheduler == 'vl_adaptive':
            # VL-Adaptive: LR 较高时为探索期
            lr_high = current_lr > 0.001
            is_exploring = lr_high
            state_desc = "EXPLORING (LR high)" if is_exploring else "CONVERGING (LR low)"
        elif config.lr_scheduler == 'warmup_cosine':
            # Warmup+Cosine: warmup 阶段和 LR 较高时为探索期
            in_warmup = epoch_idx < config.lr_warmup_epochs
            lr_high = current_lr > 0.001
            is_exploring = in_warmup or lr_high
            state_desc = f"WARMUP phase" if in_warmup else f"COSINE phase (LR={current_lr:.6f})"
        else:
            # CosineAnnealingWarmRestarts: 周期重启点附近为探索期
            lr_high = current_lr > 0.001
            restart_epochs = [5, 15, 35, 75]  # T_0=5, T_mult=2 的重启点
            near_restart = any(abs(epoch_idx + 1 - e) <= 2 for e in restart_epochs)
            is_exploring = lr_high or near_restart
            state_desc = "EXPLORING (LR high/near restart)" if is_exploring else "CONVERGING (LR low)"

        # Grace period（前几轮不早停）
        if epoch_idx < config.early_stopping_grace_period:
            print(f"[GRACE] Epoch {epoch_idx+1} < {config.early_stopping_grace_period}", flush=True)
        else:
            # 动态 patience：探索期宽松，精细期严格
            if is_exploring:
                dynamic_patience = config.early_stopping_patience * 2  # 爬山过程，允许更多波动
            else:
                dynamic_patience = config.early_stopping_patience  # 精细收敛期，严格判断

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
                    # 早停时也保存最终模型
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
    from finetune.dataset import QlibDataset
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

    # VL-IC Adaptive 学习率参数
    config.lr_scheduler = run_config['lr_scheduler']
    config.lr_max = run_config['lr_max']
    config.lr_min = run_config['lr_min']
    config.lr_decay_factor = run_config['lr_decay_factor']
    config.lr_boost_factor = run_config['lr_boost_factor']
    config.lr_boost_patience = run_config['lr_boost_patience']
    config.lr_decay_patience = run_config['lr_decay_patience']
    config.ic_improve_threshold = run_config['ic_improve_threshold']
    config.ic_test_samples = run_config['ic_test_samples']  # IC test samples during training
    config.ic_point = run_config['ic_point']  # IC calculation target point (Point+N)

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