"""
Kronos Predictor Training Entry

统一训练入口，基于 OHLCV 准确度指标：

核心指标：
- MAPE：各特征的平均绝对百分比误差（主要指标）
- Trajectory IC：轨迹形状相关性（去趋势）
- Amplitude Error Rate：振幅误差率

使用：
    # 单卡
    python finetune/predictor/train.py --norm-mode sliding_ma60 --model mini --epochs 50

    # 多卡
    torchrun --nproc_per_node=4 finetune/predictor/train.py --norm-mode sliding_ma60 --model mini
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer

from finetune.predictor.core.config import DataConfig, TrainConfig, ArtifactConfig
from finetune.predictor.core.paths import (
    get_tokenizer_path,
    get_model_path,
    get_checkpoint_path,
    get_training_info_path,
    get_summary_path,
    get_split_data_path,
    ensure_dir,
)
from finetune.predictor.core.dataset import KronosDataset, KronosWindowedDataset, collate_fn
from finetune.predictor.core.metrics import (
    FEATURE_NAMES,
    detrend_to_baseline,
    safe_trajectory_ic,
    compute_mape,
    get_step_weights,
    amplitude_error_rate,
    aggregate_ic,
)
from finetune.predictor.core.utils import (
    set_seed,
    get_device,
    format_time,
    get_model_size,
    get_rank_info,
    cleanup_ddp,
    safe_save_json,
)


# ============================================================================
# 配置
# ============================================================================

TOKENIZER_ARCH = {
    'mini': 'Kronos-Tokenizer-2k',
    'small': 'Kronos-Tokenizer-base',
    'base': 'Kronos-Tokenizer-base',
}

MODEL_PATHS = {
    'mini': 'pretrained/Kronos-mini',
    'small': 'pretrained/Kronos-small',
    'base': 'pretrained/Kronos-base',
}

MAX_CONTEXT = {
    'mini': 2048,
    'small': 512,
    'base': 512,
}


# ============================================================================
# 训练信息记录
# ============================================================================

def init_training_info(output_dir: str, config: DataConfig, train_config: TrainConfig) -> str:
    """初始化 training_info.json"""
    info = {
        'status': 'running',
        'start_time': datetime.now().isoformat(),
        'config': {
            'norm_mode': config.norm_mode,
            'lookback': config.lookback,
            'predict': config.predict,
            'split_mode': config.split_mode,
            'model_type': train_config.model_type,
            'epochs': train_config.epochs,
            'batch_size': train_config.batch_size,
            'learning_rate': train_config.learning_rate,
        },
        'data': {
            'train_samples': 0,
            'val_samples': 0,
            'data_fingerprint': '待更新',
            'target_leakage_check': '待更新',
        },
        'tokenizer': {
            'path': '',
            'data_fingerprint': '待更新',
        },
        'epochs': [],
        'best': {
            'val_loss': {'value': float('inf'), 'epoch': 0},
            'ic': {'value': -999, 'epoch': 0},
            'mape_close': {'value': float('inf'), 'epoch': 0},
            'amplitude_error_rate': {'value': None, 'epoch': 0},
        },
    }

    info_path = get_training_info_path(output_dir)
    ensure_dir(info_path)
    safe_save_json(info, info_path)

    return info_path


def update_training_info(info_path: str, epoch: int, metrics: dict, best_updates: dict = None):
    """更新 training_info.json"""
    info = {}
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        info = {}

    epoch_record = {
        'epoch': epoch,
        'train_loss': metrics.get('train_loss', 0),
        'val_loss': metrics.get('val_loss', 0),
        'ic': metrics.get('ic', 0),
        'mape_close': metrics.get('mape_close', 0),
        'amplitude_error_rate': metrics.get('amplitude_error_rate'),
        'lr': metrics.get('lr', 0),
        'time': datetime.now().isoformat(),
    }
    info.setdefault('epochs', []).append(epoch_record)

    if best_updates:
        info.setdefault('best', {})
        for metric_name, (value, best_epoch) in best_updates.items():
            info['best'][metric_name] = {'value': value, 'epoch': best_epoch}

    safe_save_json(info, info_path)


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_model(
    model,
    tokenizer,
    val_data,
    indices,
    config: DataConfig,
    device: torch.device,
    world_size: int = 1,
    rank: int = 0,
    n_samples: int = 500,
    seed: int = 42,
    model_type: str = 'mini'
) -> dict:
    """
    模型评估（基于 OHLCV 准确度）

    核心指标：MAPE、Trajectory IC、Amplitude Error Rate
    """
    from model.kronos import auto_regressive_inference

    model.eval()
    tokenizer.eval()

    raw_model = model.module if isinstance(model, DDP) else model
    rng = np.random.RandomState(seed)

    # 抽样窗口
    if n_samples > 0 and n_samples < len(indices):
        sample_idx = rng.choice(len(indices), size=n_samples, replace=False)
        eval_indices = [indices[i] for i in sample_idx]
    else:
        eval_indices = indices

    # 分片
    per_rank = len(eval_indices) // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else len(eval_indices)
    local_indices = eval_indices[start_idx:end_idx]

    # 收集结果
    mape_lists = {fn: [] for fn in FEATURE_NAMES}
    ic_lists = {fn: [] for fn in FEATURE_NAMES}
    amplitude_rates = []

    for (symbol, window_start) in local_indices:
        d = val_data[symbol]
        window_end = window_start + config.lookback + config.predict

        try:
            from finetune.predictor.core.dataset import extract_window
            normalized, original_vals, means, stds, timestamps = extract_window(d, window_start, window_end)

            if normalized is None:
                x_raw = original_vals[:config.lookback]
                x_mean = np.mean(x_raw, axis=0)
                x_std = np.std(x_raw, axis=0) + 1e-5
                x_norm = np.clip((x_raw - x_mean) / x_std, -config.clip, config.clip)
                original = original_vals
            else:
                x_norm = normalized[:config.lookback].astype(np.float32)
                original = original_vals

            baseline = original[config.lookback - 1]

            # 时间戳
            x_stamp = np.stack([
                timestamps[:config.lookback].minute.values,
                timestamps[:config.lookback].hour.values,
                timestamps[:config.lookback].weekday.values,
                timestamps[:config.lookback].day.values,
                timestamps[:config.lookback].month.values,
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                timestamps[config.lookback:].minute.values,
                timestamps[config.lookback:].hour.values,
                timestamps[config.lookback:].weekday.values,
                timestamps[config.lookback:].day.values,
                timestamps[config.lookback:].month.values,
            ], axis=1).astype(np.float32)

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, raw_model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context={'mini': 2048, 'small': 512, 'base': 512}.get(model_type, 2048),
                    pred_len=config.predict,
                    clip=config.clip,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )

                pred_norm = preds[0, config.lookback:config.lookback + config.predict, :]

                if normalized is not None:
                    pred_raw = pred_norm * stds[config.lookback:] + means[config.lookback:]
                else:
                    pred_raw = pred_norm * x_std + x_mean

                actual = original[config.lookback:config.lookback + config.predict]

            # MAPE（每步每个特征）
            mape_matrix = compute_mape(pred_raw, actual)
            for fi, fn in enumerate(FEATURE_NAMES):
                mape_lists[fn].append(mape_matrix)

            # Trajectory IC
            for fi, fn in enumerate(FEATURE_NAMES):
                pred_detrend = detrend_to_baseline(pred_raw[:, fi], baseline[fi])
                actual_detrend = detrend_to_baseline(actual[:, fi], baseline[fi])
                ic, _ = safe_trajectory_ic(pred_detrend, actual_detrend)
                if ic is not None:
                    ic_lists[fn].append(ic)

            # Amplitude Error Rate
            pred_amp = pred_raw[0, 1] - pred_raw[0, 2]
            actual_amp = actual[0, 1] - actual[0, 2]
            amplitude_rates.append(amplitude_error_rate(pred_amp, actual_amp))

            # 定期清理 GPU
            if len(amplitude_rates) % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                try:
                    print(f"[WARN eval] {symbol}@{window_start}: {type(e).__name__}", flush=True)
                except Exception:
                    pass
            continue

    # 聚合
    if world_size > 1:
        ic_result = aggregate_ic(ic_lists, world_size, device, rank == 0)

        # MAPE 聚合（简化：rank 0 处理）
        if rank == 0:
            mape_summary = _aggregate_mape(mape_lists, config.predict)
            ic_result = _finalize_ic_result(ic_lists)
        else:
            mape_summary = {}
            ic_result = {}
    else:
        ic_result = _finalize_ic_result(ic_lists)
        mape_summary = _aggregate_mape(mape_lists, config.predict)

    # 振幅统计
    valid_amp_rates = [r for r in amplitude_rates if r is not None]
    amplitude_result = {
        'mean_rate': float(np.mean(valid_amp_rates)) if valid_amp_rates else 1.0,
        'std_rate': float(np.std(valid_amp_rates)) if valid_amp_rates else 0.0,
        'usable_pct': float(np.mean(np.abs(np.array(valid_amp_rates) - 1.0) < 0.3)) if valid_amp_rates else 0.0,
    }

    return {
        'ic': ic_result,
        'mape': mape_summary,
        'amplitude': amplitude_result,
    }


def _aggregate_mape(mape_lists: dict, predict: int) -> dict:
    """聚合 MAPE（加权平均）"""
    step_weights = get_step_weights(predict)
    result = {}

    for fn in FEATURE_NAMES:
        weighted_sum = 0.0
        total_weight = 0.0
        for mape_matrix in mape_lists[fn]:
            if mape_matrix is None:
                continue
            for step_idx in range(min(len(mape_matrix), predict)):
                fi = FEATURE_NAMES.index(fn)
                val = mape_matrix[step_idx, fi]
                w = step_weights[step_idx]
                weighted_sum += val * w
                total_weight += w
        result[fn] = weighted_sum / total_weight if total_weight > 0 else 0.0

    return result


def _finalize_ic_result(ic_lists: dict) -> dict:
    """聚合 IC 结果"""
    result = {}
    for fn in FEATURE_NAMES:
        ics = ic_lists[fn]
        if ics:
            ics_arr = np.array(ics)
            result[fn] = {
                'mean': float(np.mean(ics_arr)),
                'std': float(np.std(ics_arr)) if len(ics) >= 2 else 0.0,
                'p50': float(np.percentile(ics_arr, 50)),
                'n': len(ics),
            }
        else:
            result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'n': 0}
    return result


def compute_val_loss(
    model,
    tokenizer,
    val_data,
    val_indices,
    config: DataConfig,
    device: torch.device,
    batch_size: int = 16,
    world_size: int = 1,
    rank: int = 0,
    use_ddp: bool = False,
) -> float:
    """计算 validation 损失"""
    model.eval()

    raw_model = model.module if isinstance(model, DDP) else model

    val_dataset = KronosDataset(val_data, val_indices, config, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for batch_idx, (x_norm, x_stamp, y_stamp, meta) in enumerate(val_loader):
            if use_ddp and (batch_idx % world_size != rank):
                continue

            x_norm = x_norm.to(device)
            x_stamp = x_stamp.to(device)

            token_seq_0, token_seq_1 = tokenizer.encode(x_norm, half=True)

            s1_logits, s2_logits = raw_model(token_seq_0, token_seq_1, x_stamp)

            head = raw_model.head
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            ce_loss, _, _ = head.compute_loss(
                s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
            )
            n = x_norm.size(0)
            val_loss_sum += ce_loss.item() * n
            val_count += n

    if use_ddp:
        sum_tensor = torch.tensor([val_loss_sum, val_count], device=device, dtype=torch.float64)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        val_loss_sum = sum_tensor[0].item()
        val_count = sum_tensor[1].item()

    model.train()
    return val_loss_sum / val_count if val_count > 0 else 0.0


# ============================================================================
# 主训练函数
# ============================================================================

def train(
    model,
    tokenizer,
    train_loader,
    val_data,
    val_indices,
    config: DataConfig,
    train_config: TrainConfig,
    save_dir: str,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    use_ddp: bool = False,
    resume_checkpoint: str = None,
    val_samples: int = -1,
):
    """主训练循环"""
    from safetensors.torch import load_file

    is_main = (rank == 0)
    start_time = time.time()

    info_path = init_training_info(save_dir, config, train_config) if is_main else None

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # Cosine Annealing LR
    total_steps = train_config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=train_config.lr_min
    )

    # 初始化 best 跟踪
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    best_ic = -999
    best_ic_epoch = 0
    best_mape_close = float('inf')
    best_mape_close_epoch = 0
    best_amplitude_error_rate = 1.0
    best_amplitude_error_rate_epoch = 0
    patience_counter = 0

    # Resume
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        if is_main:
            print(f"\n[RESUME] Loading from {resume_checkpoint}")

        unwrapped = model.module if use_ddp else model
        state_dict = load_file(os.path.join(resume_checkpoint, 'model.safetensors'))
        unwrapped.load_state_dict(state_dict, strict=False)

        opt_path = os.path.join(resume_checkpoint, 'optimizer.pt')
        sch_path = os.path.join(resume_checkpoint, 'scheduler.pt')
        meta_path = os.path.join(resume_checkpoint, 'resume_meta.json')

        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        if os.path.exists(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location=device))
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                resume_meta = json.load(f)
            start_epoch = resume_meta.get('epoch', 0)
            best_val_loss = resume_meta.get('best_val_loss', float('inf'))
            best_val_loss_epoch = resume_meta.get('best_val_loss_epoch', 0)
            best_ic = resume_meta.get('best_ic', -999)
            best_ic_epoch = resume_meta.get('best_ic_epoch', 0)
            best_mape_close = resume_meta.get('best_mape_close', float('inf'))
            best_mape_close_epoch = resume_meta.get('best_mape_close_epoch', 0)
            best_amplitude_error_rate = resume_meta.get('best_amplitude_error_rate', 1.0)
            best_amplitude_error_rate_epoch = resume_meta.get('best_amplitude_error_rate_epoch', 0)
            patience_counter = resume_meta.get('patience_counter', 0)

            if is_main:
                print(f"[RESUME] Starting from epoch {start_epoch + 1}")
                print(f"[RESUME] Best: val_loss={best_val_loss:.4f}@{best_val_loss_epoch}, IC={best_ic:.4f}@{best_ic_epoch}")

    history = {
        'train_loss': [],
        'val_loss': [],
        'ic': [],
        'mape_close': [],
        'lr': [],
    }

    # 训练循环
    for epoch_idx in range(start_epoch, train_config.epochs):
        epoch_start = time.time()
        model.train()

        epoch_losses = []
        current_lr = optimizer.param_groups[0]['lr']

        if is_main:
            print(f"\n=== Epoch {epoch_idx + 1}/{train_config.epochs} ===")
            print(f"LR: {current_lr:.6f}")

        total_batches = len(train_loader)
        for batch_idx, (x_norm, x_stamp, y_stamp, meta) in enumerate(train_loader):
            x_norm = x_norm.to(device, non_blocking=True)
            x_stamp = x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(x_norm, half=True)

            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            s1_logits, s2_logits = model(token_seq_0, token_seq_1, x_stamp)
            head = model.module.head if use_ddp else model.head
            recon_loss, _, _ = head.compute_loss(
                s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
            )

            optimizer.zero_grad()
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(recon_loss.item())

            if is_main and (batch_idx % 50 == 0 or batch_idx == 0):
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (batch_idx + 1) / total_batches * 100
                elapsed = time.time() - epoch_start
                print(f"  Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - Loss: {recon_loss.item():.4f}, Avg: {avg_loss:.4f} [{elapsed:.0f}s]", flush=True)

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = optimizer.param_groups[0]['lr']

        # 同源采样
        D_val = len(val_indices)
        n_eval = val_samples if val_samples > 0 else D_val
        n_eval = min(n_eval, D_val)
        rng_eval = np.random.RandomState(config.seed + epoch_idx)
        sampled_pos = rng_eval.choice(D_val, size=n_eval, replace=False)
        if use_ddp:
            pos_tensor = torch.from_numpy(sampled_pos.astype(np.int64)).to(device)
            gathered = [torch.zeros_like(pos_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, pos_tensor)
            sampled_pos = gathered[0].cpu().numpy()
        epoch_val_indices = [val_indices[i] for i in sampled_pos]

        # Val loss
        avg_val_loss = compute_val_loss(
            model, tokenizer, val_data, epoch_val_indices, config, device,
            batch_size=train_config.batch_size,
            world_size=world_size, rank=rank, use_ddp=use_ddp,
        )

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        torch.cuda.empty_cache()

        # 评估
        eval_result = evaluate_model(
            model, tokenizer, val_data, epoch_val_indices, config,
            device, world_size, rank,
            n_samples=n_eval,
            seed=config.seed + epoch_idx,
            model_type=train_config.model_type
        )

        if use_ddp:
            dist.barrier()

        if is_main:
            current_ic = eval_result['ic'].get('close', {}).get('mean', 0)
            current_mape_close = eval_result['mape'].get('close', 1.0)
            current_amplitude = eval_result['amplitude'].get('mean_rate', 1.0)

            history['ic'].append(current_ic)
            history['mape_close'].append(current_mape_close)

            epoch_time = time.time() - epoch_start
            print(f"\n  Epoch {epoch_idx + 1}/{train_config.epochs}")
            print(f"    IC: close={current_ic:.4f}, best={best_ic:.4f}@{best_ic_epoch}")
            print(f"    MAPE: close={current_mape_close:.2%}, best={best_mape_close:.2%}@{best_mape_close_epoch}")
            print(f"    Amplitude: {current_amplitude:.2f}")
            print(f"    Val_loss: {avg_val_loss:.4f}, Train: {avg_train_loss:.4f}, Time: {format_time(epoch_time)}")

            # 保存 latest
            latest_path = get_checkpoint_path(save_dir, 'latest_model')
            unwrapped = model.module if use_ddp else model
            unwrapped.save_pretrained(latest_path)

            torch.save(optimizer.state_dict(), os.path.join(latest_path, 'optimizer.pt'))
            torch.save(scheduler.state_dict(), os.path.join(latest_path, 'scheduler.pt'))

            resume_meta = {
                'epoch': epoch_idx,
                'best_val_loss': best_val_loss,
                'best_val_loss_epoch': best_val_loss_epoch,
                'best_ic': best_ic,
                'best_ic_epoch': best_ic_epoch,
                'best_mape_close': best_mape_close,
                'best_mape_close_epoch': best_mape_close_epoch,
                'best_amplitude_error_rate': best_amplitude_error_rate,
                'best_amplitude_error_rate_epoch': best_amplitude_error_rate_epoch,
                'patience_counter': patience_counter,
            }
            safe_save_json(resume_meta, os.path.join(latest_path, 'resume_meta.json'))

            # Best 跟踪
            best_updates = {}
            improved = False

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_loss_epoch = epoch_idx + 1
                improved = True
                best_updates['val_loss'] = (best_val_loss, best_val_loss_epoch)

                best_path = get_checkpoint_path(save_dir, 'best_model')
                unwrapped.save_pretrained(best_path)

            if current_ic > best_ic:
                best_ic = current_ic
                best_ic_epoch = epoch_idx + 1
                patience_counter = 0
                improved = True
                best_updates['ic'] = (best_ic, best_ic_epoch)

                ic_path = get_checkpoint_path(save_dir, 'best_ic_model')
                unwrapped.save_pretrained(ic_path)

            if current_mape_close < best_mape_close:
                best_mape_close = current_mape_close
                best_mape_close_epoch = epoch_idx + 1
                best_updates['mape_close'] = (best_mape_close, best_mape_close_epoch)

            if current_amplitude < best_amplitude_error_rate:
                best_amplitude_error_rate = current_amplitude
                best_amplitude_error_rate_epoch = epoch_idx + 1
                best_updates['amplitude_error_rate'] = (best_amplitude_error_rate, best_amplitude_error_rate_epoch)

            if not improved:
                patience_counter += 1

            # 更新 training_info
            if info_path:
                metrics_dict = {
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'ic': current_ic,
                    'mape_close': current_mape_close,
                    'amplitude_error_rate': eval_result['amplitude'],
                    'lr': current_lr,
                }
                update_training_info(info_path, epoch_idx + 1, metrics_dict, best_updates)

        # Early stopping
        if use_ddp:
            dist.barrier()
        stop_flag = torch.zeros(1, dtype=torch.float32, device=device)
        if is_main:
            if epoch_idx >= train_config.early_stopping_grace_period:
                if patience_counter >= train_config.early_stopping_patience:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                    stop_flag[0] = 1.0
        if use_ddp:
            dist.broadcast(stop_flag, src=0)
        if stop_flag[0] > 0:
            break

    # Final save
    if is_main:
        final_path = get_checkpoint_path(save_dir, 'final_model')
        unwrapped = model.module if use_ddp else model
        unwrapped.save_pretrained(final_path)

        summary = {
            'config': {
                'norm_mode': config.norm_mode,
                'lookback': config.lookback,
                'predict': config.predict,
                'split_mode': config.split_mode,
                'model_type': train_config.model_type,
            },
            'result': {
                'best_val_loss': best_val_loss,
                'best_ic': best_ic,
                'best_mape_close': best_mape_close,
                'epochs_trained': epoch_idx + 1,
            },
            'history': history,
        }
        summary_path = get_summary_path(save_dir)
        safe_save_json(summary, summary_path)

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Training completed in {format_time(total_time)}")
        print(f"Best IC: {best_ic:.4f}")
        print(f"Best MAPE (close): {best_mape_close:.2%}")
        print(f"Saved to: {save_dir}")
        print(f"{'=' * 60}")

    if use_ddp:
        dist.barrier()

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'best_mape_close': best_mape_close,
        'epochs_trained': epoch_idx + 1,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos Predictor Training')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block',
                        choices=['time', 'block'])
    parser.add_argument('--model', type=str, default='mini',
                        choices=['mini', 'small', 'base'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-folder', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--train-sample-ratio', type=float, default=0.5)
    parser.add_argument('--train-samples', type=int, default=-1)
    parser.add_argument('--n-sample-ratio', type=float, default=0.5)
    parser.add_argument('--n-samples', type=int, default=-1)
    args = parser.parse_args()

    # DDP setup
    rank, local_rank, world_size, use_ddp = get_rank_info()
    device = get_device(local_rank)
    is_main = (rank == 0)

    config = DataConfig(
        norm_mode=args.norm_mode,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
        seed=args.seed + rank,
    )

    train_config = TrainConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    set_seed(config.seed)

    train_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'train')
    val_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'val')

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Kronos Predictor Training (OHLCV Accuracy)")
        print(f"{'=' * 60}")
        print(f"norm_mode: {config.norm_mode}")
        print(f"lookback: {config.lookback}, predict: {config.predict}")
        print(f"split_mode: {config.split_mode}")
        print(f"model: {train_config.model_type}")
        print(f"batch_size: {train_config.batch_size}")
        print(f"lr: {train_config.learning_rate}")
        print(f"world_size: {world_size}")
        print(f"{'=' * 60}")

    # Tokenizer
    tokenizer_path = get_tokenizer_path(config.norm_mode, train_config.model_type)
    if not os.path.exists(os.path.join(tokenizer_path, 'model.safetensors')):
        legacy_path = 'outputs/tokenizers/final/2k-MA60' if args.model == 'mini' else 'outputs/tokenizers/final/base-MA60'
        if os.path.exists(legacy_path):
            if is_main:
                print(f"[WARNING] Tokenizer not found at {tokenizer_path}")
                print(f"[WARNING] Using legacy fallback: {legacy_path}")
            tokenizer_path = legacy_path
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)
    if is_main:
        print(f"Tokenizer: {tokenizer_path}")

    # 模型
    model_path = MODEL_PATHS[train_config.model_type]
    model = Kronos.from_pretrained(model_path)
    model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    if is_main:
        print(f"Model: {model_path}, Size: {get_model_size(model):.2f}M")

    # 数据
    import pickle
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)

    # 索引
    train_indices = []
    for symbol, d in train_data.items():
        mode = d.get('mode', 'time') if isinstance(d, dict) else None
        if mode == 'block':
            for b_start, blk in d['blocks'].items():
                for w in blk['windows']:
                    train_indices.append((symbol, int(w)))
        elif 'windows' in d:
            for w in d['windows']:
                train_indices.append((symbol, int(w)))
        elif hasattr(d, 'columns'):
            for i in range(len(d) - config.lookback - config.predict + 1):
                train_indices.append((symbol, i))
        else:
            for i in range(len(d['normalized']) - config.lookback - config.predict + 1):
                train_indices.append((symbol, i))

    val_indices = []
    for symbol, d in val_data.items():
        mode = d.get('mode', 'time') if isinstance(d, dict) else None
        if mode == 'block':
            for b_start, blk in d['blocks'].items():
                for w in blk['windows']:
                    val_indices.append((symbol, int(w)))
        elif 'windows' in d:
            for w in d['windows']:
                val_indices.append((symbol, int(w)))
        elif hasattr(d, 'columns'):
            for i in range(len(d) - config.lookback - config.predict + 1):
                val_indices.append((symbol, i))
        else:
            for i in range(len(d['normalized']) - config.lookback - config.predict + 1):
                val_indices.append((symbol, i))

    if is_main:
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    train_dataset = KronosDataset(train_data, train_indices, config, mode='train')

    D_train = len(train_dataset)
    if args.train_samples > 0:
        train_samples = min(args.train_samples, D_train)
    else:
        train_samples = max(1, int(D_train * args.train_sample_ratio))

    D_val = len(val_indices)
    if args.n_samples > 0:
        val_samples = min(args.n_samples, D_val)
    else:
        val_samples = max(1, int(D_val * args.n_sample_ratio))

    if is_main:
        print(f"Train: {D_train} windows, sampling {train_samples}/epoch")
        print(f"Val: {D_val} windows, sampling {val_samples}/epoch")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False, num_samples=train_samples),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    if args.output_folder:
        save_dir = os.path.join(project_root, 'outputs/models', args.output_folder)
    else:
        save_dir = get_model_path(config.norm_mode, config.lookback, config.predict, config.split_mode, train_config.model_type)

    if is_main:
        ensure_dir(save_dir)
        ensure_dir(get_checkpoint_path(save_dir, 'checkpoints'))

    result = train(
        model, tokenizer, train_loader, val_data, val_indices,
        config, train_config, save_dir, device,
        rank, world_size, use_ddp,
        resume_checkpoint=args.resume,
        val_samples=val_samples,
    )

    cleanup_ddp()


if __name__ == '__main__':
    main()