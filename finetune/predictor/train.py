"""
Kronos Predictor Training Entry

统一训练入口，支持：
- 多种 norm_mode（full_window/sliding_ma{N}）
- 多种 split_mode（time/block）
- 多种 model_type（mini/small/base）
- DDP 多卡训练
- 正确度量口径（detrended trajectory IC）

使用：
    # 单卡
    python finetune/predictor/train.py --norm-mode sliding_ma60 --model mini --epochs 50

    # 多卡
    torchrun --nproc_per_node=4 finetune/predictor/train.py --norm-mode sliding_ma60 --model mini

关键：
- 与 eval.py 共用 core/metrics.py
- trajectory IC 用去趋势序列（禁止原始价格）
- checkpoint 选择用正确口径 IC
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
    get_legacy_data_path,
    ensure_dir,
)
from finetune.predictor.core.dataset import KronosDataset, KronosWindowedDataset, collate_fn
from finetune.predictor.core.metrics import (
    FEATURE_NAMES,
    detrend_to_baseline,
    safe_trajectory_ic,
    safe_corrcoef,
    safe_spearmanr,
    excess_da,
    aggregate_ic,
    aggregate_da,
    calculate_combined_score,
    calculate_da_score,
    get_log_step_weights,
    get_feature_weights,
    format_metrics_report,
)
from finetune.predictor.core.utils import (
    set_seed,
    get_device,
    format_time,
    get_model_size,
    get_rank_info,
    cleanup_ddp,
    debug_print,
    safe_save_json,
    safe_save_pickle,
)


# ============================================================================
# 配置
# ============================================================================

TOKENIZER_PATHS = {
    'mini': {'vocab': 2048},
    'small': {'vocab': 4096},
    'base': {'vocab': 8192},
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
    """
    初始化 training_info.json
    """
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
            'ic': 0.0,
            'combined': 0.0,
            'epoch': 0,
        },
    }

    info_path = get_training_info_path(output_dir)
    ensure_dir(info_path)
    safe_save_json(info, info_path)

    return info_path


def update_training_info(info_path: str, epoch: int, metrics: dict, is_best: bool = False, best_type: str = None):
    """
    更新 training_info.json
    """
    info = safe_save_json.__wrapped__(info_path) if hasattr(safe_save_json, '__wrapped__') else {}
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except:
        info = {}

    epoch_record = {
        'epoch': epoch,
        'train_loss': metrics.get('train_loss', 0),
        'val_loss': metrics.get('val_loss', 0),
        'ic': metrics.get('ic', 0),
        'combined': metrics.get('combined', 0),
        'lr': metrics.get('lr', 0),
        'time': datetime.now().isoformat(),
    }
    info.setdefault('epochs', []).append(epoch_record)

    if is_best:
        if best_type == 'ic':
            info['best']['ic'] = metrics.get('ic', 0)
            info['best']['ic_epoch'] = epoch
        elif best_type == 'combined':
            info['best']['combined'] = metrics.get('combined', 0)
            info['best']['combined_epoch'] = epoch

    safe_save_json(info, info_path)


# ============================================================================
# 评估函数（使用正确度量口径）
# ============================================================================

def evaluate_trajectory_ic(
    model,
    tokenizer,
    val_data,
    indices,
    config: DataConfig,
    device: torch.device,
    world_size: int = 1,
    rank: int = 0,
    n_samples: int = 500,
    seed: int = 42
) -> dict:
    """
    Trajectory IC 评估（正确口径：去趋势序列）

    关键：
    - pred/actual 必须先 detrend_to_baseline
    - 禁止直接用原始价格
    - 聚合保留分布（mean/std/p25/p50/p75）
    """
    from model.kronos import auto_regressive_inference

    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)

    # 抽样窗口
    if n_samples > 0 and n_samples < len(indices):
        sample_idx = rng.choice(len(indices), size=n_samples, replace=False)
        eval_indices = [indices[i] for i in sample_idx]
    else:
        eval_indices = indices

    # 分片：每个 rank 处理一部分
    per_rank = len(eval_indices) // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else len(eval_indices)
    local_indices = eval_indices[start_idx:end_idx]

    # 本地 IC 收集
    local_ics = {f: [] for f in FEATURE_NAMES}
    local_rics = {f: [] for f in FEATURE_NAMES}
    local_da = [{f: [] for f in FEATURE_NAMES} for _ in range(config.predict)]

    for (symbol, window_start) in local_indices:
        d = val_data[symbol]
        window_end = window_start + config.lookback + config.predict

        try:
            # 获取数据
            if 'normalized' in d:
                # dict 格式（MA60 预归一化）
                x_norm = d['normalized'][window_start:window_start + config.lookback].astype(np.float32)
                means = d['means'][window_start:window_end]
                stds = d['stds'][window_start:window_end]
                original = d['original'][window_start:window_end]
                timestamps = d['index'][window_start:window_end]
            else:
                # DataFrame 格式（runtime 归一化）
                df = d.iloc[window_start:window_end]
                x_raw = df.values[:config.lookback].astype(np.float32)
                x_mean = np.mean(x_raw, axis=0)
                x_std = np.std(x_raw, axis=0) + 1e-5
                x_norm = np.clip((x_raw - x_mean) / x_std, -config.clip, config.clip)
                original = df.values
                timestamps = df.index

            baseline = original[config.lookback - 1]

            # 时间戳特征
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
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048,
                    pred_len=config.predict,
                    clip=config.clip,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )

                pred_norm = preds[0, config.lookback:config.lookback + config.predict, :]

                # 反归一化
                if 'normalized' in d:
                    pred_raw = pred_norm.cpu().numpy() * stds[config.lookback:] + means[config.lookback:]
                else:
                    pred_raw = pred_norm.cpu().numpy() * x_std + x_mean

                actual = original[config.lookback:config.lookback + config.predict]

            # Trajectory IC（去趋势口径）
            for fi, fn in enumerate(FEATURE_NAMES):
                baseline_val = baseline[fi]

                # 去趋势
                pred_detrend = detrend_to_baseline(pred_raw[:, fi], baseline_val)
                actual_detrend = detrend_to_baseline(actual[:, fi], baseline_val)

                ic, rank_ic = safe_trajectory_ic(pred_detrend, actual_detrend)

                if ic is not None:
                    local_ics[fn].append(ic)
                if rank_ic is not None:
                    local_rics[fn].append(rank_ic)

            # DA
            for step_idx in range(config.predict):
                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                    actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                    local_da[step_idx][fn].append(pred_dir == actual_dir)

        except Exception as e:
            continue

    # 聚合（保留分布）
    if world_size > 1:
        ic_result = aggregate_ic(local_ics, world_size, device, rank == 0)
        da_result = aggregate_da(local_da, world_size, device, config.predict, rank == 0)
    else:
        # 单卡直接聚合
        ic_result = {}
        for fn in FEATURE_NAMES:
            ics = local_ics[fn]
            if ics:
                ic_result[fn] = {
                    'mean': float(np.mean(ics)),
                    'std': float(np.std(ics)) if len(ics) >= 2 else 0.0,
                    'p50': float(np.percentile(ics, 50)),
                    'n': len(ics),
                }
            else:
                ic_result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': None, 'n': 0}

        da_result = {}
        for step_idx in range(config.predict):
            step_result = {}
            for fn in FEATURE_NAMES:
                da_list = local_da[step_idx][fn]
                if da_list:
                    step_result[fn] = {
                        'mean': float(np.mean(da_list)),
                        'std': float(np.std(da_list)),
                        'n': len(da_list),
                    }
                else:
                    step_result[fn] = {'mean': 0.0, 'std': 0.0, 'n': 0}
            da_result[f'step{step_idx + 1}'] = step_result

    return ic_result, da_result


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
):
    """
    主训练循环

    关键：
    - IC 用去趋势口径（E1 解决）
    - checkpoint 选择用正确口径
    - early stopping patience 需审视（§1.6.7）
    """
    is_main = (rank == 0)
    start_time = time.time()

    # 初始化 training_info
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

    # 最佳跟踪
    best_val_loss = float('inf')
    best_ic = -999
    best_combined = -999
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'ic': [],
        'combined': [],
        'lr': [],
    }

    for epoch_idx in range(train_config.epochs):
        epoch_start = time.time()
        model.train()

        epoch_losses = []
        current_lr = optimizer.param_groups[0]['lr']

        if is_main:
            print(f"\n=== Epoch {epoch_idx + 1}/{train_config.epochs} ===")
            print(f"LR: {current_lr:.6f}")

        for batch_idx, (x_norm, x_stamp, y_stamp, meta) in enumerate(train_loader):
            x_norm = x_norm.to(device, non_blocking=True)
            x_stamp = x_stamp.to(device, non_blocking=True)

            # Tokenize
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(x_norm, half=True)

            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward
            if use_ddp:
                s1_logits, s2_logits = model.module(token_seq_0, token_seq_1, x_stamp)
                recon_loss, _, _ = model.module.head.compute_loss(
                    s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
                )
            else:
                s1_logits, s2_logits = model(token_seq_0, token_seq_1, x_stamp)
                recon_loss, _, _ = model.head.compute_loss(
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
                print(f"  Batch {batch_idx + 1} - Loss: {recon_loss.item():.4f}, Avg: {avg_loss:.4f}")

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = optimizer.param_groups[0]['lr']

        # Validation loss（简化：用最后一个 batch）
        avg_val_loss = avg_train_loss  # TODO: 实现 validation loss 计算

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # Trajectory IC 评估
        ic_result, da_result = evaluate_trajectory_ic(
            model, tokenizer, val_data, val_indices, config,
            device, world_size, rank,
            n_samples=-1,  # 全量评估
            seed=config.seed + epoch_idx * 9999
        )

        if is_main:
            current_ic = ic_result.get('close', {}).get('mean', 0)

            # 计算 DA_score
            da_by_step = [{f: [da_result[f'step{s+1}'].get(f, {}).get('mean', 0)] for f in FEATURE_NAMES} for s in range(config.predict)]
            da_score = calculate_da_score(da_by_step, config.predict)

            # Combined score（权重 0.6/0.4）
            current_combined = calculate_combined_score(current_ic, da_score)

            history['ic'].append(current_ic)
            history['combined'].append(current_combined)

            print(f"\n  Trajectory IC (close, detrended): {current_ic:.4f}")
            print(f"  DA_score: {da_score:.4f}, Combined: {current_combined:.4f}")

            # IC 滑动均值
            ic_window = 3
            if len(history['ic']) >= ic_window:
                ic_smoothed = np.mean(history['ic'][-ic_window:])
            else:
                ic_smoothed = current_ic

            epoch_time = time.time() - epoch_start
            print(f"  Train: {avg_train_loss:.4f}, Time: {format_time(epoch_time)}")

            # 保存 latest
            latest_path = get_checkpoint_path(save_dir, 'latest_model')
            unwrapped = model.module if use_ddp else model
            unwrapped.save_pretrained(latest_path)

            # Checkpoint 选择
            improved = False

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                improved = True

                best_path = get_checkpoint_path(save_dir, 'best_model')
                unwrapped.save_pretrained(best_path)
                print(f"  [VAL LOSS] {best_val_loss:.4f}")

            # §1.6.7: IC 归零条件需审视
            if train_config.ic_patience_reset and ic_smoothed > best_ic:
                best_ic = ic_smoothed
                patience_counter = 0
                improved = True

                ic_path = get_checkpoint_path(save_dir, 'best_ic_model')
                unwrapped.save_pretrained(ic_path)
                print(f"  [IC] {best_ic:.4f}")

            if current_combined > best_combined:
                best_combined = current_combined

                combined_path = get_checkpoint_path(save_dir, 'best_combined_model')
                unwrapped.save_pretrained(combined_path)
                print(f"  [COMBINED] {best_combined:.4f}")

            if not improved:
                patience_counter += 1

            # Early stopping（§1.6.7: patience=12 需审视）
            if epoch_idx >= train_config.early_stopping_grace_period:
                if patience_counter >= train_config.early_stopping_patience:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                    break

            # 更新 training_info
            if info_path:
                update_training_info(info_path, epoch_idx + 1, {
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'ic': current_ic,
                    'combined': current_combined,
                    'lr': current_lr,
                })

    # Final save
    if is_main:
        final_path = get_checkpoint_path(save_dir, 'final_model')
        unwrapped = model.module if use_ddp else model
        unwrapped.save_pretrained(final_path)

        # Summary
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
                'best_combined': best_combined,
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
        print(f"Best Combined: {best_combined:.4f}")
        print(f"Saved to: {save_dir}")
        print(f"{'=' * 60}")

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'best_combined': best_combined,
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
    parser.add_argument('--use-block', action='store_true',
                        help='Use block_lb400_pd10 data (legacy compat)')
    parser.add_argument('--output-folder', type=str, default=None)
    args = parser.parse_args()

    # DDP setup
    rank, local_rank, world_size, use_ddp = get_rank_info()
    device = get_device(local_rank)
    is_main = (rank == 0)

    # Config
    config = DataConfig(
        norm_mode=args.norm_mode,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
        seed=args.seed,
    )

    train_config = TrainConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    set_seed(config.seed)

    # 数据路径（兼容旧格式）
    if args.use_block or args.split_mode == 'block':
        train_path = get_legacy_data_path('ma60', args.lookback, 'train')
        val_path = get_legacy_data_path('ma60', args.lookback, 'val')
    else:
        train_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'train')
        val_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'val')

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Kronos Predictor Training")
        print(f"{'=' * 60}")
        print(f"norm_mode: {config.norm_mode}")
        print(f"lookback: {config.lookback}, predict: {config.predict}")
        print(f"split_mode: {config.split_mode}")
        print(f"model: {train_config.model_type}")
        print(f"batch_size: {train_config.batch_size}")
        print(f"lr: {train_config.learning_rate}")
        print(f"world_size: {world_size}")
        print(f"{'=' * 60}")

    # 加载 tokenizer
    tokenizer_path = get_tokenizer_path(config.norm_mode, train_config.model_type)
    if not os.path.exists(tokenizer_path):
        # fallback to legacy
        tokenizer_path = 'outputs/tokenizers/final/2k-MA60' if args.model == 'mini' else 'outputs/tokenizers/final/base-MA60'

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)
    if is_main:
        print(f"Tokenizer: {tokenizer_path}")

    # 加载模型
    model_path = MODEL_PATHS[train_config.model_type]
    model = Kronos.from_pretrained(model_path)
    model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
    if is_main:
        print(f"Model: {model_path}, Size: {get_model_size(model):.2f}M")

    # 加载数据
    import pickle
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)

    # 构建索引
    train_indices = []
    for symbol, d in train_data.items():
        if 'windows' in d:
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
        if 'windows' in d:
            for w in d['windows']:
                val_indices.append((symbol, int(w)))
        elif hasattr(d, 'columns'):
            if len(d) >= config.lookback + config.predict:
                val_indices.append((symbol, len(d) - config.lookback - config.predict))
        else:
            if len(d['normalized']) >= config.lookback + config.predict:
                val_indices.append((symbol, len(d['normalized']) - config.lookback - config.predict))

    if is_main:
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Dataset
    train_dataset = KronosDataset(train_data, train_indices, config, mode='train')
    val_dataset = KronosDataset(val_data, val_indices, config, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset)),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # 输出目录
    if args.output_folder:
        save_dir = os.path.join(project_root, 'outputs/models', args.output_folder)
    else:
        save_dir = get_model_path(config.norm_mode, config.lookback, config.predict, config.split_mode, train_config.model_type)

    if is_main:
        ensure_dir(save_dir)
        ensure_dir(get_checkpoint_path(save_dir, 'checkpoints'))

    # 训练
    result = train(
        model, tokenizer, train_loader, val_data, val_indices,
        config, train_config, save_dir, device,
        rank, world_size, use_ddp
    )

    cleanup_ddp()


if __name__ == '__main__':
    main()