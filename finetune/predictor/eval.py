"""
Kronos Predictor Evaluation Entry

评估指标体系重构：基于 OHLCV 预测准确度

核心指标：
- MAPE：各特征的平均绝对百分比误差（主要指标）
- Trajectory IC：轨迹形状相关性（去趋势）
- Amplitude Error Rate：振幅误差率

使用：
    # 单模型
    python finetune/predictor/eval.py --norm-mode sliding_ma60 --model mini

    # 多卡 DDP
    torchrun --nproc_per_node=4 finetune/predictor/eval.py --model mini

    # 多模型对比
    python finetune/predictor/eval.py --models mini,small,base
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference
from safetensors.torch import load_file

from finetune.predictor.core.config import DataConfig, parse_norm_mode
from finetune.predictor.core.paths import (
    get_tokenizer_path,
    get_model_path,
    get_checkpoint_path,
    get_split_data_path,
)
from finetune.predictor.core.metrics import (
    FEATURE_NAMES,
    detrend_to_baseline,
    safe_trajectory_ic,
    safe_corrcoef,
    compute_mape,
    compute_mae,
    get_step_weights,
    amplitude_error_rate,
    compute_amplitude_stats,
    aggregate_ic,
    format_metrics_report,
)
from finetune.predictor.core.utils import get_device, format_time


# ============================================================================
# DDP 工具
# ============================================================================

def get_rank_info():
    """获取 DDP rank 信息"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        return rank, local_rank, world_size, True
    return 0, 0, 1, False

def setup_ddp(rank, local_rank, world_size):
    """初始化 DDP"""
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    """清理 DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_tokenizer(
    norm_mode: str,
    model_type: str,
    device: torch.device,
    checkpoint: str = 'best_combined_model',
    lookback: int = 400,
    predict: int = 10,
    split_mode: str = 'block',
    custom_tokenizer_path: str = None,
    custom_checkpoint_path: str = None,
):
    """加载模型和 tokenizer"""
    if custom_tokenizer_path:
        tokenizer_path = custom_tokenizer_path
    else:
        tokenizer_path = get_tokenizer_path(norm_mode, model_type)
        if not os.path.exists(tokenizer_path):
            tokenizer_path = 'outputs/tokenizers/final/2k-MA60' if model_type == 'mini' else 'outputs/tokenizers/final/base-MA60'

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    pretrained_paths = {
        'mini': 'pretrained/Kronos-mini',
        'small': 'pretrained/Kronos-small',
        'base': 'pretrained/Kronos-base',
    }

    model = Kronos.from_pretrained(pretrained_paths[model_type])
    model.eval().to(device)

    if custom_checkpoint_path:
        checkpoint_path = custom_checkpoint_path
    else:
        model_dir = get_model_path(norm_mode, lookback, predict, split_mode, model_type)
        checkpoint_path = get_checkpoint_path(model_dir, checkpoint)

    checkpoint_file = os.path.join(checkpoint_path, 'model.safetensors')
    if os.path.exists(checkpoint_file):
        state_dict = load_file(checkpoint_file)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint 与模型不匹配: missing={missing}, unexpected={unexpected}"
            )
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}")
        print(f"[WARNING] Using pretrained model instead")

    return model, tokenizer


# ============================================================================
# 评估函数
# ============================================================================

def evaluate(
    model,
    tokenizer,
    test_data,
    config: DataConfig,
    device: torch.device,
    n_samples: int = -1,
    n_sample_ratio: float = 0.5,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    model_type: str = 'mini',
):
    """
    评估模型（基于 OHLCV 准确度）

    核心指标：MAPE（各特征的百分比误差）
    """
    model.eval()
    tokenizer.eval()

    use_ddp = world_size > 1
    rng = np.random.RandomState(seed)

    # 构建窗口索引
    indices = []
    for symbol, d in test_data.items():
        mode = d.get('mode', 'time') if isinstance(d, dict) else None
        if mode == 'block':
            for b_start, blk in d['blocks'].items():
                for w in blk['windows']:
                    indices.append((symbol, int(w)))
        elif 'windows' in d:
            for w in d['windows']:
                indices.append((symbol, int(w)))
        elif hasattr(d, 'columns'):
            seq_len = len(d)
            if seq_len >= config.lookback + config.predict:
                for i in range(seq_len - config.lookback - config.predict + 1):
                    indices.append((symbol, i))
        else:
            seq_len = len(d['normalized'])
            if seq_len >= config.lookback + config.predict:
                for i in range(seq_len - config.lookback - config.predict + 1):
                    indices.append((symbol, i))

    # 抽样
    D = len(indices)
    if n_samples > 0:
        n_eval = min(n_samples, D)
    else:
        n_eval = max(1, int(D * n_sample_ratio))
    if n_eval < D:
        sample_idx = rng.choice(len(indices), size=n_eval, replace=False)
        indices = [indices[i] for i in sample_idx]

    # 收集结果
    mape_lists = {fn: [] for fn in FEATURE_NAMES}
    ic_lists = {fn: [] for fn in FEATURE_NAMES}
    amplitude_rates = []

    # DDP 分片
    for idx, (symbol, start) in enumerate(tqdm(indices, desc=f"Rank {rank} Evaluating", disable=rank != 0)):
        if idx % world_size != rank:
            continue

        d = test_data[symbol]
        end = start + config.lookback + config.predict

        try:
            from finetune.predictor.core.dataset import extract_window
            normalized, original_vals, means, stds, timestamps = extract_window(d, start, end)

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
            baseline_close = original[config.lookback - 1, 3]

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
                    tokenizer, model,
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

            # 计算 MAPE（每步每个特征）
            mape_matrix = compute_mape(pred_raw, actual)
            for fi, fn in enumerate(FEATURE_NAMES):
                mape_lists[fn].append(mape_matrix)

            # Trajectory IC（去趋势）
            for fi, fn in enumerate(FEATURE_NAMES):
                pred_detrend = detrend_to_baseline(pred_raw[:, fi], baseline[fi])
                actual_detrend = detrend_to_baseline(actual[:, fi], baseline[fi])
                ic, _ = safe_trajectory_ic(pred_detrend, actual_detrend)
                if ic is not None:
                    ic_lists[fn].append(ic)

            # 振幅误差率
            pred_amp = pred_raw[0, 1] - pred_raw[0, 2]
            actual_amp = actual[0, 1] - actual[0, 2]
            amplitude_rates.append(amplitude_error_rate(pred_amp, actual_amp))

        except Exception:
            continue

    # 聚合结果
    if use_ddp:
        ic_result = aggregate_ic(ic_lists, world_size, device, rank == 0)

        # 聚合 MAPE（简化：只用 rank 0 的结果）
        if rank == 0:
            # 计算每步 MAPE
            mape_by_step = {}
            for step_idx in range(config.predict):
                step_result = {}
                for fi, fn in enumerate(FEATURE_NAMES):
                    values = []
                    for mape_matrix in mape_lists[fn]:
                        if mape_matrix is not None and step_idx < len(mape_matrix):
                            values.append(mape_matrix[step_idx, fi])
                    if values:
                        arr = np.array(values)
                        step_result[fn] = {
                            'mean': float(np.mean(arr)),
                            'std': float(np.std(arr)),
                            'p50': float(np.percentile(arr, 50)),
                            'n': len(arr),
                        }
                    else:
                        step_result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'n': 0}
                mape_by_step[f'step{step_idx + 1}'] = step_result

            # 计算加权摘要
            mape_summary = {}
            step_weights = get_step_weights(config.predict)
            for fi, fn in enumerate(FEATURE_NAMES):
                weighted_sum = 0.0
                total_weight = 0.0
                for mape_matrix in mape_lists[fn]:
                    if mape_matrix is None:
                        continue
                    for step_idx in range(min(len(mape_matrix), config.predict)):
                        val = mape_matrix[step_idx, fi]
                        w = step_weights[step_idx]
                        weighted_sum += val * w
                        total_weight += w
                mape_summary[fn] = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            mape_by_step = {}
            mape_summary = {}
    else:
        # 单进程聚合
        ic_result = {}
        for fn in FEATURE_NAMES:
            ics = ic_lists[fn]
            if ics:
                ics_arr = np.array(ics)
                n = len(ics)
                ic_result[fn] = {
                    'mean': float(np.mean(ics_arr)),
                    'std': float(np.std(ics_arr)) if n >= 2 else 0.0,
                    'p25': float(np.percentile(ics_arr, 25)) if n >= 4 else None,
                    'p50': float(np.percentile(ics_arr, 50)),
                    'p75': float(np.percentile(ics_arr, 75)) if n >= 4 else None,
                    'n': n,
                }
            else:
                ic_result[fn] = {'mean': 0.0, 'std': 0.0, 'p25': None, 'p50': None, 'p75': None, 'n': 0}

        # 计算 MAPE
        mape_by_step = {}
        for step_idx in range(config.predict):
            step_result = {}
            for fi, fn in enumerate(FEATURE_NAMES):
                values = []
                for mape_matrix in mape_lists[fn]:
                    if mape_matrix is not None and step_idx < len(mape_matrix):
                        values.append(mape_matrix[step_idx, fi])
                if values:
                    arr = np.array(values)
                    step_result[fn] = {
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'p50': float(np.percentile(arr, 50)),
                        'n': len(arr),
                    }
                else:
                    step_result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'n': 0}
            mape_by_step[f'step{step_idx + 1}'] = step_result

        # 加权摘要
        mape_summary = {}
        step_weights = get_step_weights(config.predict)
        for fi, fn in enumerate(FEATURE_NAMES):
            weighted_sum = 0.0
            total_weight = 0.0
            for mape_matrix in mape_lists[fn]:
                if mape_matrix is None:
                    continue
                for step_idx in range(min(len(mape_matrix), config.predict)):
                    val = mape_matrix[step_idx, fi]
                    w = step_weights[step_idx]
                    weighted_sum += val * w
                    total_weight += w
            mape_summary[fn] = weighted_sum / total_weight if total_weight > 0 else 0.0

    # 振幅统计
    valid_amp_rates = [r for r in amplitude_rates if r is not None]
    amplitude_result = {
        'mean_rate': float(np.mean(valid_amp_rates)) if valid_amp_rates else 1.0,
        'std_rate': float(np.std(valid_amp_rates)) if valid_amp_rates else 0.0,
        'perfect_pct': float(np.mean(np.abs(np.array(valid_amp_rates) - 1.0) < 0.1)) if valid_amp_rates else 0.0,
        'usable_pct': float(np.mean(np.abs(np.array(valid_amp_rates) - 1.0) < 0.3)) if valid_amp_rates else 0.0,
    }

    return mape_by_step, mape_summary, ic_result, amplitude_result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos Predictor Evaluation')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60')
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block')
    parser.add_argument('--model', type=str, default='mini')
    parser.add_argument('--models', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='best_combined_model',
                        choices=['best_model', 'best_ic_model', 'best_combined_model', 'latest_model'])
    parser.add_argument('--tokenizer-path', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--test-path', type=str, default=None)
    parser.add_argument('--n-samples', type=int, default=-1)
    parser.add_argument('--n-sample-ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # DDP setup
    rank, local_rank, world_size, use_ddp = get_rank_info()
    if use_ddp:
        setup_ddp(rank, local_rank, world_size)
    device = get_device(local_rank)
    is_main = (rank == 0)

    config = DataConfig(
        norm_mode=args.norm_mode,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
    )

    if args.test_path:
        test_path = args.test_path
    else:
        test_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'test')

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Kronos Predictor Evaluation (OHLCV Accuracy)")
        print(f"{'=' * 60}")
        print(f"norm_mode: {config.norm_mode}")
        print(f"model: {args.model}")
        print(f"checkpoint: {args.checkpoint}")
        print(f"test_path: {test_path}")
        if args.n_samples > 0:
            print(f"n_samples: {args.n_samples} (absolute)")
        else:
            print(f"n_sample_ratio: {args.n_sample_ratio}")
        print(f"{'=' * 60}")

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    if is_main:
        print(f"Test data: {len(test_data)} stocks")

    if args.models:
        model_types = args.models.split(',')
    else:
        model_types = [args.model]

    for model_type in model_types:
        if is_main:
            print(f"\n--- Evaluating {model_type} ---")

        model, tokenizer = load_model_and_tokenizer(
            config.norm_mode, model_type, device, args.checkpoint,
            lookback=config.lookback,
            predict=config.predict,
            split_mode=config.split_mode,
            custom_tokenizer_path=args.tokenizer_path,
            custom_checkpoint_path=args.checkpoint_path,
        )

        mape_by_step, mape_summary, ic_result, amplitude_result = evaluate(
            model, tokenizer, test_data, config, device,
            n_samples=args.n_samples,
            n_sample_ratio=args.n_sample_ratio,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            model_type=model_type,
        )

        if is_main:
            report = format_metrics_report(
                mape_by_step, mape_summary, ic_result, amplitude_result,
                predict=config.predict
            )
            print(report)

    cleanup_ddp()


if __name__ == '__main__':
    main()