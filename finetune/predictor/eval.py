"""
Kronos Predictor Evaluation Entry

统一评估入口，支持：
- DDP 多卡评估
- 多模型批量对比
- 正确度量口径（去趋势 trajectory IC）
- 可懂指标输出

使用：
    # 单模型
    python finetune/predictor/eval.py --norm-mode sliding_ma60 --model mini

    # 多卡 DDP
    torchrun --nproc_per_node=4 finetune/predictor/eval.py --model mini

    # 多模型对比
    python finetune/predictor/eval.py --models mini,small,base

关键：
- 与 train.py 共用 core/metrics.py
- trajectory IC 用去趋势序列
- 主输出为可懂指标（方向胜率/振幅误差率/涨跌停命中率）
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
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
    calculate_combined_score,
    calculate_da_score,
    get_log_step_weights,
    get_feature_weights,
    amplitude_error_rate,
    compute_amplitude_stats,
    limit_hit_rate,
    detect_limit,
    format_metrics_report,
    excess_da,
)
from finetune.predictor.core.utils import get_device, format_time


# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_tokenizer(
    norm_mode: str,
    model_type: str,
    device: torch.device,
    checkpoint: str = 'best_combined_model'
):
    """
    加载模型和 tokenizer
    """
    # Tokenizer
    tokenizer_path = get_tokenizer_path(norm_mode, model_type)
    if not os.path.exists(tokenizer_path):
        tokenizer_path = 'outputs/tokenizers/final/2k-MA60' if model_type == 'mini' else 'outputs/tokenizers/final/base-MA60'

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    # 模型
    pretrained_paths = {
        'mini': 'pretrained/Kronos-mini',
        'small': 'pretrained/Kronos-small',
        'base': 'pretrained/Kronos-base',
    }

    model = Kronos.from_pretrained(pretrained_paths[model_type])
    model.eval().to(device)

    # Checkpoint
    model_dir = get_model_path(norm_mode, 400, 10, 'block', model_type)
    checkpoint_path = get_checkpoint_path(model_dir, checkpoint)

    if os.path.exists(os.path.join(checkpoint_path, 'model.safetensors')):
        state_dict = load_file(os.path.join(checkpoint_path, 'model.safetensors'))
        model.load_state_dict(state_dict, strict=False)

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
    seed: int = 42,
    limit_pct: float = 0.10
):
    """
    评估模型（正确口径）
    """
    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)

    # 构建窗口索引
    indices = []
    for symbol, d in test_data.items():
        if 'windows' in d:
            for w in d['windows']:
                indices.append((symbol, int(w)))
        elif hasattr(d, 'columns'):
            seq_len = len(d)
            if seq_len >= config.lookback + config.predict:
                indices.append((symbol, seq_len - config.lookback - config.predict))
        else:
            seq_len = len(d['normalized'])
            if seq_len >= config.lookback + config.predict:
                indices.append((symbol, seq_len - config.lookback - config.predict))

    # 抽样
    if n_samples > 0 and n_samples < len(indices):
        sample_idx = rng.choice(len(indices), size=n_samples, replace=False)
        indices = [indices[i] for i in sample_idx]

    # 收集结果
    ic_lists = {f: [] for f in FEATURE_NAMES}
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(config.predict)]

    # 收集 actual_dir 统计（用于计算 naive DA）
    actual_dir_by_step = [[] for _ in range(config.predict)]  # 仅 close

    amplitude_rates = []
    limit_results = {'pred_limit': [], 'actual_limit': []}

    for (symbol, start) in tqdm(indices, desc="Evaluating"):
        d = test_data[symbol]
        end = start + config.lookback + config.predict

        try:
            if 'normalized' in d:
                x_norm = d['normalized'][start:start + config.lookback].astype(np.float32)
                means = d['means'][start:end]
                stds = d['stds'][start:end]
                original = d['original'][start:end]
                timestamps = d['index'][start:end]
            else:
                df = d.iloc[start:end]
                x_raw = df.values[:config.lookback].astype(np.float32)
                x_mean = np.mean(x_raw, axis=0)
                x_std = np.std(x_raw, axis=0) + 1e-5
                x_norm = np.clip((x_raw - x_mean) / x_std, -config.clip, config.clip)
                original = df.values
                timestamps = df.index

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
                pred_detrend = detrend_to_baseline(pred_raw[:, fi], baseline[fi])
                actual_detrend = detrend_to_baseline(actual[:, fi], baseline[fi])

                ic, _ = safe_trajectory_ic(pred_detrend, actual_detrend)
                if ic is not None:
                    ic_lists[fn].append(ic)

            # DA
            for step_idx in range(config.predict):
                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                    actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                    da_by_step[step_idx][fn].append(pred_dir == actual_dir)

                    # 收集 close 的 actual_dir（用于 naive DA）
                    if fn == 'close':
                        actual_dir_by_step[step_idx].append(actual_dir)

            # 振幅误差率
            pred_amp = pred_raw[0, 1] - pred_raw[0, 2]  # high - low
            actual_amp = actual[0, 1] - actual[0, 2]
            amplitude_rates.append(amplitude_error_rate(pred_amp, actual_amp))

            # 涨跌停检测
            pred_limit = detect_limit(pred_raw, baseline_close, limit_pct)
            actual_limit = detect_limit(actual, baseline_close, limit_pct)
            limit_results['pred_limit'].append(pred_limit.any())
            limit_results['actual_limit'].append(actual_limit.any())

        except Exception:
            continue

    # 聚合结果
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

    da_result = {}
    naive_da_by_step = {}  # {step_idx: naive_da}
    for step_idx in range(config.predict):
        step_result = {}
        for fn in FEATURE_NAMES:
            da_list = da_by_step[step_idx][fn]
            if da_list:
                da_arr = np.array(da_list)
                n = len(da_list)
                step_result[fn] = {
                    'mean': float(np.mean(da_arr)),
                    'std': float(np.std(da_arr)) if n >= 2 else 0.0,
                    'p50': float(np.percentile(da_arr, 50)),
                    'n': n,
                }
            else:
                step_result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': None, 'n': 0}
        da_result[f'step{step_idx + 1}'] = step_result

        # 计算 naive DA（多数方向比例）
        actual_dirs = actual_dir_by_step[step_idx]
        if actual_dirs:
            up_ratio = np.mean(actual_dirs)
            naive_da_by_step[step_idx] = float(max(up_ratio, 1 - up_ratio))
        else:
            naive_da_by_step[step_idx] = 0.5

    # 振幅统计
    amplitude_result = {
        'mean_rate': float(np.mean(amplitude_rates)) if amplitude_rates else 1.0,
        'std_rate': float(np.std(amplitude_rates)) if amplitude_rates else 0.0,
        'perfect_pct': float(np.mean(np.abs(np.array(amplitude_rates) - 1.0) < 0.1)) if amplitude_rates else 0.0,
        'usable_pct': float(np.mean(np.abs(np.array(amplitude_rates) - 1.0) < 0.3)) if amplitude_rates else 0.0,
    }

    # 涨跌停命中率
    limit_result = limit_hit_rate(
        np.array(limit_results['pred_limit']),
        np.array(limit_results['actual_limit'])
    )

    return ic_result, da_result, amplitude_result, limit_result, naive_da_by_step


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
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated models for batch comparison')
    parser.add_argument('--checkpoint', type=str, default='best_combined_model',
                        choices=['best_model', 'best_ic_model', 'best_combined_model', 'latest_model'])
    parser.add_argument('--n-samples', type=int, default=-1,
                        help='Number of samples (-1 for full)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-pct', type=float, default=0.10,
                        help='Limit threshold (main board 10%, ChiNext 20%)')
    parser.add_argument('--use-block', action='store_true',
                        help='Use block_lb400_pd10 data')
    args = parser.parse_args()

    device = get_device()

    config = DataConfig(
        norm_mode=args.norm_mode,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
    )

    # 数据路径（由 preprocess.py 预先生成）
    test_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'test')

    print(f"\n{'=' * 60}")
    print(f"Kronos Predictor Evaluation (Detrended IC)")
    print(f"{'=' * 60}")
    print(f"norm_mode: {config.norm_mode}")
    print(f"model: {args.model}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"n_samples: {args.n_samples if args.n_samples > 0 else 'FULL'}")
    print(f"{'=' * 60}")

    # 加载测试数据
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    print(f"Test data: {len(test_data)} stocks")

    # 模型列表
    if args.models:
        model_types = args.models.split(',')
    else:
        model_types = [args.model]

    for model_type in model_types:
        print(f"\n--- Evaluating {model_type} ---")

        model, tokenizer = load_model_and_tokenizer(
            config.norm_mode, model_type, device, args.checkpoint
        )

        ic_result, da_result, amplitude_result, limit_result, naive_da_by_step = evaluate(
            model, tokenizer, test_data, config, device,
            n_samples=args.n_samples,
            seed=args.seed,
            limit_pct=args.limit_pct
        )

        # 输出报告
        report = format_metrics_report(
            ic_result, da_result, amplitude_result, limit_result,
            naive_da_by_step=naive_da_by_step,
            predict=config.predict
        )
        print(report)


if __name__ == '__main__':
    main()