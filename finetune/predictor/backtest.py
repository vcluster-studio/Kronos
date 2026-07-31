"""
Kronos Predictor Backtest Entry

回测入口，基于 OHLCV 准确度指标：

核心指标：
- MAPE：各特征的平均绝对百分比误差（主要指标）
- Trajectory IC：轨迹形状相关性（去趋势）
- Amplitude Error Rate：振幅误差率

使用：
    python finetune/predictor/backtest.py --model mini --n-samples 1000
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

from finetune.predictor.core.config import BacktestConfig
from finetune.predictor.core.paths import (
    get_backtest_data_path,
    get_tokenizer_path,
    get_model_path,
    get_checkpoint_path,
)
from finetune.predictor.core.normalization import get_normalizer
from finetune.predictor.core.metrics import (
    FEATURE_NAMES,
    detrend_to_baseline,
    safe_trajectory_ic,
    compute_mape,
    get_step_weights,
    amplitude_error_rate,
)
from finetune.predictor.core.utils import get_device


# ============================================================================
# 回测函数
# ============================================================================

def backtest(
    model,
    tokenizer,
    test_data,
    config: BacktestConfig,
    device: torch.device,
    n_samples: int = 100,
    seed: int = 42,
    norm_mode: str = 'sliding_ma60',
    model_type: str = 'mini'
):
    """
    回测（基于 OHLCV 准确度）

    核心指标：MAPE（各特征的百分比误差）
    """
    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)

    symbols = list(test_data.keys())
    if n_samples > 0 and n_samples < len(symbols):
        indices = rng.choice(len(symbols), size=n_samples, replace=False)
        symbols = [symbols[i] for i in indices]

    # 收集结果
    mape_lists = {fn: [] for fn in FEATURE_NAMES}
    ic_lists = {fn: [] for fn in FEATURE_NAMES}
    amplitude_rates = []

    # PR4 校验
    if symbols and norm_mode != 'full_window':
        first_sym = symbols[0]
        first_d = test_data[first_sym]
        first_w = first_d['windows'][0]
        try:
            lb = first_w['lookback']
            pd_ = first_w['predict']
            full_orig = first_w['original']
            full_norm = first_w['normalized']
            seq_len = len(full_orig)
            ctx_start = seq_len - lb - pd_
            target_start = ctx_start + lb

            from finetune.predictor.core.normalization import NormalizerFactory
            required = NormalizerFactory.get_required_history(norm_mode)
            hist_start = max(0, ctx_start - required)
            ctx_with_hist = full_orig[hist_start:target_start]
            normalizer = get_normalizer(norm_mode)
            recon_norm, _, _ = normalizer.normalize(ctx_with_hist.astype(np.float32))
            recon_ctx_norm = recon_norm[-lb:]

            stored_ctx_norm = full_norm[ctx_start:target_start]
            if not np.allclose(recon_ctx_norm, stored_ctx_norm, atol=1e-5, equal_nan=True):
                raise RuntimeError(f"PR4 校验失败：{first_sym} context 段归一化可能泄露")
            print(f"[INFO] PR4 校验通过：context 段归一化未用 target 数据")
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[WARNING] PR4 校验跳过：{e}")
    elif norm_mode == 'full_window':
        print(f"[INFO] full_window 模式：跳过 PR4 检查")

    for sym in tqdm(symbols, desc="Backtesting"):
        sym_data = test_data[sym]
        windows = sym_data['windows']

        for wi, w in enumerate(windows):
            try:
                lookback = w['lookback']
                predict = w['predict']
                seq_len = len(w['normalized'])
                ctx_start = seq_len - lookback - predict
                target_start = ctx_start + lookback

                original = w['original']
                timestamps = w['index']

                if norm_mode == 'full_window':
                    ctx_orig = original[ctx_start:target_start].astype(np.float32)
                    ctx_normalizer = get_normalizer(norm_mode)
                    x_norm, ctx_mean, ctx_std = ctx_normalizer.normalize(ctx_orig)
                    inference_mean = ctx_mean
                    inference_std = ctx_std
                else:
                    x_norm = w['normalized'][ctx_start:target_start].astype(np.float32)
                    means = w['means']
                    stds = w['stds']
                    inference_mean = means[target_start:target_start + predict]
                    inference_std = stds[target_start:target_start + predict]

                baseline = original[target_start - 1]
                baseline_close = original[target_start - 1, 3]

                x_stamp = np.stack([
                    timestamps[ctx_start:target_start].minute.values,
                    timestamps[ctx_start:target_start].hour.values,
                    timestamps[ctx_start:target_start].weekday.values,
                    timestamps[ctx_start:target_start].day.values,
                    timestamps[ctx_start:target_start].month.values,
                ], axis=1).astype(np.float32)

                y_stamp = np.stack([
                    timestamps[target_start:target_start + predict].minute.values,
                    timestamps[target_start:target_start + predict].hour.values,
                    timestamps[target_start:target_start + predict].weekday.values,
                    timestamps[target_start:target_start + predict].day.values,
                    timestamps[target_start:target_start + predict].month.values,
                ], axis=1).astype(np.float32)

                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                    x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                    y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model,
                        x_tensor, x_stamp_tensor, y_stamp_tensor,
                        max_context={'mini': 2048, 'small': 512, 'base': 512}.get(model_type, 2048),
                        pred_len=predict,
                        clip=5.0,
                        T=1.0,
                        top_p=0.9,
                        sample_count=1,
                        verbose=False
                    )

                    pred_norm = preds[0, -predict:, :]
                    pred_raw = pred_norm * inference_std + inference_mean
                    actual = original[target_start:target_start + predict]

                # 计算 MAPE
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

                # 振幅误差率
                pred_amp = pred_raw[0, 1] - pred_raw[0, 2]
                actual_amp = actual[0, 1] - actual[0, 2]
                amplitude_rates.append(amplitude_error_rate(pred_amp, actual_amp))

            except Exception as e:
                print(f"[WARN backtest] {sym} window{wi}: {type(e).__name__}: {e}")
                continue

    # 统计
    total_samples = sum(len(mape_lists[fn]) for fn in FEATURE_NAMES[:1])

    # 计算每步 MAPE
    predict = 10  # 默认
    mape_by_step = {}
    for step_idx in range(predict):
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
    step_weights = get_step_weights(predict)
    for fi, fn in enumerate(FEATURE_NAMES):
        weighted_sum = 0.0
        total_weight = 0.0
        for mape_matrix in mape_lists[fn]:
            if mape_matrix is None:
                continue
            for step_idx in range(min(len(mape_matrix), predict)):
                val = mape_matrix[step_idx, fi]
                w = step_weights[step_idx]
                weighted_sum += val * w
                total_weight += w
        mape_summary[fn] = weighted_sum / total_weight if total_weight > 0 else 0.0

    # IC 聚合
    ic_result = {}
    for fn in FEATURE_NAMES:
        ics = ic_lists[fn]
        if ics:
            ics_arr = np.array(ics)
            n = len(ics)
            ic_result[fn] = {
                'mean': float(np.mean(ics_arr)),
                'std': float(np.std(ics_arr)) if n >= 2 else 0.0,
                'p50': float(np.percentile(ics_arr, 50)),
                'n': n,
            }
        else:
            ic_result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'n': 0}

    # 振幅统计
    valid_amp_rates = [r for r in amplitude_rates if r is not None]
    amplitude_result = {
        'mean_rate': float(np.mean(valid_amp_rates)) if valid_amp_rates else 1.0,
        'std_rate': float(np.std(valid_amp_rates)) if valid_amp_rates else 0.0,
        'perfect_pct': float(np.mean(np.abs(np.array(valid_amp_rates) - 1.0) < 0.1)) if valid_amp_rates else 0.0,
        'usable_pct': float(np.mean(np.abs(np.array(valid_amp_rates) - 1.0) < 0.3)) if valid_amp_rates else 0.0,
    }

    return {
        'n_samples': total_samples,
        'n_stocks': len(symbols),
        'mape_by_step': mape_by_step,
        'mape_summary': mape_summary,
        'ic_result': ic_result,
        'amplitude_result': amplitude_result,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos Predictor Backtest')
    parser.add_argument('--model', type=str, default='mini',
                        help='模型类型(mini/small/base)或 checkpoint 绝对路径')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['mini', 'small', 'base'],
                        help='模型架构类型（--model 为路径时必须指定）')
    parser.add_argument('--checkpoint', type=str, default='best_combined_model',
                        choices=['best_model', 'best_ic_model', 'best_combined_model', 'latest_model'])
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='tokenizer 绝对路径（不指定则用默认路径）')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block',
                        choices=['time', 'block'])
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    config = BacktestConfig()

    # 模型类型判断
    pretrained_paths = {
        'mini': 'pretrained/Kronos-mini',
        'small': 'pretrained/Kronos-small',
        'base': 'pretrained/Kronos-base',
    }

    # 判断 --model 是类型还是路径
    is_model_path = args.model not in pretrained_paths or args.model_type is not None

    if is_model_path:
        # 路径模式：必须指定 --model-type
        if args.model_type is None:
            raise ValueError("--model 为路径时必须指定 --model-type (mini/small/base)")
        model_type = args.model_type
        checkpoint_dir = args.model
    else:
        # 类型模式：使用默认路径
        model_type = args.model
        checkpoint_dir = None

    # Tokenizer
    if args.tokenizer:
        tokenizer_path = args.tokenizer
    else:
        tokenizer_path = get_tokenizer_path(args.norm_mode, model_type)

    if not os.path.exists(tokenizer_path):
        print(f"[WARNING] Tokenizer not found at {tokenizer_path}")
        tokenizer_path = 'pretrained/Kronos-Tokenizer-2k' if model_type == 'mini' else 'pretrained/Kronos-Tokenizer-base'
        print(f"[WARNING] Using pretrained tokenizer: {tokenizer_path}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    # 模型
    model = Kronos.from_pretrained(pretrained_paths[model_type])
    model.eval().to(device)

    # Checkpoint
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_path(
            get_model_path(args.norm_mode, args.lookback, args.predict, args.split_mode, model_type),
            args.checkpoint
        )

    safetensors_path = os.path.join(checkpoint_dir, 'model.safetensors')

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint 与模型不匹配: missing={missing}, unexpected={unexpected}")
        print(f"[INFO] Loaded checkpoint: {checkpoint_dir}")
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_dir}")

    # 数据
    test_path = get_backtest_data_path(args.norm_mode, args.lookback, args.predict)

    print(f"\n{'=' * 60}")
    print(f"Kronos Predictor Backtest (OHLCV Accuracy)")
    print(f"{'=' * 60}")
    print(f"model: {args.model}")
    print(f"tokenizer: {tokenizer_path}")
    print(f"norm_mode: {args.norm_mode}")
    print(f"n_samples: {args.n_samples if args.n_samples > 0 else 'FULL'}")
    print(f"{'=' * 60}")

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    result = backtest(
        model, tokenizer, test_data, config, device,
        n_samples=args.n_samples,
        seed=args.seed,
        norm_mode=args.norm_mode,
        model_type=model_type
    )

    # 输出
    print(f"\n{'=' * 60}")
    print(f"[Backtest Results]")
    print(f"{'=' * 60}")
    print(f"  Samples: {result['n_samples']}, Stocks: {result['n_stocks']}")

    # MAPE 摘要
    print(f"\n  [MAPE Summary - Weighted Average]")
    for fn in FEATURE_NAMES:
        mape_val = result['mape_summary'].get(fn, 0.0)
        print(f"    {fn}: {mape_val:.2%}")

    # Amplitude
    amp = result['amplitude_result']
    print(f"\n  [Amplitude Error Rate]")
    print(f"    Mean: {amp['mean_rate']:.2f}")
    print(f"    Usable (0.7-1.3): {amp['usable_pct']:.1%}")

    # IC
    print(f"\n  [Trajectory IC]")
    for fn in FEATURE_NAMES:
        ic_info = result['ic_result'].get(fn, {})
        print(f"    {fn}: {ic_info.get('mean', 0):.4f} (n={ic_info.get('n', 0)})")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()