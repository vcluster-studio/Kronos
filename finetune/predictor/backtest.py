"""
Kronos Predictor Backtest Entry

回测入口，支持：
- 批量回测
- 可懂指标（方向胜率/振幅误差率/涨跌停命中率）
- per-stock IC 聚合（不跨股票混算）

使用：
    python finetune/predictor/backtest.py --model mini --n-samples 1000

关键：
- backtest IC 逐股票算再聚合（E8 解决）
- sigmoid 参数配置化（E7 解决）
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
from finetune.predictor.core.paths import get_legacy_data_path
from finetune.predictor.core.metrics import (
    safe_corrcoef,
    safe_spearmanr,
    detrend_to_baseline,
    excess_da,
    amplitude_error_rate,
    limit_hit_rate,
    detect_limit,
)
from finetune.predictor.core.utils import get_device


# ============================================================================
# 配置
# ============================================================================

LOOKBACK = 400
PREDICT = 10


# ============================================================================
# 回测函数
# ============================================================================

def backtest(
    model,
    tokenizer,
    test_data,
    config: BacktestConfig,
    device: torch.device,
    raw_data=None,
    n_samples: int = 100,
    seed: int = 42,
    norm_mode: str = 'ma60'
):
    """
    回测（正确口径）

    关键：
    - per-stock IC（不跨股票混算）
    - sigmoid 参数从配置读取
    """
    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)

    # 构建窗口
    if norm_mode == 'ma60':
        windows = []
        for sym, d in test_data.items():
            if 'windows' in d:
                for start in d['windows']:
                    windows.append((sym, int(start)))
            else:
                seq_len = len(d['normalized'])
                if seq_len >= LOOKBACK + PREDICT:
                    windows.append((sym, seq_len - LOOKBACK - PREDICT))
    else:
        # full_window 需要从 raw 借用历史
        windows = [(sym, 0) for sym in test_data.keys()]

    # 抽样
    if n_samples > 0 and n_samples < len(windows):
        indices = rng.choice(len(windows), size=n_samples, replace=False)
        windows = [windows[i] for i in indices]

    # 结果收集（per-symbol）
    results_by_symbol = {}

    for (sym, start) in tqdm(windows, desc="Backtesting"):
        d = test_data[sym]

        try:
            if norm_mode == 'ma60':
                end = start + LOOKBACK + PREDICT
                x_norm = d['normalized'][start:start + LOOKBACK].astype(np.float32)
                means = d['means'][start:end]
                stds = d['stds'][start:end]
                original = d['original'][start:end]
                timestamps = d['index'][start:end]
                baseline = original[LOOKBACK - 1]
                baseline_close = original[LOOKBACK - 1, 3]
            else:
                # full_window: 从 raw 借用历史
                raw = raw_data.get(sym)
                if raw is None:
                    continue

                # 找到 test 在 raw 中的位置
                test_start_date = d.index[0]
                raw_idx = raw['index']
                test_pos = None
                for i, dt in enumerate(raw_idx):
                    if dt == test_start_date:
                        test_pos = i
                        break

                if test_pos is None or test_pos < LOOKBACK:
                    continue

                window_start = test_pos - LOOKBACK
                window_values = raw['values'][window_start:test_pos + PREDICT]
                window_ts = raw['index'][window_start:test_pos + PREDICT]

                x_raw = window_values[:LOOKBACK]
                x_mean = np.mean(x_raw, axis=0)
                x_std = np.std(x_raw, axis=0) + 1e-5
                x_norm = np.clip((x_raw - x_mean) / x_std, -5.0, 5.0)

                baseline = window_values[LOOKBACK - 1]
                baseline_close = window_values[LOOKBACK - 1, 3]
                timestamps = window_ts

            # 时间戳
            x_stamp = np.stack([
                timestamps[:LOOKBACK].minute.values,
                timestamps[:LOOKBACK].hour.values,
                timestamps[:LOOKBACK].weekday.values,
                timestamps[:LOOKBACK].day.values,
                timestamps[:LOOKBACK].month.values,
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                timestamps[LOOKBACK:LOOKBACK + PREDICT].minute.values,
                timestamps[LOOKBACK:LOOKBACK + PREDICT].hour.values,
                timestamps[LOOKBACK:LOOKBACK + PREDICT].weekday.values,
                timestamps[LOOKBACK:LOOKBACK + PREDICT].day.values,
                timestamps[LOOKBACK:LOOKBACK + PREDICT].month.values,
            ], axis=1).astype(np.float32)

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048,
                    pred_len=PREDICT,
                    clip=5.0,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )

                pred_norm = preds[0, LOOKBACK:LOOKBACK + PREDICT, :]

                # 反归一化
                if norm_mode == 'ma60':
                    pred_raw = pred_norm.cpu().numpy() * stds[LOOKBACK:] + means[LOOKBACK:]
                    actual = original[LOOKBACK:LOOKBACK + PREDICT]
                else:
                    pred_raw = pred_norm.cpu().numpy() * x_std + x_mean
                    actual = window_values[LOOKBACK:LOOKBACK + PREDICT]

            # 增益计算（用 close）
            pred_close = pred_raw[0, 3]
            actual_close = actual[0, 3]
            pred_gain = (pred_close - baseline_close) / (abs(baseline_close) + 1e-8)
            actual_gain = (actual_close - baseline_close) / (abs(baseline_close) + 1e-8)

            # sigmoid 分值
            score = config.sigmoid_score(pred_gain)

            # 方向
            pred_dir = pred_gain > 0
            actual_dir = actual_gain > 0
            direction_correct = pred_dir == actual_dir

            # 振幅
            pred_amp = pred_raw[0, 1] - pred_raw[0, 2]
            actual_amp = actual[0, 1] - actual[0, 2]
            amp_rate = amplitude_error_rate(pred_amp, actual_amp)

            # 涨跌停
            pred_limit = detect_limit(pred_raw, baseline_close, config.limit_pct)
            actual_limit = detect_limit(actual, baseline_close, config.limit_pct)

            # 按股票收集
            if sym not in results_by_symbol:
                results_by_symbol[sym] = {
                    'pred_gains': [],
                    'actual_gains': [],
                    'scores': [],
                    'directions': [],
                    'amp_rates': [],
                    'pred_limit': [],
                    'actual_limit': [],
                }

            results_by_symbol[sym]['pred_gains'].append(pred_gain)
            results_by_symbol[sym]['actual_gains'].append(actual_gain)
            results_by_symbol[sym]['scores'].append(score)
            results_by_symbol[sym]['directions'].append(direction_correct)
            results_by_symbol[sym]['amp_rates'].append(amp_rate)
            results_by_symbol[sym]['pred_limit'].append(pred_limit.any())
            results_by_symbol[sym]['actual_limit'].append(actual_limit.any())

        except Exception:
            continue

    # 聚合（per-stock IC）
    per_stock_ics = []
    per_stock_rank_ics = []

    for sym, res in results_by_symbol.items():
        if len(res['pred_gains']) >= 3:
            ic = safe_corrcoef(res['pred_gains'], res['actual_gains'])
            rank_ic = safe_spearmanr(res['pred_gains'], res['actual_gains'])
            if ic is not None:
                per_stock_ics.append(ic)
            if rank_ic is not None:
                per_stock_rank_ics.append(rank_ic)

    # 统计
    all_pred_gains = []
    all_actual_gains = []
    all_directions = []
    all_amp_rates = []
    all_pred_limit = []
    all_actual_limit = []

    for sym, res in results_by_symbol.items():
        all_pred_gains.extend(res['pred_gains'])
        all_actual_gains.extend(res['actual_gains'])
        all_directions.extend(res['directions'])
        all_amp_rates.extend(res['amp_rates'])
        all_pred_limit.extend(res['pred_limit'])
        all_actual_limit.extend(res['actual_limit'])

    # 结果
    result = {
        'n_samples': len(all_pred_gains),
        'n_stocks': len(results_by_symbol),
    }

    # IC（per-stock 聚合）
    if per_stock_ics:
        result['backtest_ic_mean'] = float(np.mean(per_stock_ics))
        result['backtest_ic_std'] = float(np.std(per_stock_ics))
        result['backtest_ic_p50'] = float(np.percentile(per_stock_ics, 50))
    else:
        result['backtest_ic_mean'] = 0.0
        result['backtest_ic_std'] = 0.0

    if per_stock_rank_ics:
        result['backtest_rank_ic_mean'] = float(np.mean(per_stock_rank_ics))

    # 方向胜率（DA）
    if all_directions:
        model_da = float(np.mean(all_directions))
        result['direction_accuracy'] = model_da
        result['excess_da'] = excess_da(model_da, 0.5)

    # 振幅
    if all_amp_rates:
        result['amplitude_mean'] = float(np.mean(all_amp_rates))
        result['amplitude_std'] = float(np.std(all_amp_rates))
        result['amplitude_usable_pct'] = float(np.mean(np.abs(np.array(all_amp_rates) - 1.0) < 0.3))

    # 涨跌停
    limit_result = limit_hit_rate(np.array(all_pred_limit), np.array(all_actual_limit))
    result['limit_hit_rate'] = limit_result['hit_rate']
    result['limit_pred_count'] = limit_result['n_pred_limit']
    result['limit_actual_count'] = limit_result['n_actual_limit']

    # 分布
    if all_pred_gains:
        result['pred_gain_mean'] = float(np.mean(all_pred_gains))
        result['pred_gain_std'] = float(np.std(all_pred_gains))
        result['actual_gain_mean'] = float(np.mean(all_actual_gains))
        result['actual_gain_std'] = float(np.std(all_actual_gains))
        result['win_rate'] = float(np.mean(np.array(all_actual_gains) > 0))

    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos Predictor Backtest')
    parser.add_argument('--model', type=str, default='final_models/Kronos-mini-MA60')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--norm-mode', type=str, default='ma60',
                        choices=['ma60', 'full_window'])
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples (-1 for full)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-pct', type=float, default=0.10)
    parser.add_argument('--signal-center', type=float, default=0.084)
    parser.add_argument('--signal-steepness', type=float, default=21.0)
    args = parser.parse_args()

    device = get_device()

    config = BacktestConfig(
        signal_center=args.signal_center,
        signal_steepness=args.signal_steepness,
        limit_pct=args.limit_pct,
    )

    # Tokenizer
    if args.tokenizer:
        tokenizer_path = args.tokenizer
    elif args.norm_mode == 'full_window':
        tokenizer_path = 'final_models/Kronos-Tokenizer-2k'
    else:
        tokenizer_path = 'outputs/tokenizers/final/2k-MA60'

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    # 模型
    model = Kronos.from_pretrained('pretrained/Kronos-mini')
    model.eval().to(device)

    model_dir = os.path.join(project_root, args.model)
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)

    # 数据
    if args.norm_mode == 'ma60':
        test_path = get_legacy_data_path('ma60', 400, 'final_test')
    else:
        test_path = 'finetune/data/global_norm/full_series/test_data.pkl'

    print(f"\n{'=' * 60}")
    print(f"Kronos Predictor Backtest")
    print(f"{'=' * 60}")
    print(f"model: {args.model}")
    print(f"tokenizer: {tokenizer_path}")
    print(f"norm_mode: {args.norm_mode}")
    print(f"n_samples: {args.n_samples if args.n_samples > 0 else 'FULL'}")
    print(f"signal_center: {config.signal_center}")
    print(f"signal_steepness: {config.signal_steepness}")
    print(f"{'=' * 60}")

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # full_window 需要 raw
    raw_data = None
    if args.norm_mode == 'full_window':
        raw_path = 'data/kline_daily_raw.pkl'
        if os.path.exists(raw_path):
            with open(raw_path, 'rb') as f:
                raw_data = pickle.load(f)

    # 回测
    result = backtest(
        model, tokenizer, test_data, config, device,
        raw_data=raw_data,
        n_samples=args.n_samples,
        seed=args.seed,
        norm_mode=args.norm_mode
    )

    # 输出
    print(f"\n{'=' * 60}")
    print(f"[Backtest Results]")
    print(f"{'=' * 60}")
    print(f"  Samples: {result['n_samples']}, Stocks: {result['n_stocks']}")

    print(f"\n  [Direction Accuracy]")
    print(f"    DA: {result.get('direction_accuracy', 0):.1%}")
    print(f"    Excess DA: {result.get('excess_da', 0):.1%} (vs 50% random)")

    print(f"\n  [Amplitude Error Rate]")
    print(f"    Mean: {result.get('amplitude_mean', 1):.2f}")
    print(f"    Usable (0.7-1.3): {result.get('amplitude_usable_pct', 0):.1%}")

    print(f"\n  [Limit Hit Rate]")
    hit_rate = result.get('limit_hit_rate')
    if hit_rate is not None:
        print(f"    Hit Rate: {hit_rate:.1%} (random ~1-3%, signal >10%)")
        print(f"    Predicted: {result['limit_pred_count']}, Actual: {result['limit_actual_count']}")
    else:
        print(f"    No predicted limit")

    print(f"\n  [Backtest IC] (per-stock aggregated)")
    print(f"    Mean: {result.get('backtest_ic_mean', 0):.4f}")
    print(f"    Std: {result.get('backtest_ic_std', 0):.4f}")
    print(f"    Rank IC Mean: {result.get('backtest_rank_ic_mean', 0):.4f}")

    print(f"\n  [Distribution]")
    print(f"    Pred gain: mean={result.get('pred_gain_mean', 0):+.2%}, std={result.get('pred_gain_std', 0):.2%}")
    print(f"    Actual gain: mean={result.get('actual_gain_mean', 0):+.2%}, std={result.get('actual_gain_std', 0):.2%}")
    print(f"    Win rate: {result.get('win_rate', 0):.1%}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()