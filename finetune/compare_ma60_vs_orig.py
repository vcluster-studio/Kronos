"""
MA60 Pipeline vs Original Pipeline 对比测试

测试组合：
1. 原始 tokenizer + 原始 predictor (full_window 归一化)
2. MA60 tokenizer + MA60 predictor (sliding_ma60 归一化)

使用预归一化 MA60 数据进行公平测试。
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# 模型路径
TOKENIZER_ORIG = 'final_models/Kronos-Tokenizer-2k'
TOKENIZER_MA60 = 'outputs/models/ma60_tokenizer_v1/checkpoints/best_model'
PREDICTOR_ORIG = 'final_models/Kronos-mini'
PREDICTOR_MA60 = 'outputs/models/ma60_predictor_v1/checkpoints/best_ic_model'

# 数据路径
TEST_DATA_ORIG = 'finetune/data/processed_datasets/test_data.pkl'
TEST_DATA_MA60 = 'finetune/data/processed_datasets_ma60/test_data.pkl'

# 测试参数
lookback = 400
pred_len = 10
clip = 5.0
ic_point = 3
n_samples = 500
seed = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def full_window_normalize(values, clip=5.0):
    """全窗口归一化"""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0) + 1e-5
    normalized = (values - mean) / std
    normalized = np.clip(normalized, -clip, clip)
    return normalized, mean, std


def test_orig_pipeline(tokenizer, predictor, test_data, n_samples):
    """测试原始 pipeline (full_window)"""
    tokenizer.to(device)
    predictor.to(device)
    tokenizer.eval()
    predictor.eval()

    symbols = list(test_data.keys())[:n_samples]
    np.random.seed(seed)
    sampled = np.random.choice(symbols, size=min(n_samples, len(symbols)), replace=False)

    pred_returns = []
    actual_returns = []

    for symbol in sampled:
        df = test_data[symbol]
        values = df[['open', 'high', 'low', 'close', 'vol', 'amt']].values.astype(np.float32)

        if len(values) < lookback + pred_len:
            continue

        try:
            # 取 lookback 窗口
            x_values = values[-(lookback + pred_len):-pred_len]
            x_norm, mean, std = full_window_normalize(x_values)

            # 时间戳
            timestamps = df.index[-(lookback + pred_len):-pred_len]
            x_ts = timestamps
            y_ts = df.index[-pred_len:]

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            # 基准 close
            baseline_close = values[-pred_len - 1, 3]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, predictor,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048, pred_len=pred_len,
                    clip=clip, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                pred_close_norm = preds[0, -pred_len:, 3]
                pred_close = pred_close_norm * std[3] + mean[3]
                pred_ret = (pred_close[ic_point - 1] - baseline_close) / baseline_close

            actual_close = values[-pred_len:, 3]
            actual_ret = (actual_close[ic_point - 1] - baseline_close) / baseline_close

            pred_returns.append(pred_ret)
            actual_returns.append(actual_ret)

        except Exception as e:
            continue

    if len(pred_returns) < 20:
        return None

    pred_returns = np.array(pred_returns)
    actual_returns = np.array(actual_returns)

    ic = np.corrcoef(pred_returns, actual_returns)[0, 1]
    rank_ic, _ = spearmanr(pred_returns, actual_returns)
    dir_acc = np.mean((pred_returns > 0) == (actual_returns > 0))
    mse = np.mean((pred_returns - actual_returns) ** 2)

    return {
        'ic': ic,
        'rank_ic': rank_ic,
        'dir_acc': dir_acc,
        'mse': mse,
        'samples': len(pred_returns)
    }


def test_ma60_pipeline(tokenizer, predictor, test_data, n_samples):
    """测试 MA60 pipeline (预归一化数据)"""
    tokenizer.to(device)
    predictor.to(device)
    tokenizer.eval()
    predictor.eval()

    symbols = list(test_data.keys())[:n_samples]
    np.random.seed(seed)
    sampled = np.random.choice(symbols, size=min(n_samples, len(symbols)), replace=False)

    pred_returns = []
    actual_returns = []

    for symbol in sampled:
        data = test_data[symbol]
        seq_len = len(data['normalized'])

        if seq_len < lookback + pred_len:
            continue

        try:
            full_len = lookback + pred_len
            end_idx = seq_len

            x_norm_full = data['normalized'][end_idx - full_len:end_idx].astype(np.float32)
            means_full = data['means'][end_idx - full_len:end_idx]
            stds_full = data['stds'][end_idx - full_len:end_idx]
            timestamps_full = data['index'][end_idx - full_len:end_idx]

            x_norm = x_norm_full[:lookback]
            x_ts = timestamps_full[:lookback]
            y_ts = timestamps_full[lookback:]

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            # 原始 close
            original_close = data['original'][:, 3]
            baseline_close = original_close[end_idx - pred_len - 1]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, predictor,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048, pred_len=pred_len,
                    clip=clip, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                pred_close_norm = preds[0, -pred_len:, 3]
                pred_close = pred_close_norm * stds_full[lookback:, 3] + means_full[lookback:, 3]
                pred_ret = (pred_close[ic_point - 1] - baseline_close) / baseline_close

            actual_close = original_close[end_idx - pred_len:]
            actual_ret = (actual_close[ic_point - 1] - baseline_close) / baseline_close

            pred_returns.append(pred_ret)
            actual_returns.append(actual_ret)

        except Exception as e:
            continue

    if len(pred_returns) < 20:
        return None

    pred_returns = np.array(pred_returns)
    actual_returns = np.array(actual_returns)

    ic = np.corrcoef(pred_returns, actual_returns)[0, 1]
    rank_ic, _ = spearmanr(pred_returns, actual_returns)
    dir_acc = np.mean((pred_returns > 0) == (actual_returns > 0))
    mse = np.mean((pred_returns - actual_returns) ** 2)

    return {
        'ic': ic,
        'rank_ic': rank_ic,
        'dir_acc': dir_acc,
        'mse': mse,
        'samples': len(pred_returns)
    }


def main():
    print("\n" + "="*60)
    print("MA60 Pipeline vs Original Pipeline 对比测试")
    print("="*60)

    # 加载模型
    print("\n加载模型...")
    tokenizer_orig = KronosTokenizer.from_pretrained(TOKENIZER_ORIG)
    tokenizer_ma60 = KronosTokenizer.from_pretrained(TOKENIZER_MA60)
    predictor_orig = Kronos.from_pretrained(PREDICTOR_ORIG)
    predictor_ma60 = Kronos.from_pretrained(PREDICTOR_MA60)

    # 加载测试数据
    print("\n加载测试数据...")
    with open(TEST_DATA_ORIG, 'rb') as f:
        test_data_orig = pickle.load(f)
    print(f"  原始数据: {len(test_data_orig)} 股票")

    with open(TEST_DATA_MA60, 'rb') as f:
        test_data_ma60 = pickle.load(f)
    print(f"  MA60数据: {len(test_data_ma60)} 股票")

    print(f"\n测试样本数: {n_samples}")
    print(f"IC计算点: Point+{ic_point}")

    # 测试原始 pipeline
    print("\n测试: 原始 pipeline (full_window)...")
    r_orig = test_orig_pipeline(tokenizer_orig, predictor_orig, test_data_orig, n_samples)

    # 测试 MA60 pipeline
    print("\n测试: MA60 pipeline (sliding_ma60)...")
    r_ma60 = test_ma60_pipeline(tokenizer_ma60, predictor_ma60, test_data_ma60, n_samples)

    # 输出结果
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)

    print("\n| Pipeline | IC | Rank IC | 方向准确率 | MSE | 样本数 |")
    print("|----------|------|---------|-----------|------|--------|")

    if r_orig:
        print(f"| 原始 (full_window) | {r_orig['ic']:.4f} | {r_orig['rank_ic']:.4f} | {r_orig['dir_acc']:.2%} | {r_orig['mse']:.6f} | {r_orig['samples']} |")

    if r_ma60:
        print(f"| MA60 (sliding) | {r_ma60['ic']:.4f} | {r_ma60['rank_ic']:.4f} | {r_ma60['dir_acc']:.2%} | {r_ma60['mse']:.6f} | {r_ma60['samples']} |")

    # 分析
    if r_orig and r_ma60:
        print("\n" + "-"*60)
        ic_diff = r_ma60['ic'] - r_orig['ic']
        print(f"IC 差异: {ic_diff:+.4f}")

        if ic_diff > 0:
            print("结论: MA60 pipeline 更优")
        elif ic_diff < -0.05:
            print("结论: 原始 pipeline 更优，MA60 可能需要更多训练")
        else:
            print("结论: 两种 pipeline 效果相近")

    print("="*60)


if __name__ == '__main__':
    main()