"""
MA60 Tokenizer + Original Predictor 对比验证

测试组合：
1. 原始 tokenizer + 原始 predictor (baseline, full_window归一化)
2. MA60 tokenizer + 原始 predictor (sliding_ma60归一化)
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

# 数据集路径
DATASETS = {
    'Full': 'finetune/data/processed_datasets_new',
    'Mid': 'finetune/data/processed_datasets_mid',
    'Small': 'finetune/data/processed_datasets_small',
}

feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
lookback = 400
predict = 10
clip = 5.0
seed = 42
max_samples = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def sliding_ma60_normalize(values, window=60):
    """滑动MA60归一化"""
    n = len(values)
    normalized = np.zeros_like(values)
    means = np.zeros((n, values.shape[1]))
    stds = np.zeros((n, values.shape[1]))

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = values[start:i+1]
        means[i] = np.mean(window_data, axis=0)
        stds[i] = np.std(window_data, axis=0) + 1e-5

    normalized = (values - means) / stds
    normalized = np.clip(normalized, -clip, clip)
    return normalized, means, stds

def full_window_normalize(values):
    """全窗口归一化"""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0) + 1e-5
    normalized = (values - mean) / std
    normalized = np.clip(normalized, -clip, clip)
    return normalized, mean, std

def prepare_timestamps(df, lookback, predict):
    if 'datetime' in df.columns:
        dates = df['datetime'].values[-(lookback + predict):]
    else:
        dates = df.index.values[-(lookback + predict):]
    dates = pd.to_datetime(dates)

    x_ts = dates[:lookback]
    y_ts = dates[lookback:]

    x_stamp = np.stack([
        x_ts.minute, x_ts.hour, x_ts.weekday, x_ts.day, x_ts.month
    ], axis=1).astype(np.float32)

    y_stamp = np.stack([
        y_ts.minute, y_ts.hour, y_ts.weekday, y_ts.day, y_ts.month
    ], axis=1).astype(np.float32)

    return x_stamp, y_stamp

def test_combo(tokenizer, predictor, data, norm_mode='full_window'):
    """测试组合"""
    tokenizer.to(device)
    predictor.to(device)
    tokenizer.eval()
    predictor.eval()

    symbols = sorted(list(data.keys()))
    np.random.seed(seed)
    sampled = np.random.choice(symbols, size=min(max_samples, len(symbols)), replace=False)

    pred_returns = []
    actual_returns = []

    for symbol in sampled:
        df = data[symbol]
        values = df[feature_cols].values.astype(np.float32)

        if len(values) < lookback + predict:
            continue

        # 取数据窗口
        window_values = values[-(lookback + predict):]
        x_values = window_values[:lookback]

        # 归一化
        if norm_mode == 'sliding_ma60':
            x_norm, means, stds = sliding_ma60_normalize(x_values)
        else:
            x_norm, mean, std = full_window_normalize(x_values)
            means = np.tile(mean, (lookback, 1))
            stds = np.tile(std, (lookback, 1))

        # 时间戳
        x_stamp, y_stamp = prepare_timestamps(df, lookback, predict)

        # 基准close
        baseline_close = values[-predict - 1, 3]
        actual_close = values[-predict:, 3]
        actual_ret = (actual_close[-1] - baseline_close) / baseline_close

        try:
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm[np.newaxis, :, :]).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, :, :]).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, :, :]).to(device)

                preds = auto_regressive_inference(
                    tokenizer, predictor,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048, pred_len=predict,
                    clip=clip, T=0.6, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                # 反归一化
                pred_close = preds[0, -predict:, 3].cpu().numpy()
                pred_close_denorm = pred_close * stds[-predict:, 3] + means[-predict:, 3]

                # 预测收益率
                pred_ret = (pred_close_denorm[-1] - baseline_close) / baseline_close

                pred_returns.append(pred_ret)
                actual_returns.append(actual_ret)

        except Exception as e:
            continue

    if len(pred_returns) < 20:
        return None

    pred_returns = np.array(pred_returns)
    actual_returns = np.array(actual_returns)

    # IC (Spearman)
    ic, _ = spearmanr(pred_returns, actual_returns)

    # 方向准确率
    correct_dir = np.mean((pred_returns > 0) == (actual_returns > 0))

    # MSE
    mse = np.mean((pred_returns - actual_returns) ** 2)

    return {
        'ic': ic,
        'dir_acc': correct_dir,
        'mse': mse,
        'samples': len(pred_returns)
    }

def main():
    print("加载模型...")
    tokenizer_orig = KronosTokenizer.from_pretrained(TOKENIZER_ORIG)
    tokenizer_ma60 = KronosTokenizer.from_pretrained(TOKENIZER_MA60)
    predictor = Kronos.from_pretrained(PREDICTOR_ORIG)

    print("\n" + "="*60)
    print("MA60 Tokenizer + Original Predictor 对比验证")
    print("="*60)

    results = {}

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n数据集: {dataset_name}")

        # 加载测试数据
        test_path = f"{dataset_path}/test_data.pkl"
        if not os.path.exists(test_path):
            test_path = f"{dataset_path}/val_data.pkl"

        with open(test_path, 'rb') as f:
            data = pickle.load(f)

        print(f"  股票数: {len(data)}")

        # 测试原始组合
        print("  测试: 原始 tokenizer + 原始 predictor (full_window)...")
        r_orig = test_combo(tokenizer_orig, predictor, data, 'full_window')

        # 测试MA60组合
        print("  测试: MA60 tokenizer + 原始 predictor (sliding_ma60)...")
        r_ma60 = test_combo(tokenizer_ma60, predictor, data, 'sliding_ma60')

        results[dataset_name] = {
            'orig': r_orig,
            'ma60': r_ma60
        }

        if r_orig and r_ma60:
            print(f"\n  {dataset_name} 结果:")
            print(f"  | 组合 | IC | 方向准确率 | MSE | 样本数 |")
            print(f"  | 原始 (full_window) | {r_orig['ic']:.4f} | {r_orig['dir_acc']:.2%} | {r_orig['mse']:.6f} | {r_orig['samples']} |")
            print(f"  | MA60 (sliding) | {r_ma60['ic']:.4f} | {r_ma60['dir_acc']:.2%} | {r_ma60['mse']:.6f} | {r_ma60['samples']} |")

    print("\n" + "="*60)
    print("汇总对比")
    print("="*60)
    print("| 数据集 | 原始IC | MA60 IC | IC变化 |")
    for ds, r in results.items():
        if r['orig'] and r['ma60']:
            change = (r['ma60']['ic'] - r['orig']['ic']) / abs(r['orig']['ic']) * 100 if r['orig']['ic'] != 0 else 0
            print(f"| {ds} | {r['orig']['ic']:.4f} | {r['ma60']['ic']:.4f} | {change:+.1f}% |")

if __name__ == '__main__':
    main()