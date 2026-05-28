"""
全市场5组模型对比测试

测试组合：
1. Original (原始tokenizer + 原始predictor)
2. V1 IC Best (MA60 tokenizer + V1 best_ic_model)
3. V2 IC Best (MA60 tokenizer + V2 best_ic_model)
4. V1 Loss Best (MA60 tokenizer + V1 best_loss_model)
5. V2 Loss Best (MA60 tokenizer + V2 best_loss_model)
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# 模型路径
TOKENIZER_ORIG = 'final_models/Kronos-Tokenizer-2k'
TOKENIZER_MA60 = 'outputs/models/ma60_tokenizer_v1/checkpoints/best_model'

PREDICTOR_ORIG = 'final_models/Kronos-mini'
PREDICTOR_V1_IC = 'outputs/models/ma60_predictor_v1/checkpoints/best_ic_model'
PREDICTOR_V2_IC = 'outputs/models/ma60_predictor_v2/checkpoints/best_ic_model'
PREDICTOR_V1_LOSS = 'outputs/models/ma60_predictor_v1/checkpoints/best_model'  # best VL
PREDICTOR_V2_LOSS = 'outputs/models/ma60_predictor_v2/checkpoints/best_model'  # best VL

# 数据路径
TEST_DATA_ORIG = 'finetune/data/processed_datasets/test_data.pkl'
TEST_DATA_MA60 = 'finetune/data/processed_datasets_ma60/test_data.pkl'

# 测试参数
lookback = 400
pred_len = 10
clip = 5.0
ic_point = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def full_window_normalize(values, clip=5.0):
    """全窗口归一化"""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0) + 1e-5
    normalized = (values - mean) / std
    normalized = np.clip(normalized, -clip, clip)
    return normalized, mean, std


def test_orig_pipeline(tokenizer, predictor, test_data):
    """测试原始 pipeline (full_window)"""
    tokenizer.to(device)
    predictor.to(device)
    tokenizer.eval()
    predictor.eval()

    pred_returns = []
    actual_returns = []

    symbols = list(test_data.keys())

    for symbol in tqdm(symbols, desc="Testing Original"):
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

    return analyze_results(pred_returns, actual_returns, "Original")


def test_ma60_pipeline(tokenizer, predictor, test_data, name):
    """测试 MA60 pipeline (预归一化数据)"""
    tokenizer.to(device)
    predictor.to(device)
    tokenizer.eval()
    predictor.eval()

    pred_returns = []
    actual_returns = []

    symbols = list(test_data.keys())

    for symbol in tqdm(symbols, desc=f"Testing {name}"):
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

    return analyze_results(pred_returns, actual_returns, name)


def analyze_results(pred_returns, actual_returns, name):
    """分析结果"""
    if len(pred_returns) < 20:
        return None

    pred_returns = np.array(pred_returns)
    actual_returns = np.array(actual_returns)

    # 计算指标
    ic = np.corrcoef(pred_returns, actual_returns)[0, 1]
    rank_ic, _ = spearmanr(pred_returns, actual_returns)
    dir_acc = np.mean((pred_returns > 0) == (actual_returns > 0))
    mse = np.mean((pred_returns - actual_returns) ** 2)

    # 分组分析
    sorted_idx = np.argsort(actual_returns)
    n = len(sorted_idx)
    group_size = n // 5

    group_accs = []
    for i in range(5):
        group_idx = sorted_idx[i*group_size:(i+1)*group_size]
        group_pred = pred_returns[group_idx]
        group_actual = actual_returns[group_idx]
        if i == 0:  # 最差组
            acc = np.mean(group_pred < 0)
        elif i == 4:  # 最优组
            acc = np.mean(group_pred > 0)
        else:
            acc = np.mean((group_pred > 0) == (group_actual > 0))
        group_accs.append(acc)

    return {
        'name': name,
        'samples': len(pred_returns),
        'ic': ic,
        'rank_ic': rank_ic,
        'dir_acc': dir_acc,
        'mse': mse,
        'group_accs': group_accs,
        'pred_returns': pred_returns,
        'actual_returns': actual_returns
    }


def main():
    print("="*70)
    print("全市场5组模型对比测试")
    print("="*70)

    # 加载模型
    print("\n加载模型...")
    tokenizer_orig = KronosTokenizer.from_pretrained(TOKENIZER_ORIG)
    tokenizer_ma60 = KronosTokenizer.from_pretrained(TOKENIZER_MA60)

    predictor_orig = Kronos.from_pretrained(PREDICTOR_ORIG)
    predictor_v1_ic = Kronos.from_pretrained(PREDICTOR_V1_IC)
    predictor_v2_ic = Kronos.from_pretrained(PREDICTOR_V2_IC)
    predictor_v1_loss = Kronos.from_pretrained(PREDICTOR_V1_LOSS)
    predictor_v2_loss = Kronos.from_pretrained(PREDICTOR_V2_LOSS)

    # 加载测试数据
    print("\n加载测试数据...")
    with open(TEST_DATA_ORIG, 'rb') as f:
        test_data_orig = pickle.load(f)
    print(f"  原始数据: {len(test_data_orig)} 股票")

    with open(TEST_DATA_MA60, 'rb') as f:
        test_data_ma60 = pickle.load(f)
    print(f"  MA60数据: {len(test_data_ma60)} 股票")

    # 测试5组模型
    results = []

    print("\n[1/5] 测试 Original Pipeline...")
    r_orig = test_orig_pipeline(tokenizer_orig, predictor_orig, test_data_orig)
    if r_orig:
        results.append(r_orig)

    print("\n[2/5] 测试 V1 IC Best...")
    r_v1_ic = test_ma60_pipeline(tokenizer_ma60, predictor_v1_ic, test_data_ma60, "V1-IC")
    if r_v1_ic:
        results.append(r_v1_ic)

    print("\n[3/5] 测试 V2 IC Best...")
    r_v2_ic = test_ma60_pipeline(tokenizer_ma60, predictor_v2_ic, test_data_ma60, "V2-IC")
    if r_v2_ic:
        results.append(r_v2_ic)

    print("\n[4/5] 测试 V1 Loss Best...")
    r_v1_loss = test_ma60_pipeline(tokenizer_ma60, predictor_v1_loss, test_data_ma60, "V1-Loss")
    if r_v1_loss:
        results.append(r_v1_loss)

    print("\n[5/5] 测试 V2 Loss Best...")
    r_v2_loss = test_ma60_pipeline(tokenizer_ma60, predictor_v2_loss, test_data_ma60, "V2-Loss")
    if r_v2_loss:
        results.append(r_v2_loss)

    # 输出结果
    print("\n" + "="*70)
    print("测试结果对比")
    print("="*70)

    print("\n| 模型 | 样本数 | IC | Rank IC | 方向准确率 | MSE |")
    print("|------|--------|------|---------|-----------|------|")
    for r in results:
        print(f"| {r['name']} | {r['samples']} | {r['ic']:.4f} | {r['rank_ic']:.4f} | {r['dir_acc']:.2%} | {r['mse']:.6f} |")

    # 分组准确率
    print("\n分组方向准确率 (按实际收益率分5组):")
    print("| 组别 |", end="")
    for r in results:
        print(f" {r['name']} |", end="")
    print()
    print("|------|", end="")
    for r in results:
        print("------|", end="")
    print()
    groups = ["最差(跌)", "较差", "中等", "较好", "最优(涨)"]
    for i, g in enumerate(groups):
        print(f"| {g} |", end="")
        for r in results:
            print(f" {r['group_accs'][i]:.2%} |", end="")
        print()

    # 排名
    print("\n" + "-"*70)
    print("IC 排名:")
    sorted_by_ic = sorted(results, key=lambda x: x['ic'], reverse=True)
    for i, r in enumerate(sorted_by_ic, 1):
        print(f"  {i}. {r['name']}: IC={r['ic']:.4f}")

    print("\n方向准确率排名:")
    sorted_by_acc = sorted(results, key=lambda x: x['dir_acc'], reverse=True)
    for i, r in enumerate(sorted_by_acc, 1):
        print(f"  {i}. {r['name']}: {r['dir_acc']:.2%}")

    print("="*70)

    # 保存详细结果
    output_path = 'finetune/full_comparison_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n详细结果已保存至: {output_path}")


if __name__ == '__main__':
    main()