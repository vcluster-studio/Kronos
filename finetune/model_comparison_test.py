"""
全市场对比测试：MA60微调模型 vs 原始Pretrained模型

测试组合：
1. Kronos-mini (pretrained, full_window归一化)
2. Kronos-small (pretrained, full_window归一化)
3. Kronos-base (pretrained, full_window归一化)
4. Kronos-mini-MA60-IC (微调, sliding_ma60归一化)
5. Kronos-small-MA60 (微调, sliding_ma60归一化, group_size=4)

全市场2555股票，IC=P+3
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import spearmanr
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# 参数
LOOKBACK = 400
PRED_LEN = 10
IC_POINT = 3
CLIP = 5.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def full_window_normalize(x, clip=5.0):
    """全窗口归一化"""
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0) + 1e-5
    x_norm = (x - x_mean) / x_std
    x_norm = np.clip(x_norm, -clip, clip)
    return x_norm, x_mean, x_std


def test_pretrained_pipeline(tokenizer, model, test_data, name, max_context=512):
    """测试原始 pretrained pipeline (full_window归一化)"""
    tokenizer.to(device)
    model.to(device)
    tokenizer.eval()
    model.eval()

    feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
    pred_returns = []
    actual_returns = []

    symbols = list(test_data.keys())

    for symbol in tqdm(symbols, desc=f"Testing {name}"):
        df = test_data[symbol]
        values = df[feature_cols].values.astype(np.float32)

        if len(values) < LOOKBACK + PRED_LEN:
            continue

        try:
            x_values = values[-(LOOKBACK + PRED_LEN):-PRED_LEN]
            x_norm, mean, std = full_window_normalize(x_values)

            x_ts = df.index[-(LOOKBACK + PRED_LEN):-PRED_LEN]
            y_ts = df.index[-PRED_LEN:]

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            baseline_close = values[-PRED_LEN - 1, 3]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=max_context, pred_len=PRED_LEN,
                    clip=CLIP, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                pred_close_norm = preds[0, -PRED_LEN:, 3]
                pred_close = pred_close_norm * std[3] + mean[3]
                pred_ret = (pred_close[IC_POINT - 1] - baseline_close) / baseline_close

            actual_close = values[-PRED_LEN:, 3]
            actual_ret = (actual_close[IC_POINT - 1] - baseline_close) / baseline_close

            pred_returns.append(pred_ret)
            actual_returns.append(actual_ret)

        except Exception as e:
            continue

    return analyze_results(pred_returns, actual_returns, name)


def test_ma60_pipeline(tokenizer, model, test_data, name, max_context=2048):
    """测试 MA60 pipeline (预归一化数据)"""
    tokenizer.to(device)
    model.to(device)
    tokenizer.eval()
    model.eval()

    pred_returns = []
    actual_returns = []

    symbols = list(test_data.keys())

    for symbol in tqdm(symbols, desc=f"Testing {name}"):
        data = test_data[symbol]
        seq_len = len(data['normalized'])

        if seq_len < LOOKBACK + PRED_LEN:
            continue

        try:
            full_len = LOOKBACK + PRED_LEN
            end_idx = seq_len

            x_norm_full = data['normalized'][end_idx - full_len:end_idx].astype(np.float32)
            means_full = data['means'][end_idx - full_len:end_idx]
            stds_full = data['stds'][end_idx - full_len:end_idx]
            timestamps_full = data['index'][end_idx - full_len:end_idx]

            x_norm = x_norm_full[:LOOKBACK]
            x_ts = timestamps_full[:LOOKBACK]
            y_ts = timestamps_full[LOOKBACK:]

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            original_close = data['original'][:, 3]
            baseline_close = original_close[end_idx - PRED_LEN - 1]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=max_context, pred_len=PRED_LEN,
                    clip=CLIP, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                pred_close_norm = preds[0, -PRED_LEN:, 3]
                pred_close = pred_close_norm * stds_full[LOOKBACK:, 3] + means_full[LOOKBACK:, 3]
                pred_ret = (pred_close[IC_POINT - 1] - baseline_close) / baseline_close

            actual_close = original_close[end_idx - PRED_LEN:]
            actual_ret = (actual_close[IC_POINT - 1] - baseline_close) / baseline_close

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

    ic = np.corrcoef(pred_returns, actual_returns)[0, 1]
    rank_ic, _ = spearmanr(pred_returns, actual_returns)
    dir_acc = np.mean((pred_returns > 0) == (actual_returns > 0))
    mse = np.mean((pred_returns - actual_returns) ** 2)

    sorted_idx = np.argsort(actual_returns)
    n = len(sorted_idx)
    group_size = n // 5

    group_accs = []
    for i in range(5):
        group_idx = sorted_idx[i*group_size:(i+1)*group_size]
        group_pred = pred_returns[group_idx]
        group_actual = actual_returns[group_idx]
        if i == 0:
            acc = np.mean(group_pred < 0)
        elif i == 4:
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
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model_comparison_{timestamp}.log")

    def log_print(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log_print("=" * 70)
    log_print("全市场对比测试：MA60微调 vs Pretrained原始模型")
    log_print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"设备: {device}")
    log_print("=" * 70)

    # 加载原始测试数据
    log_print("\n加载原始测试数据...")
    with open('finetune/data/processed_datasets/test_data.pkl', 'rb') as f:
        test_data_orig = pickle.load(f)
    log_print(f"  原始数据: {len(test_data_orig)} 股票")

    # 加载MA60测试数据
    log_print("加载MA60测试数据...")
    with open('finetune/data/processed_datasets_ma60/test_data.pkl', 'rb') as f:
        test_data_ma60 = pickle.load(f)
    log_print(f"  MA60数据: {len(test_data_ma60)} 股票")

    results = []

    # === 1. Kronos-mini (pretrained) ===
    log_print("\n[1/4] 测试 Kronos-mini (pretrained)...")
    tk_mini = KronosTokenizer.from_pretrained('pretrained/Kronos-Tokenizer-2k')
    m_mini = Kronos.from_pretrained('pretrained/Kronos-mini')
    r = test_pretrained_pipeline(tk_mini, m_mini, test_data_orig, "Kronos-mini")
    if r:
        results.append(r)
        log_print(f"  IC={r['ic']:.4f}, DirAcc={r['dir_acc']:.2%}, Samples={r['samples']}")
    del tk_mini, m_mini
    torch.cuda.empty_cache()

    # === 2. Kronos-small (pretrained) ===
    log_print("\n[2/4] 测试 Kronos-small (pretrained)...")
    tk_small = KronosTokenizer.from_pretrained('pretrained/Kronos-Tokenizer-2k')
    m_small = Kronos.from_pretrained('pretrained/Kronos-small')
    r = test_pretrained_pipeline(tk_small, m_small, test_data_orig, "Kronos-small")
    if r:
        results.append(r)
        log_print(f"  IC={r['ic']:.4f}, DirAcc={r['dir_acc']:.2%}, Samples={r['samples']}")
    del tk_small, m_small
    torch.cuda.empty_cache()

    # === 3. Kronos-base (pretrained) ===
    log_print("\n[3/4] 测试 Kronos-base (pretrained)...")
    tk_base = KronosTokenizer.from_pretrained('pretrained/Kronos-Tokenizer-base')
    m_base = Kronos.from_pretrained('pretrained/Kronos-base')
    # base模型max_context=512，lookback=400已经超出，需要截断
    r = test_pretrained_pipeline(tk_base, m_base, test_data_orig, "Kronos-base", max_context=512)
    if r:
        results.append(r)
        log_print(f"  IC={r['ic']:.4f}, DirAcc={r['dir_acc']:.2%}, Samples={r['samples']}")
    del tk_base, m_base
    torch.cuda.empty_cache()

    # === 4. MA60-mini-IC (微调) ===
    log_print("\n[4/5] 测试 Kronos-mini-MA60-IC (微调)...")
    tk_ma60 = KronosTokenizer.from_pretrained('final_models/Kronos-Tokenizer-2k-MA60')
    m_ma60 = Kronos.from_pretrained('final_models/Kronos-mini-MA60')
    r = test_ma60_pipeline(tk_ma60, m_ma60, test_data_ma60, "MA60-mini-IC")
    if r:
        results.append(r)
        log_print(f"  IC={r['ic']:.4f}, DirAcc={r['dir_acc']:.2%}, Samples={r['samples']}")
    del tk_ma60, m_ma60
    torch.cuda.empty_cache()

    # === 5. MA60-small (微调, group_size=4) ===
    log_print("\n[5/5] 测试 Kronos-small-MA60 (微调, group_size=4)...")
    tk_ma60_base = KronosTokenizer.from_pretrained('outputs/models/ma60_tokenizer_base_v1/checkpoints/best_model')
    m_ma60_small = Kronos.from_pretrained('outputs/models/ma60_predictor_small_v2/checkpoints/best_ic_model')
    r = test_ma60_pipeline(tk_ma60_base, m_ma60_small, test_data_ma60, "MA60-small", max_context=512)
    if r:
        results.append(r)
        log_print(f"  IC={r['ic']:.4f}, DirAcc={r['dir_acc']:.2%}, Samples={r['samples']}")
    del tk_ma60_base, m_ma60_small
    torch.cuda.empty_cache()

    # 输出结果
    log_print("\n" + "=" * 70)
    log_print("测试结果对比")
    log_print("=" * 70)

    log_print("\n| 模型 | 样本数 | IC | Rank IC | 方向准确率 | MSE |")
    log_print("|------|--------|------|---------|-----------|------|")
    for r in results:
        log_print(f"| {r['name']} | {r['samples']} | {r['ic']:.4f} | {r['rank_ic']:.4f} | {r['dir_acc']:.2%} | {r['mse']:.6f} |")

    # 分组准确率
    log_print("\n分组方向准确率 (按实际收益率分5组):")
    header = "| 组别 |"
    sep = "|------|"
    for r in results:
        header += f" {r['name']} |"
        sep += "------|"
    log_print(header)
    log_print(sep)
    groups = ["最差(跌)", "较差", "中等", "较好", "最优(涨)"]
    for i, g in enumerate(groups):
        row = f"| {g} |"
        for r in results:
            row += f" {r['group_accs'][i]:.2%} |"
        log_print(row)

    # IC排名
    log_print("\n" + "-" * 70)
    log_print("IC 排名:")
    sorted_by_ic = sorted(results, key=lambda x: x['ic'], reverse=True)
    for i, r in enumerate(sorted_by_ic, 1):
        log_print(f"  {i}. {r['name']}: IC={r['ic']:.4f}, DirAcc={r['dir_acc']:.2%}")

    log_print(f"\n日志文件: {log_file}")
    log_print("=" * 70)

    # 保存结果
    output_path = 'finetune/model_comparison_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    log_print(f"详细结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
