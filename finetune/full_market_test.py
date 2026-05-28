"""
全市场全时段测试 - MA60 Pipeline V1 vs V2

对全部测试数据（2555股票，约411时间步）进行全面测试
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
TOKENIZER_MA60 = 'outputs/models/ma60_tokenizer_v1/checkpoints/best_model'
PREDICTOR_V1 = 'outputs/models/ma60_predictor_v1/checkpoints/best_ic_model'
PREDICTOR_V2 = 'outputs/models/ma60_predictor_v2/checkpoints/best_ic_model'

# 测试数据
TEST_DATA = 'finetune/data/processed_datasets_ma60/test_data.pkl'

# 参数
lookback = 400
pred_len = 10
ic_point = 3
clip = 5.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def test_full_market(predictor, tokenizer, test_data, name):
    """全市场全时段测试"""
    predictor.to(device)
    tokenizer.to(device)
    predictor.eval()
    tokenizer.eval()

    all_pred_returns = []
    all_actual_returns = []

    # 每只股票取多个时间点测试
    symbols = list(test_data.keys())

    for symbol in tqdm(symbols, desc=f"Testing {name}"):
        data = test_data[symbol]
        seq_len = len(data['normalized'])

        if seq_len < lookback + pred_len:
            continue

        # 取最后可用的时间点
        end_idx = seq_len

        try:
            full_len = lookback + pred_len
            x_norm = data['normalized'][end_idx - full_len:end_idx].astype(np.float32)
            means = data['means'][end_idx - full_len:end_idx]
            stds = data['stds'][end_idx - full_len:end_idx]
            timestamps = data['index'][end_idx - full_len:end_idx]

            x_ts = timestamps[:lookback]
            y_ts = timestamps[lookback:]

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            original_close = data['original'][:, 3]
            baseline_close = original_close[end_idx - pred_len - 1]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm[:lookback]).unsqueeze(0).to(device)
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
                pred_close = pred_close_norm * stds[lookback:, 3] + means[lookback:, 3]
                pred_ret = (pred_close[ic_point - 1] - baseline_close) / baseline_close

            actual_close = original_close[end_idx - pred_len:]
            actual_ret = (actual_close[ic_point - 1] - baseline_close) / baseline_close

            all_pred_returns.append(pred_ret)
            all_actual_returns.append(actual_ret)

        except Exception as e:
            continue

    all_pred_returns = np.array(all_pred_returns)
    all_actual_returns = np.array(all_actual_returns)

    # 计算指标
    ic = np.corrcoef(all_pred_returns, all_actual_returns)[0, 1]
    rank_ic, _ = spearmanr(all_pred_returns, all_actual_returns)
    dir_acc = np.mean((all_pred_returns > 0) == (all_actual_returns > 0))
    mse = np.mean((all_pred_returns - all_actual_returns) ** 2)

    # 分组分析
    # 按 actual_returns 分5组，看预测准确率
    sorted_idx = np.argsort(all_actual_returns)
    n = len(sorted_idx)
    group_size = n // 5

    group_accs = []
    for i in range(5):
        group_idx = sorted_idx[i*group_size:(i+1)*group_size]
        group_pred = all_pred_returns[group_idx]
        group_actual = all_actual_returns[group_idx]
        # 组内方向准确率
        if i == 0:  # 最差组（跌最多）
            acc = np.mean(group_pred < 0)  # 预测跌
        elif i == 4:  # 最优组（涨最多）
            acc = np.mean(group_pred > 0)  # 预测涨
        else:
            acc = np.mean((group_pred > 0) == (group_actual > 0))
        group_accs.append(acc)

    return {
        'name': name,
        'samples': len(all_pred_returns),
        'ic': ic,
        'rank_ic': rank_ic,
        'dir_acc': dir_acc,
        'mse': mse,
        'group_accs': group_accs,  # [最差组, ..., 最优组]
        'pred_returns': all_pred_returns,
        'actual_returns': all_actual_returns
    }

def main():
    print("="*60)
    print("全市场全时段测试 - MA60 Pipeline")
    print("="*60)

    # 加载模型
    print("\n加载模型...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_MA60)
    predictor_v1 = Kronos.from_pretrained(PREDICTOR_V1)
    predictor_v2 = Kronos.from_pretrained(PREDICTOR_V2)

    # 加载测试数据
    print("\n加载测试数据...")
    with open(TEST_DATA, 'rb') as f:
        test_data = pickle.load(f)
    print(f"股票数: {len(test_data)}")

    # 测试 V1
    print("\n测试 V1 (训练 IC=0.2602)...")
    result_v1 = test_full_market(predictor_v1, tokenizer, test_data, "V1")

    # 测试 V2
    print("\n测试 V2 (训练 IC=0.2944)...")
    result_v2 = test_full_market(predictor_v2, tokenizer, test_data, "V2")

    # 输出结果
    print("\n" + "="*60)
    print("测试结果对比")
    print("="*60)

    print("\n| 模型 | 样本数 | IC | Rank IC | 方向准确率 | MSE |")
    print("|------|--------|------|---------|-----------|------|")
    print(f"| V1 (IC=0.2602) | {result_v1['samples']} | {result_v1['ic']:.4f} | {result_v1['rank_ic']:.4f} | {result_v1['dir_acc']:.2%} | {result_v1['mse']:.6f} |")
    print(f"| V2 (IC=0.2944) | {result_v2['samples']} | {result_v2['ic']:.4f} | {result_v2['rank_ic']:.4f} | {result_v2['dir_acc']:.2%} | {result_v2['mse']:.6f} |")

    # 分组准确率
    print("\n分组方向准确率 (按实际收益率分5组):")
    print("| 组别 | V1准确率 | V2准确率 |")
    print("|------|----------|----------|")
    groups = ["最差(跌最多)", "较差", "中等", "较好", "最优(涨最多)"]
    for i, g in enumerate(groups):
        print(f"| {g} | {result_v1['group_accs'][i]:.2%} | {result_v2['group_accs'][i]:.2%} |")

    # 结论
    print("\n" + "-"*60)
    if result_v1['ic'] > result_v2['ic']:
        print(f"结论: V1 测试效果更好 (IC={result_v1['ic']:.4f} > {result_v2['ic']:.4f})")
    else:
        print(f"结论: V2 测试效果更好 (IC={result_v2['ic']:.4f} > {result_v1['ic']:.4f})")

    print("="*60)

if __name__ == '__main__':
    main()