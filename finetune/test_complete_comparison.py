"""
完整对比测试：原始模型 vs 微调模型 × 三种归一化方式

测试组合：
- pretrained + full_window
- pretrained + sliding_ma20
- pretrained + sliding_ma60
- final_models + full_window
- final_models + sliding_ma20
- final_models + sliding_ma60
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr
import torch

# Add project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference


def full_window_normalize(x, clip=5.0):
    """
    全窗口归一化：使用整个lookback窗口的mean/std
    """
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0) + 1e-5
    x_norm = (x - x_mean) / x_std
    x_norm = np.clip(x_norm, -clip, clip)
    return x_norm, x_mean, x_std


def sliding_ma_normalize(x, ma_window=20, clip=5.0):
    """
    滑动MA归一化：每个点根据自己的前N步计算mean/std
    """
    seq_len = len(x)

    x_norm = np.zeros_like(x)
    means = np.zeros_like(x)
    stds = np.zeros_like(x)

    for t in range(seq_len):
        start_idx = max(0, t - ma_window)
        window_data = x[start_idx:t]

        if len(window_data) == 0:
            x_mean_t = x[t]
            x_std_t = np.ones(6) * 1e-5
        else:
            x_mean_t = np.mean(window_data, axis=0)
            x_std_t = np.std(window_data, axis=0) + 1e-5

        means[t] = x_mean_t
        stds[t] = x_std_t
        x_norm[t] = (x[t] - x_mean_t) / x_std_t

    x_norm = np.clip(x_norm, -clip, clip)
    return x_norm, means[-1], stds[-1]  # 返回最后一点的参数用于反归一化


def calc_time_stamps(timestamps):
    """Calculate time features from timestamps."""
    if isinstance(timestamps, pd.DatetimeIndex):
        time_df = pd.DataFrame()
        time_df['minute'] = timestamps.minute
        time_df['hour'] = timestamps.hour
        time_df['weekday'] = timestamps.weekday
        time_df['day'] = timestamps.day
        time_df['month'] = timestamps.month
    else:
        time_df = pd.DataFrame()
        time_df['minute'] = timestamps.dt.minute
        time_df['hour'] = timestamps.dt.hour
        time_df['weekday'] = timestamps.dt.weekday
        time_df['day'] = timestamps.dt.day
        time_df['month'] = timestamps.dt.month
    return time_df


def test_single_symbol(df, tokenizer, model, device, lookback, pred_len, ic_point, norm_mode, ma_window, feature_cols):
    """Test a single symbol with given normalization."""
    if len(df) < lookback + pred_len:
        return None

    try:
        # 时间戳
        dates = df.index[-(lookback + pred_len):]
        dates = pd.to_datetime(dates)
        x_timestamp = dates[:lookback]
        y_timestamp = dates[lookback:]

        # 时间特征
        time_df_x = calc_time_stamps(x_timestamp)
        time_df_y = calc_time_stamps(y_timestamp)

        # 数据准备
        values = df[feature_cols].values.astype(np.float32)

        # 原始数据
        full_data = values[-(lookback + pred_len):]
        x_original = full_data[:lookback]
        y_original = full_data[lookback:]

        # 基准价格
        baseline_close = x_original[-1, 3]

        # 归一化
        if norm_mode == 'full_window':
            x_norm, mean, std = full_window_normalize(x_original, clip=5.0)
        elif norm_mode == 'sliding_ma':
            x_norm, mean, std = sliding_ma_normalize(x_original, ma_window=ma_window, clip=5.0)
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
            x_stamp_tensor = torch.from_numpy(time_df_x.values.astype(np.float32)).unsqueeze(0).to(device)
            y_stamp_tensor = torch.from_numpy(time_df_y.values.astype(np.float32)).unsqueeze(0).to(device)

            preds = auto_regressive_inference(
                tokenizer, model,
                x_tensor, x_stamp_tensor, y_stamp_tensor,
                max_context=2048,
                pred_len=pred_len,
                clip=5.0,
                T=1.0, top_k=0, top_p=0.9,
                sample_count=1, verbose=False
            )

            preds = preds.squeeze(0)
            preds_denorm = preds * std + mean

        # 收益率计算
        pred_close_p3 = preds_denorm[ic_point, 3]
        pred_return = (pred_close_p3 - baseline_close) / baseline_close

        actual_close_p3 = y_original[ic_point, 3]
        actual_return = (actual_close_p3 - baseline_close) / baseline_close

        return pred_return, actual_return

    except Exception as e:
        return None


def main():
    """Main test function."""
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"complete_comparison_{timestamp}.log")

    def log_print(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log_print("=" * 70)
    log_print("完整对比测试：原始模型 vs 微调模型 × 三种归一化")
    log_print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"日志文件: {log_file}")
    log_print("=" * 70)

    # 测试参数
    lookback = 400
    pred_len = 10
    n_samples = 100
    ic_point = 3  # P+3
    feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']

    log_print(f"\n测试参数:")
    log_print(f"  lookback: {lookback}")
    log_print(f"  pred_len: {pred_len}")
    log_print(f"  n_samples: {n_samples}")
    log_print(f"  IC计算点: P+{ic_point}")

    # 加载测试数据
    test_path = os.path.join(project_root, "finetune/data/processed_datasets/test_data.pkl")
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    log_print(f"\n加载测试数据...")
    log_print(f"股票数: {len(test_data)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Device: {device}")

    # 模型配置
    models_config = [
        {
            'name': 'pretrained',
            'tokenizer_path': 'pretrained/Kronos-Tokenizer-2k',
            'model_path': 'pretrained/Kronos-mini',
        },
        {
            'name': 'final_models',
            'tokenizer_path': 'final_models/Kronos-Tokenizer-2k',
            'model_path': 'final_models/Kronos-mini',
        },
    ]

    # 归一化配置
    norms_config = [
        {'mode': 'full_window', 'ma_window': None, 'label': 'full_window'},
        {'mode': 'sliding_ma', 'ma_window': 20, 'label': 'sliding_ma20'},
        {'mode': 'sliding_ma', 'ma_window': 60, 'label': 'sliding_ma60'},
    ]

    symbols = list(test_data.keys())[:n_samples]
    all_results = []

    # 遍历所有组合
    for model_cfg in models_config:
        log_print(f"\n{'='*70}")
        log_print(f"加载模型: {model_cfg['name']}")
        log_print(f"Tokenizer: {model_cfg['tokenizer_path']}")
        log_print(f"Predictor: {model_cfg['model_path']}")
        log_print("=" * 70)

        tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, model_cfg['tokenizer_path']))
        model = Kronos.from_pretrained(os.path.join(project_root, model_cfg['model_path']))
        tokenizer = tokenizer.eval().to(device)
        model = model.eval().to(device)

        for norm_cfg in norms_config:
            log_print(f"\n测试: {model_cfg['name']} + {norm_cfg['label']}")

            predictions = []
            actuals = []

            for i, symbol in enumerate(symbols):
                df = test_data[symbol]
                result = test_single_symbol(
                    df, tokenizer, model, device,
                    lookback, pred_len, ic_point,
                    norm_cfg['mode'], norm_cfg['ma_window'],
                    feature_cols
                )

                if result is not None:
                    predictions.append(result[0])
                    actuals.append(result[1])

            # 计算指标
            if len(predictions) > 0:
                predictions = np.array(predictions)
                actuals = np.array(actuals)

                ic, _ = spearmanr(predictions, actuals)

                pred_direction = predictions > 0
                actual_direction = actuals > 0
                dir_acc = np.mean(pred_direction == actual_direction) * 100

                mse = np.mean((predictions - actuals) ** 2)

                log_print(f"  有效样本: {len(predictions)}")
                log_print(f"  方向准确率: {dir_acc:.2f}%")
                log_print(f"  IC: {ic:.4f}")
                log_print(f"  MSE: {mse:.6f}")

                all_results.append({
                    'model': model_cfg['name'],
                    'norm': norm_cfg['label'],
                    'samples': len(predictions),
                    'dir_acc': dir_acc,
                    'ic': ic,
                    'mse': mse
                })

    # 总结表格
    log_print(f"\n{'='*70}")
    log_print("完整对比结果")
    log_print("=" * 70)
    log_print(f"\n模型            归一化方式      Samples  DirAcc%  IC       MSE")
    log_print("-" * 70)
    for r in all_results:
        log_print(f"{r['model']:14}  {r['norm']:14}  {r['samples']:6}  {r['dir_acc']:6.2f}  {r['ic']:7.4f}  {r['mse']:10.6f}")

    # 分析结论
    log_print(f"\n{'='*70}")
    log_print("分析结论")
    log_print("=" * 70)

    # 按IC排序
    sorted_results = sorted(all_results, key=lambda x: x['ic'], reverse=True)
    log_print(f"\n按IC排序（从高到低）：")
    for i, r in enumerate(sorted_results, 1):
        log_print(f"  {i}. {r['model']}+{r['norm']}: IC={r['ic']:.4f}, DirAcc={r['dir_acc']:.2f}%")

    # 最佳组合
    best = sorted_results[0]
    log_print(f"\n最佳组合：{best['model']} + {best['norm']}")
    log_print(f"  IC: {best['ic']:.4f}")
    log_print(f"  方向准确率: {best['dir_acc']:.2f}%")

    # 归一化对比
    log_print(f"\n归一化方式对比：")
    for norm in ['full_window', 'sliding_ma20', 'sliding_ma60']:
        norm_results = [r for r in all_results if r['norm'] == norm]
        avg_ic = np.mean([r['ic'] for r in norm_results])
        avg_dir = np.mean([r['dir_acc'] for r in norm_results])
        log_print(f"  {norm}: 平均IC={avg_ic:.4f}, 平均DirAcc={avg_dir:.2f}%")

    # 模型对比
    log_print(f"\n模型对比：")
    for model in ['pretrained', 'final_models']:
        model_results = [r for r in all_results if r['model'] == model]
        avg_ic = np.mean([r['ic'] for r in model_results])
        avg_dir = np.mean([r['dir_acc'] for r in model_results])
        log_print(f"  {model}: 平均IC={avg_ic:.4f}, 平均DirAcc={avg_dir:.2f}%")

    log_print("\n测试完成！")


if __name__ == '__main__':
    main()