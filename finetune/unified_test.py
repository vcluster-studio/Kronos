"""
统一测试脚本：所有模型组合在所有分层数据集上的 IC 对比

支持多 lookback 组对比，验证不同上下文长度下的预测效果。

采样策略：固定随机种子，确保可重复
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# ============================================================================
# 待验证模型配置（新增模型时只需更新此变量）
# ============================================================================
MODEL_CONFIGS = {
    # === 原始模型 baseline ===
    'mini-2k-orig': {
        'name': 'mini (2k tok + orig pred)',
        'tokenizer': 'pretrained/Kronos-Tokenizer-2k',
        'predictor': 'pretrained/Kronos-mini',
        'params': '4.1M',
        'note': 'original 2k tokenizer baseline'
    },
    # === MA20 tokenizer + orig predictor ===
    'mini-fulltok2k-ma20-orig': {
        'name': 'mini (full 2k tok MA20 + orig pred)',
        'tokenizer': 'outputs/models/full_tokenizer_2k_v1/checkpoints/best_model',
        'predictor': 'pretrained/Kronos-mini',
        'params': '4.1M',
        'note': 'finetuned 2k tokenizer with MA20 norm + orig predictor'
    },
    # === Hidden direction training 结果 ===
    'mini-hidden-bestic': {
        'name': 'mini (hidden dir, best IC)',
        'tokenizer': 'outputs/models/full_tokenizer_2k_v1/checkpoints/best_model',
        'predictor': 'outputs/models/full_predictor_v1/checkpoints/best_ic_model',
        'params': '4.1M',
        'note': 'hidden direction training - best IC model'
    },
    'mini-hidden-bestvl': {
        'name': 'mini (hidden dir, best VL)',
        'tokenizer': 'outputs/models/full_tokenizer_2k_v1/checkpoints/best_model',
        'predictor': 'outputs/models/full_predictor_v1/checkpoints/best_model',
        'params': '4.1M',
        'note': 'hidden direction training - best val loss model'
    },
    'mini-hidden-final': {
        'name': 'mini (hidden dir, final)',
        'tokenizer': 'outputs/models/full_tokenizer_2k_v1/checkpoints/best_model',
        'predictor': 'outputs/models/full_predictor_v1/checkpoints/latest_model',
        'params': '4.1M',
        'note': 'hidden direction training - final model'
    },
    # === v7 训练结果 ===
    'mini-v7-bestvl': {
        'name': 'mini (v7, best VL)',
        'tokenizer': 'outputs/models/full_tokenizer_2k_v1/checkpoints/best_model',
        'predictor': 'outputs/models/full_predictor_v7/checkpoints/best_model',
        'params': '4.1M',
        'note': 'v7 training - best val loss (VL=2.725)'
    },
    'mini-v7-bestic': {
        'name': 'mini (v7, best IC)',
        'tokenizer': 'outputs/models/full_tokenizer_2k_v1/checkpoints/best_model',
        'predictor': 'outputs/models/full_predictor_v7/checkpoints/best_ic_model',
        'params': '4.1M',
        'note': 'v7 training - best IC (IC=0.22)'
    },
    # === 之前最佳 ===
    'mini-final-best': {
        'name': 'mini (final_models best)',
        'tokenizer': 'outputs/models/full_tokenizer_2k_v1/checkpoints/best_model',
        'predictor': 'final_models/Kronos-mini',
        'params': '4.1M',
        'note': 'previous best model from final_models'
    },
}

# 数据集配置
DATASET_CONFIGS = {
    'Full': {'path': 'finetune/data/processed_datasets', 'desc': 'Full A-share (large+mid+small)'},
    'Mid': {'path': 'finetune/data/processed_datasets_mid', 'desc': 'Mid-cap stocks'},
    'Small': {'path': 'finetune/data/processed_datasets_small', 'desc': 'Small-cap stocks'},
    'Mid+Small': {'path': 'finetune/data/processed_datasets_mid_small', 'desc': 'Mid+Small mixed'},
}

# 测试参数
TEST_PARAMS = {
    'seed': 42,
    'max_samples': 300,
    'predict': 10,
}

# 多 lookback 对比组
LOOKBACK_GROUPS = [100, 200, 400]

def load_test_data(data_path):
    test_path = f"{data_path}/test_data.pkl"
    if not os.path.exists(test_path):
        test_path = f"{data_path}/val_data.pkl"
    with open(test_path, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_timestamps(df, lookback=90, predict=10):
    if 'datetime' in df.columns:
        dates = df['datetime'].values[-(lookback + predict):]
    else:
        dates = pd.date_range(start='2025-01-01', periods=lookback + predict, freq='1min')
    dates = pd.to_datetime(dates)
    x_timestamp = dates[:lookback]
    y_timestamp = dates[lookback:]

    time_df_x = pd.DataFrame()
    time_df_x['minute'] = x_timestamp.minute
    time_df_x['hour'] = x_timestamp.hour
    time_df_x['weekday'] = x_timestamp.weekday
    time_df_x['day'] = x_timestamp.day
    time_df_x['month'] = x_timestamp.month

    time_df_y = pd.DataFrame()
    time_df_y['minute'] = y_timestamp.minute
    time_df_y['hour'] = y_timestamp.hour
    time_df_y['weekday'] = y_timestamp.weekday
    time_df_y['day'] = y_timestamp.day
    time_df_y['month'] = y_timestamp.month

    return time_df_x.values, time_df_y.values

def prepare_sample(df, lookback=90, predict=10):
    values = df[['open', 'high', 'low', 'close', 'vol', 'amt']].values.astype(np.float32)
    if len(values) < lookback + predict:
        return None

    # === MA20 归一化（市场交易者视角）===
    # 价格列（open, high, low, close）共享 close 的 MA20 作为基准
    # vol 和 amt 各自用自己的 MA20
    ma_window = 20

    if lookback < ma_window:
        # 降级为全窗口归一化
        x = values[-(lookback + predict):-predict]
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0) + 1e-5
    else:
        # MA20 归一化：用 lookback 窗口最后 20 步计算
        window_data = values[-(lookback + predict):-predict]  # lookback 窗口数据
        ma_data = window_data[-ma_window:]  # 取最后 20 步

        # 价格列共享 close 的 MA20
        close_ma = np.mean(ma_data[:, 3])
        close_std = np.std(ma_data[:, 3]) + 1e-5

        # vol 用自己的 MA20
        vol_ma = np.mean(ma_data[:, 4])
        vol_std = np.std(ma_data[:, 4]) + 1e-5

        # amt 用自己的 MA20
        amt_ma = np.mean(ma_data[:, 5])
        amt_std = np.std(ma_data[:, 5]) + 1e-5

        mean = np.array([close_ma, close_ma, close_ma, close_ma, vol_ma, amt_ma])
        std = np.array([close_std, close_std, close_std, close_std, vol_std, amt_std])

        # 输入窗口
        x = window_data

    x = (x - mean) / std
    x = np.clip(x, -5, 5)

    # 基准值：预测窗口前的最后一个close（即 lookback 窗口的最后一个点）
    baseline_close = values[-predict - 1, 3]

    # 实际值：预测窗口各点的close（用于对比预测）
    actual_close_series = values[-predict:, 3]  # 预测窗口的10个实际close

    # 实际收益率：预测窗口最后一个close相对于基准的变化
    actual_return = (actual_close_series[-1] - baseline_close) / baseline_close

    x_stamp, y_stamp = prepare_timestamps(df, lookback, predict)

    norm_params = {
        'mean': mean,
        'std': std,
        'baseline_close': baseline_close,
        'actual_close_series': actual_close_series,
    }

    return x, x_stamp, y_stamp, actual_return, norm_params

def evaluate_combo(tokenizer, predictor, data, device, max_samples=300, seed=42, lookback=90, predict=10):
    tokenizer.to(device)
    predictor.to(device)
    tokenizer.eval()
    predictor.eval()

    # 根据 lookback 确定 max_context
    max_context = 2048 if lookback > 512 else 512

    # 固定采样顺序
    symbols = sorted(list(data.keys()))
    np.random.seed(seed)
    sampled_symbols = np.random.choice(symbols, size=min(max_samples, len(symbols)), replace=False)

    # 存储各种数据
    all_pred_series = []  # 预测序列
    all_actual_series = []  # 实际序列
    all_returns = {'3': [], '5': [], '10': []}  # 特定点收益率
    all_actual_returns = {'3': [], '5': [], '10': []}
    n_samples = 0

    for symbol in sampled_symbols:
        df = data[symbol]
        sample = prepare_sample(df, lookback, predict)
        if sample is None:
            continue

        x, x_stamp, y_stamp, actual_return, norm_params = sample

        try:
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                x_stamp_tensor = torch.tensor(x_stamp, dtype=torch.float32).unsqueeze(0).to(device)
                y_stamp_tensor = torch.tensor(y_stamp, dtype=torch.float32).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, predictor,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=max_context, pred_len=predict,
                    clip=5, T=0.6, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                # 预测序列（反归一化）- 只取最后 predict 个时间步
                pred_series = preds[0, -predict:, 3]  # 取最后10个预测close
                pred_series_denorm = pred_series * norm_params['std'][3] + norm_params['mean'][3]

                # 实际序列
                actual_series = norm_params['actual_close_series']

                # 基准值
                baseline_close = norm_params['baseline_close']

                # 序列收益率（相对于基准）
                pred_returns = (pred_series_denorm - baseline_close) / baseline_close
                actual_returns = (actual_series - baseline_close) / baseline_close

                all_pred_series.append(pred_returns)
                all_actual_series.append(actual_returns)

                # 特定点收益率（+3, +5, +10）
                for point in ['3', '5', '10']:
                    idx = int(point) - 1
                    all_returns[point].append(pred_returns[idx])
                    all_actual_returns[point].append(actual_returns[idx])

                n_samples += 1

        except Exception as e:
            print(f"    Error on {symbol}: {e}")
            continue

    print(f"  Collected {n_samples} samples")
    if n_samples < 20:
        return None

    # 转换为数组
    all_pred_series = np.array(all_pred_series)  # (n_samples, 10)
    all_actual_series = np.array(all_actual_series)  # (n_samples, 10)

    # ===== 序列层面指标 =====
    # MSE/MAE：每个样本序列的平均误差，再跨样本平均
    series_mse = np.mean((all_pred_series - all_actual_series) ** 2)
    series_mae = np.mean(np.abs(all_pred_series - all_actual_series))

    # Pearson：每个样本序列的 Pearson 相关，再跨样本平均
    series_pearson_list = []
    for i in range(n_samples):
        if np.std(all_pred_series[i]) > 0 and np.std(all_actual_series[i]) > 0:
            corr, _ = pearsonr(all_pred_series[i], all_actual_series[i])
            series_pearson_list.append(corr)
    series_pearson = np.mean(series_pearson_list) if series_pearson_list else 0

    # Spearman IC：每个样本序列的 Spearman 相关，再跨样本平均
    series_spearman_list = []
    for i in range(n_samples):
        if np.std(all_pred_series[i]) > 0 and np.std(all_actual_series[i]) > 0:
            corr, _ = spearmanr(all_pred_series[i], all_actual_series[i])
            series_spearman_list.append(corr)
    series_spearman = np.mean(series_spearman_list) if series_spearman_list else 0

    # ===== 特定点指标 =====
    point_metrics = {}
    for point in ['3', '5', '10']:
        preds_point = np.array(all_returns[point])
        actuals_point = np.array(all_actual_returns[point])

        # MSE/MAE
        point_mse = np.mean((preds_point - actuals_point) ** 2)
        point_mae = np.mean(np.abs(preds_point - actuals_point))

        # Spearman IC
        if np.std(preds_point) > 0 and np.std(actuals_point) > 0:
            point_ic, _ = spearmanr(preds_point, actuals_point)
        else:
            point_ic = 0

        # Direction Accuracy
        pred_dir = preds_point > 0
        actual_dir = actuals_point > 0
        point_dir_acc = np.mean(pred_dir == actual_dir)

        point_metrics[point] = {
            'mse': point_mse,
            'mae': point_mae,
            'ic': point_ic,
            'dir_acc': point_dir_acc,
        }

    return {
        'n_samples': n_samples,
        'series': {
            'mse': series_mse,
            'mae': series_mae,
            'pearson': series_pearson,
            'spearman': series_spearman,
        },
        'point': point_metrics,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Kronos Unified Test')
    parser.add_argument('--lookbacks', type=str, default=None,
                        help='Comma-separated lookback values (e.g., "100,200,400"). Default: use LOOKBACK_GROUPS')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Override max samples per dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用配置变量
    seed = TEST_PARAMS['seed']
    max_samples = args.max_samples or TEST_PARAMS['max_samples']
    predict = TEST_PARAMS['predict']

    # lookback groups
    if args.lookbacks:
        lookback_groups = [int(x) for x in args.lookbacks.split(',')]
    else:
        lookback_groups = LOOKBACK_GROUPS

    np.random.seed(seed)

    print(f"Device: {device}")
    print(f"Random seed: {seed} (fixed for reproducibility)")
    print(f"Lookback groups: {lookback_groups}")

    # Print model configurations
    print("\n" + "="*80)
    print("Model Configurations to Validate")
    print("="*80)
    for key, config in MODEL_CONFIGS.items():
        status = "enabled" if not key.startswith('#') else "disabled"
        print(f"{key}: {config['name']} ({config['params']}) - {config['note']} [{status}]")

    # 加载模型（使用 MODEL_CONFIGS）
    loaded_models = {}
    for key, config in MODEL_CONFIGS.items():
        tok_path = os.path.join(project_root, config['tokenizer'])
        pred_path = os.path.join(project_root, config['predictor'])

        if not os.path.exists(tok_path):
            print(f"Skip {config['name']}: tokenizer not found")
            continue
        if not os.path.exists(pred_path):
            print(f"Skip {config['name']}: predictor not found")
            continue

        print(f"Loading {config['name']}...")
        tokenizer = KronosTokenizer.from_pretrained(tok_path)
        predictor = Kronos.from_pretrained(pred_path)
        loaded_models[key] = {
            'name': config['name'],
            'tokenizer': tokenizer,
            'predictor': predictor,
            'params': config['params'],
            'note': config['note']
        }

    # ===== 按 lookback 分组测试 =====
    all_results = {}  # {lookback: [results_list]}

    for lookback in lookback_groups:
        print("\n" + "="*80)
        print(f"Lookback = {lookback} | Unified Model Evaluation")
        print("="*80)
        print(f"Seed: {seed}, Samples: {max_samples}, Lookback: {lookback}, Predict: {predict}")
        print("="*80)

        results = []

        for ds_name, ds_config in DATASET_CONFIGS.items():
            data_path = os.path.join(project_root, ds_config['path'])
            if not os.path.exists(data_path):
                print(f"Skip {ds_name}: dataset not found at {data_path}")
                continue

            print(f"\n--- {ds_name} ({ds_config['desc']}) ---")
            data = load_test_data(data_path)
            print(f"Stocks: {len(data)}")

            for model_key, model_config in loaded_models.items():
                metrics = evaluate_combo(
                    model_config['tokenizer'],
                    model_config['predictor'],
                    data, device,
                    max_samples, seed, lookback, predict
                )
                if metrics is None:
                    continue

                n = metrics['n_samples']
                series = metrics['series']
                point = metrics['point']

                # 打印结果
                print(f"  {model_config['name']}:")
                print(f"    Series: MSE={series['mse']:.6f}, MAE={series['mae']:.6f}, Pearson={series['pearson']:.4f}, Spearman={series['spearman']:.4f}")
                print(f"    Point+3: IC={point['3']['ic']:.4f}, DirAcc={point['3']['dir_acc']:.2%}, MSE={point['3']['mse']:.6f}")
                print(f"    Point+5: IC={point['5']['ic']:.4f}, DirAcc={point['5']['dir_acc']:.2%}, MSE={point['5']['mse']:.6f}")
                print(f"    Point+10: IC={point['10']['ic']:.4f}, DirAcc={point['10']['dir_acc']:.2%}, MSE={point['10']['mse']:.6f} ({n} samples)")

                results.append({
                    'dataset': ds_name,
                    'model': model_key,
                    'name': model_config['name'],
                    'params': model_config['params'],
                    'series': series,
                    'point': point,
                    'samples': n
                })

        all_results[lookback] = results

    # ===== 跨 lookback 对比汇总 =====
    print("\n" + "="*80)
    print("Cross-Lookback Comparison")
    print("="*80)

    ds_names = list(DATASET_CONFIGS.keys())
    model_keys = list(MODEL_CONFIGS.keys())

    # === Point+10 IC 跨 lookback 对比 ===
    print("\n--- Point+10 IC (Spearman) by Lookback ---")
    header = f"{'Dataset':<12} {'Model':<30}"
    for lb in lookback_groups:
        header += f" {'LB='+str(lb):<12}"
    print(header)
    print("-" * (12 + 30 + 12 * len(lookback_groups)))

    for ds in ds_names:
        for key in model_keys:
            if key in loaded_models:
                row = f"{ds:<12} {loaded_models[key]['name']:<30}"
                for lb in lookback_groups:
                    matching = [r for r in all_results.get(lb, []) if r['dataset'] == ds and r['model'] == key]
                    if matching:
                        ic = matching[0]['point']['10']['ic']
                        row += f" {ic:<12.4f}"
                    else:
                        row += f" {'N/A':<12}"
                print(row)

    # === Point+10 Direction Accuracy 跨 lookback 对比 ===
    print("\n--- Point+10 Direction Accuracy by Lookback ---")
    header = f"{'Dataset':<12} {'Model':<30}"
    for lb in lookback_groups:
        header += f" {'LB='+str(lb):<12}"
    print(header)
    print("-" * (12 + 30 + 12 * len(lookback_groups)))

    for ds in ds_names:
        for key in model_keys:
            if key in loaded_models:
                row = f"{ds:<12} {loaded_models[key]['name']:<30}"
                for lb in lookback_groups:
                    matching = [r for r in all_results.get(lb, []) if r['dataset'] == ds and r['model'] == key]
                    if matching:
                        da = matching[0]['point']['10']['dir_acc']
                        row += f" {da:<12.2%}"
                    else:
                        row += f" {'N/A':<12}"
                print(row)

    # === Series Spearman 跨 lookback 对比 ===
    print("\n--- Series Spearman by Lookback ---")
    header = f"{'Dataset':<12} {'Model':<30}"
    for lb in lookback_groups:
        header += f" {'LB='+str(lb):<12}"
    print(header)
    print("-" * (12 + 30 + 12 * len(lookback_groups)))

    for ds in ds_names:
        for key in model_keys:
            if key in loaded_models:
                row = f"{ds:<12} {loaded_models[key]['name']:<30}"
                for lb in lookback_groups:
                    matching = [r for r in all_results.get(lb, []) if r['dataset'] == ds and r['model'] == key]
                    if matching:
                        sp = matching[0]['series']['spearman']
                        row += f" {sp:<12.4f}"
                    else:
                        row += f" {'N/A':<12}"
                print(row)

    # === Series MSE 跨 lookback 对比 ===
    print("\n--- Series MSE by Lookback ---")
    header = f"{'Dataset':<12} {'Model':<30}"
    for lb in lookback_groups:
        header += f" {'LB='+str(lb):<12}"
    print(header)
    print("-" * (12 + 30 + 12 * len(lookback_groups)))

    for ds in ds_names:
        for key in model_keys:
            if key in loaded_models:
                row = f"{ds:<12} {loaded_models[key]['name']:<30}"
                for lb in lookback_groups:
                    matching = [r for r in all_results.get(lb, []) if r['dataset'] == ds and r['model'] == key]
                    if matching:
                        mse = matching[0]['series']['mse']
                        row += f" {mse:<12.6f}"
                    else:
                        row += f" {'N/A':<12}"
                print(row)

    # ===== 各 lookback 下的 Best Model =====
    print("\n" + "="*80)
    print("Best Model by Metric (per Lookback)")
    print("="*80)

    for lb in lookback_groups:
        results = all_results.get(lb, [])
        if not results:
            continue

        print(f"\n--- Lookback = {lb} ---")
        for ds in ds_names:
            ds_results = [r for r in results if r['dataset'] == ds]
            if ds_results:
                best_point10_ic = max(ds_results, key=lambda x: x['point']['10']['ic'])
                best_point10_da = max(ds_results, key=lambda x: x['point']['10']['dir_acc'])
                best_series_sp = max(ds_results, key=lambda x: x['series']['spearman'])
                best_mse = min(ds_results, key=lambda x: x['series']['mse'])

                print(f"  {ds}:")
                print(f"    Best Point+10 IC:     {best_point10_ic['name']} ({best_point10_ic['point']['10']['ic']:.4f})")
                print(f"    Best Point+10 DirAcc: {best_point10_da['name']} ({best_point10_da['point']['10']['dir_acc']:.2%})")
                print(f"    Best Series Spearman: {best_series_sp['name']} ({best_series_sp['series']['spearman']:.4f})")
                print(f"    Best Series MSE:      {best_mse['name']} ({best_mse['series']['mse']:.6f})")

if __name__ == '__main__':
    main()
