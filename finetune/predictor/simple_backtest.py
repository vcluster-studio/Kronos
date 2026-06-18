"""
批量回测脚本 - 使用 final_test_data.pkl 验证因子分值效果

用法：
    python finetune/predictor/simple_backtest.py --n-samples 100   # 快速测试
    python finetune/predictor/simple_backtest.py --n-samples -1    # 全量评估
    python finetune/predictor/simple_backtest.py --batch-size 32  # 指定batch大小
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from safetensors.torch import load_file
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference

# ============================================================================
# 配置
# ============================================================================

MODEL_PATH = 'outputs/models/mode_mini_lb400/checkpoints/best_combined_model'
TOKENIZER_PATH = 'outputs/tokenizers/final/2k-MA60'
TEST_DATA_PATH = 'finetune/data/ma60_norm/block_lb400_pd10/final_test_data.pkl'

# 因子参数
STEP_INDEX = 3  # step+3 作为预测涨幅
SIGNAL_CENTER = 0.084  # sigmoid 中心
SIGNAL_STEEPNESS = 21  # sigmoid 陡度

LOOKBACK = 400
PREDICT = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sigmoid_score(gain, center=0.084, steepness=21):
    """将涨幅转换为sigmoid分值"""
    return 1 / (1 + np.exp(-steepness * (gain - center)))


def prepare_batch_inputs(test_data, window_list, lookback=LOOKBACK, predict=PREDICT):
    """准备批量输入数据"""
    x_norms = []
    x_stamps = []
    y_stamps = []
    means_list = []
    stds_list = []
    originals_list = []
    valid_indices = []

    for idx, (sym, start) in enumerate(window_list):
        try:
            d = test_data[sym]
            normalized = d['normalized']
            original = d['original']
            means = d['means']
            stds = d['stds']
            timestamps = d['index']

            end = start + lookback + predict
            if end > len(normalized):
                continue

            # 输入数据
            x_norm = normalized[start:start + lookback]
            x_ts = timestamps[start:start + lookback]
            y_ts = timestamps[start + lookback:start + lookback + predict]

            # 时间戳
            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            x_norms.append(x_norm)
            x_stamps.append(x_stamp)
            y_stamps.append(y_stamp)
            means_list.append(means[start + lookback:start + lookback + predict])
            stds_list.append(stds[start + lookback:start + lookback + predict])
            originals_list.append(original[start + lookback:start + lookback + predict])
            valid_indices.append(idx)

        except Exception:
            continue

    return {
        'x_norms': np.array(x_norms, dtype=np.float32),
        'x_stamps': np.array(x_stamps, dtype=np.float32),
        'y_stamps': np.array(y_stamps, dtype=np.float32),
        'means': np.array(means_list, dtype=np.float32),
        'stds': np.array(stds_list, dtype=np.float32),
        'originals': np.array(originals_list, dtype=np.float32),
        'valid_indices': valid_indices,
        'window_list': window_list,
    }


def batch_predict(tokenizer, model, batch_data, batch_size=32, device=DEVICE):
    """批量预测"""
    results = []
    n_samples = len(batch_data['x_norms'])

    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Batch predicting"):
        end_idx = min(start_idx + batch_size, n_samples)
        actual_batch_size = end_idx - start_idx

        # 准备batch tensor
        x_batch = torch.from_numpy(batch_data['x_norms'][start_idx:end_idx]).to(device)
        x_stamp_batch = torch.from_numpy(batch_data['x_stamps'][start_idx:end_idx]).to(device)
        y_stamp_batch = torch.from_numpy(batch_data['y_stamps'][start_idx:end_idx]).to(device)

        # 批量推理
        with torch.no_grad():
            preds = auto_regressive_inference(
                tokenizer, model,
                x_batch, x_stamp_batch, y_stamp_batch,
                max_context=2048, pred_len=PREDICT,
                clip=5.0, T=1.0, top_k=0, top_p=0.9,
                sample_count=1, verbose=False
            )

        # 解码每个样本
        for i in range(actual_batch_size):
            batch_i = start_idx + i
            pred_norm = preds[i, LOOKBACK:LOOKBACK + PREDICT, :]
            if hasattr(pred_norm, 'cpu'):
                pred_norm = pred_norm.cpu().numpy()

            pred_raw = pred_norm * batch_data['stds'][batch_i] + batch_data['means'][batch_i]
            actual = batch_data['originals'][batch_i]

            # 当前价格 (lookback最后一步)
            # 注意: originals[i] 是预测区间，当前价格需要从原窗口获取
            # 这里用预测区间第一步的前一步，即 actual[0] 的前一步
            # 实际上我们需要从 test_data 获取，但为了简化，用 batch_data 中的信息

            results.append({
                'pred_raw': pred_raw,
                'actual': actual,
                'batch_idx': batch_i,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description='Batch Backtest')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples (-1 for full)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--step', type=int, default=3,
                        help='Step index for gain calculation')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--model', type=str, default='outputs/models/mode_mini_lb400/checkpoints/best_combined_model',
                        help='Model checkpoint path')
    args = parser.parse_args()

    model_path_arg = args.model
    step_idx = args.step
    n_samples = args.n_samples
    batch_size = args.batch_size

    print("=" * 60)
    print("Batch Backtest - Factor Score Validation")
    print("=" * 60)
    print(f"Model: {model_path_arg}")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Step index: +{step_idx}")
    print(f"Signal center: {SIGNAL_CENTER}")
    print(f"Signal steepness: {SIGNAL_STEEPNESS}")
    print(f"Samples: {n_samples if n_samples > 0 else 'FULL'}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer_path = os.path.join(project_root, TOKENIZER_PATH)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(DEVICE)
    print(f"Tokenizer loaded")

    # 加载模型
    print("\nLoading model...")
    model_path = os.path.join(project_root, 'pretrained/Kronos-mini')
    model = Kronos.from_pretrained(model_path)
    model.eval().to(DEVICE)

    # 加载 checkpoint
    safetensors_path = os.path.join(project_root, model_path_arg, 'model.safetensors')
    if not os.path.exists(safetensors_path):
        # 尝试直接路径
        safetensors_path = os.path.join(project_root, model_path_arg)
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded: {safetensors_path}")

    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model size: {model_size:.2f}M")

    # 加载测试数据
    print("\nLoading test data...")
    test_path = os.path.join(project_root, TEST_DATA_PATH)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # 构建窗口索引列表
    all_windows = []
    for sym, d in test_data.items():
        windows = d.get('windows', [])
        for start in windows:
            all_windows.append((sym, start))

    total_windows = len(all_windows)
    print(f"Total windows: {total_windows}")

    # 抽样
    rng = np.random.RandomState(args.seed)
    if n_samples > 0 and n_samples < total_windows:
        sampled_indices = rng.choice(total_windows, size=n_samples, replace=False)
        sampled_windows = [all_windows[i] for i in sampled_indices]
        print(f"Sampled: {n_samples} windows")
    else:
        sampled_windows = all_windows
        print(f"Using ALL windows")

    # 准备批量数据
    print("\nPreparing batch data...")
    batch_data = prepare_batch_inputs(test_data, sampled_windows)
    valid_count = len(batch_data['x_norms'])
    print(f"Valid windows: {valid_count}")

    # 批量预测
    print("\nPredicting...")
    pred_results = batch_predict(tokenizer, model, batch_data, batch_size=batch_size)

    # 计算涨幅和分值
    print("\nCalculating gains...")
    results = []

    for i, res in enumerate(pred_results):
        pred_raw = res['pred_raw']
        actual = res['actual']
        batch_idx = res['batch_idx']

        # 获取原始窗口信息以计算当前价格
        valid_idx = batch_data['valid_indices'][batch_idx]
        sym, start = sampled_windows[valid_idx]
        d = test_data[sym]

        # 当前价格 = lookback最后一步的close
        current_close = d['original'][start + LOOKBACK - 1, 3]

        # 预测涨幅
        pred_close = pred_raw[step_idx, 3]
        pred_gain = (pred_close - current_close) / current_close

        # 实际涨幅
        actual_close = actual[step_idx, 3]
        actual_gain = (actual_close - current_close) / current_close

        # sigmoid分值
        score = sigmoid_score(pred_gain)

        results.append({
            'symbol': sym,
            'pred_gain': pred_gain,
            'actual_gain': actual_gain,
            'score': score,
        })

    print(f"Completed: {len(results)} predictions")

    # 统计分析
    if len(results) == 0:
        print("No valid results!")
        return

    df = pd.DataFrame(results)

    # IC计算
    pred_gains = df['pred_gain'].values
    actual_gains = df['actual_gain'].values

    valid_mask = np.isfinite(pred_gains) & np.isfinite(actual_gains)
    if valid_mask.sum() < 3:
        print("Too few valid samples for IC!")
        return

    pred_gains = pred_gains[valid_mask]
    actual_gains = actual_gains[valid_mask]

    # 过滤极端值和零方差
    pred_std = np.std(pred_gains)
    actual_std = np.std(actual_gains)
    if pred_std > 1e-8 and actual_std > 1e-8:
        ic = np.corrcoef(pred_gains, actual_gains)[0, 1]
        rank_ic, _ = spearmanr(pred_gains, actual_gains)
    else:
        ic = 0
        rank_ic = 0

    print(f"\n{'=' * 60}")
    print(f"[Gain IC] (step+{step_idx})")
    print(f"{'=' * 60}")
    print(f"  Pearson IC: {ic:.4f}")
    print(f"  Rank IC: {rank_ic:.4f}")

    # 分值分组分析
    print(f"\n[Score Group Analysis]")
    df['score_group'] = pd.cut(df['score'], bins=[0, 0.3, 0.5, 0.7, 1.0],
                               labels=['Low(0-30)', 'Mid(30-50)', 'High(50-70)', 'Top(70-100)'])

    for group in ['Low(0-30)', 'Mid(30-50)', 'High(50-70)', 'Top(70-100)']:
        group_df = df[df['score_group'] == group]
        if len(group_df) > 0:
            avg_actual = group_df['actual_gain'].mean()
            win_rate = (group_df['actual_gain'] > 0).mean()
            pct = len(group_df) / len(df) * 100
            print(f"  {group}: n={len(group_df)} ({pct:.1f}%), avg={avg_actual:+.2%}, win={win_rate:.1%}")

    # 高分股票表现
    print(f"\n[Score > 50%]")
    high_score = df[df['score'] > 0.5]
    if len(high_score) > 0:
        avg_actual = high_score['actual_gain'].mean()
        win_rate = (high_score['actual_gain'] > 0).mean()
        pct = len(high_score) / len(df) * 100
        print(f"  Count: {len(high_score)} ({pct:.1f}%)")
        print(f"  Avg actual: {avg_actual:+.2%}")
        print(f"  Win rate: {win_rate:.1%}")
    else:
        print("  None")

    # 分布统计
    print(f"\n[Distribution]")
    print(f"  Pred gain: mean={df['pred_gain'].mean():+.2%}, std={df['pred_gain'].std():.2%}")
    print(f"  Actual gain: mean={df['actual_gain'].mean():+.2%}, std={df['actual_gain'].std():.2%}")
    print(f"  Pred > 8.4%: {(df['pred_gain'] > SIGNAL_CENTER).mean():.1%}")
    print(f"  Actual > 0: {(df['actual_gain'] > 0).mean():.1%}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()