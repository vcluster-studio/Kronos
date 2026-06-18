"""
Mode6 模型评估 - MA60 lb200_pd10
使用测试集计算 Trajectory IC / MAE / DA

用法:
    python eval.py --model outputs/models/mode6_lb200_pd10/checkpoints/best_ic_model
"""

import os
import sys
import pickle
import argparse
import torch
import numpy as np
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']


def load_windowed_data(data_dir, lookback, predict):
    """加载预生成的窗口数据"""
    test_path = os.path.join(data_dir, 'test_data.pkl')

    with open(test_path, 'rb') as f:
        test_raw = pickle.load(f)

    window = lookback + predict
    all_data = {}
    test_indices = []

    for sym in test_raw:
        d = test_raw[sym]
        all_data[sym] = {
            'normalized': d['normalized'].astype(np.float32),
            'original': d['original'].astype(np.float32),
            'means': d['means'].astype(np.float32),
            'stds': d['stds'].astype(np.float32),
            'index': d['index'],
        }
        for start in d['windows']:
            test_indices.append((sym, int(start)))

    return all_data, test_indices


def evaluate_trajectory_ic(model, tokenizer, device, all_data, test_indices,
                           n_samples=500, lookback=200, pred_len=10, clip=5.0, rng=None):
    """Trajectory IC 测试"""

    model.eval()

    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(pred_len)]

    if rng is not None:
        sample_indices = rng.choice(len(test_indices), size=min(n_samples, len(test_indices)), replace=False)
    else:
        sample_indices = np.arange(min(n_samples, len(test_indices)))

    for idx in sample_indices:
        symbol, start_idx = test_indices[idx]
        data = all_data[symbol]
        end_idx = start_idx + lookback + pred_len

        try:
            x_norm_full = data['normalized'][start_idx:end_idx]
            means_full = data['means'][start_idx:end_idx]
            stds_full = data['stds'][start_idx:end_idx]
            timestamps_full = data['index'][start_idx:end_idx]
            orig_full = data['original'][start_idx:end_idx]

            x_norm = x_norm_full[:lookback]
            x_ts = timestamps_full[:lookback]
            y_ts = timestamps_full[lookback:]

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values, x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values, y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            baseline = orig_full[lookback - 1]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048, pred_len=pred_len,
                    clip=clip, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                pred_norm = preds[0, lookback:lookback + pred_len, :]
                pred_raw = pred_norm * stds_full[lookback:lookback + pred_len] + means_full[lookback:lookback + pred_len]
                actual = orig_full[lookback:lookback + pred_len]

            for fi, fn in enumerate(FEATURE_NAMES):
                pred_traj = pred_raw[:, fi]
                actual_traj = actual[:, fi]

                if len(pred_traj) >= 3:
                    # 检查轨迹方差，避免除零警告
                    pred_std = np.std(pred_traj)
                    actual_std = np.std(actual_traj)
                    if pred_std > 1e-8 and actual_std > 1e-8:
                        traj_ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
                        if np.isfinite(traj_ic):
                            trajectory_ics[fn].append(traj_ic)

                        traj_ric, _ = spearmanr(pred_traj, actual_traj)
                        if np.isfinite(traj_ric):
                            trajectory_rics[fn].append(traj_ric)

            for step_idx in range(pred_len):
                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                    actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                    da_by_step[step_idx][fn].append(pred_dir == actual_dir)

        except Exception as e:
            if len(trajectory_ics['close']) == 0:
                print(f"[Trajectory IC TEST] First error: {e}")
            continue

    result = {'n_samples': len(trajectory_ics['close'])}

    for fn in FEATURE_NAMES:
        tics = trajectory_ics[fn]
        trics = trajectory_rics[fn]
        result[f'{fn}_trajectory_ic'] = float(np.mean(tics)) if tics else 0.0
        result[f'{fn}_trajectory_rank_ic'] = float(np.mean(trics)) if trics else 0.0

    for step_idx in range(pred_len):
        for fn in FEATURE_NAMES:
            da_list = da_by_step[step_idx][fn]
            result[f'{fn}_da_step{step_idx+1}'] = float(np.mean(da_list)) if da_list else 0.0

    return result


def print_evaluation_result(result, predict, title="Evaluation Results"):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"\nSamples evaluated: {result['n_samples']}")

    print("\nTrajectory IC (各特征):")
    for fn in FEATURE_NAMES:
        ic = result.get(f'{fn}_trajectory_ic', 0)
        ric = result.get(f'{fn}_trajectory_rank_ic', 0)
        print(f"  {fn}: IC={ic:.4f}, RankIC={ric:.4f}")

    print("\nDirection Accuracy (各步):")
    for step_idx in range(predict):
        print(f"  Step {step_idx+1}:")
        for fn in FEATURE_NAMES:
            da = result.get(f'{fn}_da_step{step_idx+1}', 0)
            print(f"    {fn}: {da:.1%}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Mode6 Model Evaluation')
    parser.add_argument('--model', type=str,
                        default='outputs/models/mode6_lb200_pd10/checkpoints/best_ic_model')
    parser.add_argument('--tokenizer', type=str,
                        default='outputs/tokenizers/ma60_tokenizer_base_v1/checkpoints/best_model')
    parser.add_argument('--data-dir', type=str, default='finetune/data/ma60_norm/windowed_lb200_pd10')
    parser.add_argument('--lookback', type=int, default=200)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Mode6 Model Evaluation (MA60 lb200)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Lookback: {args.lookback}, Predict: {args.predict}")
    print(f"Test samples: {args.n_samples}")

    # 加载 tokenizer
    tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, args.tokenizer))
    tokenizer.eval().to(DEVICE)

    # 加载模型
    model = Kronos.from_pretrained(os.path.join(project_root, "pretrained/Kronos-mini"))
    model.eval().to(DEVICE)

    # 加载 checkpoint
    checkpoint_path = os.path.join(project_root, args.model)
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded safetensors from {safetensors_path}")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {bin_path}")
    else:
        print("Warning: No checkpoint found, using pretrained model")

    # 加载测试数据
    data_dir = os.path.join(project_root, args.data_dir)
    all_data, test_indices = load_windowed_data(data_dir, args.lookback, args.predict)
    print(f"Test indices: {len(test_indices)} windows")

    # 评估
    print("\nEvaluating...")
    rng = np.random.RandomState(args.seed)
    result = evaluate_trajectory_ic(
        model, tokenizer, DEVICE, all_data, test_indices,
        n_samples=args.n_samples, lookback=args.lookback,
        pred_len=args.predict, rng=rng
    )

    # 打印结果
    print_evaluation_result(result, predict=args.predict, title="Mode6 Test Set Evaluation")


if __name__ == '__main__':
    main()