"""
统一评估脚本 - 评估 Mode7/8/9/10 模型
"""

import os
import sys
import pickle
import argparse
import torch
import numpy as np
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']

# 模型配置（已更新到新目录结构）
MODEL_CONFIGS = {
    # ===== final/ 最佳模型 =====
    'final_mini': {
        'model_type': 'mini',
        'pretrained': 'pretrained/Kronos-mini',
        'checkpoint': 'outputs/models/final/mini/best_ic_model',
        'data_dir': 'finetune/data/ma60_norm/windowed_lb400_pd10',
        'lookback': 400,
        'max_context': 2048,
    },
    'final_small': {
        'model_type': 'small',
        'pretrained': 'pretrained/Kronos-small',
        'checkpoint': 'outputs/models/final/small/best_ic_model',
        'data_dir': 'finetune/data/ma60_norm/windowed_lb246_pd10',
        'lookback': 246,
        'max_context': 512,
    },

    # ===== experiments/ 进行中实验 =====
    'experiments_base': {
        'model_type': 'base',
        'pretrained': 'pretrained/Kronos-base',
        'checkpoint': 'outputs/models/experiments/base_ma60_lb400_ddp/checkpoints/best_ic_model',
        'data_dir': 'finetune/data/ma60_norm/windowed_lb400_pd10',
        'lookback': 400,
        'max_context': 512,
    },

    # ===== deprecated/ 低IC模型（保留旧名称兼容）=====
    'mode8': {
        'model_type': 'small',
        'pretrained': 'pretrained/Kronos-small',
        'checkpoint': 'outputs/models/deprecated/small_ma60_lb400_pd10_ic145/checkpoints/best_ic_model',
        'data_dir': 'finetune/data/ma60_norm/windowed_lb400_pd10',
        'lookback': 400,
        'max_context': 512,
    },
    'mode9': {
        'model_type': 'small',
        'pretrained': 'pretrained/Kronos-small',
        'checkpoint': 'outputs/models/deprecated/small_ma60_lb60_pd10_ic108/checkpoints/best_ic_model',
        'data_dir': 'finetune/data/ma60_norm/windowed_lb60_pd10',
        'lookback': 60,
        'max_context': 512,
    },
}

TOKENIZER_PATH = 'outputs/tokenizers/ma60_tokenizer_base_v1/checkpoints/best_model'


def load_data(data_dir):
    test_path = os.path.join(project_root, data_dir, 'test_data.pkl')
    with open(test_path, 'rb') as f:
        test_raw = pickle.load(f)

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


def evaluate_model(model, tokenizer, all_data, test_indices, lookback, pred_len=10,
                   max_context=512, n_samples=500, rng=None):
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
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            baseline = orig_full[lookback - 1]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(DEVICE)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(DEVICE)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=max_context, pred_len=pred_len,
                    clip=5.0, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                pred_norm = preds[0, lookback:lookback + pred_len, :]
                pred_raw = pred_norm * stds_full[lookback:lookback + pred_len] + means_full[lookback:lookback + pred_len]
                actual = orig_full[lookback:lookback + pred_len]

            for fi, fn in enumerate(FEATURE_NAMES):
                pred_traj = pred_raw[:, fi]
                actual_traj = actual[:, fi]

                if len(pred_traj) >= 3:
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Mode7/8/9/10 models')
    parser.add_argument('--mode', type=str, nargs='+', default=['final_mini', 'final_small'],
                        help='Modes to evaluate (e.g., final_mini, final_small, experiments_base)')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Model Evaluation - Trajectory IC on Test Set")
    print("=" * 70)

    # Load tokenizer
    tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, TOKENIZER_PATH))
    tokenizer.eval().to(DEVICE)

    rng = np.random.RandomState(args.seed)

    results = {}

    for mode in args.mode:
        if mode not in MODEL_CONFIGS:
            print(f"Unknown mode: {mode}")
            continue

        config = MODEL_CONFIGS[mode]
        print(f"\n{'='*70}")
        print(f"{mode.upper()} - {config['model_type']} + lb{config['lookback']}")
        print(f"{'='*70}")

        # Load model
        model = Kronos.from_pretrained(os.path.join(project_root, config['pretrained']))
        model.eval().to(DEVICE)

        # Load checkpoint
        checkpoint_path = os.path.join(project_root, config['checkpoint'])
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded: {safetensors_path}")

        # Load data
        all_data, test_indices = load_data(config['data_dir'])
        print(f"Test windows: {len(test_indices)}")

        # Evaluate
        result = evaluate_model(
            model, tokenizer, all_data, test_indices,
            lookback=config['lookback'],
            max_context=config['max_context'],
            n_samples=args.n_samples,
            rng=rng
        )

        results[mode] = result

        print(f"\nSamples: {result['n_samples']}")
        print(f"\nTrajectory IC (6 features):")
        for fn in FEATURE_NAMES:
            ic = result[f'{fn}_trajectory_ic']
            ric = result[f'{fn}_trajectory_rank_ic']
            print(f"  {fn:<8}: IC={ic:.4f}, RankIC={ric:.4f}")

        print(f"\nDirection Accuracy (close):")
        for step in range(1, 11):
            da = result[f'close_da_step{step}']
            print(f"  Step {step}: {da:.1%}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Mode':<10} {'Model':<10} {'lb':<8} {'close_IC':<10} {'close_DA1':<10}")
    print("-" * 50)
    for mode, res in results.items():
        config = MODEL_CONFIGS[mode]
        ic = res['close_trajectory_ic']
        da1 = res['close_da_step1']
        print(f"{mode:<10} {config['model_type']:<10} {config['lookback']:<8} {ic:<10.4f} {da1:<10.1%}")
    print("=" * 70)


if __name__ == '__main__':
    main()