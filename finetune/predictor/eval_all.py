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

# 模型配置
MODEL_CONFIGS = {
    # ===== 最新训练模型 =====
    'latest_mini_lb400': {
        'model_type': 'mini',
        'pretrained': 'pretrained/Kronos-mini',
        'checkpoint': 'outputs/models/mode_mini_lb400/checkpoints/best_combined_model',
        'tokenizer': 'outputs/tokenizers/final/2k-MA60',
        'data_dir': 'finetune/data/ma60_norm/block_lb400_pd10',
        'test_data': 'finetune/data/ma60_norm/block_lb400_pd10/final_test_data.pkl',  # 最终测试集
        'lookback': 400,
        'max_context': 2048,
    },
    'latest_mini_lb400_ic': {
        'model_type': 'mini',
        'pretrained': 'pretrained/Kronos-mini',
        'checkpoint': 'outputs/models/mode_mini_lb400/checkpoints/best_ic_model',
        'tokenizer': 'outputs/tokenizers/final/2k-MA60',
        'data_dir': 'finetune/data/ma60_norm/block_lb400_pd10',
        'test_data': 'finetune/data/ma60_norm/block_lb400_pd10/final_test_data.pkl',
        'lookback': 400,
        'max_context': 2048,
    },

    # ===== final/ 最佳模型 =====
    'final_mini': {
        'model_type': 'mini',
        'pretrained': 'pretrained/Kronos-mini',
        'checkpoint': 'outputs/models/final/mini',
        'tokenizer': 'outputs/tokenizers/final/2k-MA60',
        'data_dir': 'finetune/data/ma60_norm/block_lb400_pd10',
        'lookback': 400,
        'max_context': 2048,
    },
    'final_small': {
        'model_type': 'small',
        'pretrained': 'pretrained/Kronos-small',
        'checkpoint': 'outputs/models/final/small',
        'tokenizer': 'outputs/tokenizers/final/base-MA60',
        'data_dir': 'finetune/data/ma60_norm/block_lb400_pd10',
        'lookback': 400,
        'max_context': 512,
    },
}

# 默认tokenizer
DEFAULT_TOKENIZER_PATH = 'outputs/tokenizers/final/2k-MA60'


def load_data(data_dir, test_data_path=None):
    """加载测试数据

    Args:
        data_dir: 数据目录（用于默认 test_data.pkl）
        test_data_path: 直接指定的测试数据路径（优先使用）
    """
    if test_data_path:
        test_path = os.path.join(project_root, test_data_path)
    else:
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
                    # 检查轨迹是否有足够方差（避免除0警告）
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
    parser = argparse.ArgumentParser(description='Evaluate Kronos models')
    # 预定义模式
    parser.add_argument('--models', type=str, nargs='+', default=['latest_mini_lb400'],
                        help='Predefined models (e.g., latest_mini_lb400, final_mini)')
    # 参数化路径（可覆盖mode配置）
    parser.add_argument('--model-path', type=str, default=None,
                        help='Direct model checkpoint path (overrides mode)')
    parser.add_argument('--model-type', type=str, default='mini', choices=['mini', 'small', 'base'],
                        help='Model type for tokenizer selection')
    parser.add_argument('--tokenizer-path', type=str, default=None,
                        help='Direct tokenizer path (overrides default)')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Direct test data path (overrides mode)')
    parser.add_argument('--n-samples', type=int, default=-1,
                        help='Number of samples to evaluate (-1 for full test set)')
    parser.add_argument('--checkpoint', type=str, default='best_combined_model',
                        help='Checkpoint type for predefined models')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Model Evaluation - Trajectory IC on Test Set")
    print("=" * 70)

    rng = np.random.RandomState(args.seed)

    results = {}

    # 如果指定了直接路径，使用参数化配置
    if args.model_path:
        config = {
            'model_type': args.model_type,
            'pretrained': f'pretrained/Kronos-{args.model_type}',
            'checkpoint': args.model_path,
            'tokenizer': args.tokenizer_path or (f'outputs/tokenizers/final/2k-MA60' if args.model_type == 'mini' else 'outputs/tokenizers/final/base-MA60'),
            'data_dir': 'finetune/data/ma60_norm/block_lb400_pd10',
            'test_data': args.test_data,
            'lookback': 400,
            'max_context': 2048 if args.model_type == 'mini' else 512,
        }
        modes = ['custom']
        MODEL_CONFIGS['custom'] = config  # 添加临时配置
    else:
        modes = args.models

    for mode in modes:
        if mode not in MODEL_CONFIGS:
            print(f"Unknown mode: {mode}")
            continue

        config = MODEL_CONFIGS[mode]
        print(f"\n{'='*70}")
        print(f"{mode.upper()} - {config['model_type']} + lb{config['lookback']}")
        print(f"{'='*70}")

        # Load tokenizer (model-specific)
        tokenizer_path = config.get('tokenizer', DEFAULT_TOKENIZER_PATH)
        tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, tokenizer_path))
        tokenizer.eval().to(DEVICE)
        print(f"Tokenizer: {tokenizer_path}")

        # Load model
        print(f"{mode.upper()} - {config['model_type']} + lb{config['lookback']}")
        print(f"{'='*70}")

        # Load model
        model = Kronos.from_pretrained(os.path.join(project_root, config['pretrained']))
        model.eval().to(DEVICE)

        # Load checkpoint
        if args.checkpoint and mode != 'custom':
            # 使用 --checkpoint 参数覆盖预定义模型的checkpoint类型
            base_checkpoint = config['checkpoint']
            # 如果原路径包含 checkpoints/，替换为新的checkpoint类型
            if '/checkpoints/' in base_checkpoint:
                checkpoint_path = base_checkpoint.rsplit('/checkpoints/', 1)[0] + '/checkpoints/' + args.checkpoint
            else:
                checkpoint_path = base_checkpoint
        else:
            checkpoint_path = config['checkpoint']

        checkpoint_full_path = os.path.join(project_root, checkpoint_path)
        safetensors_path = os.path.join(checkpoint_full_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded: {safetensors_path}")
        else:
            print(f"Checkpoint not found: {safetensors_path}")

        # Load data (使用最终测试集或默认测试集)
        test_data_path = config.get('test_data')
        all_data, test_indices = load_data(config['data_dir'], test_data_path)
        print(f"Test data: {test_data_path or config['data_dir'] + '/test_data.pkl'}")
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

    # 保存结果
    if args.output:
        import json
        output_path = os.path.join(project_root, args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()