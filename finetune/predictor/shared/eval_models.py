"""
标准模型评估脚本 - 测试集完整评估

输出完整的 Trajectory IC、MAE、DA 表格（6特征 x 10步）。

Usage:
    # 抽样评估（快速）
    python eval_models.py --models mode2_mini_lb400 mode10_small_lb246 --n-samples 1000

    # 全量评估（精确）
    python eval_models.py --models mode2_mini_lb400 --n-samples -1

    # 保存结果
    python eval_models.py --models mode2_mini_lb400 --output results.json
"""

import os
import sys
import argparse
import json
import pickle
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference

# ============================================================================
# 配置
# ============================================================================

TOKENIZER_PATH = 'outputs/tokenizers/ma60_tokenizer_base_v1/checkpoints/best_model'
DATA_DIR_TEMPLATE = 'finetune/data/ma60_norm/windowed_lb{lookback}_pd10'
MODELS_DIR = 'outputs/models'

FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']

# 模型配置映射（自动检测 lookback）
MODEL_CONFIGS = {
    'mode2_mini_lb400': {'model_type': 'mini', 'lookback': 400, 'max_context': 2048},
    'mode2_lb400_pd10': {'model_type': 'mini', 'lookback': 400, 'max_context': 2048},
    'mode7_base_lb400': {'model_type': 'base', 'lookback': 400, 'max_context': 512},
    'mode8_small_lb400': {'model_type': 'small', 'lookback': 400, 'max_context': 512},
    'mode9_small_lb60': {'model_type': 'small', 'lookback': 60, 'max_context': 512},
    'mode10_small_lb246': {'model_type': 'small', 'lookback': 246, 'max_context': 512},
}

PRETRAINED_PATHS = {
    'mini': 'pretrained/Kronos-mini',
    'small': 'pretrained/Kronos-small',
    'base': 'pretrained/Kronos-base',
}


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_model_full(model, tokenizer, test_data, indices, lookback, predict,
                        n_samples=-1, seed=42, clip=5.0, device='cuda'):
    """
    全量评估 - Trajectory IC / MAE / DA（6特征 x predict步）

    Args:
        n_samples: -1 表示全量评估
    """
    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)

    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}
    mae_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]

    # 抽样或全量
    if n_samples > 0 and n_samples < len(indices):
        sample_idx = rng.choice(len(indices), n_samples, replace=False)
        indices = [indices[i] for i in sample_idx]

    for (sym, start) in tqdm(indices, desc="Evaluating"):
        d = test_data[sym]
        window = lookback + predict
        end = start + window

        if end > len(d['normalized']):
            continue

        norm = d['normalized'][start:end]
        orig = d['original'][start:end]
        means = d['means'][start:end]
        stds = d['stds'][start:end]
        ts = d['index'][start:end]

        stamp = np.stack([
            ts.minute.values.astype(np.float32),
            ts.hour.values.astype(np.float32),
            ts.weekday.values.astype(np.float32),
            ts.day.values.astype(np.float32),
            ts.month.values.astype(np.float32),
        ], axis=1)

        x_norm = norm[:lookback]
        baseline = orig[lookback - 1]

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
            x_stamp = torch.from_numpy(stamp[:lookback]).unsqueeze(0).to(device)
            y_stamp = torch.from_numpy(stamp[lookback:lookback+predict]).unsqueeze(0).to(device)

            preds = auto_regressive_inference(
                tokenizer, model, x_tensor, x_stamp, y_stamp,
                max_context=2048, pred_len=predict, clip=clip, T=1.0, top_p=0.9, sample_count=1
            )

        pred_norm = preds[0, lookback:lookback+predict]
        pred_raw = pred_norm * stds[lookback:lookback+predict] + means[lookback:lookback+predict]
        actual = orig[lookback:lookback+predict]

        # Trajectory IC
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

        # MAE / DA by step
        for step_idx in range(predict):
            for fi, fn in enumerate(FEATURE_NAMES):
                mae_by_step[step_idx][fn].append(abs(pred_raw[step_idx, fi] - actual[step_idx, fi]))
                pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                da_by_step[step_idx][fn].append(pred_dir == actual_dir)

    # 聚合结果
    result = {'n_samples': len(indices)}

    for fn in FEATURE_NAMES:
        tics = trajectory_ics[fn]
        trics = trajectory_rics[fn]
        result[f'{fn}_trajectory_ic'] = float(np.mean(tics)) if tics else 0.0
        result[f'{fn}_trajectory_ic_std'] = float(np.std(tics)) if len(tics) >= 2 else 0.0
        result[f'{fn}_trajectory_ic_pos_pct'] = float(sum(t > 0 for t in tics) / len(tics)) if tics else 0.0
        result[f'{fn}_trajectory_rank_ic'] = float(np.mean(trics)) if trics else 0.0

    for step_idx in range(predict):
        suffix = f'_step{step_idx+1}'
        for fn in FEATURE_NAMES:
            result[f'{fn}_mae{suffix}'] = float(np.mean(mae_by_step[step_idx][fn])) if mae_by_step[step_idx][fn] else 0.0
            result[f'{fn}_da{suffix}'] = float(np.mean(da_by_step[step_idx][fn])) if da_by_step[step_idx][fn] else 0.0

    return result


def print_full_result(result, predict=10, title="Evaluation Results"):
    """
    打印完整评估结果表格
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Trajectory IC 表
    print(f"\n[Trajectory IC]")
    print(f"{'Feature':<8} {'IC':>10} {'std':>10} {'pos%':>8} {'RankIC':>10}")
    print("-" * 46)
    for fn in FEATURE_NAMES:
        ic = result.get(f'{fn}_trajectory_ic', 0)
        std = result.get(f'{fn}_trajectory_ic_std', 0)
        pos = result.get(f'{fn}_trajectory_ic_pos_pct', 0)
        ric = result.get(f'{fn}_trajectory_rank_ic', 0)
        print(f"{fn:<8} {ic:>10.4f} {std:>10.4f} {pos:>7.1%} {ric:>10.4f}")
    print(f"\nN Samples: {result.get('n_samples', 0)}")

    # MAE 表
    print(f"\n[Per-step MAE]")
    print(f"{'Step':<5} {'open':>8} {'high':>8} {'low':>8} {'close':>8} {'vol':>12} {'amt':>12}")
    print("-" * 60)
    for step_idx in range(predict):
        suffix = f'_step{step_idx+1}'
        print(f"+{step_idx+1:<4} "
              f"{result.get(f'open_mae{suffix}', 0):>8.2f} "
              f"{result.get(f'high_mae{suffix}', 0):>8.2f} "
              f"{result.get(f'low_mae{suffix}', 0):>8.2f} "
              f"{result.get(f'close_mae{suffix}', 0):>8.2f} "
              f"{result.get(f'vol_mae{suffix}', 0):>12.0f} "
              f"{result.get(f'amt_mae{suffix}', 0):>12.0f}")

    # DA 表
    print(f"\n[Per-step DA]")
    print(f"{'Step':<5} {'open':>8} {'high':>8} {'low':>8} {'close':>8} {'vol':>8} {'amt':>8}")
    print("-" * 50)
    for step_idx in range(predict):
        suffix = f'_step{step_idx+1}'
        print(f"+{step_idx+1:<4} "
              f"{result.get(f'open_da{suffix}', 0):>7.0%} "
              f"{result.get(f'high_da{suffix}', 0):>7.0%} "
              f"{result.get(f'low_da{suffix}', 0):>7.0%} "
              f"{result.get(f'close_da{suffix}', 0):>7.0%} "
              f"{result.get(f'vol_da{suffix}', 0):>7.0%} "
              f"{result.get(f'amt_da{suffix}', 0):>7.0%}")

    print(f"{'='*80}")


def get_test_indices(test_data, lookback, predict):
    """获取测试集窗口索引"""
    indices = []
    window = lookback + predict

    for symbol in test_data.keys():
        d = test_data[symbol]
        if 'windows' in d:
            for start in d['windows']:
                indices.append((symbol, int(start)))
        else:
            seq_len = len(d['normalized'])
            if seq_len >= window:
                for i in range(seq_len - window + 1):
                    indices.append((symbol, i))

    return indices


# ============================================================================
# 主函数
# ============================================================================

def evaluate_single_model(model_name, checkpoint_type='best_ic_model',
                          n_samples=-1, seed=42, device='cuda'):
    """评估单个模型"""

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        return None

    config = MODEL_CONFIGS[model_name]
    model_type = config['model_type']
    lookback = config['lookback']
    max_context = config['max_context']
    predict = 10

    # 数据路径
    data_dir = DATA_DIR_TEMPLATE.format(lookback=lookback)
    test_path = os.path.join(project_root, data_dir, 'test_data.pkl')

    # 模型路径
    model_path = os.path.join(project_root, MODELS_DIR, model_name, 'checkpoints', checkpoint_type)

    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {checkpoint_type}")
    print(f"Model type: {model_type}, Lookback: {lookback}")
    print(f"Test data: {test_path}")
    print(f"{'='*80}")

    # 加载 tokenizer
    tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, TOKENIZER_PATH))
    tokenizer.to(device)
    tokenizer.eval()

    # 加载模型
    model = Kronos.from_pretrained(model_path)
    model.to(device)
    model.eval()

    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model size: {model_size:.2f}M")

    # 加载测试数据
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    indices = get_test_indices(test_data, lookback, predict)
    print(f"Test windows: {len(indices)}")

    if n_samples > 0:
        print(f"Sampling: {n_samples} windows")
    else:
        print(f"Full evaluation: {len(indices)} windows")

    # 评估
    result = evaluate_model_full(
        model, tokenizer, test_data, indices,
        lookback=lookback, predict=predict,
        n_samples=n_samples, seed=seed, device=device
    )

    # 打印结果
    print_full_result(result, predict=predict, title=f"{model_name} Test Results")

    return result


def main():
    parser = argparse.ArgumentParser(description='Standard model evaluation on test set')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['mode2_mini_lb400'],
                        help='Models to evaluate')
    parser.add_argument('--checkpoint', type=str, default='best_ic_model',
                        help='Checkpoint type: best_ic_model, best_model, latest_model')
    parser.add_argument('--n-samples', type=int, default=-1,
                        help='Number of samples (-1 for full evaluation)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    for model_name in args.models:
        result = evaluate_single_model(
            model_name,
            checkpoint_type=args.checkpoint,
            n_samples=args.n_samples,
            seed=args.seed,
            device=str(device)
        )
        if result:
            results[model_name] = result

    # 汇总对比
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'close_IC':>10} {'close_DA1':>10} {'open_IC':>10} {'open_DA1':>10}")
    print("-" * 60)
    for model_name, r in results.items():
        close_ic = r.get('close_trajectory_ic', 0)
        close_da1 = r.get('close_da_step1', 0)
        open_ic = r.get('open_trajectory_ic', 0)
        open_da1 = r.get('open_da_step1', 0)
        print(f"{model_name:<20} {close_ic:>10.4f} {close_da1:>9.1%} {open_ic:>10.4f} {open_da1:>9.1%}")
    print(f"{'='*80}")

    # 保存结果
    if args.output:
        output_path = os.path.join(project_root, args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()