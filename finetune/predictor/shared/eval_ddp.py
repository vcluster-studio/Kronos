"""
多GPU并行评估脚本 - 全量测试集评估

利用DDP并行加速全量评估，每个GPU处理部分窗口。

Usage:
    torchrun --nproc_per_node=4 eval_ddp.py --models mode2_mini_lb400 mode8_small_lb400
"""

import os
import sys
import argparse
import json
import pickle
import torch
import torch.distributed as dist
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

MODEL_CONFIGS = {
    'mode2_mini_lb400': {'model_type': 'mini', 'lookback': 400, 'max_context': 2048},
    'mode2_lb400_pd10': {'model_type': 'mini', 'lookback': 400, 'max_context': 2048},
    'mode7_base_lb400': {'model_type': 'base', 'lookback': 400, 'max_context': 512},
    'mode8_small_lb400': {'model_type': 'small', 'lookback': 400, 'max_context': 512},
    'mode9_small_lb60': {'model_type': 'small', 'lookback': 60, 'max_context': 512},
    'mode10_small_lb246': {'model_type': 'small', 'lookback': 246, 'max_context': 512},
    # Full window 归一化模型（mode1）
    'mode1_global': {
        'model_type': 'mini', 'lookback': 200, 'max_context': 2048,
        'norm_mode': 'full_window',
        'model_path': 'outputs/models/archived/mode1_global_predictor/best_ic_model',
        'tokenizer_path': 'pretrained/Kronos-Tokenizer-2k',
        'data_path': 'finetune/data/global_norm/full_series/test_data.pkl',
    },
}


def setup_ddp():
    """初始化DDP"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if world_size == 1:
            return 0, 0, 1, False
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_ddp():
    """清理DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_test_indices(test_data, lookback, predict):
    """获取测试集窗口索引（支持MA60 windowed格式和 full_window DataFrame格式）"""
    indices = []
    window = lookback + predict

    sample_val = next(iter(test_data.values()))
    # DataFrame 格式（full_window 归一化）
    if hasattr(sample_val, 'columns'):
        for symbol in test_data.keys():
            df = test_data[symbol]
            if len(df) >= window:
                # full_window格式：每个股票取最后1个窗口
                indices.append((symbol, len(df) - window))
    # Dict 格式（MA60 预归一化）
    elif 'windows' in sample_val:
        for symbol in test_data.keys():
            d = test_data[symbol]
            for start in d['windows']:
                indices.append((symbol, int(start)))
    else:
        for symbol in test_data.keys():
            d = test_data[symbol]
            seq_len = len(d['normalized'])
            if seq_len >= window:
                for i in range(seq_len - window + 1):
                    indices.append((symbol, i))

    return indices


def evaluate_windows(model, tokenizer, test_data, indices, lookback, predict,
                     clip=5.0, device='cuda', desc="Evaluating", norm_mode='ma60'):
    """
    评估指定窗口列表（单GPU）

    norm_mode: 'ma60' 预归一化数据 或 'full_window' 原始数据动态归一化

    Returns:
        trajectory_ics: 各特征的IC列表
        trajectory_rics: 各特征的Rank IC列表
        da_by_step: 各步各特征的DA列表
    """
    model.eval()
    tokenizer.eval()

    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]

    for (sym, start) in tqdm(indices, desc=desc, disable=len(indices) < 100):
        d = test_data[sym]
        window = lookback + predict
        end = start + window

        try:
            # full_window 归一化（DataFrame 格式）
            if norm_mode == 'full_window':
                df = d
                slice_df = df.iloc[start:end]
                x = slice_df.values[:lookback]
                y = slice_df.values[lookback:]
                baseline = slice_df.values[lookback - 1]

                x_mean = np.mean(x, axis=0)
                x_std = np.std(x, axis=0) + 1e-5
                x_norm_raw = (x - x_mean) / x_std
                x_norm_raw = np.clip(x_norm_raw, -clip, clip)

                ts = df.index[start:end]
                x_norm = torch.from_numpy(x_norm_raw).unsqueeze(0).to(device)
                stamp = np.stack([
                    ts.minute.values.astype(np.float32),
                    ts.hour.values.astype(np.float32),
                    ts.weekday.values.astype(np.float32),
                    ts.day.values.astype(np.float32),
                    ts.month.values.astype(np.float32),
                ], axis=1)

                with torch.no_grad():
                    x_stamp_tensor = torch.from_numpy(stamp[:lookback]).unsqueeze(0).to(device)
                    y_stamp_tensor = torch.from_numpy(stamp[lookback:]).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model, x_norm, x_stamp_tensor, y_stamp_tensor,
                        max_context=2048, pred_len=predict, clip=clip,
                        T=1.0, top_p=0.9, sample_count=1
                    )

                pred_norm = preds[0, lookback:lookback+predict]
                pred_raw = pred_norm * x_std + x_mean
                actual = y

            # MA60 预归一化数据
            else:
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
                    x_stamp_tensor = torch.from_numpy(stamp[:lookback]).unsqueeze(0).to(device)
                    y_stamp_tensor = torch.from_numpy(stamp[lookback:lookback+predict]).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model, x_tensor, x_stamp_tensor, y_stamp_tensor,
                        max_context=2048, pred_len=predict, clip=clip,
                        T=1.0, top_p=0.9, sample_count=1
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

            # DA by step
            for step_idx in range(predict):
                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                    actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                    da_by_step[step_idx][fn].append(pred_dir == actual_dir)

        except Exception:
            continue

    return trajectory_ics, trajectory_rics, da_by_step


def aggregate_results(local_ics, local_rics, local_da, predict, world_size, device):
    """
    聚合各GPU的评估结果
    """
    # 收集各特征的IC均值和样本数
    results = {}

    for fn in FEATURE_NAMES:
        local_ic_list = local_ics[fn]
        local_ric_list = local_rics[fn]
        n_local = len(local_ic_list)

        # 发送样本数
        n_tensor = torch.tensor([n_local], device=device)
        if world_size > 1:
            gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_n, n_tensor)
            total_n = sum(t.item() for t in gathered_n)
        else:
            total_n = n_local

        # 发送IC均值
        local_ic_mean = np.mean(local_ic_list) if local_ic_list else 0.0
        ic_mean_tensor = torch.tensor([local_ic_mean * n_local], device=device)  # 加权

        if world_size > 1:
            gathered_ic = [torch.zeros_like(ic_mean_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_ic, ic_mean_tensor)
            total_ic_sum = sum(t.item() for t in gathered_ic)
        else:
            total_ic_sum = local_ic_mean * n_local

        # 发送RIC均值
        local_ric_mean = np.mean(local_ric_list) if local_ric_list else 0.0
        ric_mean_tensor = torch.tensor([local_ric_mean * n_local], device=device)

        if world_size > 1:
            gathered_ric = [torch.zeros_like(ric_mean_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_ric, ric_mean_tensor)
            total_ric_sum = sum(t.item() for t in gathered_ric)
        else:
            total_ric_sum = local_ric_mean * n_local

        results[f'{fn}_trajectory_ic'] = total_ic_sum / total_n if total_n > 0 else 0.0
        results[f'{fn}_trajectory_rank_ic'] = total_ric_sum / total_n if total_n > 0 else 0.0

    # 聚合DA
    for step_idx in range(predict):
        for fn in FEATURE_NAMES:
            local_da_list = local_da[step_idx][fn]
            n_local = len(local_da_list)

            # 发送样本数
            n_tensor = torch.tensor([n_local], device=device)
            if world_size > 1:
                gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_n, n_tensor)
                total_n = sum(t.item() for t in gathered_n)
            else:
                total_n = n_local

            # 发送DA均值
            local_da_mean = np.mean(local_da_list) if local_da_list else 0.0
            da_mean_tensor = torch.tensor([local_da_mean * n_local], device=device)

            if world_size > 1:
                gathered_da = [torch.zeros_like(da_mean_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_da, da_mean_tensor)
                total_da_sum = sum(t.item() for t in gathered_da)
            else:
                total_da_sum = local_da_mean * n_local

            results[f'{fn}_da_step{step_idx+1}'] = total_da_sum / total_n if total_n > 0 else 0.0

    results['n_samples'] = total_n

    return results


def evaluate_model_ddp(model_name, checkpoint_type='best_ic_model',
                       rank=0, local_rank=0, world_size=1, device='cuda'):
    """多GPU评估单个模型"""

    if model_name not in MODEL_CONFIGS:
        if rank == 0:
            print(f"Unknown model: {model_name}")
        return None

    cfg = MODEL_CONFIGS[model_name]
    lookback = cfg['lookback']
    predict = 10
    norm_mode = cfg.get('norm_mode', 'ma60')
    max_context = cfg.get('max_context', 2048)

    # 数据路径（支持自定义或模板）
    if 'data_path' in cfg:
        test_path = os.path.join(project_root, cfg['data_path'])
    else:
        data_dir = DATA_DIR_TEMPLATE.format(lookback=lookback)
        test_path = os.path.join(project_root, data_dir, 'test_data.pkl')

    # 模型路径（支持自定义或模板）
    if 'model_path' in cfg:
        # 如果自定义路径已包含 checkpoint 目录（含 config.json），直接使用
        full_model_path = os.path.join(project_root, cfg['model_path'])
        if os.path.isdir(full_model_path) and os.path.exists(os.path.join(full_model_path, 'config.json')):
            model_path = full_model_path
        else:
            model_path = os.path.join(project_root, cfg['model_path'], 'checkpoints', checkpoint_type)
    else:
        model_path = os.path.join(project_root, MODELS_DIR, model_name, 'checkpoints', checkpoint_type)

    # Tokenizer路径（支持自定义或默认MA60）
    if 'tokenizer_path' in cfg:
        tokenizer_path = os.path.join(project_root, cfg['tokenizer_path'])
    else:
        tokenizer_path = os.path.join(project_root, TOKENIZER_PATH)

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name} ({world_size} GPUs)")
        print(f"Checkpoint: {checkpoint_type}")
        print(f"Norm mode: {norm_mode}")
        print(f"Model: {model_path}")
        print(f"Data: {test_path}")
        print(f"Tokenizer: {tokenizer_path}")
        print(f"{'='*80}")

    # 加载tokenizer
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.to(device)
    tokenizer.eval()

    # 加载模型
    model = Kronos.from_pretrained(model_path)
    model.to(device)
    model.eval()

    if rank == 0:
        model_size = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model size: {model_size:.2f}M")

    # 加载测试数据
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # 获取所有测试窗口
    all_indices = get_test_indices(test_data, lookback, predict)

    if rank == 0:
        print(f"Total test windows: {len(all_indices)}")

    # 分片：每个rank处理一部分
    per_rank = len(all_indices) // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else len(all_indices)
    local_indices = all_indices[start_idx:end_idx]

    if rank == 0:
        print(f"Per-GPU windows: {len(local_indices)} (GPU 0), ~{per_rank} (others)")

    # 评估本地窗口
    local_ics, local_rics, local_da = evaluate_windows(
        model, tokenizer, test_data, local_indices,
        lookback=lookback, predict=predict,
        device=device, desc=f"GPU{rank}", norm_mode=norm_mode
    )

    # 同步
    if world_size > 1:
        dist.barrier()

    # 聚合结果
    results = aggregate_results(local_ics, local_rics, local_da, predict, world_size, device)

    return results


def print_full_result(result, predict=10, title="Evaluation Results"):
    """打印完整评估结果"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Trajectory IC
    print(f"\n[Trajectory IC]")
    print(f"{'Feature':<8} {'IC':>10} {'RankIC':>10}")
    print("-" * 28)
    for fn in FEATURE_NAMES:
        ic = result.get(f'{fn}_trajectory_ic', 0)
        ric = result.get(f'{fn}_trajectory_rank_ic', 0)
        print(f"{fn:<8} {ic:>10.4f} {ric:>10.4f}")

    print(f"\nN Samples: {result.get('n_samples', 0)}")

    # DA
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


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU evaluation on test set')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['mode2_mini_lb400'],
                        help='Models to evaluate')
    parser.add_argument('--checkpoint', type=str, default='best_ic_model',
                        help='Checkpoint type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    # DDP setup
    rank, local_rank, world_size, use_ddp = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"Using {world_size} GPUs")

    results = {}

    for model_name in args.models:
        result = evaluate_model_ddp(
            model_name,
            checkpoint_type=args.checkpoint,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device
        )

        if result and rank == 0:
            results[model_name] = result
            print_full_result(result, title=f"{model_name} Test Results")

    # 汇总（only rank 0）
    if rank == 0 and len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'close_IC':>10} {'close_DA1':>10} {'open_IC':>10}")
        print("-" * 50)
        for model_name, r in results.items():
            close_ic = r.get('close_trajectory_ic', 0)
            close_da1 = r.get('close_da_step1', 0)
            open_ic = r.get('open_trajectory_ic', 0)
            print(f"{model_name:<20} {close_ic:>10.4f} {close_da1:>9.1%} {open_ic:>10.4f}")

    # 保存（only rank 0）
    if rank == 0 and args.output:
        output_path = os.path.join(project_root, args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    cleanup_ddp()


if __name__ == '__main__':
    main()