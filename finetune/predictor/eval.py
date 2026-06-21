"""
Kronos Predictor Evaluation Entry

统一评估入口，支持：
- DDP 多卡评估
- 多模型批量对比
- 正确度量口径（去趋势 trajectory IC）
- 可懂指标输出

使用：
    # 单模型
    python finetune/predictor/eval.py --norm-mode sliding_ma60 --model mini

    # 多卡 DDP
    torchrun --nproc_per_node=4 finetune/predictor/eval.py --model mini

    # 多模型对比
    python finetune/predictor/eval.py --models mini,small,base

关键：
- 与 train.py 共用 core/metrics.py
- trajectory IC 用去趋势序列
- 主输出为可懂指标（方向胜率/振幅误差率/涨跌停命中率）
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference
from safetensors.torch import load_file

from finetune.predictor.core.config import DataConfig, parse_norm_mode
from finetune.predictor.core.paths import (
    get_tokenizer_path,
    get_model_path,
    get_checkpoint_path,
    get_split_data_path,
)
from finetune.predictor.core.metrics import (
    FEATURE_NAMES,
    detrend_to_baseline,
    safe_trajectory_ic,
    safe_corrcoef,
    calculate_combined_score,
    calculate_da_score,
    get_log_step_weights,
    get_feature_weights,
    amplitude_error_rate,
    compute_amplitude_stats,
    limit_hit_rate,
    detect_limit,
    format_metrics_report,
    excess_da,
    aggregate_ic,
    aggregate_da,
)
from finetune.predictor.core.utils import get_device, format_time


# ============================================================================
# DDP 工具（从 train.py 复用）
# ============================================================================

def get_rank_info():
    """获取 DDP rank 信息"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        return rank, local_rank, world_size, True
    return 0, 0, 1, False

def setup_ddp(rank, local_rank, world_size):
    """初始化 DDP"""
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    """清理 DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_tokenizer(
    norm_mode: str,
    model_type: str,
    device: torch.device,
    checkpoint: str = 'best_combined_model',
    lookback: int = 400,      # EV1 修复：从 CLI 传入
    predict: int = 10,        # EV1 修复
    split_mode: str = 'block', # EV1 修复
    custom_tokenizer_path: str = None,
    custom_checkpoint_path: str = None,
):
    """
    加载模型和 tokenizer

    EV1 修复：lookback/predict/split_mode 从 CLI 传入，避免硬编码导致数据模型错配

    支持自定义路径：
    - custom_tokenizer_path: 自定义 tokenizer 路径
    - custom_checkpoint_path: 自定义 checkpoint 目录路径
    """
    # Tokenizer：支持自定义路径
    if custom_tokenizer_path:
        tokenizer_path = custom_tokenizer_path
    else:
        tokenizer_path = get_tokenizer_path(norm_mode, model_type)
        if not os.path.exists(tokenizer_path):
            tokenizer_path = 'outputs/tokenizers/final/2k-MA60' if model_type == 'mini' else 'outputs/tokenizers/final/base-MA60'

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    # 模型
    pretrained_paths = {
        'mini': 'pretrained/Kronos-mini',
        'small': 'pretrained/Kronos-small',
        'base': 'pretrained/Kronos-base',
    }

    model = Kronos.from_pretrained(pretrained_paths[model_type])
    model.eval().to(device)

    # Checkpoint：支持自定义路径
    if custom_checkpoint_path:
        checkpoint_path = custom_checkpoint_path
    else:
        model_dir = get_model_path(norm_mode, lookback, predict, split_mode, model_type)
        checkpoint_path = get_checkpoint_path(model_dir, checkpoint)

    checkpoint_file = os.path.join(checkpoint_path, 'model.safetensors')
    if os.path.exists(checkpoint_file):
        state_dict = load_file(checkpoint_file)
        # EV2 修复：checkpoint 与模型对不上直接报错（不静默用错权重）
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint 与模型不匹配，拒绝加载: missing={missing}, unexpected={unexpected}"
            )
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    else:
        # EV3 修复：显式警告找不到 checkpoint，使用预训练
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}")
        print(f"[WARNING] Using pretrained model instead - evaluation may not reflect fine-tuned performance")

    return model, tokenizer


# ============================================================================
# 评估函数
# ============================================================================

def evaluate(
    model,
    tokenizer,
    test_data,
    config: DataConfig,
    device: torch.device,
    n_samples: int = -1,
    seed: int = 42,
    limit_pct: float = 0.10,
    rank: int = 0,
    world_size: int = 1,
    model_type: str = 'mini',
):
    """
    评估模型（正确口径）

    DDP 模式：各 rank 按轮询分配样本，最后用 all_gather 聚合
    """
    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)  # EV6 修复：所有 rank 同 seed 抽同样本，再按 idx%world_size 分片

    # 构建窗口索引
    indices = []
    for symbol, d in test_data.items():
        mode = d.get('mode', 'time') if isinstance(d, dict) else None
        if mode == 'block':
            # block 模式：遍历各 block 的 windows
            for b_start, blk in d['blocks'].items():
                for w in blk['windows']:
                    indices.append((symbol, int(w)))
        elif 'windows' in d:
            # time 模式（或有预分配 windows）
            for w in d['windows']:
                indices.append((symbol, int(w)))
        elif hasattr(d, 'columns'):
            seq_len = len(d)
            if seq_len >= config.lookback + config.predict:
                for i in range(seq_len - config.lookback - config.predict + 1):
                    indices.append((symbol, i))
        else:
            seq_len = len(d['normalized'])
            if seq_len >= config.lookback + config.predict:
                for i in range(seq_len - config.lookback - config.predict + 1):
                    indices.append((symbol, i))

    # 抽样
    if n_samples > 0 and n_samples < len(indices):
        sample_idx = rng.choice(len(indices), size=n_samples, replace=False)
        indices = [indices[i] for i in sample_idx]

    # 收集结果
    ic_lists = {f: [] for f in FEATURE_NAMES}
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(config.predict)]

    # 收集 actual_dir 统计（用于计算 naive DA）
    actual_dir_by_step = [[] for _ in range(config.predict)]  # 仅 close

    amplitude_rates = []
    limit_results = {'pred_limit': [], 'actual_limit': []}

    # DDP 分片：每个 rank 只处理属于它的样本（按轮询分配）
    for idx, (symbol, start) in enumerate(tqdm(indices, desc=f"Rank {rank} Evaluating", disable=rank != 0)):
        if idx % world_size != rank:
            continue  # 跳过不属于该 rank 的样本

        d = test_data[symbol]
        end = start + config.lookback + config.predict

        try:
            if 'normalized' in d:
                x_norm = d['normalized'][start:start + config.lookback].astype(np.float32)
                means = d['means'][start:end]
                stds = d['stds'][start:end]
                original = d['original'][start:end]
                timestamps = d['index'][start:end]
            else:
                df = d.iloc[start:end]
                x_raw = df.values[:config.lookback].astype(np.float32)
                x_mean = np.mean(x_raw, axis=0)
                x_std = np.std(x_raw, axis=0) + 1e-5
                x_norm = np.clip((x_raw - x_mean) / x_std, -config.clip, config.clip)
                original = df.values
                timestamps = df.index

            baseline = original[config.lookback - 1]
            baseline_close = original[config.lookback - 1, 3]

            # 时间戳
            x_stamp = np.stack([
                timestamps[:config.lookback].minute.values,
                timestamps[:config.lookback].hour.values,
                timestamps[:config.lookback].weekday.values,
                timestamps[:config.lookback].day.values,
                timestamps[:config.lookback].month.values,
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                timestamps[config.lookback:].minute.values,
                timestamps[config.lookback:].hour.values,
                timestamps[config.lookback:].weekday.values,
                timestamps[config.lookback:].day.values,
                timestamps[config.lookback:].month.values,
            ], axis=1).astype(np.float32)

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context={'mini': 2048, 'small': 512, 'base': 512}.get(model_type, 2048),  # EV9 修复：按 model_type
                    pred_len=config.predict,
                    clip=config.clip,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )

                pred_norm = preds[0, config.lookback:config.lookback + config.predict, :]

                # 反归一化
                if 'normalized' in d:
                    pred_raw = pred_norm.cpu().numpy() * stds[config.lookback:] + means[config.lookback:]
                else:
                    pred_raw = pred_norm.cpu().numpy() * x_std + x_mean

                actual = original[config.lookback:config.lookback + config.predict]

            # Trajectory IC（去趋势口径）
            for fi, fn in enumerate(FEATURE_NAMES):
                pred_detrend = detrend_to_baseline(pred_raw[:, fi], baseline[fi])
                actual_detrend = detrend_to_baseline(actual[:, fi], baseline[fi])

                ic, _ = safe_trajectory_ic(pred_detrend, actual_detrend)
                if ic is not None:
                    ic_lists[fn].append(ic)

            # DA
            for step_idx in range(config.predict):
                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                    actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                    da_by_step[step_idx][fn].append(pred_dir == actual_dir)

                    # 收集 close 的 actual_dir（用于 naive DA）
                    if fn == 'close':
                        actual_dir_by_step[step_idx].append(actual_dir)

            # 振幅误差率
            pred_amp = pred_raw[0, 1] - pred_raw[0, 2]  # high - low
            actual_amp = actual[0, 1] - actual[0, 2]
            amplitude_rates.append(amplitude_error_rate(pred_amp, actual_amp))

            # 涨跌停检测
            pred_limit = detect_limit(pred_raw, baseline_close, limit_pct)
            actual_limit = detect_limit(actual, baseline_close, limit_pct)
            limit_results['pred_limit'].append(pred_limit.any())
            limit_results['actual_limit'].append(actual_limit.any())

        except Exception:
            continue

    # 聚合结果（DDP 模式用 all_gather）
    use_ddp = world_size > 1

    if use_ddp:
        # DDP 聚合
        ic_result = aggregate_ic(ic_lists, world_size, device, rank == 0)
        da_result = aggregate_da(da_by_step, world_size, device, config.predict, rank == 0)

        # 聚合 actual_dir 用于 naive DA（EV7 修复：计数法，避免列表 padding）
        # 与 train.py:386-412 同源：all_gather up_count/n 标量，rank0 算 up_ratio
        naive_da_by_step = {}
        for step_idx in range(config.predict):
            local_actual = actual_dir_by_step[step_idx]
            local_up_count = int(sum(local_actual))
            local_n = len(local_actual)

            up_count_tensor = torch.tensor([local_up_count], device=device)
            n_tensor = torch.tensor([local_n], device=device)

            gathered_up = [torch.zeros_like(up_count_tensor) for _ in range(world_size)]
            gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]

            # 所有 rank 执行 all_gather
            dist.all_gather(gathered_up, up_count_tensor)
            dist.all_gather(gathered_n, n_tensor)

            # 只 rank 0 组装
            if rank == 0:
                total_up = sum(t.item() for t in gathered_up)
                total_n = sum(t.item() for t in gathered_n)
                if total_n > 0:
                    up_ratio = total_up / total_n
                    naive_da_by_step[step_idx] = float(max(up_ratio, 1 - up_ratio))
                else:
                    naive_da_by_step[step_idx] = 0.5
    else:
        # 单进程聚合
        ic_result = {}
        for fn in FEATURE_NAMES:
            ics = ic_lists[fn]
            if ics:
                ics_arr = np.array(ics)
                n = len(ics)
                ic_result[fn] = {
                    'mean': float(np.mean(ics_arr)),
                    'std': float(np.std(ics_arr)) if n >= 2 else 0.0,
                    'p25': float(np.percentile(ics_arr, 25)) if n >= 4 else None,
                    'p50': float(np.percentile(ics_arr, 50)),
                    'p75': float(np.percentile(ics_arr, 75)) if n >= 4 else None,
                    'n': n,
                }
            else:
                ic_result[fn] = {'mean': 0.0, 'std': 0.0, 'p25': None, 'p50': None, 'p75': None, 'n': 0}

        da_result = {}
        naive_da_by_step = {}
        for step_idx in range(config.predict):
            step_result = {}
            for fn in FEATURE_NAMES:
                da_list = da_by_step[step_idx][fn]
                if da_list:
                    da_arr = np.array(da_list)
                    n = len(da_list)
                    step_result[fn] = {
                        'mean': float(np.mean(da_arr)),
                        'std': float(np.std(da_arr)) if n >= 2 else 0.0,
                        'p25': float(np.percentile(da_arr, 25)) if n >= 4 else None,
                        'p50': float(np.percentile(da_arr, 50)),
                        'p75': float(np.percentile(da_arr, 75)) if n >= 4 else None,
                        'n': n,
                    }
                else:
                    step_result[fn] = {'mean': 0.0, 'std': 0.0, 'p25': None, 'p50': None, 'p75': None, 'n': 0}
            da_result[f'step{step_idx + 1}'] = step_result

            # 计算 naive DA
            actual_dirs = actual_dir_by_step[step_idx]
            if actual_dirs:
                up_ratio = np.mean(actual_dirs)
                naive_da_by_step[step_idx] = float(max(up_ratio, 1 - up_ratio))
            else:
                naive_da_by_step[step_idx] = 0.5

    # 振幅统计
    amplitude_result = {
        'mean_rate': float(np.mean(amplitude_rates)) if amplitude_rates else 1.0,
        'std_rate': float(np.std(amplitude_rates)) if amplitude_rates else 0.0,
        'perfect_pct': float(np.mean(np.abs(np.array(amplitude_rates) - 1.0) < 0.1)) if amplitude_rates else 0.0,
        'usable_pct': float(np.mean(np.abs(np.array(amplitude_rates) - 1.0) < 0.3)) if amplitude_rates else 0.0,
    }

    # 涨跌停命中率
    limit_result = limit_hit_rate(
        np.array(limit_results['pred_limit']),
        np.array(limit_results['actual_limit'])
    )

    return ic_result, da_result, amplitude_result, limit_result, naive_da_by_step


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos Predictor Evaluation')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60')
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block')
    parser.add_argument('--model', type=str, default='mini')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated models for batch comparison')
    parser.add_argument('--checkpoint', type=str, default='best_combined_model',
                        choices=['best_model', 'best_ic_model', 'best_combined_model', 'latest_model'])
    parser.add_argument('--tokenizer-path', type=str, default=None,
                        help='自定义 tokenizer 路径（优先级最高）')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='自定义 checkpoint 目录路径（优先级最高）')
    parser.add_argument('--test-path', type=str, default=None,
                        help='自定义测试数据路径（优先级最高）')
    parser.add_argument('--n-samples', type=int, default=-1,
                        help='Number of samples (-1 for full)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-pct', type=float, default=0.10,
                        help='Limit threshold (main board 10%, ChiNext 20%)')
    parser.add_argument('--use-block', action='store_true',
                        help='Use block_lb400_pd10 data')
    args = parser.parse_args()

    # DDP setup
    rank, local_rank, world_size, use_ddp = get_rank_info()
    if use_ddp:
        setup_ddp(rank, local_rank, world_size)
    device = get_device(local_rank)
    is_main = (rank == 0)

    config = DataConfig(
        norm_mode=args.norm_mode,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
    )

    # 数据路径：支持自定义（优先级最高）
    if args.test_path:
        test_path = args.test_path
    else:
        test_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'test')

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Kronos Predictor Evaluation (Detrended IC)")
        print(f"{'=' * 60}")
        print(f"norm_mode: {config.norm_mode}")
        print(f"model: {args.model}")
        print(f"checkpoint: {args.checkpoint}")
        if args.tokenizer_path:
            print(f"tokenizer_path: {args.tokenizer_path} (custom)")
        if args.checkpoint_path:
            print(f"checkpoint_path: {args.checkpoint_path} (custom)")
        print(f"test_path: {test_path}")
        if args.test_path:
            print(f"  (custom data path)")
        print(f"n_samples: {args.n_samples if args.n_samples > 0 else 'FULL'}")
        if use_ddp:
            print(f"DDP: {world_size} GPUs")
        print(f"{'=' * 60}")

    # 加载测试数据
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    if is_main:
        print(f"Test data: {len(test_data)} stocks")

    # 模型列表
    if args.models:
        model_types = args.models.split(',')
    else:
        model_types = [args.model]

    for model_type in model_types:
        if is_main:
            print(f"\n--- Evaluating {model_type} ---")

        model, tokenizer = load_model_and_tokenizer(
            config.norm_mode, model_type, device, args.checkpoint,
            lookback=config.lookback,
            predict=config.predict,
            split_mode=config.split_mode,
            custom_tokenizer_path=args.tokenizer_path,
            custom_checkpoint_path=args.checkpoint_path,
        )

        ic_result, da_result, amplitude_result, limit_result, naive_da_by_step = evaluate(
            model, tokenizer, test_data, config, device,
            n_samples=args.n_samples,
            seed=args.seed,
            limit_pct=args.limit_pct,
            rank=rank,
            world_size=world_size,
            model_type=model_type,
        )

        # 输出报告（只 rank 0）
        if is_main:
            report = format_metrics_report(
                ic_result, da_result, amplitude_result, limit_result,
                naive_da_by_step=naive_da_by_step,
                predict=config.predict
            )
            print(report)

    cleanup_ddp()


if __name__ == '__main__':
    main()