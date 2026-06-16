"""
Kronos Predictor Training Script - DDP多卡版本

支持两种归一化模式：
  - ma60: 使用 MA60 预归一化数据 + MA60 tokenizer（默认）
  - full_window: 使用原始数据 + pretrained tokenizer，训练时动态归一化

Usage:
    # MA60 模式（默认）
    torchrun --nproc_per_node=N train_ddp.py --model mini --lookback 400

    # Full window 模式（mode1）
    torchrun --nproc_per_node=N train_ddp.py --model mini --norm-mode full_window --lookback 200

数据格式：
    MA60: {symbol: {'normalized', 'means', 'stds', 'original', 'index', 'windows'?}}
    Full window: {symbol: DataFrame with DatetimeIndex, columns=[open, high, low, close, vol, amt]}
"""

import os
import sys
import json
import time
import argparse
from time import gmtime, strftime
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# ============================================================================
# 配置
# ============================================================================

# MA60 tokenizer 路径
TOKENIZER_MA60 = 'outputs/tokenizers/ma60_tokenizer_base_v1/checkpoints/best_model'

# Full window 归一化使用的 tokenizer（pretrained，按模型类型匹配）
TOKENIZER_FULL_WINDOW = {
    'mini': 'pretrained/Kronos-Tokenizer-2k',
    'small': 'pretrained/Kronos-Tokenizer-base',
    'base': 'pretrained/Kronos-Tokenizer-base',
}

# Predictor 预训练路径
MODEL_PATHS = {
    'mini': 'pretrained/Kronos-mini',
    'small': 'pretrained/Kronos-small',
    'base': 'pretrained/Kronos-base',
}

# max_context 配置
MAX_CONTEXT = {
    'mini': 2048,
    'small': 512,
    'base': 512,
}

# 训练参数
TRAINING_PARAMS = {
    'epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.003,
    'weight_decay': 0.01,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'seed': 100,
    'lookback': 400,
    'predict': 10,
    'clip': 5.0,
    'max_context': 2048,
    # 早停
    'early_stopping_patience': 12,
    'early_stopping_grace_period': 8,
    # IC 测试（epoch间评估）
    # -1：全量评估（稳定决策，4卡并行约3分钟/epoch）
    # 1000：抽样评估（较快，IC置信区间±0.02）
    'ic_test_samples': -1,
    'ic_point': 3,
    # 方向损失
    'direction_loss_weight': 0.1,
    # Cosine Annealing LR
    'lr_scheduler': 'cosine',
    'lr_min': 1e-5,
    'warmup_epochs': 2,
    # 冻结层
    'freeze_layers': 0,
    'freeze_embedding': False,
}


def setup_ddp():
    """初始化DDP"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # 单卡时不启用DDP
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


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


class MA60Dataset(Dataset):
    """
    支持两种归一化模式的数据集：

    1. MA60 预归一化数据:
       {symbol: {'normalized', 'means', 'stds', 'original', 'index', 'windows'?}}
       直接使用预归一化值

    2. Full window 原始数据（DataFrame 格式）:
       {symbol: DataFrame with DatetimeIndex, columns=[open, high, low, close, vol, amt]}
       训练时动态计算 full_window 归一化
    """

    def __init__(self, data_type='train', config=None, data_paths=None):
        self.config = config
        self.data_type = data_type
        self.py_rng = np.random.RandomState(config.seed)
        self.norm_mode = getattr(config, 'norm_mode', 'ma60')

        # 加载数据
        data_path = data_paths[data_type]
        print(f"[{data_type.upper()}] Loading: {data_path}")
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.symbols = list(self.raw_data.keys())
        self.window = config.lookback + config.predict
        self.lookback = config.lookback

        # 检测数据格式
        sample_val = next(iter(self.raw_data.values()))
        self.is_dataframe = hasattr(sample_val, 'columns')

        # 预计算索引
        print(f"[{data_type.upper()}] Pre-computing indices (norm_mode={self.norm_mode})...")
        self.indices = []

        if self.is_dataframe:
            # Full window 格式（DataFrame with DatetimeIndex）
            for symbol in self.symbols:
                df = self.raw_data[symbol]
                seq_len = len(df)
                if seq_len >= self.window:
                    for i in range(seq_len - self.window + 1):
                        self.indices.append((symbol, i))
            print(f"[{data_type.upper()}] DataFrame format: {len(self.indices)} windows")
        else:
            # MA60 格式
            has_windows = 'windows' in sample_val
            if has_windows:
                for symbol in self.symbols:
                    data = self.raw_data[symbol]
                    for start_idx in data['windows']:
                        self.indices.append((symbol, int(start_idx)))
                print(f"[{data_type.upper()}] Windowed format: using pre-assigned {len(self.indices)} windows")
            else:
                for symbol in self.symbols:
                    data = self.raw_data[symbol]
                    seq_len = len(data['normalized'])
                    if seq_len >= self.window:
                        for i in range(seq_len - self.window + 1):
                            self.indices.append((symbol, i))

        # 样本数（-1 表示全量）
        n_iter = config.n_train_iter if data_type == 'train' else config.n_val_iter
        if n_iter <= 0 or n_iter >= len(self.indices):
            self.n_samples = len(self.indices)
        else:
            self.n_samples = n_iter
        print(f"[{data_type.upper()}] {len(self.indices)} windows, using {self.n_samples} samples/epoch")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 随机采样
        rand_idx = self.py_rng.randint(0, len(self.indices))
        symbol, start_idx = self.indices[rand_idx]
        end_idx = start_idx + self.window

        if self.is_dataframe:
            # === Full window 归一化（动态计算） ===
            df = self.raw_data[symbol]
            slice_df = df.iloc[start_idx:end_idx]
            x = slice_df.values.astype(np.float32)  # (window, 6)

            # 时间戳特征（从 DatetimeIndex 提取）
            ts = slice_df.index
            x_stamp = np.stack([
                ts.minute.values.astype(np.float32),
                ts.hour.values.astype(np.float32),
                ts.weekday.values.astype(np.float32),
                ts.day.values.astype(np.float32),
                ts.month.values.astype(np.float32),
            ], axis=1)

            # 动态 full_window 归一化：使用 lookback 窗口的 mean/std
            past_x = x[:self.lookback]
            x_mean = np.mean(past_x, axis=0)
            x_std = np.std(past_x, axis=0) + 1e-5
            x_norm = (x - x_mean) / x_std
            x_norm = np.clip(x_norm, -self.config.clip, self.config.clip)

            # 方向标签（用原始 close 值）
            original_close = x[:, 3]  # close 列
            baseline_close = original_close[self.lookback - 1]
            pred_close_end = original_close[self.lookback + self.config.predict - 1]
            direction = pred_close_end > baseline_close

            return torch.from_numpy(x_norm), torch.from_numpy(x_stamp), torch.tensor(direction, dtype=torch.float32)

        else:
            # === MA60 预归一化数据（原有逻辑） ===
            data = self.raw_data[symbol]

            x_norm = data['normalized'][start_idx:end_idx].astype(np.float32)

            timestamps = data['index'][start_idx:end_idx]
            x_stamp = np.stack([
                timestamps.minute.values,
                timestamps.hour.values,
                timestamps.weekday.values,
                timestamps.day.values,
                timestamps.month.values,
            ], axis=1).astype(np.float32)

            original_close = data['original'][start_idx:end_idx, 3]
            baseline_close = original_close[self.config.lookback - 1]
            pred_close_end = original_close[self.config.lookback + self.config.predict - 1]
            direction = pred_close_end > baseline_close

            return torch.from_numpy(x_norm), torch.from_numpy(x_stamp), torch.tensor(direction, dtype=torch.float32)


def quick_trajectory_ic_test(model, tokenizer, device, val_data, n_samples=500,
                              lookback=400, pred_len=10, clip=5.0, rng=None):
    """
    Trajectory IC 测试（新方法）

    计算同一股票内预测轨迹 vs 实际轨迹的相关性（不是 Return IC）

    返回:
        - trajectory_ics: 各特征的 trajectory IC
        - trajectory_rank_ics: 各特征的 trajectory rank IC
        - da_by_step: 各步各特征的 direction accuracy
    """
    from finetune.predictor.shared.eval import FEATURE_NAMES

    model.eval()

    # 检测是否为窗口化格式
    sample_symbol = list(val_data.keys())[0]
    has_windows = 'windows' in val_data[sample_symbol]

    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(pred_len)]

    if has_windows:
        # 窗口化格式：从 val 的预分配窗口中随机采样
        all_windows = []
        for sym in val_data:
            for start in val_data[sym]['windows']:
                all_windows.append((sym, int(start)))

        # n_samples <= 0 表示全量评估
        if n_samples <= 0 or n_samples >= len(all_windows):
            sample_indices = np.arange(len(all_windows))
        elif rng is not None:
            sample_indices = rng.choice(len(all_windows), size=n_samples, replace=False)
        else:
            sample_indices = np.arange(n_samples)

        for idx in sample_indices:
            symbol, start_idx = all_windows[idx]
            data = val_data[symbol]
            end_idx = start_idx + lookback + pred_len

            try:
                x_norm_full = data['normalized'][start_idx:end_idx].astype(np.float32)
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

                baseline = orig_full[lookback - 1]  # 最后一个历史位置

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

                    # pred shape: (1, lookback + pred_len, 6)
                    pred_norm = preds[0, lookback:lookback + pred_len, :]  # (pred_len, 6)
                    pred_raw = pred_norm * stds_full[lookback:lookback + pred_len] + means_full[lookback:lookback + pred_len]
                    actual = orig_full[lookback:lookback + pred_len]

                # 计算 Trajectory IC（各特征）
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

                # 计算 DA（各步各特征）
                for step_idx in range(pred_len):
                    for fi, fn in enumerate(FEATURE_NAMES):
                        pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                        actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                        da_by_step[step_idx][fn].append(pred_dir == actual_dir)

            except Exception as e:
                if len(trajectory_ics['close']) == 0:
                    print(f"[Trajectory IC TEST] First error: {e}")
                continue
    else:
        # 传统格式：从每只股票末尾采样
        all_symbols = list(val_data.keys())
        # n_samples <= 0 表示全量评估
        if n_samples <= 0 or n_samples >= len(all_symbols):
            symbols = all_symbols
        elif rng is not None:
            symbols = rng.choice(all_symbols, size=n_samples, replace=False).tolist()
        else:
            symbols = all_symbols[:n_samples]

        for symbol in symbols:
            data = val_data[symbol]
            seq_len = len(data['normalized'])

            if seq_len < lookback + pred_len:
                continue

            try:
                end_idx = seq_len
                start_idx = end_idx - (lookback + pred_len)

                full_len = lookback + pred_len
                x_norm_full = data['normalized'][end_idx - full_len:end_idx].astype(np.float32)
                means_full = data['means'][end_idx - full_len:end_idx]
                stds_full = data['stds'][end_idx - full_len:end_idx]
                timestamps_full = data['index'][end_idx - full_len:end_idx]
                orig_full = data['original'][end_idx - full_len:end_idx]

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
                    x_tensor = torch.from_numpy(x_norm[:lookback]).unsqueeze(0).to(device)
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

                # 计算 Trajectory IC
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

                # 计算 DA
                for step_idx in range(pred_len):
                    for fi, fn in enumerate(FEATURE_NAMES):
                        pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                        actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                        da_by_step[step_idx][fn].append(pred_dir == actual_dir)

            except Exception as e:
                if len(trajectory_ics['close']) == 0:
                    print(f"[Trajectory IC TEST] First error: {e}")
                continue

    # 聚合结果
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


def quick_trajectory_ic_test_distributed(model, tokenizer, device, val_data,
                                          local_indices, lookback=400, pred_len=10, clip=5.0,
                                          rank=0, desc="Eval", norm_mode='ma60'):
    """
    多GPU分布式Trajectory IC评估

    每个rank评估自己的窗口子集，返回局部结果供聚合

    Args:
        local_indices: 该rank负责评估的窗口列表 [(symbol, start), ...]
        rank: 当前rank编号（用于进度条显示）
        norm_mode: 'ma60' 预归一化数据 或 'full_window' 动态归一化

    Returns:
        trajectory_ics: 各特征的IC列表（用于后续聚合）
        trajectory_rics: 各特征的Rank IC列表
        da_by_step: 各步各特征的DA列表
        n_evaluated: 实际评估的窗口数
    """
    from finetune.predictor.shared.eval import FEATURE_NAMES

    model.eval()
    tokenizer.eval()

    # 检测数据格式
    sample_val = next(iter(val_data.values()))
    is_dataframe = hasattr(sample_val, 'columns')

    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(pred_len)]

    for (symbol, start_idx) in tqdm(local_indices, desc=f"GPU{rank}", disable=len(local_indices) < 50):
        end_idx = start_idx + lookback + pred_len

        try:
            if is_dataframe:
                # === Full window 归一化（DataFrame 格式） ===
                df = val_data[symbol]
                if end_idx > len(df):
                    continue

                slice_df = df.iloc[start_idx:end_idx]
                x = slice_df.values.astype(np.float32)

                # 时间戳
                ts = slice_df.index
                stamp = np.stack([
                    ts.minute.values.astype(np.float32),
                    ts.hour.values.astype(np.float32),
                    ts.weekday.values.astype(np.float32),
                    ts.day.values.astype(np.float32),
                    ts.month.values.astype(np.float32),
                ], axis=1)

                x_lookback = x[:lookback]
                x_mean = np.mean(x_lookback, axis=0)
                x_std = np.std(x_lookback, axis=0) + 1e-5
                x_norm = (x - x_mean) / x_std
                x_norm = np.clip(x_norm, -clip, clip)

                baseline = x[lookback - 1]

                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_norm[:lookback]).unsqueeze(0).to(device)
                    x_stamp_tensor = torch.from_numpy(stamp[:lookback]).unsqueeze(0).to(device)
                    y_stamp_tensor = torch.from_numpy(stamp[lookback:lookback + pred_len]).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model,
                        x_tensor, x_stamp_tensor, y_stamp_tensor,
                        max_context=2048, pred_len=pred_len,
                        clip=clip, T=1.0, top_k=0, top_p=0.9,
                        sample_count=1, verbose=False
                    )

                    pred_norm = preds[0, lookback:lookback + pred_len, :]
                    pred_raw = pred_norm * x_std + x_mean
                    actual = x[lookback:lookback + pred_len]

            else:
                # === MA60 预归一化数据（原有逻辑） ===
                data = val_data[symbol]
                if end_idx > len(data['normalized']):
                    continue

                x_norm_full = data['normalized'][start_idx:end_idx].astype(np.float32)
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

            # DA
            for step_idx in range(pred_len):
                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                    actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                    da_by_step[step_idx][fn].append(pred_dir == actual_dir)

        except Exception:
            continue

    return trajectory_ics, trajectory_rics, da_by_step, len(trajectory_ics['close'])


def aggregate_ic_results(local_ics, local_rics, local_da, pred_len,
                         world_size, device, is_main=True):
    """
    聚合各GPU的IC评估结果

    使用加权平均：IC = sum(IC_i * n_i) / sum(n_i)
    """
    from finetune.predictor.shared.eval import FEATURE_NAMES

    result = {}

    # 各特征IC聚合
    for fn in FEATURE_NAMES:
        local_ic_list = local_ics[fn]
        local_ric_list = local_rics[fn]
        n_local = len(local_ic_list)

        # 加权聚合
        n_tensor = torch.tensor([n_local], device=device)
        ic_sum_tensor = torch.tensor([sum(local_ic_list)], device=device)
        ric_sum_tensor = torch.tensor([sum(local_ric_list)], device=device)

        if world_size > 1:
            # 收集各rank的结果
            gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]
            gathered_ic = [torch.zeros_like(ic_sum_tensor) for _ in range(world_size)]
            gathered_ric = [torch.zeros_like(ric_sum_tensor) for _ in range(world_size)]

            dist.all_gather(gathered_n, n_tensor)
            dist.all_gather(gathered_ic, ic_sum_tensor)
            dist.all_gather(gathered_ric, ric_sum_tensor)

            total_n = sum(t.item() for t in gathered_n)
            total_ic = sum(t.item() for t in gathered_ic)
            total_ric = sum(t.item() for t in gathered_ric)
        else:
            total_n = n_local
            total_ic = sum(local_ic_list)
            total_ric = sum(local_ric_list)

        result[f'{fn}_trajectory_ic'] = total_ic / total_n if total_n > 0 else 0.0
        result[f'{fn}_trajectory_rank_ic'] = total_ric / total_n if total_n > 0 else 0.0

    # DA聚合
    for step_idx in range(pred_len):
        for fn in FEATURE_NAMES:
            local_da_list = local_da[step_idx][fn]
            n_local = len(local_da_list)
            da_sum = sum(local_da_list)

            n_tensor = torch.tensor([n_local], device=device)
            da_sum_tensor = torch.tensor([da_sum], device=device)

            if world_size > 1:
                gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]
                gathered_da = [torch.zeros_like(da_sum_tensor) for _ in range(world_size)]

                dist.all_gather(gathered_n, n_tensor)
                dist.all_gather(gathered_da, da_sum_tensor)

                total_n = sum(t.item() for t in gathered_n)
                total_da = sum(t.item() for t in gathered_da)
            else:
                total_n = n_local
                total_da = da_sum

            result[f'{fn}_da_step{step_idx+1}'] = total_da / total_n if total_n > 0 else 0.0

    result['n_samples'] = total_n

    return result


def freeze_model_layers(model, freeze_layers=2, freeze_embedding=True, use_ddp=False):
    """冻结模型前 N 层 transformer 和 embedding"""
    unwrapped = model.module if use_ddp else model

    # 冻结 embedding
    if freeze_embedding:
        for param in unwrapped.embedding.parameters():
            param.requires_grad = False
        for param in unwrapped.time_emb.parameters():
            param.requires_grad = False
        print(f"[FREEZE] Embedding + TemporalEmb frozen")

    # 冻结前 N 层 transformer
    for i in range(min(freeze_layers, len(unwrapped.transformer))):
        for param in unwrapped.transformer[i].parameters():
            param.requires_grad = False
        print(f"[FREEZE] Transformer layer {i} frozen")

    # 统计可训练参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FREEZE] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)")


def train_model(model, tokenizer, device, config, save_dir, data_paths=None, val_data=None,
                rank=0, local_rank=0, world_size=1, use_ddp=False):
    """训练 MA60 predictor"""
    start_time = time.time()
    is_main = (rank == 0)

    if is_main:
        print(f"BATCHSIZE: {config.batch_size}")
        print(f"LR: {config.learning_rate}")
        print(f"LR Scheduler: {config.lr_scheduler}")
        print(f"Weight Decay: {config.weight_decay}")
        print(f"Lookback: {config.lookback}")
        print(f"Predict: {config.predict}")

    # 冻结层
    if config.freeze_layers > 0 or config.freeze_embedding:
        freeze_model_layers(model, config.freeze_layers, config.freeze_embedding, use_ddp)

    # 方向预测头（从 hidden state 预测 close 涨跌）
    unwrapped_model = model.module if use_ddp else model
    d_model = unwrapped_model.d_model
    close_direction_head = torch.nn.Linear(d_model, 1).to(device)
    if use_ddp:
        close_direction_head = DDP(close_direction_head, device_ids=[local_rank])

    # 数据集
    train_dataset = MA60Dataset('train', config=config, data_paths=data_paths)
    val_dataset = MA60Dataset('val', config=config, data_paths=data_paths)

    # DDP时也用RandomSampler，各rank种子不同
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                              sampler=val_sampler, num_workers=0, pin_memory=True)

    if is_main:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 优化器（只优化可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(close_direction_head.parameters())
    all_params = trainable_params + head_params
    optimizer = torch.optim.AdamW(all_params, lr=config.learning_rate,
                                   weight_decay=config.weight_decay,
                                   betas=(config.adam_beta1, config.adam_beta2))

    # Cosine Annealing LR（无 warmup，直接 cosine）
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.lr_min
    )

    best_val_loss = float('inf')
    best_ic = -999
    patience_counter = 0

    # 预计算 bit_mask 用于 soft decode（只需一次）
    tokenizer_module = tokenizer.module if hasattr(tokenizer, 'module') else tokenizer
    vocab_s1 = 2 ** tokenizer_module.s1_bits
    vocab_s2 = 2 ** tokenizer_module.s2_bits
    s1_bit_mask = torch.zeros(vocab_s1, tokenizer_module.s1_bits, device=device)
    s2_bit_mask = torch.zeros(vocab_s2, tokenizer_module.s2_bits, device=device)
    for idx in range(vocab_s1):
        for b in range(tokenizer_module.s1_bits):
            if (idx >> b) & 1:
                s1_bit_mask[idx, b] = 1.0
    for idx in range(vocab_s2):
        for b in range(tokenizer_module.s2_bits):
            if (idx >> b) & 1:
                s2_bit_mask[idx, b] = 1.0
    codebook_dim = tokenizer_module.codebook_dim
    q_scale = 1.0 / (codebook_dim ** 0.5)

    history = {'train_loss': [], 'val_loss': [], 'ic': [], 'lr': []}

    for epoch_idx in range(config.epochs):
        epoch_start = time.time()

        model.train()
        close_direction_head.train()

        # 每个rank种子不同，避免重复采样
        train_dataset.py_rng.seed(config.seed + epoch_idx * 10000 + rank)

        epoch_losses = []
        current_lr = optimizer.param_groups[0]['lr']

        if is_main:
            print(f"\n=== Epoch {epoch_idx+1}/{config.epochs} ===")
            print(f"LR: {current_lr:.6f}")

        for i, (batch_x, batch_stamp, batch_direction) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_stamp = batch_stamp.to(device, non_blocking=True)
            batch_direction = batch_direction.to(device, non_blocking=True)

            # Tokenize
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward
            if use_ddp:
                s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
                    token_seq_0, token_seq_1, batch_stamp
                )
                recon_loss, ce_s1, ce_s2 = model.module.head.compute_loss(
                    s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
                )
            else:
                s1_logits, s2_logits, hidden = model.forward_with_hidden(
                    token_seq_0, token_seq_1, batch_stamp
                )
                recon_loss, ce_s1, ce_s2 = model.head.compute_loss(
                    s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
                )

            # 方向损失（从 hidden state 预测 close 涨跌）
            pred_hidden = hidden[:, -1, :]
            direction_logits = close_direction_head(pred_hidden).squeeze(-1)
            direction_pred = torch.sigmoid(direction_logits)
            direction_loss = F.binary_cross_entropy(direction_pred, batch_direction)

            total_loss = recon_loss + config.direction_loss_weight * direction_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
            optimizer.step()

            scheduler.step()

            epoch_losses.append(total_loss.item())

            if is_main and ((i + 1) % 50 == 0 or i == 0):
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (i + 1) / len(train_loader) * 100
                print(f"  Batch {i+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {total_loss.item():.4f}, Avg: {avg_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # Validation
        model.eval()
        close_direction_head.eval()

        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_stamp, batch_direction in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_stamp = batch_stamp.to(device, non_blocking=True)
                batch_direction = batch_direction.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                if use_ddp:
                    s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
                        token_seq_0, token_seq_1, batch_stamp
                    )
                    recon_loss, _, _ = model.module.head.compute_loss(
                        s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
                    )
                else:
                    s1_logits, s2_logits, hidden = model.forward_with_hidden(
                        token_seq_0, token_seq_1, batch_stamp
                    )
                    recon_loss, _, _ = model.head.compute_loss(
                        s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
                    )

                # 方向损失（与训练一致）
                pred_hidden = hidden[:, -1, :]
                direction_logits = close_direction_head(pred_hidden).squeeze(-1)
                direction_pred = torch.sigmoid(direction_logits)
                direction_loss = F.binary_cross_entropy(direction_pred, batch_direction)

                val_loss = recon_loss + config.direction_loss_weight * direction_loss

                val_loss_sum += val_loss.item()
                val_batches += 1

        # 同步 validation loss
        if use_ddp:
            val_loss_tensor = torch.tensor([val_loss_sum, val_batches], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss_sum = val_loss_tensor[0].item()
            val_batches = val_loss_tensor[1].item()

        avg_val_loss = val_loss_sum / val_batches
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # ===== 多GPU分布式 Trajectory IC 评估 =====
        # 所有GPU都参与评估，每个GPU处理部分窗口

        # 1. 准备窗口索引（所有rank都需要，用相同seed确保一致性）
        sample_val = val_data[list(val_data.keys())[0]]
        is_df = hasattr(sample_val, 'columns')
        all_windows = []
        if is_df:
            # Full window 格式：每只股票取最后 1 个窗口（与 eval_ddp 一致）
            window_total = config.lookback + config.predict
            for sym in val_data:
                df = val_data[sym]
                if len(df) >= window_total:
                    all_windows.append((sym, len(df) - window_total))
        elif 'windows' in sample_val:
            # MA60 窗口化格式
            for sym in val_data:
                for start in val_data[sym]['windows']:
                    all_windows.append((sym, int(start)))
        else:
            # MA60 传统格式（所有滑动窗口）
            for sym in val_data:
                seq_len = len(val_data[sym]['normalized'])
                if seq_len >= config.lookback + config.predict:
                    for i in range(seq_len - config.lookback - config.predict + 1):
                        all_windows.append((sym, i))

        # 2. 用固定seed抽样（确保各rank抽样相同集合）
        ic_rng = np.random.RandomState(config.seed + epoch_idx * 9999)
        n_samples = config.ic_test_samples
        if n_samples <= 0 or n_samples >= len(all_windows):
            sampled_indices = np.arange(len(all_windows))
        else:
            sampled_indices = ic_rng.choice(len(all_windows), size=n_samples, replace=False)

        sampled_windows = [all_windows[i] for i in sampled_indices]

        # 3. 分片：每个rank处理一部分
        per_rank = len(sampled_windows) // world_size
        start_idx = rank * per_rank
        end_idx = start_idx + per_rank if rank < world_size - 1 else len(sampled_windows)
        local_windows = sampled_windows[start_idx:end_idx]

        # 4. 各rank并行评估
        local_ics, local_rics, local_da, n_local = quick_trajectory_ic_test_distributed(
            unwrapped_model, tokenizer, device, val_data,
            local_windows, lookback=config.lookback,
            pred_len=config.predict, clip=config.clip,
            rank=rank, desc=f"Epoch{epoch_idx+1}",
            norm_mode=config.norm_mode
        )

        # 5. 同步并聚合结果
        if use_ddp:
            dist.barrier()

        traj_result = aggregate_ic_results(
            local_ics, local_rics, local_da, config.predict,
            world_size, device, is_main=is_main
        )

        current_ic = 0
        if traj_result and is_main:
            current_ic = traj_result.get('close_trajectory_ic', 0)
            history['ic'].append(current_ic)

            # 各特征的 Trajectory IC 汇总
            print(f"\n  {'Feature':<8} {'Traj_IC':>8} {'RankIC':>8}")
            for fn in ['open', 'high', 'low', 'close', 'vol', 'amt']:
                tic = traj_result.get(f'{fn}_trajectory_ic', 0)
                ric = traj_result.get(f'{fn}_trajectory_rank_ic', 0)
                print(f"  {fn:<8} {tic:>8.4f} {ric:>8.4f}")

            # 各步各特征 DA 表格
            print(f"\n  {'Step':<6} {'open_DA':>7} {'high_DA':>7} {'low_DA':>7} {'close_DA':>7} {'vol_DA':>7} {'amt_DA':>7}")
            for step_idx in range(config.predict):
                suffix = f'_step{step_idx+1}'
                print(f"  +{step_idx+1:<5} "
                      f"{traj_result.get(f'open_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'high_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'low_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'close_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'vol_da{suffix}', 0):>7.0%} "
                      f"{traj_result.get(f'amt_da{suffix}', 0):>7.0%}")

        # 评估完成后再同步（确保所有rank完成）
        if use_ddp:
            dist.barrier()

        # 广播 IC 结果（让非main rank也知道当前IC）
        if use_ddp:
            ic_tensor = torch.tensor([current_ic], device=device)
            dist.broadcast(ic_tensor, src=0)
            current_ic = ic_tensor[0].item()

        # IC 滑动均值（用于决策，减少噪声）
        ic_window = 3
        if len(history['ic']) >= ic_window:
            ic_smoothed = np.mean(history['ic'][-ic_window:])
        else:
            ic_smoothed = current_ic

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        if is_main:
            print(f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
            print(f"Trajectory IC (close): {current_ic:.4f}, IC_smoothed: {ic_smoothed:.4f} (best: {best_ic:.4f})")
            print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}")
            print(f"[LR] {current_lr:.6f}")

        # Save latest (only main)
        if is_main:
            latest_path = f"{save_dir}/checkpoints/latest_model"
            unwrapped_model.save_pretrained(latest_path)

        # Check improvement
        improved = False

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            improved = True
            if is_main:
                save_path = f"{save_dir}/checkpoints/best_model"
                unwrapped_model.save_pretrained(save_path)
                print(f"[VAL LOSS SAVED] {best_val_loss:.4f}")

        if ic_smoothed > best_ic:
            best_ic = ic_smoothed
            patience_counter = 0
            improved = True
            if is_main:
                ic_save_path = f"{save_dir}/checkpoints/best_ic_model"
                unwrapped_model.save_pretrained(ic_save_path)
                print(f"[IC SAVED] {best_ic:.4f}")

        if not improved:
            patience_counter += 1

        # Early stopping
        if epoch_idx >= config.early_stopping_grace_period:
            if patience_counter >= config.early_stopping_patience:
                if is_main:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                    final_path = f"{save_dir}/checkpoints/final_model"
                    unwrapped_model.save_pretrained(final_path)
                break

        if is_main:
            print(flush=True)

    # Final save
    if is_main:
        final_path = f"{save_dir}/checkpoints/final_model"
        unwrapped_model.save_pretrained(final_path)

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'epochs_trained': epoch_idx + 1,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='MA60 Predictor Training (DDP)')
    parser.add_argument('--model', type=str, default='mini', choices=['mini', 'small', 'base'],
                        help='Model type: mini (4.1M), small (24.7M), base (102M)')
    parser.add_argument('--lookback', type=int, default=None,
                        help='Lookback window size (default: 400 for ma60, 200 for full_window)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--n-samples', type=int, default=-1,
                        help='Number of samples for IC evaluation (-1 for full)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--save-folder', type=str, default=None,
                        help='Save folder name (default: auto generate from model and lookback)')
    parser.add_argument('--norm-mode', type=str, default='ma60',
                        choices=['ma60', 'full_window'],
                        help='Normalization mode: ma60 (pre-normalized) or full_window (dynamic)')
    args = parser.parse_args()

    # 动态构建数据路径（基于norm_mode）
    if args.lookback is None:
        args.lookback = 200 if args.norm_mode == 'full_window' else 400
    if args.norm_mode == 'full_window':
        data_dir = 'finetune/data/global_norm/full_series'
    else:
        data_dir = f'finetune/data/ma60_norm/windowed_lb{args.lookback}_pd10'
    data_paths = {
        'train': f'{data_dir}/train_data.pkl',
        'val': f'{data_dir}/val_data.pkl',
        'test': f'{data_dir}/test_data.pkl',
    }

    # 自动生成保存目录名
    if args.save_folder is None:
        args.save_folder = f'mode_{args.model}_lb{args.lookback}'

    # DDP setup
    rank, local_rank, world_size, use_ddp = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)

    if is_main:
        print("\n" + "="*60)
        print(f"MA60 Predictor Training (DDP) - {args.model} + lb{args.lookback}")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Norm mode: {args.norm_mode}")
        print(f"Tokenizer: {TOKENIZER_FULL_WINDOW[args.model] if args.norm_mode == 'full_window' else TOKENIZER_MA60}")
        print(f"Model: {MODEL_PATHS[args.model]}")
        print(f"Data: {data_paths['train']}")
        print(f"Device: {device}")
        if args.resume:
            print(f"Resume from: {args.resume}")
        print("="*60)

    set_seed(TRAINING_PARAMS['seed'])

    # 保存目录
    save_dir = os.path.join(project_root, "outputs/models", args.save_folder)
    if is_main:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # 加载 tokenizer（根据norm_mode选择）
    if args.norm_mode == 'full_window':
        tokenizer_path = os.path.join(project_root, TOKENIZER_FULL_WINDOW[args.model])
    else:
        tokenizer_path = os.path.join(project_root, TOKENIZER_MA60)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)
    if is_main:
        print(f"Tokenizer loaded from: {tokenizer_path}")

    # 加载 predictor (支持 resume)
    if args.resume:
        predictor_path = args.resume
        if is_main:
            print(f"Predictor loaded from (resume): {predictor_path}")
    else:
        predictor_path = os.path.join(project_root, MODEL_PATHS[args.model])
        if is_main:
            print(f"Predictor loaded from: {MODEL_PATHS[args.model]}")
    model = Kronos.from_pretrained(predictor_path)
    model.to(device)
    if is_main:
        print(f"Model size: {get_model_size(model):.2f}M")

    # DDP wrap
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # 配置
    class Config:
        pass

    config = Config()
    config.seed = TRAINING_PARAMS['seed']
    config.lookback = args.lookback
    config.predict = TRAINING_PARAMS['predict']
    config.clip = TRAINING_PARAMS['clip']
    config.max_context = MAX_CONTEXT[args.model]
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.adam_beta1 = TRAINING_PARAMS['adam_beta1']
    config.adam_beta2 = TRAINING_PARAMS['adam_beta2']
    config.n_train_iter = -1  # 全量训练
    config.n_val_iter = -1    # 全量验证
    config.early_stopping_patience = TRAINING_PARAMS['early_stopping_patience']
    config.early_stopping_grace_period = TRAINING_PARAMS['early_stopping_grace_period']
    config.ic_test_samples = args.n_samples
    config.ic_point = TRAINING_PARAMS['ic_point']
    config.direction_loss_weight = TRAINING_PARAMS['direction_loss_weight']
    config.lr_scheduler = TRAINING_PARAMS['lr_scheduler']
    config.lr_min = TRAINING_PARAMS['lr_min']
    config.warmup_epochs = TRAINING_PARAMS['warmup_epochs']
    config.freeze_layers = TRAINING_PARAMS['freeze_layers']
    config.freeze_embedding = TRAINING_PARAMS['freeze_embedding']
    config.norm_mode = args.norm_mode

    # 加载验证数据（IC 评估用，所有rank都需要）
    val_path = data_paths['val']
    if is_main:
        print(f"Loading val data for IC evaluation: {val_path}")
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    if is_main:
        print(f"Val: {len(val_data)} stocks")

    # 训练
    result = train_model(model, tokenizer, device, config, save_dir, data_paths, val_data,
                         rank, local_rank, world_size, use_ddp)

    # 保存结果 (only main)
    if is_main:
        summary = {
            'tokenizer': tokenizer_path,
            'norm_mode': args.norm_mode,
            'data_path': data_paths['train'],
            'config': {
                'batch_size': config.batch_size,
                'world_size': world_size,
                'total_batch_size': config.batch_size * world_size,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'epochs': config.epochs,
            },
            'result': {
                'best_val_loss': result['best_val_loss'],
                'best_ic': result['best_ic'],
                'epochs_trained': result['epochs_trained'],
            }
        }

        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4, default=float)

        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"Best IC: {result['best_ic']:.4f}")
        print(f"Epochs: {result['epochs_trained']}")
        print(f"Saved to: {save_dir}")
        print("="*60)

    cleanup_ddp()


if __name__ == '__main__':
    main()