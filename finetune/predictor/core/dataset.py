"""
Kronos Predictor Core Dataset Module

统一数据集实现（runtime 归一化）

支持两种数据格式：
1. DataFrame（full_window 动态归一化）
2. dict（MA60 预归一化，含 means/stds）

关键：
- runtime 归一化（pkl 存原始 OHLCV）
- 与现有 dataset.py 兼容（min_periods=1）
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset

from .config import DataConfig
from .normalization import get_normalizer, FullWindowNormalizer, SlidingMANormalizer
from .schema import SampleSchema, dict_to_sample
from .utils import extract_time_features


class KronosDataset(Dataset):
    """
    Kronos 统一数据集

    支持：
    - runtime 归一化（多种 norm_mode）
    - DataFrame 和 dict 两种数据格式
    - 窗口化预分配索引

    与现有 dataset.py 兼容：
    - min_periods=1（滑动归一化边界）
    - clip=5.0
    """

    def __init__(
        self,
        data: Dict[str, Any],
        indices: List[Tuple[str, int]],
        config: DataConfig,
        mode: str = 'train'
    ):
        """
        Args:
            data: 原始数据 {symbol: DataFrame or dict}
            indices: 窗口索引列表 [(symbol, start)]
            config: DataConfig
            mode: 'train' | 'val' | 'test'
        """
        self.data = data
        self.indices = indices
        self.config = config
        self.mode = mode

        self.lookback = config.lookback
        self.predict = config.predict
        self.window_size = self.lookback + self.predict
        self.norm_mode = config.norm_mode
        self.clip = config.clip

        # 归一化器
        self.normalizer = get_normalizer(self.norm_mode, clip=self.clip)

        # 随机采样器（训练时随机采样）
        self.py_rng = np.random.RandomState(config.seed)

        # 检测数据格式
        sample_val = next(iter(data.values()))
        self.is_dataframe = hasattr(sample_val, 'columns')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            (x_norm, x_stamp, y_stamp, meta)
            - x_norm: (lookback, n_features) 归一化后的输入
            - x_stamp: (lookback, 5) 时间戳特征
            - y_stamp: (predict, 5) 目标时间戳特征
            - meta: {symbol, start, baseline, means, stds}
        """
        # 直接使用传入 idx，让 DataLoader 的 sampler 控制采样顺序
        # DDP 时 DistributedSampler 或 RandomSampler 保证各 rank 数据不同
        symbol, start = self.indices[idx]

        end = start + self.window_size

        if self.is_dataframe:
            return self._get_dataframe_sample(symbol, start, end)
        else:
            return self._get_dict_sample(symbol, start, end)

    def _get_dataframe_sample(self, symbol: str, start: int, end: int) -> Tuple:
        """
        DataFrame 格式样本（runtime 归一化）

        用于 full_window 模式
        """
        df = self.data[symbol]
        window_df = df.iloc[start:end]

        values = window_df.values.astype(np.float32)  # (window, 6)
        timestamps = window_df.index

        # 动态归一化
        if self.norm_mode == 'full_window':
            # Full window: 基于 lookback 窗口归一化
            x_raw = values[:self.lookback]
            x_norm, x_mean, x_std = self.normalizer.normalize(x_raw)

            # 整个窗口用相同的 mean/std
            full_norm = (values - x_mean) / (x_std + 1e-5)
            full_norm = np.clip(full_norm, -self.clip, self.clip)
            x_norm = full_norm[:self.lookback]

            baseline = values[self.lookback - 1]

        else:
            # Sliding MA: 需要完整历史
            # DataFrame 格式通常用于 full_window，这里兼容
            full_norm, means, stds = self.normalizer.normalize(values)
            x_norm = full_norm[:self.lookback]

            baseline = values[self.lookback - 1]
            x_mean = means[self.lookback - 1]
            x_std = stds[self.lookback - 1]

        # 时间戳特征
        x_stamp = extract_time_features(timestamps[:self.lookback])
        y_stamp = extract_time_features(timestamps[self.lookback:self.window_size])

        meta = {
            'symbol': symbol,
            'start': start,
            'baseline': baseline,
            'mean': x_mean,
            'std': x_std,
        }

        return (
            x_norm.astype(np.float32),
            x_stamp.astype(np.float32),
            y_stamp.astype(np.float32),
            meta
        )

    def _get_dict_sample(self, symbol: str, start: int, end: int) -> Tuple:
        """
        dict 格式样本（MA60 预归一化数据）

        数据结构：
        {
            'normalized': np.ndarray,
            'original': np.ndarray,
            'means': np.ndarray,
            'stds': np.ndarray,
            'index': DatetimeIndex,
            'windows': List[int] (optional)
        }
        """
        d = self.data[symbol]

        normalized = d['normalized'][start:end].astype(np.float32)
        original = d['original'][start:end]
        means = d['means'][start:end]
        stds = d['stds'][start:end]
        timestamps = d['index'][start:end]

        # 归一化数据已预计算，直接使用
        x_norm = normalized[:self.lookback]

        # 反归一化元数据
        baseline = original[self.lookback - 1]

        # 时间戳特征
        x_stamp = extract_time_features(timestamps[:self.lookback])
        y_stamp = extract_time_features(timestamps[self.lookback:self.window_size])

        meta = {
            'symbol': symbol,
            'start': start,
            'baseline': baseline,
            'means_target': means[self.lookback:self.window_size],
            'stds_target': stds[self.lookback:self.window_size],
            'original_target': original[self.lookback:self.window_size],
        }

        return (
            x_norm,
            x_stamp.astype(np.float32),
            y_stamp.astype(np.float32),
            meta
        )


class KronosWindowedDataset(KronosDataset):
    """
    窗口化数据集（使用预分配 windows）

    用于 block_lb400_pd10 格式
    """

    def __init__(
        self,
        data: Dict[str, Any],
        config: DataConfig,
        mode: str = 'train'
    ):
        """
        Args:
            data: 数据（含 windows 字段）
            config: DataConfig
            mode: 'train' | 'val' | 'test'
        """
        # 从预分配 windows 构建索引
        indices = []
        for symbol, d in data.items():
            windows = d.get('windows', [])
            for w in windows:
                indices.append((symbol, int(w)))

        super().__init__(data, indices, config, mode)


def load_split_data(
    data_path: str,
    config: DataConfig,
    mode: str = 'train'
) -> KronosDataset:
    """
    加载分割数据并创建 Dataset

    Args:
        data_path: pkl 文件路径
        config: DataConfig
        mode: 'train' | 'val' | 'test'

    Returns:
        KronosDataset
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 检测是否有预分配 windows
    sample_val = next(iter(data.values()))
    has_windows = isinstance(sample_val, dict) and 'windows' in sample_val

    if has_windows:
        return KronosWindowedDataset(data, config, mode)

    # 无预分配 windows，构建全量索引
    indices = []
    for symbol, d in data.items():
        if hasattr(d, 'columns'):
            # DataFrame
            seq_len = len(d)
        else:
            # dict
            seq_len = len(d.get('normalized', d.get('values', [])))

        if seq_len >= config.lookback + config.predict:
            for i in range(seq_len - config.lookback - config.predict + 1):
                indices.append((symbol, i))

    return KronosDataset(data, indices, config, mode)


def collate_fn(batch):
    """
    自定义 collate 函数

    Args:
        batch: [(x_norm, x_stamp, y_stamp, meta)]

    Returns:
        (x_norm_batch, x_stamp_batch, y_stamp_batch, meta_list)
    """
    import torch

    x_norms = []
    x_stamps = []
    y_stamps = []
    metas = []

    for x_norm, x_stamp, y_stamp, meta in batch:
        x_norms.append(x_norm)
        x_stamps.append(x_stamp)
        y_stamps.append(y_stamp)
        metas.append(meta)

    return (
        torch.from_numpy(np.array(x_norms)),
        torch.from_numpy(np.array(x_stamps)),
        torch.from_numpy(np.array(y_stamps)),
        metas
    )