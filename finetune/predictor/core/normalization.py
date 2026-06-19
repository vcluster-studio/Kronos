"""
Kronos Predictor Core Normalization Module

归一化器实现

两种模式：
1. FullWindowNormalizer: 基于 lookback 窗口的 mean/std
2. SlidingMANormalizer: 每点基于前 N 步的 rolling mean/std

关键：min_periods=1 与现有 dataset.py:185 对齐
"""

import numpy as np
from typing import Optional, Tuple


class FullWindowNormalizer:
    """
    全窗口归一化器

    基于 lookback 窗口的 mean/std，对整个窗口归一化。
    用于 mode1/full_window 模式。
    """

    def __init__(self, clip: float = 5.0):
        """
        Args:
            clip: 归一化后截断范围
        """
        self.clip = clip

    def normalize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对输入序列归一化

        Args:
            x: (lookback, n_features) 原始数据

        Returns:
            (x_norm, mean, std)
        """
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0) + 1e-5
        x_norm = (x - x_mean) / x_std
        x_norm = np.clip(x_norm, -self.clip, self.clip)
        return x_norm, x_mean, x_std

    def denormalize(self, x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        反归一化

        Args:
            x_norm: 归一化后的数据
            mean: 均值
            std: 标准差

        Returns:
            原始数据
        """
        return x_norm * std + mean


class SlidingMANormalizer:
    """
    滑动均线归一化器

    每点基于前 N 步的 rolling mean/std 归一化。
    用于 sliding_ma{20,60,120} 模式。

    关键：min_periods=1 与现有 dataset.py:185 对齐
    - 窗口前若干点用 expanding window（数据不足时逐步扩大）
    - 保证新旧数据输出一致
    """

    def __init__(self, window: int = 60, clip: float = 5.0, min_periods: int = 1):
        """
        Args:
            window: 滑动窗口大小（20/60/120）
            clip: 归一化后截断范围
            min_periods: 最小计算点数（=1 与现有实现一致）
        """
        self.window = window
        self.clip = clip
        self.min_periods = min_periods  # 对齐 dataset.py:185

    def normalize(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对序列进行滑动归一化

        Args:
            series: (seq_len, n_features) 原始数据

        Returns:
            (normalized, means, stds)
            - normalized: 归一化后的数据
            - means: 各点的滚动均值
            - stds: 各点的滚动标准差
        """
        import pandas as pd

        seq_len, n_features = series.shape

        normalized = np.zeros_like(series)
        means = np.zeros_like(series)
        stds = np.zeros_like(series)

        for fi in range(n_features):
            # 使用 pandas rolling 计算
            s = pd.Series(series[:, fi])

            # rolling mean/std，min_periods=1 与现有实现一致
            rolling_mean = s.rolling(window=self.window, min_periods=self.min_periods).mean().values.copy()
            rolling_std = s.rolling(window=self.window, min_periods=self.min_periods).std().values.copy()

            # 前若干点（min_periods 之前）的 expanding 处理
            # min_periods=1 时，第一个点用自己的值作为 mean，std=0
            # 但 std=0 会导致除零，这里用 expanding 填补

            # 对 min_periods 之前的点用 expanding mean/std
            if self.min_periods > 1:
                expanding_mean = s.expanding(min_periods=1).mean().values[:self.min_periods-1]
                expanding_std = s.expanding(min_periods=1).std().values[:self.min_periods-1]
                rolling_mean[:self.min_periods-1] = expanding_mean
                rolling_std[:self.min_periods-1] = expanding_std

            # 填充 std=0 的点（使用 expanding std 或设为 1）
            zero_std_mask = rolling_std < 1e-8
            if zero_std_mask.any():
                # 使用 expanding std 填充
                expanding_std_full = s.expanding(min_periods=1).std().values
                rolling_std[zero_std_mask] = expanding_std_full[zero_std_mask]
                # 如果仍为 0（单点），设为 1（保持原值）
                rolling_std[rolling_std < 1e-8] = 1.0

            means[:, fi] = rolling_mean
            stds[:, fi] = rolling_std + 1e-5

            normalized[:, fi] = np.clip(
                (series[:, fi] - rolling_mean) / (rolling_std + 1e-5),
                -self.clip, self.clip
            )

        return normalized, means, stds

    def normalize_target_window(
        self,
        series: np.ndarray,
        lookback: int,
        predict: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        仅归一化 target 区间（用于推理）

        Args:
            series: (lookback + predict, n_features) 完整窗口
            lookback: 回看长度
            predict: 预测长度

        Returns:
            (target_normalized, target_means, target_stds)
            仅返回 target 区间的归一化结果
        """
        normalized, means, stds = self.normalize(series)
        return normalized[lookback:], means[lookback:], stds[lookback:]


class NormalizerFactory:
    """
    归一化器工厂

    根据 norm_mode 创建对应的归一化器
    """

    @staticmethod
    def create(norm_mode: str, clip: float = 5.0) -> object:
        """
        创建归一化器

        Args:
            norm_mode: 'full_window' | 'sliding_ma{N}'
            clip: 截断范围

        Returns:
            对应的归一化器实例
        """
        if norm_mode == 'full_window':
            return FullWindowNormalizer(clip=clip)
        elif norm_mode.startswith('sliding_ma'):
            window = int(norm_mode.split('_ma')[-1])
            return SlidingMANormalizer(window=window, clip=clip, min_periods=1)
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

    @staticmethod
    def get_required_history(norm_mode: str) -> int:
        """
        获取 norm_mode 需要的历史数据长度

        Args:
            norm_mode: 归一化模式

        Returns:
            需要的历史长度（full_window=0, sliding_ma60=60）
        """
        if norm_mode == 'full_window':
            return 0
        elif norm_mode.startswith('sliding_ma'):
            return int(norm_mode.split('_ma')[-1])
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")


# ============================================================================
# 辅助函数
# ============================================================================

def get_normalizer(norm_mode: str, clip: float = 5.0):
    """
    获取归一化器（便捷函数）

    Args:
        norm_mode: 归一化模式
        clip: 截断范围

    Returns:
        归一化器实例
    """
    return NormalizerFactory.create(norm_mode, clip)