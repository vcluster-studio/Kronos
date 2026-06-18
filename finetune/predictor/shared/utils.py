"""
评估工具函数
"""

import numpy as np
from scipy.stats import spearmanr


def safe_corrcoef(x, y, default=0.0):
    """
    安全计算 Pearson 相关系数，避免常量序列的除零警告

    Args:
        x: 预测序列
        y: 实际序列
        default: 当相关系数无法计算时的默认值

    Returns:
        float: 相关系数，或默认值
    """
    x_std = np.std(x)
    y_std = np.std(y)

    if x_std < 1e-8 or y_std < 1e-8:
        return default

    corr = np.corrcoef(x, y)[0, 1]
    return corr if np.isfinite(corr) else default


def safe_spearmanr(x, y, default=0.0):
    """
    安全计算 Spearman 秩相关系数，避免常量序列的警告

    Args:
        x: 预测序列
        y: 实际序列
        default: 当相关系数无法计算时的默认值

    Returns:
        float: 秩相关系数，或默认值
    """
    x_std = np.std(x)
    y_std = np.std(y)

    if x_std < 1e-8 or y_std < 1e-8:
        return default

    corr, _ = spearmanr(x, y)
    return corr if np.isfinite(corr) else default


def calc_trajectory_ic(pred_traj, actual_traj):
    """
    计算轨迹 IC 和 Rank IC

    Args:
        pred_traj: 预测轨迹 (numpy array)
        actual_traj: 实际轨迹 (numpy array)

    Returns:
        tuple: (ic, rank_ic)，无效时返回 (None, None)
    """
    if len(pred_traj) < 3:
        return None, None

    pred_std = np.std(pred_traj)
    actual_std = np.std(actual_traj)

    if pred_std < 1e-8 or actual_std < 1e-8:
        return None, None

    ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
    rank_ic, _ = spearmanr(pred_traj, actual_traj)

    ic = ic if np.isfinite(ic) else None
    rank_ic = rank_ic if np.isfinite(rank_ic) else None

    return ic, rank_ic