"""
Kronos Predictor Core Metrics Module

指标体系重构：基于 OHLCV 预测准确度

核心指标：
- MAPE：各特征的平均绝对百分比误差
- MAE：各特征的平均绝对误差
- Trajectory IC：轨迹形状相关性（去趋势）
- Amplitude Error Rate：振幅误差率

删除指标：
- DA (Direction Accuracy)：只看方向，忽略数值，无意义
- Limit Hit Rate：设计有误，目标错位

聚合规则：
- 温和线性衰减：近端重要，远端不舍弃
- w(t) = 1.5 - 0.05 * (t - 1)
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.distributed as dist


# ============================================================================
# 常量
# ============================================================================

FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']


# ============================================================================
# 权重方案
# ============================================================================

def get_step_weights(predict: int = 10) -> np.ndarray:
    """
    温和线性衰减权重：近端重要，远端不舍弃

    w(t) = 1.5 - 0.05 * (t - 1)
    +1: 1.50 (14.5%), +10: 1.05 (10.1%)

    Args:
        predict: 预测步数

    Returns:
        权重数组，sum = 1.0
    """
    steps = np.arange(1, predict + 1)
    weights = 1.5 - 0.05 * (steps - 1)
    weights = weights / weights.sum()
    return weights


# ============================================================================
# OHLCV 准确度指标
# ============================================================================

def compute_mape(pred: np.ndarray, actual: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    计算每步每个特征的 MAPE

    Args:
        pred: (pred_len, 6) 预测值
        actual: (pred_len, 6) 实际值
        eps: 防止除零

    Returns:
        (pred_len, 6) MAPE 矩阵
    """
    pred = np.asarray(pred)
    actual = np.asarray(actual)

    # 分母用实际值的绝对值
    denominator = np.abs(actual) + eps
    mape = np.abs(pred - actual) / denominator

    return mape


def compute_mae(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    计算每步每个特征的 MAE

    Args:
        pred: (pred_len, 6) 预测值
        actual: (pred_len, 6) 实际值

    Returns:
        (pred_len, 6) MAE 矩阵
    """
    pred = np.asarray(pred)
    actual = np.asarray(actual)

    mae = np.abs(pred - actual)

    return mae


def aggregate_mape_by_step(
    mape_lists: Dict[str, List[np.ndarray]],
    predict: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    聚合每步的 MAPE（per-step 统计）

    Args:
        mape_lists: {feature: [mape_per_sample]} 每个样本的 MAPE 矩阵 (pred_len, 6)
        predict: 预测步数

    Returns:
        {step: {feature: {mean, std, p50, n}}}
    """
    step_weights = get_step_weights(predict)
    result = {}

    for step_idx in range(predict):
        step_result = {}
        for fi, fn in enumerate(FEATURE_NAMES):
            values = []
            for mape_matrix in mape_lists[fn]:
                if mape_matrix is not None and step_idx < len(mape_matrix):
                    values.append(mape_matrix[step_idx, fi])

            if values:
                arr = np.array(values)
                step_result[fn] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)) if len(arr) >= 2 else 0.0,
                    'p50': float(np.percentile(arr, 50)),
                    'n': len(arr),
                }
            else:
                step_result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'n': 0}

        result[f'step{step_idx + 1}'] = step_result

    return result


def compute_weighted_summary(
    mape_lists: Dict[str, List[np.ndarray]],
    predict: int = 10
) -> Dict[str, float]:
    """
    计算加权聚合的 MAPE 摘要

    Args:
        mape_lists: {feature: [mape_per_sample]}
        predict: 预测步数

    Returns:
        {feature: weighted_mape}
    """
    step_weights = get_step_weights(predict)
    result = {}

    for fi, fn in enumerate(FEATURE_NAMES):
        weighted_sum = 0.0
        total_weight = 0.0

        for mape_matrix in mape_lists[fn]:
            if mape_matrix is None:
                continue
            for step_idx in range(min(len(mape_matrix), predict)):
                val = mape_matrix[step_idx, fi]
                w = step_weights[step_idx]
                weighted_sum += val * w
                total_weight += w

        result[fn] = weighted_sum / total_weight if total_weight > 0 else 0.0

    return result


# ============================================================================
# 去趋势口径
# ============================================================================

def detrend_to_baseline(series: np.ndarray, baseline: float) -> np.ndarray:
    """
    去趋势：相对 baseline 的相对序列

    trajectory IC 的设计意图是「同一股票内预测轨迹形状与实际轨迹形状的相似度」。
    原始价格序列强自相关，即使预测完全持平也会因与实际趋势同向拿到高 IC。
    必须对去趋势序列算相关，才能测形状。

    Args:
        series: (pred_len,) 或 (pred_len, n_features) 的原始序列
        baseline: 标量或 (n_features,)，取 lookback 最后一根对应特征值

    Returns:
        去趋势后的相对序列: (series - baseline) / (|baseline| + eps)
    """
    eps = 1e-8
    return (series - baseline) / (np.abs(baseline) + eps)


# ============================================================================
# Trajectory IC
# ============================================================================

def safe_corrcoef(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    """安全计算 Pearson 相关系数"""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) < 3 or len(y) < 3:
        return default

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std < 1e-8 or y_std < 1e-8:
        return default

    corr = np.corrcoef(x, y)[0, 1]
    return corr if np.isfinite(corr) else default


def safe_spearmanr(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    """安全计算 Spearman 秩相关系数"""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) < 3 or len(y) < 3:
        return default

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std < 1e-8 or y_std < 1e-8:
        return default

    corr, _ = spearmanr(x, y)
    return corr if np.isfinite(corr) else default


def safe_trajectory_ic(
    pred_traj: np.ndarray,
    actual_traj: np.ndarray,
    min_len: int = 3
) -> Tuple[Optional[float], Optional[float]]:
    """
    安全计算轨迹 IC（去趋势序列上的 Pearson + Spearman）

    Args:
        pred_traj: 去趋势后的预测序列（相对 baseline）
        actual_traj: 去趋势后的实际序列（相对 baseline）
        min_len: 最小序列长度

    Returns:
        (ic, rank_ic)，无法计算时返回 (None, None)
    """
    pred_traj = np.asarray(pred_traj).flatten()
    actual_traj = np.asarray(actual_traj).flatten()

    if len(pred_traj) < min_len or len(actual_traj) < min_len:
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


# ============================================================================
# Amplitude Error Rate
# ============================================================================

def amplitude_error_rate(
    pred_high_low: float,
    actual_high_low: float
) -> Optional[float]:
    """
    预测振幅误差率

    振幅 = high - low（日内波动范围）
    error_rate = 预测振幅 ÷ 实际振幅
    标尺：1.0=完美，0.8-1.2=可用，偏离>30%=失真

    actual_high_low 接近 0 时（一字涨跌停/停牌，全天无波动）返回 None

    Args:
        pred_high_low: 预测的 (high - low)
        actual_high_low: 实际的 (high - low)

    Returns:
        振幅误差率，1.0 表示完美匹配；actual 无波动时返回 None
    """
    if actual_high_low <= 1e-3:
        return None
    return pred_high_low / actual_high_low


def compute_amplitude_stats(
    pred_values: np.ndarray,
    actual_values: np.ndarray
) -> Dict[str, float]:
    """
    计算振幅统计

    Args:
        pred_values: (pred_len, 6) 预测值
        actual_values: (pred_len, 6) 实际值

    Returns:
        {mean_rate, std_rate, perfect_pct, usable_pct}
    """
    pred_high = pred_values[:, 1]
    pred_low = pred_values[:, 2]
    actual_high = actual_values[:, 1]
    actual_low = actual_values[:, 2]

    pred_amp = pred_high - pred_low
    actual_amp = actual_high - actual_low

    rates = []
    for i in range(len(pred_amp)):
        if actual_amp[i] > 1e-8:
            rate = pred_amp[i] / actual_amp[i]
            rates.append(rate)

    if not rates:
        return {'mean_rate': 1.0, 'std_rate': 0.0, 'perfect_pct': 0.0, 'usable_pct': 0.0}

    rates_arr = np.array(rates)

    return {
        'mean_rate': float(np.mean(rates_arr)),
        'std_rate': float(np.std(rates_arr)),
        'perfect_pct': float(np.mean(np.abs(rates_arr - 1.0) < 0.1)),
        'usable_pct': float(np.mean(np.abs(rates_arr - 1.0) < 0.3)),
    }


# ============================================================================
# DDP 聚合
# ============================================================================

def aggregate_ic(
    local_ic_lists: Dict[str, List[float]],
    world_size: int,
    device: torch.device,
    is_main: bool = True
) -> Dict[str, Any]:
    """
    聚合各 GPU 的 IC 列表，保留分布

    Args:
        local_ic_lists: {feature_name: [ic_values]}
        world_size: GPU 数量
        device: 当前设备
        is_main: 是否主进程

    Returns:
        {feature: {mean, std, p25, p50, p75, n}}
    """
    result = {}

    for fn in FEATURE_NAMES:
        local_list = local_ic_lists.get(fn, [])
        n_local = len(local_list)

        if world_size > 1:
            n_tensor = torch.tensor([n_local], device=device)
            gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_n, n_tensor)
            lengths = [t.item() for t in gathered_n]
            total_n = sum(lengths)

            max_len = max(lengths) if lengths else 0
            if max_len == 0:
                full_list = []
            else:
                send_buffer = torch.zeros(max_len, device=device)
                for i, val in enumerate(local_list):
                    send_buffer[i] = val

                gathered_buffers = [torch.zeros(max_len, device=device) for _ in range(world_size)]
                dist.all_gather(gathered_buffers, send_buffer)

                full_list = []
                for rank_idx, buf in enumerate(gathered_buffers):
                    for i in range(lengths[rank_idx]):
                        full_list.append(buf[i].item())
        else:
            full_list = list(local_list)
            total_n = n_local

        if not is_main:
            continue

        if total_n == 0:
            result[fn] = {'mean': 0.0, 'std': 0.0, 'p25': None, 'p50': None, 'p75': None, 'n': 0}
            continue

        full_arr = np.array(full_list)

        result[fn] = {
            'mean': float(np.mean(full_arr)),
            'std': float(np.std(full_arr)) if total_n >= 2 else 0.0,
            'p25': float(np.percentile(full_arr, 25)) if total_n >= 4 else None,
            'p50': float(np.percentile(full_arr, 50)),
            'p75': float(np.percentile(full_arr, 75)) if total_n >= 4 else None,
            'n': total_n,
        }

    return result


# ============================================================================
# 报告格式化
# ============================================================================

def format_metrics_report(
    mape_by_step: Dict[str, Dict[str, Dict[str, float]]],
    mape_summary: Dict[str, float],
    ic_result: Dict[str, Dict[str, float]],
    amplitude_result: Dict[str, float],
    predict: int = 10
) -> str:
    """
    格式化度量报告

    主输出：
    - MAPE 摘要（加权聚合）
    - 各步 MAPE
    - Trajectory IC
    - Amplitude Error Rate
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Evaluation Results (OHLCV Accuracy)")
    lines.append("=" * 70)

    # 1. MAPE 摘要（加权聚合）
    lines.append("\n[MAPE Summary - Weighted Average (Linear Decay)]")
    lines.append(f"{'Feature':<8} {'MAPE':>8} {'Weight':>8}")
    weights = get_step_weights(predict)
    lines.append(f"{'(weights)':<8} {'':<8} {'+1=' + f'{weights[0]:.1%}':>8} → {'+10=' + f'{weights[-1]:.1%}':<8}")
    lines.append("-" * 26)

    for fn in FEATURE_NAMES:
        mape_val = mape_summary.get(fn, 0.0)
        lines.append(f"{fn:<8} {mape_val:>7.2%}")

    # 2. 各步 MAPE（close 为例）
    lines.append("\n[MAPE by Step - close]")
    lines.append(f"{'Step':<6} {'MAPE':>8} {'std':>8} {'p50':>8}")

    for step_idx in range(predict):
        step_key = f'step{step_idx + 1}'
        if step_key in mape_by_step and 'close' in mape_by_step[step_key]:
            info = mape_by_step[step_key]['close']
            lines.append(f"+{step_idx + 1:<5} {info['mean']:>7.2%} {info['std']:>7.2%} {info['p50']:>7.2%}")

    # 3. Amplitude Error Rate
    if amplitude_result:
        lines.append("\n[Amplitude Error Rate]")
        lines.append(f"Mean: {amplitude_result['mean_rate']:.2f}, Std: {amplitude_result['std_rate']:.2f}")
        lines.append(f"Perfect (0.9-1.1): {amplitude_result['perfect_pct']:.1%}")
        lines.append(f"Usable (0.7-1.3): {amplitude_result['usable_pct']:.1%}")

    # 4. Trajectory IC
    lines.append("\n[Trajectory IC - Detrended]")
    lines.append(f"{'Feature':<8} {'mean':>8} {'std':>8} {'p50':>8} {'n':>8}")

    for fn in FEATURE_NAMES:
        if fn in ic_result:
            info = ic_result[fn]
            p50 = info.get('p50')
            p50_str = f"{p50:.4f}" if p50 is not None else "N/A"
            lines.append(f"{fn:<8} {info['mean']:>8.4f} {info['std']:>8.4f} {p50_str:>8} {info['n']:>8}")

    lines.append("=" * 70)

    return "\n".join(lines)


