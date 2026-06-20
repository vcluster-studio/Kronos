"""
Kronos Predictor Core Metrics Module

度量口径定稿（Phase -1）

核心约束：
- train.py 与 eval.py 必须共用此模块的同一组度量函数
- trajectory IC 必须用去趋势序列，禁止原始价格
- checkpoint 选择 IC 与 eval 报告 IC 同口径

关键修订（E1/E8/E2/E4 解决）：
- E1: trajectory IC 改用去趋势序列（detrend_to_baseline）
- E8: backtest IC 逐股票算再聚合（见 backtest.py）
- E2: 新增 excess_da，让 DA 数字可解读
- E4: aggregate_ic 保留分布（mean/std/p25/p50/p75）

可懂指标三件套（eval 主输出）：
- 方向胜率（DA）
- 振幅误差率
- 涨跌停命中率

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
# 去趋势口径（E1 解决）
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

    示例:
        # 正确用法
        baseline_close = orig[lookback - 1, 3]  # close 特征
        pred_detrended = detrend_to_baseline(pred_close_series, baseline_close)
        actual_detrended = detrend_to_baseline(actual_close_series, baseline_close)
        ic, rank_ic = safe_trajectory_ic(pred_detrended, actual_detrended)

        # 错误用法（禁止）
        ic = safe_trajectory_ic(pred_raw, actual_raw)  # 直接用原始价格 → 虚高 IC
    """
    eps = 1e-8
    return (series - baseline) / (np.abs(baseline) + eps)


# ============================================================================
# 安全相关系数计算（除零保护）
# ============================================================================

def safe_corrcoef(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    """
    安全计算 Pearson 相关系数

    检查双方标准差，避免除零警告

    Args:
        x: 一维数组
        y: 一维数组
        default: 无法计算时的默认返回值

    Returns:
        Pearson 相关系数，或 default
    """
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
    """
    安全计算 Spearman 秩相关系数

    检查双方标准差，避免除零警告

    Args:
        x: 一维数组
        y: 一维数组
        default: 无法计算时的默认返回值

    Returns:
        Spearman 秩相关系数，或 default
    """
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

    约束: 调用方必须先 detrend_to_baseline，禁止传入原始价格。

    Args:
        pred_traj: 去趋势后的预测序列（相对 baseline）
        actual_traj: 去趋势后的实际序列（相对 baseline）
        min_len: 最小序列长度

    Returns:
        (ic, rank_ic)，无法计算时返回 (None, None)

    使用示例:
        baseline_close = orig[lookback - 1, 3]
        pred_detrended = detrend_to_baseline(pred_close_series, baseline_close)
        actual_detrended = detrend_to_baseline(actual_close_series, baseline_close)
        ic, rank_ic = safe_trajectory_ic(pred_detrended, actual_detrended)
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
# Direction Accuracy（E2 解决）
# ============================================================================

def excess_da(model_da: float, naive_da: float) -> float:
    """
    计算 excess DA

    DA 基线为 lookback 末根（baseline = values[lookback-1]）。
    naive baseline = 持平预测（pred == baseline）的 DA，即实际方向中「与 baseline 同向」的比例。

    excess DA = model DA − naive DA
    让 DA 数字可解读：excess > 0 才是真 alpha，excess ≈ 0 等同朴素预测。

    Args:
        model_da: 模型的方向准确率（与 baseline 比较方向）
        naive_da: 持平预测的 DA（实际数据中与 baseline 同向的比例）

    Returns:
        excess DA，正值表示模型优于持平预测
    """
    return model_da - naive_da


def compute_naive_da(actual_values: np.ndarray, baseline: np.ndarray) -> float:
    """
    计算 naive DA（多数方向预测的 DA）

    naive baseline 策略：预测多数方向（假设市场延续多数趋势）。
    naive DA = max(上涨比例, 下跌比例)

    例如：
    - 实际上涨 60% → naive DA = 60%（预测全部涨）
    - 实际上涨 50% → naive DA = 50%（涨跌对半）
    - 实际上涨 30% → naive DA = 70%（预测全部跌）

    模型 DA 超过 naive DA 才有意义（excess > 0 = 真 alpha）。

    Args:
        actual_values: 实际值序列 (pred_len,) 或 (pred_len, n_features)
        baseline: lookback 末根值，标量或 (n_features,)

    Returns:
        naive DA = 多数方向的比例
    """
    actual_values = np.asarray(actual_values)
    baseline = np.asarray(baseline)

    # 方向：actual > baseline 为上涨
    actual_dir = actual_values > baseline

    # 上涨比例（True 的比例）
    up_ratio = np.mean(actual_dir)

    # naive DA = 多数方向的比例
    naive_da = max(up_ratio, 1 - up_ratio)

    return float(naive_da)


# ============================================================================
# 聚合保留分布（E4 解决）
# ============================================================================

def aggregate_ic(
    local_ic_lists: Dict[str, List[float]],
    world_size: int,
    device: torch.device,
    is_main: bool = True
) -> Dict[str, Any]:
    """
    聚合各 GPU 的 IC 列表，保留分布

    不只传 mean×n，保留 std/分位数，让 IC 稳健性可判断。
    IC 0.21 的 std 是 0.05 还是 0.3 决定数字是否可信。

    Args:
        local_ic_lists: {feature_name: [ic_values]} 各 GPU 本地的 IC 列表
        world_size: GPU 数量
        device: 当前设备
        is_main: 是否主进程

    Returns:
        {feature: {mean, std, p25, p50, p75, n, list}} 聚合结果
        若 is_main=False，返回空 dict
    """
    result = {}

    for fn in FEATURE_NAMES:
        local_list = local_ic_lists.get(fn, [])
        n_local = len(local_list)

        if world_size > 1:
            # 需要收集各 rank 的完整列表以计算分位数
            # 方法：all_gather 各 rank 的列表长度和列表内容

            # 先 gather 长度
            n_tensor = torch.tensor([n_local], device=device)
            gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_n, n_tensor)
            lengths = [t.item() for t in gathered_n]
            total_n = sum(lengths)

            # 空 rank 边界：所有 rank 的 local_list 全空时 max_len=0，
            # all_gather 空 tensor 在个别 NCCL 版本会异常，跳过。
            # 该判断不依赖 rank（lengths 各 rank 一致），所有 rank 同步进入。
            max_len = max(lengths) if lengths else 0
            if max_len == 0:
                full_list = []
            else:
                # 准备发送缓冲区（padding 到最大长度）
                send_buffer = torch.zeros(max_len, device=device)
                for i, val in enumerate(local_list):
                    send_buffer[i] = val

                # all_gather
                gathered_buffers = [torch.zeros(max_len, device=device) for _ in range(world_size)]
                dist.all_gather(gathered_buffers, send_buffer)

                # 重组完整列表
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
            result[fn] = {
                'mean': 0.0,
                'std': 0.0,
                'p25': None,
                'p50': None,
                'p75': None,
                'n': 0,
            }
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


def aggregate_da(
    local_da_lists: List[Dict[str, List[float]]],
    world_size: int,
    device: torch.device,
    pred_len: int,
    is_main: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    聚合各 GPU 的 DA 列表，保留分布

    Args:
        local_da_lists: [{feature: [da_values]}] 各步的 DA 数据
        world_size: GPU 数量
        device: 当前设备
        pred_len: 预测步数
        is_main: 是否主进程

    Returns:
        {step_idx: {feature: {mean, std, p50, n}}} 聚合结果
    """
    result = {}

    for step_idx in range(pred_len):
        step_result = {}

        for fn in FEATURE_NAMES:
            local_list = local_da_lists[step_idx].get(fn, [])
            n_local = len(local_list)

            if world_size > 1:
                # gather 长度
                n_tensor = torch.tensor([n_local], device=device)
                gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_n, n_tensor)
                lengths = [t.item() for t in gathered_n]
                total_n = sum(lengths)

                # 空 rank 边界：max_len=0 时跳过 all_gather（同 aggregate_ic）
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
                step_result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': None, 'n': 0}
                continue

            full_arr = np.array(full_list)
            step_result[fn] = {
                'mean': float(np.mean(full_arr)),
                'std': float(np.std(full_arr)) if total_n >= 2 else 0.0,
                'p50': float(np.percentile(full_arr, 50)),
                'n': total_n,
            }

        if is_main:
            result[f'step{step_idx + 1}'] = step_result

    return result


# ============================================================================
# 可懂指标三件套（§1.6.5）
# ============================================================================

def amplitude_error_rate(
    pred_high_low: float,
    actual_high_low: float
) -> float:
    """
    预测振幅误差率

    振幅 = high - low（日内波动范围）
    error_rate = 预测振幅 ÷ 实际振幅
    标尺：1.0=完美，0.8-1.2=可用，偏离>30%=失真

    Args:
        pred_high_low: 预测的 (high - low)
        actual_high_low: 实际的 (high - low)

    Returns:
        振幅误差率，1.0 表示完美匹配
    """
    eps = 1e-8
    return pred_high_low / (actual_high_low + eps)


def compute_amplitude_stats(
    pred_values: np.ndarray,
    actual_values: np.ndarray
) -> Dict[str, float]:
    """
    计算振幅统计

    Args:
        pred_values: (pred_len, 6) 预测值，特征顺序 [open, high, low, close, vol, amt]
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
        if actual_amp[i] > 1e-8:  # 有实际波动
            rate = pred_amp[i] / actual_amp[i]
            rates.append(rate)

    if not rates:
        return {'mean_rate': 1.0, 'std_rate': 0.0, 'perfect_pct': 0.0, 'usable_pct': 0.0}

    rates_arr = np.array(rates)

    return {
        'mean_rate': float(np.mean(rates_arr)),
        'std_rate': float(np.std(rates_arr)),
        'perfect_pct': float(np.mean(np.abs(rates_arr - 1.0) < 0.1)),  # 0.9-1.1
        'usable_pct': float(np.mean(np.abs(rates_arr - 1.0) < 0.3)),   # 0.7-1.3
    }


def limit_hit_rate(
    pred_limit_flags: np.ndarray,
    actual_limit_flags: np.ndarray
) -> Dict[str, float]:
    """
    涨跌停命中率

    预测涨停的样本中实际涨停比例。
    标尺：随机≈1-3%，10%+=有信号，30%+=强。

    Args:
        pred_limit_flags: bool 数组，模型预测触及涨停的样本
        actual_limit_flags: bool 数组，实际涨停的样本

    Returns:
        {hit_rate, n_pred_limit, n_actual_limit}
    """
    pred_limit_flags = np.asarray(pred_limit_flags)
    actual_limit_flags = np.asarray(actual_limit_flags)

    n_pred = pred_limit_flags.sum()
    n_actual = actual_limit_flags.sum()

    if n_pred == 0:
        return {'hit_rate': None, 'n_pred_limit': 0, 'n_actual_limit': int(n_actual)}

    hit_rate = actual_limit_flags[pred_limit_flags].mean()

    return {
        'hit_rate': float(hit_rate),
        'n_pred_limit': int(n_pred),
        'n_actual_limit': int(n_actual),
    }


def detect_limit(
    values: np.ndarray,
    baseline_close: float,
    limit_pct: float = 0.10
) -> np.ndarray:
    """
    检测涨跌停

    Args:
        values: (pred_len, 6) OHLCV 值
        baseline_close: lookback 末根 close
        limit_pct: 涨跌停阈值（主板 10%，创业板 20%）

    Returns:
        bool 数组，是否触及涨跌停（涨停或跌停）
    """
    close_values = values[:, 3]
    gain = (close_values - baseline_close) / (np.abs(baseline_close) + 1e-8)
    return np.abs(gain) >= limit_pct


# ============================================================================
# 综合评分（§1.6.6）
# ============================================================================

def get_log_step_weights(predict: int = 10) -> np.ndarray:
    """
    对数递减步权重：短期权重高，远期平滑递减

    Args:
        predict: 预测步数

    Returns:
        权重数组，sum = 1.0
    """
    steps = np.arange(1, predict + 1)
    weights = 1 / np.log(steps + 1)
    weights = weights / weights.sum()
    return weights


def get_feature_weights() -> Dict[str, float]:
    """
    特征权重：价格类 0.7，交易量类 0.3

    Returns:
        {feature: weight}
    """
    return {
        'open': 0.15, 'high': 0.15, 'low': 0.15, 'close': 0.25,  # 价格类 0.70
        'vol': 0.15, 'amt': 0.15                                   # 交易量类 0.30
    }


def calculate_da_score(
    da_by_step: List[Dict[str, Any]],
    predict: int = 10
) -> float:
    """
    计算综合 DA 评分

    两级加权：
    1. 同特征内：对数步权重（+1=24.5% → +10=7.0%）
    2. 特征间：价格类0.7 + 交易量类0.3

    Args:
        da_by_step: [{feature: DA_value}] for each step
            DA_value 可以是标量（已聚合的均值）或列表（原始 0/1 值）；
            列表时取 np.mean，标量时直接用。数值结果一致。
        predict: 预测步数

    Returns:
        da_score: 综合DA评分 (0-1)
    """
    step_weights = get_log_step_weights(predict)
    feature_weights = get_feature_weights()

    da_score = 0.0
    for feature, f_weight in feature_weights.items():
        feature_da = 0.0
        for step in range(predict):
            da_val = da_by_step[step].get(feature, [])
            if isinstance(da_val, (list, tuple, np.ndarray)):
                if len(da_val) == 0:
                    continue
                step_da = float(np.mean(da_val))
            else:
                # 标量（已聚合的均值）
                step_da = float(da_val)
            feature_da += step_da * step_weights[step]
        da_score += feature_da * f_weight

    return float(da_score)


def calculate_combined_score(
    ic: float,
    da_score: float,
    ic_weight: float = 0.6,
    da_weight: float = 0.4
) -> float:
    """
    综合评分：IC + DA（线性归一化）

    注意：去趋势后 IC 量级可能变化（原始价格口径的虚高 IC 消失），
    ic_weight/da_weight 需在 Phase -1 用正确口径重新标定。

    当前权重 0.6/0.4 是基于旧口径（原始价格 IC）的经验值，
    新口径下可能不再合适。

    Args:
        ic: close 的 trajectory IC（必须用去趋势序列）
        da_score: 综合 DA 评分 (0~1)
        ic_weight: IC 权重（Phase -1 后可能调整）
        da_weight: DA 权重

    Returns:
        combined_score: 综合评分 (0~1)
    """
    # IC 归一化: -1~1 → 0~1
    ic_norm = (ic + 1) / 2
    # DA 已在 0~1 范围

    return ic_norm * ic_weight + da_score * da_weight


# ============================================================================
# 辅助函数
# ============================================================================

def format_metrics_report(
    ic_result: Dict[str, Dict[str, float]],
    da_result: Dict[str, Dict[str, float]],
    amplitude_result: Dict[str, float] = None,
    limit_result: Dict[str, float] = None,
    naive_da_by_step: Dict[int, float] = None,
    predict: int = 10
) -> str:
    """
    格式化度量报告（eval 主输出）

    按可懂性优先原则，主输出用：
    - 方向胜率（各步 DA）
    - 振幅误差率
    - 涨跌停命中率
    IC/RankIC 降为附录

    Args:
        ic_result: {feature: {mean, std, p25, p50, p75, n}}
        da_result: {step: {feature: {mean, std, p50, n}}}
        amplitude_result: {mean_rate, std_rate, perfect_pct, usable_pct}
        limit_result: {hit_rate, n_pred_limit, n_actual_limit}
        naive_da_by_step: {step_idx: naive_da} 各步的 naive DA（多数方向比例）
        predict: 预测步数

    Returns:
        格式化的报告字符串
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Evaluation Results (Detrended Trajectory IC)")
    lines.append("=" * 70)

    # 1. 可懂指标：方向胜率（close）
    lines.append("\n[Direction Accuracy - close]")
    lines.append(f"{'Step':<6} {'DA':>8} {'std':>8} {'p50':>8} {'naive':>8} {'excess':>8}")

    for step_idx in range(predict):
        step_key = f'step{step_idx + 1}'
        # 获取该步的 naive DA（如果提供）
        naive_da = naive_da_by_step.get(step_idx, 0.5) if naive_da_by_step else 0.5
        # 确保 naive_da 不为 None（防止 dict 中显式存储了 None）
        if naive_da is None:
            naive_da = 0.5

        if step_key in da_result and 'close' in da_result[step_key]:
            da_info = da_result[step_key]['close']
            da_mean = da_info.get('mean', 0)
            da_std = da_info.get('std', 0)
            da_p50 = da_info.get('p50')
            # 确保 da_p50 有值（可能为 None）
            if da_p50 is None:
                da_p50 = da_mean
            excess = excess_da(da_mean, naive_da)
            lines.append(f"+{step_idx+1:<5} {da_mean:>8.1%} {da_std:>8.1%} {da_p50:>8.1%} {naive_da:>8.1%} {excess:>8.1%}")

    # 2. 可懂指标：振幅误差率
    if amplitude_result:
        lines.append("\n[Amplitude Error Rate]")
        lines.append(f"Mean: {amplitude_result['mean_rate']:.2f}, Std: {amplitude_result['std_rate']:.2f}")
        lines.append(f"Perfect (0.9-1.1): {amplitude_result['perfect_pct']:.1%}")
        lines.append(f"Usable (0.7-1.3): {amplitude_result['usable_pct']:.1%}")

    # 3. 可懂指标：涨跌停命中率
    if limit_result:
        lines.append("\n[Limit Hit Rate]")
        hit_rate = limit_result.get('hit_rate')
        if hit_rate is not None:
            lines.append(f"Hit Rate: {hit_rate:.1%} (random ~1-3%, signal >10%)")
        else:
            lines.append("Hit Rate: N/A (no predicted limit)")
        lines.append(f"Predicted: {limit_result['n_pred_limit']}, Actual: {limit_result['n_actual_limit']}")

    # 4. IC 附录
    lines.append("\n[Trajectory IC - Appendix (Detrended)]")
    lines.append(f"{'Feature':<8} {'mean':>8} {'std':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'n':>6}")

    for fn in FEATURE_NAMES:
        if fn in ic_result:
            info = ic_result[fn]
            p25 = info.get('p25')
            p50 = info.get('p50')
            p75 = info.get('p75')
            p25_str = f"{p25:.4f}" if p25 is not None else "N/A"
            p50_str = f"{p50:.4f}" if p50 is not None else "N/A"
            p75_str = f"{p75:.4f}" if p75 is not None else "N/A"
            lines.append(f"{fn:<8} {info['mean']:>8.4f} {info['std']:>8.4f} {p25_str:>8} {p50_str:>8} {p75_str:>8} {info['n']:>6}")

    lines.append("=" * 70)

    return "\n".join(lines)