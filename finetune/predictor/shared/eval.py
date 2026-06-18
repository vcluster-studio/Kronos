"""
统一评估模块 - 支持 full_window 和 MA60 两种数据格式

评估指标（6个特征各自计算）：
1. Trajectory IC: 同一股票内，预测轨迹 vs 实际轨迹的相关性
2. MAE: 预测绝对误差（各步）
3. Direction Acc: 方向准确率（各步）
4. DA_score: 综合方向准确率（对数步权重 + 特征权重）

注意：
- Return IC (旧): 跨股票收益率相关性 → 不符合项目目标
- Trajectory IC (新): 同股票内轨迹相关性 → 符合"预测未来K线轨迹"目标
"""

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']


# ============================================================================
# DA 综合评分计算
# ============================================================================

def get_log_step_weights(predict=10):
    """对数递减步权重：短期权重高，远期平滑递减"""
    steps = np.arange(1, predict + 1)
    weights = 1 / np.log(steps + 1)
    weights = weights / weights.sum()
    return weights


def get_feature_weights():
    """特征权重：价格类 0.7，交易量类 0.3"""
    return {
        'open': 0.15, 'high': 0.15, 'low': 0.15, 'close': 0.25,  # 价格类 0.70
        'vol': 0.15, 'amt': 0.15                                   # 交易量类 0.30
    }


def calculate_da_score(da_by_step, predict=10):
    """
    计算综合 DA 评分

    两级加权：
    1. 同特征内：对数步权重（+1=24.5% → +10=7.0%）
    2. 特征间：价格类0.7 + 交易量类0.3

    Args:
        da_by_step: [{feature: [DA_values]}] for each step
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
            if da_by_step[step][feature]:
                step_da = np.mean(da_by_step[step][feature])
                feature_da += step_da * step_weights[step]
        da_score += feature_da * f_weight

    return da_score


def calculate_combined_score(ic, da_score, ic_weight=0.6, da_weight=0.4):
    """
    综合评分：IC + DA（线性归一化）

    Args:
        ic: close 的 trajectory IC (-1~1)
        da_score: 综合 DA 评分 (0~1)
        ic_weight: IC 权重
        da_weight: DA 权重

    Returns:
        combined_score: 综合评分 (0~1)
    """
    # IC 归一化: -1~1 → 0~1
    ic_norm = (ic + 1) / 2
    # DA 已在 0~1 范围

    return ic_norm * ic_weight + da_score * da_weight


# ============================================================================
# 测试集完整评估（详细输出）
# ============================================================================

def evaluate_full_window(model, tokenizer, test_data, lookback, predict, n_samples=500, seed=42, clip=5.0):
    """
    评估 - full_window 归一化（mode1）

    数据格式：test_data[symbol] = DataFrame，动态归一化
    """
    from model.kronos import auto_regressive_inference
    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)

    # 按特征收集 trajectory IC
    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}

    # 按步按特征收集 MAE 和 DA
    mae_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]

    symbols = list(test_data.keys())
    # n_samples <= 0 表示全量评估
    if n_samples <= 0 or n_samples >= len(symbols):
        sample_symbols = symbols
    else:
        sample_symbols = rng.choice(symbols, size=n_samples, replace=False)

    for symbol in tqdm(sample_symbols, desc="Evaluating"):
        df = test_data[symbol]
        if len(df) < lookback + predict:
            continue

        values = df.values[-(lookback + predict):]
        index = df.index[-(lookback + predict):]

        x = values[:lookback]
        y = values[lookback:]
        baseline = values[lookback - 1]

        # Full window normalize
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0) + 1e-5
        x_norm = (x - x_mean) / x_std
        x_norm = np.clip(x_norm, -clip, clip)

        # Time features
        x_stamp = np.stack([
            index[:lookback].minute.values.astype(np.float32),
            index[:lookback].hour.values.astype(np.float32),
            index[:lookback].weekday.values.astype(np.float32),
            index[:lookback].day.values.astype(np.float32),
            index[:lookback].month.values.astype(np.float32),
        ], axis=1)

        y_stamp = np.stack([
            index[lookback:].minute.values.astype(np.float32),
            index[lookback:].hour.values.astype(np.float32),
            index[lookback:].weekday.values.astype(np.float32),
            index[lookback:].day.values.astype(np.float32),
            index[lookback:].month.values.astype(np.float32),
        ], axis=1)

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
            x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(DEVICE)
            y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(DEVICE)

            preds = auto_regressive_inference(
                tokenizer, model, x_tensor, x_stamp_tensor, y_stamp_tensor,
                max_context=2048, pred_len=predict, T=1.0, top_p=0.9, sample_count=1
            )

        pred_norm = preds[0, -predict:]
        pred_values = pred_norm * x_std + x_mean

        # 各特征 Trajectory IC
        for fi, fn in enumerate(FEATURE_NAMES):
            pred_traj = pred_values[:, fi]
            actual_traj = y[:, fi]

            if len(pred_traj) >= 3:
                # 检查轨迹方差，避免除零警告
                pred_std = np.std(pred_traj)
                actual_std = np.std(actual_traj)
                if pred_std > 1e-8 and actual_std > 1e-8:
                    traj_ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
                    if np.isfinite(traj_ic):
                        trajectory_ics[fn].append(traj_ic)

                    traj_ric, _ = spearmanr(pred_traj, actual_traj)
                    if np.isfinite(traj_ric):
                        trajectory_rics[fn].append(traj_ric)

        # 各步 MAE 和 DA
        for step_idx in range(predict):
            pred_raw = pred_values[step_idx]
            actual = y[step_idx]

            for fi, fn in enumerate(FEATURE_NAMES):
                mae_by_step[step_idx][fn].append(abs(pred_raw[fi] - actual[fi]))
                pred_dir = (pred_raw[fi] - baseline[fi]) > 0
                actual_dir = (actual[fi] - baseline[fi]) > 0
                da_by_step[step_idx][fn].append(pred_dir == actual_dir)

    # 聚合结果
    result = {'n_samples': len(sample_symbols)}

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

    # 综合评分
    result['da_score'] = calculate_da_score(da_by_step, predict)
    result['combined_score'] = calculate_combined_score(
        result['close_trajectory_ic'], result['da_score']
    )

    return result


def evaluate_ma60(model, tokenizer, all_data, indices, lookback, predict, n_samples=500, seed=42, clip=5.0):
    """
    评估 - MA60 预归一化数据（mode2/3/4）

    数据格式：all_data[symbol] = {
        'normalized': np.ndarray,  # 预归一化数据
        'original': np.ndarray,    # 原始数据
        'means': np.ndarray,       # MA60 mean
        'stds': np.ndarray,        # MA60 std
        'index': DatetimeIndex
    }
    """
    from model.kronos import auto_regressive_inference
    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)

    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}
    mae_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]

    if n_samples > 0 and n_samples < len(indices):
        sample_idx = rng.choice(len(indices), n_samples, replace=False)
        indices = [indices[i] for i in sample_idx]

    for (sym, start) in tqdm(indices, desc="Evaluating"):
        d = all_data[sym]
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
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
            x_stamp = torch.from_numpy(stamp[:lookback]).unsqueeze(0).to(DEVICE)
            y_stamp = torch.from_numpy(stamp[lookback:lookback+predict]).unsqueeze(0).to(DEVICE)

            preds = auto_regressive_inference(
                tokenizer, model, x_tensor, x_stamp, y_stamp,
                max_context=2048, pred_len=predict, clip=clip, T=1.0, top_p=0.9, sample_count=1
            )

        pred_norm = preds[0, lookback:lookback+predict]
        pred_raw = pred_norm * stds[lookback:lookback+predict] + means[lookback:lookback+predict]
        actual = orig[lookback:lookback+predict]

        # 各特征 Trajectory IC
        for fi, fn in enumerate(FEATURE_NAMES):
            pred_traj = pred_raw[:, fi]
            actual_traj = actual[:, fi]

            if len(pred_traj) >= 3:
                # 检查轨迹方差，避免除零警告
                pred_std = np.std(pred_traj)
                actual_std = np.std(actual_traj)
                if pred_std > 1e-8 and actual_std > 1e-8:
                    traj_ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
                    if np.isfinite(traj_ic):
                        trajectory_ics[fn].append(traj_ic)

                    traj_ric, _ = spearmanr(pred_traj, actual_traj)
                    if np.isfinite(traj_ric):
                        trajectory_rics[fn].append(traj_ric)

        # 各步 MAE 和 DA
        for step_idx in range(predict):
            for fi, fn in enumerate(FEATURE_NAMES):
                mae_by_step[step_idx][fn].append(abs(pred_raw[step_idx, fi] - actual[step_idx, fi]))
                pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                da_by_step[step_idx][fn].append(pred_dir == actual_dir)

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

    # 综合评分
    result['da_score'] = calculate_da_score(da_by_step, predict)
    result['combined_score'] = calculate_combined_score(
        result['close_trajectory_ic'], result['da_score']
    )

    return result


def evaluate_ma60_with_buckets(model, tokenizer, all_data, indices, lookback, predict,
                                n_samples=500, seed=42, clip=5.0, buckets=None):
    """
    评估 - MA60 数据 + 分桶指标（mode4专用）

    保留原有的分桶结构，同时增加 Trajectory IC
    """
    from model.kronos import auto_regressive_inference
    model.eval()
    tokenizer.eval()

    if buckets is None:
        buckets = {
            "<1%": (0, 0.01),
            "1%~2%": (0.01, 0.02),
            "2%~5%": (0.02, 0.05),
            ">5%": (0.05, float("inf"))
        }

    rng = np.random.RandomState(seed)

    # Trajectory IC 收集
    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}

    # 分桶收集（基于 close delta）
    bucket_preds = {k: [] for k in buckets.keys()}
    bucket_actuals = {k: [] for k in buckets.keys()}
    bucket_trajectories = {k: [] for k in buckets.keys()}  # 新增: 分桶 trajectory

    # MAE/DA 收集
    mae_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]

    if n_samples > 0 and n_samples < len(indices):
        sample_idx = rng.choice(len(indices), n_samples, replace=False)
        indices = [indices[i] for i in sample_idx]

    for (sym, start) in tqdm(indices, desc="Evaluating"):
        d = all_data[sym]
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
        baseline_close = orig[lookback - 1, 3]

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
            x_stamp = torch.from_numpy(stamp[:lookback]).unsqueeze(0).to(DEVICE)
            y_stamp = torch.from_numpy(stamp[lookback:lookback+predict]).unsqueeze(0).to(DEVICE)

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
                # 检查轨迹方差，避免除零警告
                pred_std = np.std(pred_traj)
                actual_std = np.std(actual_traj)
                if pred_std > 1e-8 and actual_std > 1e-8:
                    traj_ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
                    if np.isfinite(traj_ic):
                        trajectory_ics[fn].append(traj_ic)

                    traj_ric, _ = spearmanr(pred_traj, actual_traj)
                    if np.isfinite(traj_ric):
                        trajectory_rics[fn].append(traj_ric)

        # MAE/DA by step
        for step_idx in range(predict):
            for fi, fn in enumerate(FEATURE_NAMES):
                mae_by_step[step_idx][fn].append(abs(pred_raw[step_idx, fi] - actual[step_idx, fi]))
                pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                da_by_step[step_idx][fn].append(pred_dir == actual_dir)

        # 分桶（基于 t+1 close delta）
        actual_close_delta = (actual[0, 3] - baseline_close) / (abs(baseline_close) + 1e-8)
        abs_return = abs(actual_close_delta)

        for bucket_name, (low, high) in buckets.items():
            if low <= abs_return < high:
                pred_close_delta = (pred_raw[:, 3] - baseline_close) / (abs(baseline_close) + 1e-8)
                actual_delta_full = (actual[:, 3] - baseline_close) / (abs(baseline_close) + 1e-8)

                bucket_preds[bucket_name].append(pred_close_delta.tolist())
                bucket_actuals[bucket_name].append(actual_delta_full.tolist())

                # 分桶 trajectory IC（close）
                if len(pred_close_delta) >= 3:
                    bucket_traj_ic = np.corrcoef(pred_close_delta, actual_delta_full)[0, 1]
                    if np.isfinite(bucket_traj_ic):
                        bucket_trajectories[bucket_name].append(bucket_traj_ic)
                break

    # 聚合结果
    result = {'n_samples': len(indices)}

    # 全局 Trajectory IC
    for fn in FEATURE_NAMES:
        tics = trajectory_ics[fn]
        trics = trajectory_rics[fn]
        result[f'{fn}_trajectory_ic'] = float(np.mean(tics)) if tics else 0.0
        result[f'{fn}_trajectory_ic_std'] = float(np.std(tics)) if len(tics) >= 2 else 0.0
        result[f'{fn}_trajectory_rank_ic'] = float(np.mean(trics)) if trics else 0.0

    # MAE/DA by step
    for step_idx in range(predict):
        suffix = f'_step{step_idx+1}'
        for fn in FEATURE_NAMES:
            result[f'{fn}_mae{suffix}'] = float(np.mean(mae_by_step[step_idx][fn])) if mae_by_step[step_idx][fn] else 0.0
            result[f'{fn}_da{suffix}'] = float(np.mean(da_by_step[step_idx][fn])) if da_by_step[step_idx][fn] else 0.0

    # 分桶指标
    for bucket_name in buckets.keys():
        preds_arr = np.array(bucket_preds[bucket_name])
        actuals_arr = np.array(bucket_actuals[bucket_name])
        traj_ics = bucket_trajectories[bucket_name]
        n = len(preds_arr)

        result[f'bucket_{bucket_name}_n'] = n

        if n >= 5:
            # t+1 单步 IC/DA/MAE/bias
            if n >= 10:
                t1_ic = np.corrcoef(preds_arr[:, 0], actuals_arr[:, 0])[0, 1]
                result[f'bucket_{bucket_name}_ic'] = float(t1_ic) if np.isfinite(t1_ic) else None
            else:
                result[f'bucket_{bucket_name}_ic'] = None

            t1_da = np.mean(np.sign(preds_arr[:, 0]) == np.sign(actuals_arr[:, 0]))
            t1_mae = np.mean(np.abs(preds_arr[:, 0] - actuals_arr[:, 0]))
            t1_bias = np.mean(preds_arr[:, 0] - actuals_arr[:, 0])

            result[f'bucket_{bucket_name}_da'] = float(t1_da)
            result[f'bucket_{bucket_name}_mae'] = float(t1_mae)
            result[f'bucket_{bucket_name}_bias'] = float(t1_bias)

            # 分桶 Trajectory IC（close trajectory）
            result[f'bucket_{bucket_name}_trajectory_ic'] = float(np.mean(traj_ics)) if traj_ics else None
        else:
            result[f'bucket_{bucket_name}_ic'] = None
            result[f'bucket_{bucket_name}_da'] = None
            result[f'bucket_{bucket_name}_mae'] = None
            result[f'bucket_{bucket_name}_bias'] = None
            result[f'bucket_{bucket_name}_trajectory_ic'] = None

    return result


# ============================================================================
# Epoch 快速评估（精简输出）
# ============================================================================

def quick_eval_full_window(model, tokenizer, test_data, lookback, predict,
                            n_samples=200, seed=42, clip=5.0):
    """
    Epoch间快速评估 - full_window（mode1）
    只返回核心指标，不打印详细表格
    """
    result = evaluate_full_window(model, tokenizer, test_data, lookback, predict,
                                   n_samples=n_samples, seed=seed, clip=clip)

    # 精简结果
    quick_result = {
        'n_samples': result['n_samples'],
        'close_trajectory_ic': result.get('close_trajectory_ic', 0),
        'close_trajectory_rank_ic': result.get('close_trajectory_rank_ic', 0),
    }

    # 各特征 Trajectory IC 汇总
    for fn in FEATURE_NAMES:
        quick_result[f'{fn}_trajectory_ic'] = result.get(f'{fn}_trajectory_ic', 0)

    # 第1步 DA（关键指标）
    quick_result['close_da_step1'] = result.get('close_da_step1', 0)
    quick_result['open_da_step1'] = result.get('open_da_step1', 0)

    return quick_result


def quick_eval_ma60(model, tokenizer, all_data, indices, lookback, predict,
                     n_samples=200, seed=42, clip=5.0):
    """
    Epoch间快速评估 - MA60（mode2/3）
    只返回核心指标
    """
    result = evaluate_ma60(model, tokenizer, all_data, indices, lookback, predict,
                            n_samples=n_samples, seed=seed, clip=clip)

    # 精简结果
    quick_result = {
        'n_samples': result['n_samples'],
        'close_trajectory_ic': result.get('close_trajectory_ic', 0),
        'close_trajectory_rank_ic': result.get('close_trajectory_rank_ic', 0),
    }

    # 各特征 Trajectory IC
    for fn in FEATURE_NAMES:
        quick_result[f'{fn}_trajectory_ic'] = result.get(f'{fn}_trajectory_ic', 0)

    # 各步 DA
    for step_idx in range(predict):
        quick_result[f'close_da_step{step_idx+1}'] = result.get(f'close_da_step{step_idx+1}', 0)

    return quick_result


def quick_eval_ma60_with_buckets(model, tokenizer, all_data, indices, lookback, predict,
                                  n_samples=200, seed=42, clip=5.0, buckets=None):
    """
    Epoch间快速评估 - MA60 + 分桶（mode4）
    返回核心指标 + 分桶指标
    """
    result = evaluate_ma60_with_buckets(model, tokenizer, all_data, indices, lookback, predict,
                                         n_samples=n_samples, seed=seed, clip=clip, buckets=buckets)

    # 精简结果
    quick_result = {
        'n_samples': result['n_samples'],
        'close_trajectory_ic': result.get('close_trajectory_ic', 0),
    }

    # 分桶核心指标（2%~5% bucket）
    core_bucket = '2%~5%'
    quick_result['core_bucket_n'] = result.get(f'bucket_{core_bucket}_n', 0)
    quick_result['core_bucket_ic'] = result.get(f'bucket_{core_bucket}_ic')
    quick_result['core_bucket_da'] = result.get(f'bucket_{core_bucket}_da')
    quick_result['core_bucket_mae'] = result.get(f'bucket_{core_bucket}_mae')
    quick_result['core_bucket_trajectory_ic'] = result.get(f'bucket_{core_bucket}_trajectory_ic')

    # Extreme bucket bias
    quick_result['extreme_bucket_bias'] = result.get('bucket_>5%_bias')
    quick_result['extreme_bucket_n'] = result.get('bucket_>5%_n', 0)

    return quick_result


# ============================================================================
# 输出格式化
# ============================================================================

def print_evaluation_result(result, predict=10, title="Evaluation Results"):
    """
    打印完整评估结果（测试集评估用）
    """
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")

    # Trajectory IC
    print(f"\nTrajectory IC (6 features):")
    print(f"  {'Feature':<8} {'Traj_IC':>8} {'std':>8} {'pos%':>6}")
    for fn in FEATURE_NAMES:
        tic = result.get(f'{fn}_trajectory_ic', 0)
        tic_std = result.get(f'{fn}_trajectory_ic_std', 0)
        tic_pos = result.get(f'{fn}_trajectory_ic_pos_pct', 0)
        print(f"  {fn:<8} {tic:>8.4f} {tic_std:>8.4f} {tic_pos:>6.1%}")

    print(f"\n  N Samples: {result.get('n_samples', 0)}")

    # MAE
    print(f"\nPer-step MAE ({predict} steps, 6 features):")
    print(f"{'Step':<6} {'open_MAE':>10} {'high_MAE':>10} {'low_MAE':>10} {'close_MAE':>10} {'vol_MAE':>12} {'amt_MAE':>12}")
    for step_idx in range(predict):
        suffix = f'_step{step_idx+1}'
        print(f"+{step_idx+1:<5} "
              f"{result.get(f'open_mae{suffix}', 0):>10.2f} "
              f"{result.get(f'high_mae{suffix}', 0):>10.2f} "
              f"{result.get(f'low_mae{suffix}', 0):>10.2f} "
              f"{result.get(f'close_mae{suffix}', 0):>10.2f} "
              f"{result.get(f'vol_mae{suffix}', 0):>12.0f} "
              f"{result.get(f'amt_mae{suffix}', 0):>12.0f}")

    # DA
    print(f"\nPer-step DA ({predict} steps, 6 features):")
    print(f"{'Step':<6} {'open_DA':>8} {'high_DA':>8} {'low_DA':>8} {'close_DA':>8} {'vol_DA':>8} {'amt_DA':>8}")
    for step_idx in range(predict):
        suffix = f'_step{step_idx+1}'
        print(f"+{step_idx+1:<5} "
              f"{result.get(f'open_da{suffix}', 0):>8.0%} "
              f"{result.get(f'high_da{suffix}', 0):>8.0%} "
              f"{result.get(f'low_da{suffix}', 0):>8.0%} "
              f"{result.get(f'close_da{suffix}', 0):>8.0%} "
              f"{result.get(f'vol_da{suffix}', 0):>8.0%} "
              f"{result.get(f'amt_da{suffix}', 0):>8.0%}")

    print(f"{'=' * 80}")


def print_epoch_result(result, predict=10, title="Epoch Eval"):
    """
    打印Epoch间评估结果（精简格式）
    """
    print(f"\n[{title}] Samples: {result.get('n_samples', 0)}")

    # Trajectory IC 汇总
    print(f"  Trajectory IC:")
    for fn in FEATURE_NAMES:
        tic = result.get(f'{fn}_trajectory_ic', 0)
        print(f"    {fn}: {tic:>6.4f}")

    # 关键 DA
    if predict > 1:
        print(f"  DA by step (close):")
        for step_idx in range(predict):
            da = result.get(f'close_da_step{step_idx+1}', 0)
            print(f"    +{step_idx+1}: {da:>5.1%}")
    else:
        da = result.get('close_da_step1', result.get('close_da', 0))
        print(f"  close DA: {da:>5.1%}")


def print_bucket_result(result, title="Bucket Eval"):
    """
    打印分桶评估结果（mode4专用）
    """
    print(f"\n[{title}] Samples: {result.get('n_samples', 0)}")
    print(f"  Overall Trajectory IC: close={result.get('close_trajectory_ic', 0):.4f}")

    # Core bucket (2%~5%)
    core_bucket = '2%~5%'
    n_core = result.get(f'bucket_{core_bucket}_n', 0)
    ic_core = result.get(f'bucket_{core_bucket}_ic')
    da_core = result.get(f'bucket_{core_bucket}_da')
    traj_ic_core = result.get(f'bucket_{core_bucket}_trajectory_ic')

    ic_str = f"{ic_core:.4f}" if ic_core is not None else "N/A"
    da_str = f"{da_core:.1%}" if da_core is not None else "N/A"
    traj_str = f"{traj_ic_core:.4f}" if traj_ic_core is not None else "N/A"

    print(f"  Core Bucket (2%~5%): N={n_core}, IC={ic_str}, DA={da_str}, TrajIC={traj_str}")

    # Extreme bucket
    n_ext = result.get('bucket_>5%_n', 0)
    bias_ext = result.get('bucket_>5%_bias')
    if bias_ext is not None:
        bias_str = f"{bias_ext:.2%}"
        print(f"  Extreme (>5%): N={n_ext}, bias={bias_str}")
        if abs(bias_ext) > 0.05:
            print(f"    WARNING: Large bias in extreme bucket!")