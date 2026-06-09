"""
模型能力对比测试 - 多维度评估

两种模式：
1. multi_step (默认): 400+10 自回归预测，v3 数据集
2. single_step: 60+1 单步预测，全序列滑动窗口数据集

评估指标：
1. IC / Rank IC — 因子预测能力
2. ICIR — IC 稳定性
3. DA (Directional Accuracy) — 涨跌方向准确率
4. DDA (Directional Change Accuracy) — 转折点捕获率
5. 多步 IC — 1~10 步的 IC 曲线（仅 multi_step）
6. 分位数覆盖率 — 概率预测质量（仅 multi_step）
7. NMSE — 归一化均方误差
8. 波动率指标 — VolCorr, Weighted IC, Vol Binned, Vol DA
9. 朴素指标 — Up/Down Hit, Balance

用法：
    # 多步模式（原有）
    python -u finetune/benchmark_models.py

    # 单步模式
    python -u finetune/benchmark_models.py --mode single_step --checkpoint outputs/models/ma60_predictor_lb60_pd1/checkpoint-696418
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# ============================================================================
# 配置
# ============================================================================

PRED_LEN = 10
CLIP = 5.0
N_SAMPLES = 500
SEED = 42
N_QUANTILE_SAMPLES = 10  # 概率预测采样次数

# Tokenizer（不同模型可能使用不同的分词器）
TOKENIZER_MA60_V1 = 'outputs/models/ma60_tokenizer_v1/checkpoints/best_model'
TOKENIZER_MA60_BASE = 'outputs/models/ma60_tokenizer_base_v1/checkpoints/best_model'
TOKENIZER_ORIG_BASE = 'pretrained/Kronos-Tokenizer-base'

# 数据集路径
DATA_V3 = 'finetune/data/processed_datasets_ma60_windowed_v3/val_data.pkl'
DATA_V3_SMALL = 'finetune/data/processed_datasets_ma60_windowed_v3_small/val_data.pkl'

# 待测模型配置: (name, model_path, tokenizer_path, max_context, lookback, val_data_path)
MODELS = [
    # 原始模型: 原始分词器
    ('mini-orig-400', 'pretrained/Kronos-mini', TOKENIZER_ORIG_BASE, 2048, 400, DATA_V3),
    ('small-orig-400', 'pretrained/Kronos-small', TOKENIZER_ORIG_BASE, 512, 400, DATA_V3),
    ('base-orig-400', 'pretrained/Kronos-base', TOKENIZER_ORIG_BASE, 512, 400, DATA_V3),
    # mini 微调模型: ma60_tokenizer_v1（训练时使用的分词器）
    ('mini-v5-best_ic', 'outputs/models/ma60_predictor_mini_v5/checkpoints/best_ic_model', TOKENIZER_MA60_V1, 2048, 400, DATA_V3),
    # small 微调模型: ma60_tokenizer_base_v1（训练时使用的分词器）
    ('small-v5a-best_ic', 'outputs/models/ma60_predictor_small_v5a/checkpoints/best_ic_model', TOKENIZER_MA60_BASE, 512, 200, DATA_V3_SMALL),
    # small V5b 微调模型: freeze 4层 + embedding
    ('small-v5b-best_ic', 'outputs/models/ma60_predictor_small_v5b/checkpoints/best_ic_model', TOKENIZER_MA60_BASE, 512, 200, DATA_V3_SMALL),
]

# ============================================================================
# 单步模式配置
# ============================================================================

SINGLE_STEP_DATA = 'finetune/data/kline_daily_ma60.pkl'
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']


def load_single_step_data(data_path, lookback, predict):
    """加载 kline_daily_ma60.pkl 并返回时间切分的 (stock, start) 索引"""
    with open(data_path, 'rb') as f:
        raw = pickle.load(f)

    all_data = {}
    train_indices, val_indices, test_indices = [], [], []
    window = lookback + predict

    for sym in sorted(raw.keys()):
        d = raw[sym]
        seq_len = len(d['normalized'])
        if seq_len < window + 1:
            continue

        all_data[sym] = {
            'normalized': d['normalized'].astype(np.float32),
            'original': d['original'].astype(np.float32),
            'means': d['means'].astype(np.float32),
            'stds': d['stds'].astype(np.float32),
            'index': d['index'],
        }

        n_windows = seq_len - window
        tr_end = int(n_windows * 0.70)
        val_end = int(n_windows * 0.85)

        for i in range(n_windows):
            idx = (sym, i)
            if i < tr_end:
                train_indices.append(idx)
            elif i < val_end:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

    return all_data, train_indices, val_indices, test_indices


def compute_da(predictions, actuals):
    """DA: 方向准确率"""
    return np.mean(np.sign(predictions) == np.sign(actuals))


def compute_dda(predictions, actuals, actuals_prev):
    """DDA: 方向变化准确率"""
    actual_change = np.sign(actuals) != np.sign(actuals_prev)
    pred_change = np.sign(predictions) != np.sign(actuals_prev)
    change_mask = actual_change
    if change_mask.sum() == 0:
        return np.nan
    return np.mean(pred_change[change_mask] == actual_change[change_mask])


def compute_icir(ic_series):
    """ICIR: IC 均值 / IC 标准差"""
    if len(ic_series) < 2:
        return np.nan
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    if ic_std < 1e-8:
        return np.nan
    return ic_mean / ic_std


def compute_nmse(predictions, actuals):
    """NMSE: 归一化均方误差"""
    var_actual = np.var(actuals)
    if var_actual < 1e-8:
        return np.nan
    return np.mean((predictions - actuals) ** 2) / var_actual


def compute_vol_corr(predictions, actuals):
    """波动率预测准确性：预测幅度与真实幅度的相关性"""
    pred_abs = np.abs(predictions)
    actual_abs = np.abs(actuals)
    if len(pred_abs) < 10:
        return np.nan
    return np.corrcoef(pred_abs, actual_abs)[0, 1]


def compute_vol_binned(predictions, actuals, n_bins=2):
    """分档波动率验证：高/低波动预测组的实际波动差异"""
    if len(predictions) < 20:
        return {'high_mean': np.nan, 'low_mean': np.nan, 'ratio': np.nan, 'n': len(predictions)}
    pred_abs = np.abs(predictions)
    actual_abs = np.abs(actuals)
    median = np.median(pred_abs)
    high_mask = pred_abs >= median
    low_mask = pred_abs < median
    high_mean = np.mean(actual_abs[high_mask]) if high_mask.sum() > 0 else np.nan
    low_mean = np.mean(actual_abs[low_mask]) if low_mask.sum() > 0 else np.nan
    ratio = high_mean / low_mean if (low_mean and low_mean > 1e-12) else np.nan
    return {'high_mean': float(high_mean), 'low_mean': float(low_mean),
            'ratio': float(ratio), 'n': len(predictions)}


def compute_vol_weighted_ic(predictions, actuals):
    """波动率加权 IC：以预测幅度为权重计算加权 Pearson 相关"""
    if len(predictions) < 10:
        return np.nan
    weights = np.abs(predictions)
    w_sum = weights.sum()
    if w_sum < 1e-12:
        return np.nan
    w_mean_pred = np.average(predictions, weights=weights)
    w_mean_actual = np.average(actuals, weights=weights)
    w_cov = np.average((predictions - w_mean_pred) * (actuals - w_mean_actual), weights=weights)
    w_var_pred = np.average((predictions - w_mean_pred) ** 2, weights=weights)
    w_var_actual = np.average((actuals - w_mean_actual) ** 2, weights=weights)
    denom = np.sqrt(w_var_pred * w_var_actual)
    if denom < 1e-12:
        return np.nan
    return float(w_cov / denom)


def compute_vol_da(predictions, actuals):
    """波动率方向准确率：预测波动大小 vs 实际波动大小，对了几次（朴素指标）

    预测波动 = abs(预测收益)，实际波动 = abs(真实收益)
    各取中位数分高/低两组，计算分类准确率。
    返回: (accuracy, up_hit, down_hit) — 总体准确率、预测高波动的命中率、预测低波动的命中率
    """
    if len(predictions) < 20:
        return np.nan, np.nan, np.nan
    pred_abs = np.abs(predictions)
    actual_abs = np.abs(actuals)
    pred_median = np.median(pred_abs)
    actual_median = np.median(actual_abs)
    pred_high = pred_abs >= pred_median
    actual_high = actual_abs >= actual_median
    accuracy = np.mean(pred_high == actual_high)
    # Up-hit: 预测高波动时，实际也高波动的比例
    up_hit = np.mean(actual_high[pred_high]) if pred_high.sum() > 0 else np.nan
    # Down-hit: 预测低波动时，实际也低波动的比例
    down_hit = np.mean(~actual_high[~pred_high]) if (~pred_high).sum() > 0 else np.nan
    return float(accuracy), float(up_hit), float(down_hit)


def compute_up_down_hits(predictions, actuals):
    """涨跌命中率：预测涨的里面实际涨的比例，预测跌的里面实际跌的比例（朴素指标）

    返回: (up_hit, down_hit, balance)
    - up_hit: 预测上涨的股票中实际涨了的比例
    - down_hit: 预测下跌的股票中实际跌了的比例
    - balance: up_hit 和 down_hit 的调和平均（衡量多空双向判断力是否均衡）
    """
    if len(predictions) < 20:
        return np.nan, np.nan, np.nan
    pred_up = predictions > 0
    pred_down = predictions < 0
    actual_up = actuals > 0
    actual_down = actuals < 0
    up_hit = np.mean(actual_up[pred_up]) if pred_up.sum() > 0 else np.nan
    down_hit = np.mean(actual_down[pred_down]) if pred_down.sum() > 0 else np.nan
    # 调和平均：如果多空严重失衡（比如只会喊涨），balance 会很低
    if not np.isnan(up_hit) and not np.isnan(down_hit) and up_hit + down_hit > 1e-12:
        balance = 2 * up_hit * down_hit / (up_hit + down_hit)
    else:
        balance = np.nan
    return float(up_hit), float(down_hit), float(balance)


def compute_quantile_coverage(sample_preds, actuals, quantiles=[0.1, 0.5, 0.9]):
    """分位数覆盖率"""
    coverages = {}
    for q in quantiles:
        q_vals = np.quantile(sample_preds, q, axis=1)
        if q <= 0.5:
            coverages[f'q{q:.1f}'] = np.mean(actuals >= q_vals)
        else:
            coverages[f'q{q:.1f}'] = np.mean(actuals <= q_vals)
    q10 = np.quantile(sample_preds, 0.1, axis=1)
    q90 = np.quantile(sample_preds, 0.9, axis=1)
    coverages['q10_q90'] = np.mean((actuals >= q10) & (actuals <= q90))
    return coverages


def benchmark_model(model, tokenizer, device, val_data, lookback, pred_len,
                    clip, max_context, n_samples, rng, n_quantile_samples=10):
    """多维度模型评估 — 全 6 维特征"""
    model.eval()

    feature_names = ['open', 'high', 'low', 'close', 'vol', 'amt']

    all_symbols = list(val_data.keys())
    symbols = rng.choice(all_symbols, size=min(n_samples, len(all_symbols)), replace=False).tolist()

    # 每步每特征的预测和实际收益率
    step_preds_all = {s: {f: [] for f in range(6)} for s in range(1, pred_len + 1)}
    step_actuals_all = {s: {f: [] for f in range(6)} for s in range(1, pred_len + 1)}
    step_actuals_prev_all = {s: {f: [] for f in range(6)} for s in range(1, pred_len + 1)}

    # 概率预测采样 (close only)
    quantile_preds = {s: [] for s in range(1, pred_len + 1)}
    quantile_actuals = {s: [] for s in range(1, pred_len + 1)}

    for symbol in symbols:
        data = val_data[symbol]
        seq_len = len(data['normalized'])

        if seq_len < lookback + pred_len:
            continue

        try:
            full_len = lookback + pred_len
            end_idx = seq_len

            x_norm_full = data['normalized'][end_idx - full_len:end_idx].astype(np.float32)
            means_full = data['means'][end_idx - full_len:end_idx]
            stds_full = data['stds'][end_idx - full_len:end_idx]
            timestamps_full = data['index'][end_idx - full_len:end_idx]

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

            # 单次推理
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm[:lookback]).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=max_context, pred_len=pred_len,
                    clip=clip, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                # 全 6 维 denormalize
                pred_raw = preds[0, -pred_len:, :] * stds_full[lookback:, :] + means_full[lookback:, :]

            actual_raw = data['original'][end_idx - pred_len:, :]
            baseline_vals = data['original'][end_idx - pred_len - 1, :]

            # 多步多特征收益率
            for s in range(1, pred_len + 1):
                for f in range(6):
                    base = float(baseline_vals[f])
                    pred_return = (float(pred_raw[s - 1, f]) - base) / base if abs(base) > 1e-8 else 0.0
                    actual_return = (float(actual_raw[s - 1, f]) - base) / base if abs(base) > 1e-8 else 0.0
                    if s == 1:
                        actual_prev = 0.0
                    else:
                        prev_val = float(actual_raw[s - 2, f])
                        actual_prev = (prev_val - base) / base if abs(base) > 1e-8 else 0.0

                    step_preds_all[s][f].append(pred_return)
                    step_actuals_all[s][f].append(actual_return)
                    step_actuals_prev_all[s][f].append(actual_prev)

            # 概率预测采样 (close only)
            if n_quantile_samples > 1:
                baseline_close = float(baseline_vals[3])
                sample_returns = []
                with torch.no_grad():
                    multi_preds = auto_regressive_inference(
                        tokenizer, model,
                        x_tensor, x_stamp_tensor, y_stamp_tensor,
                        max_context=max_context, pred_len=pred_len,
                        clip=clip, T=1.0, top_k=0, top_p=0.9,
                        sample_count=n_quantile_samples, verbose=False
                    )
                for k in range(n_quantile_samples):
                    pred_norm_k = multi_preds[k, -pred_len:, 3]
                    pred_raw_k = pred_norm_k * stds_full[lookback:, 3] + means_full[lookback:, 3]
                    sample_returns.append(pred_raw_k)

                for s in range(1, pred_len + 1):
                    step_returns = [(sr[s - 1] - baseline_close) / baseline_close for sr in sample_returns]
                    quantile_preds[s].append(step_returns)
                    actual_return = (float(actual_raw[s - 1, 3]) - baseline_close) / baseline_close
                    quantile_actuals[s].append(float(actual_return))

        except Exception as e:
            if len(step_preds_all[1][3]) == 0:
                print(f"  [WARN] First error: {e}")
            continue

    # === 汇总指标 ===
    results = {}

    # 全特征指标
    for f in range(6):
        fname = feature_names[f]
        step_metrics_f = {}
        step_ics_f = []
        for s in range(1, pred_len + 1):
            preds_arr = np.array(step_preds_all[s][f])
            actuals_arr = np.array(step_actuals_all[s][f])
            prevs_arr = np.array(step_actuals_prev_all[s][f])

            if len(preds_arr) < 10:
                continue

            ic = np.corrcoef(preds_arr, actuals_arr)[0, 1]
            rank_ic, _ = spearmanr(preds_arr, actuals_arr)
            da = compute_da(preds_arr, actuals_arr)
            dda = compute_dda(preds_arr, actuals_arr, prevs_arr)
            nmse = compute_nmse(preds_arr, actuals_arr)
            pred_bias = float(np.mean(preds_arr) - np.mean(actuals_arr))
            var_actual = np.var(actuals_arr)
            var_ratio = float(np.var(preds_arr) / var_actual) if var_actual > 1e-8 else np.nan

            step_metrics_f[s] = {
                'ic': ic, 'rank_ic': rank_ic,
                'da': da, 'dda': dda, 'nmse': nmse,
                'pred_bias': pred_bias, 'var_ratio': var_ratio,
                'n': len(preds_arr)
            }
            step_ics_f.append(ic)

        results[f'{fname}_step_metrics'] = step_metrics_f
        # 代表步 (step 3)
        rep_step = 3
        if rep_step in step_metrics_f:
            m = step_metrics_f[rep_step]
            results[f'{fname}_ic'] = m['ic']
            results[f'{fname}_rank_ic'] = m['rank_ic']
            results[f'{fname}_da'] = m['da']
            results[f'{fname}_dda'] = m['dda']
            results[f'{fname}_nmse'] = m['nmse']
            # 波动率指标
            step_preds_f = np.array(step_preds_all[rep_step][f])
            step_actuals_f = np.array(step_actuals_all[rep_step][f])
            results[f'{fname}_vol_corr'] = compute_vol_corr(step_preds_f, step_actuals_f)
            results[f'{fname}_vol_binned'] = compute_vol_binned(step_preds_f, step_actuals_f)
            results[f'{fname}_vol_weighted_ic'] = compute_vol_weighted_ic(step_preds_f, step_actuals_f)
            # 朴素指标
            vol_da, vol_up, vol_down = compute_vol_da(step_preds_f, step_actuals_f)
            results[f'{fname}_vol_da'] = vol_da
            results[f'{fname}_vol_up_hit'] = vol_up
            results[f'{fname}_vol_down_hit'] = vol_down
            up_hit, down_hit, balance = compute_up_down_hits(step_preds_f, step_actuals_f)
            results[f'{fname}_up_hit'] = up_hit
            results[f'{fname}_down_hit'] = down_hit
            results[f'{fname}_hit_balance'] = balance
        if len(step_ics_f) >= 2:
            results[f'{fname}_icir'] = compute_icir(step_ics_f)

    # 兼容顶层 key（close）
    for key in ['ic', 'rank_ic', 'da', 'dda', 'nmse', 'icir']:
        results[key] = results.get(f'close_{key}', np.nan)
    results['step_metrics'] = results.get('close_step_metrics', {})

    if quantile_preds[1]:
        coverage = {}
        for s in [1, 3, 5, 10]:
            if s in quantile_preds and quantile_preds[s]:
                sample_arr = np.array(quantile_preds[s])
                actual_arr = np.array(quantile_actuals[s])
                cov = compute_quantile_coverage(sample_arr, actual_arr)
                coverage[s] = cov
        results['quantile_coverage'] = coverage

    results['n_samples'] = len(step_preds_all.get(1, {}).get(3, []))
    results['feature_names'] = feature_names
    return results


def benchmark_single_step(model, tokenizer, device, all_data, indices,
                          lookback, n_samples, rng):
    """单步预测综合评估 — 60+1 模式，所有指标仅 step=1"""
    import torch.nn.functional as F

    model.eval()
    feature_names = FEATURE_NAMES

    # 采样
    n_avail = min(n_samples, len(indices))
    sampled = rng.choice(len(indices), size=n_avail, replace=False)

    preds_all = {f: [] for f in range(6)}
    actuals_all = {f: [] for f in range(6)}
    actuals_prev_all = {f: [] for f in range(6)}

    for idx in sampled:
        sym, start = indices[idx]
        d = all_data[sym]
        end = start + lookback + 1  # 60 input + 1 target

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

        try:
            with torch.no_grad():
                norm_t = torch.from_numpy(norm).unsqueeze(0).to(device)
                stamp_t = torch.from_numpy(stamp).unsqueeze(0).to(device)

                t0, t1 = tokenizer.encode(norm_t, half=True)
                s1_logits, s2_logits = model(t0[:, :-1], t1[:, :-1], stamp=stamp_t[:, :-1, :])

                s1_p = torch.argmax(s1_logits[:, -1, :], dim=-1)
                s2_p = torch.argmax(s2_logits[:, -1, :], dim=-1)
                z = tokenizer.decode([s1_p.unsqueeze(1), s2_p.unsqueeze(1)], half=True)
                z = z.squeeze(1).cpu().numpy()
        except Exception:
            continue

        pred_px = z * stds[-1] + means[-1]
        base_px = orig[-2]
        actual_px = orig[-1]

        # 上一期的实际收益率 (for DDA)
        actual_prev_base = orig[-3] if len(orig) >= 3 else orig[-2]
        actual_prev_return = (orig[-2] - actual_prev_base) / (abs(actual_prev_base) + 1e-8)

        for f in range(6):
            base = float(base_px[f])
            if abs(base) < 1e-8:
                continue
            pr = float((pred_px[0, f] - base) / base)
            ar = float((actual_px[f] - base) / base)
            apr = float(actual_prev_return[f])

            if np.isfinite(pr) and np.isfinite(ar):
                preds_all[f].append(pr)
                actuals_all[f].append(ar)
                actuals_prev_all[f].append(apr)

    # 汇总
    results = {}
    for f in range(6):
        fname = feature_names[f]
        preds_arr = np.array(preds_all[f])
        actuals_arr = np.array(actuals_all[f])
        prevs_arr = np.array(actuals_prev_all[f])

        if len(preds_arr) < 10:
            continue

        ic = np.corrcoef(preds_arr, actuals_arr)[0, 1]
        rank_ic, _ = spearmanr(preds_arr, actuals_arr)
        da = compute_da(preds_arr, actuals_arr)
        dda = compute_dda(preds_arr, actuals_arr, prevs_arr)
        nmse = compute_nmse(preds_arr, actuals_arr)
        pred_bias = float(np.mean(preds_arr) - np.mean(actuals_arr))
        var_actual = np.var(actuals_arr)
        var_ratio = float(np.var(preds_arr) / var_actual) if var_actual > 1e-8 else np.nan

        results[f'{fname}_ic'] = ic
        results[f'{fname}_rank_ic'] = rank_ic
        results[f'{fname}_da'] = da
        results[f'{fname}_dda'] = dda
        results[f'{fname}_nmse'] = nmse
        results[f'{fname}_pred_bias'] = pred_bias
        results[f'{fname}_var_ratio'] = var_ratio
        results[f'{fname}_icir'] = ic  # 单步 ICIR = IC（仅一步无法计算稳定性）
        results[f'{fname}_n'] = len(preds_arr)

        # 波动率指标
        results[f'{fname}_vol_corr'] = compute_vol_corr(preds_arr, actuals_arr)
        results[f'{fname}_vol_binned'] = compute_vol_binned(preds_arr, actuals_arr)
        results[f'{fname}_vol_weighted_ic'] = compute_vol_weighted_ic(preds_arr, actuals_arr)

        vol_da, vol_up, vol_down = compute_vol_da(preds_arr, actuals_arr)
        results[f'{fname}_vol_da'] = vol_da
        results[f'{fname}_vol_up_hit'] = vol_up
        results[f'{fname}_vol_down_hit'] = vol_down

        up_hit, down_hit, balance = compute_up_down_hits(preds_arr, actuals_arr)
        results[f'{fname}_up_hit'] = up_hit
        results[f'{fname}_down_hit'] = down_hit
        results[f'{fname}_hit_balance'] = balance

    # 兼容顶层 key (close)
    for key in ['ic', 'rank_ic', 'da', 'dda', 'nmse', 'icir']:
        results[key] = results.get(f'close_{key}', np.nan)

    results['n_samples'] = len(preds_all[3])
    results['feature_names'] = feature_names
    return results


def print_single_step_results(result, label):
    """打印单步评估结果"""
    feature_names = result.get('feature_names', FEATURE_NAMES)

    print(f"\n  === {label} Summary ===")
    print(f"  IC:       {result.get('ic', np.nan):.4f}")
    print(f"  Rank IC:  {result.get('rank_ic', np.nan):.4f}")
    print(f"  DA:       {result.get('da', np.nan):.4f}")
    print(f"  DDA:      {result.get('dda', np.nan):.4f}")
    print(f"  NMSE:     {result.get('nmse', np.nan):.4f}")
    print(f"  Samples:  {result['n_samples']}")

    # 全特征
    print(f"\n  === All Features ===")
    print(f"  {'Feat':>5s} {'IC':>8s} {'RankIC':>8s} {'DA':>8s} {'DDA':>8s} {'NMSE':>8s} {'Bias':>8s} {'VarR':>8s}")
    for fname in feature_names:
        f_ic = result.get(f'{fname}_ic', np.nan)
        f_ric = result.get(f'{fname}_rank_ic', np.nan)
        f_da = result.get(f'{fname}_da', np.nan)
        f_dda = result.get(f'{fname}_dda', np.nan)
        f_nmse = result.get(f'{fname}_nmse', np.nan)
        f_bias = result.get(f'{fname}_pred_bias', np.nan)
        f_varr = result.get(f'{fname}_var_ratio', np.nan)
        bias_str = f"{f_bias:+.4f}" if not np.isnan(f_bias) else "    N/A"
        varr_str = f"{f_varr:.4f}" if not np.isnan(f_varr) else "    N/A"
        dda_str = f"{f_dda:.4f}" if not np.isnan(f_dda) else "    N/A"
        print(f"  {fname:>5s} {f_ic:>8.4f} {f_ric:>8.4f} {f_da:>8.4f} {dda_str:>8s} {f_nmse:>8.4f} {bias_str:>8s} {varr_str:>8s}")

    # 波动率指标
    print(f"\n  === Volatility Metrics ===")
    print(f"  {'Feat':>5s} {'VolCorr':>8s} {'W-IC':>8s} {'RawIC':>8s} {'Delta':>8s} {'Hi/Lo':>8s}")
    for fname in feature_names:
        f_vc = result.get(f'{fname}_vol_corr', np.nan)
        f_wic = result.get(f'{fname}_vol_weighted_ic', np.nan)
        f_ic = result.get(f'{fname}_ic', np.nan)
        f_delta = f_wic - f_ic if not (np.isnan(f_wic) or np.isnan(f_ic)) else np.nan
        f_vb = result.get(f'{fname}_vol_binned', {})
        f_ratio = f_vb.get('ratio', np.nan)
        vc_str = f"{f_vc:.4f}" if not np.isnan(f_vc) else "    N/A"
        wic_str = f"{f_wic:.4f}" if not np.isnan(f_wic) else "    N/A"
        ic_str = f"{f_ic:.4f}" if not np.isnan(f_ic) else "    N/A"
        delta_str = f"{f_delta:+.4f}" if not np.isnan(f_delta) else "    N/A"
        ratio_str = f"{f_ratio:.4f}" if not np.isnan(f_ratio) else "    N/A"
        print(f"  {fname:>5s} {vc_str:>8s} {wic_str:>8s} {ic_str:>8s} {delta_str:>8s} {ratio_str:>8s}")

    # 朴素指标
    print(f"\n  === Plain Metrics — 'out of 100 correct' ===")
    print(f"  {'Feat':>5s} {'DA':>7s} {'UpHit':>7s} {'DnHit':>7s} {'Bal':>7s} {'VolDA':>7s} {'VUp':>7s} {'VDn':>7s}")
    for fname in feature_names:
        f_da = result.get(f'{fname}_da', np.nan)
        f_up = result.get(f'{fname}_up_hit', np.nan)
        f_dn = result.get(f'{fname}_down_hit', np.nan)
        f_bal = result.get(f'{fname}_hit_balance', np.nan)
        f_vda = result.get(f'{fname}_vol_da', np.nan)
        f_vup = result.get(f'{fname}_vol_up_hit', np.nan)
        f_vdn = result.get(f'{fname}_vol_down_hit', np.nan)
        da_str = f"{f_da:.3f}" if not np.isnan(f_da) else "   N/A"
        up_str = f"{f_up:.3f}" if not np.isnan(f_up) else "   N/A"
        dn_str = f"{f_dn:.3f}" if not np.isnan(f_dn) else "   N/A"
        bal_str = f"{f_bal:.3f}" if not np.isnan(f_bal) else "   N/A"
        vda_str = f"{f_vda:.3f}" if not np.isnan(f_vda) else "   N/A"
        vup_str = f"{f_vup:.3f}" if not np.isnan(f_vup) else "   N/A"
        vdn_str = f"{f_vdn:.3f}" if not np.isnan(f_vdn) else "   N/A"
        print(f"  {fname:>5s} {da_str:>7s} {up_str:>7s} {dn_str:>7s} {bal_str:>7s} {vda_str:>7s} {vup_str:>7s} {vdn_str:>7s}")


def main():
    parser = argparse.ArgumentParser(description='Model Benchmark - Multi-metric Evaluation')
    parser.add_argument('--mode', choices=['multi_step', 'single_step'], default='multi_step')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path for single_step mode')
    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_MA60_V1)
    parser.add_argument('--data', type=str, default=SINGLE_STEP_DATA)
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--eval-split', choices=['val', 'test', 'both'], default='both')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(args.seed)

    # ========================================================================
    # 单步模式
    # ========================================================================
    if args.mode == 'single_step':
        print("=" * 80)
        print("Model Benchmark - Single-Step Mode (60+1)")
        print("=" * 80)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Tokenizer: {args.tokenizer}")
        print(f"Data: {args.data}")
        print(f"Samples: {args.n_samples}, Seed: {args.seed}")
        print(f"Eval split: {args.eval_split}")
        print(f"Device: {device}")

        if not args.checkpoint:
            print("[ERROR] --checkpoint is required for single_step mode")
            return

        lookback, predict = 60, 1

        tok_path = os.path.join(project_root, args.tokenizer)
        print(f"\nLoading tokenizer: {tok_path}")
        tokenizer = KronosTokenizer.from_pretrained(tok_path).eval().to(device)

        ckpt_path = os.path.join(project_root, args.checkpoint)
        print(f"Loading model: {ckpt_path}")
        model = Kronos.from_pretrained(ckpt_path).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {n_params:.1f}M")

        data_path = os.path.join(project_root, args.data)
        print(f"\nLoading data: {data_path}")
        all_data, train_idx, val_idx, test_idx = load_single_step_data(
            data_path, lookback, predict)
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        for split_name, split_idx in [('Val', val_idx), ('Test', test_idx)]:
            if args.eval_split != 'both' and args.eval_split != split_name.lower():
                continue
            print(f"\n{'=' * 80}")
            print(f"Evaluating on {split_name} set ({len(split_idx)} windows)")
            print(f"{'=' * 80}")

            t0 = time.time()
            result = benchmark_single_step(model, tokenizer, device, all_data,
                                           split_idx, lookback, args.n_samples, rng)
            elapsed = time.time() - t0

            if result and result['n_samples'] > 0:
                print_single_step_results(result, split_name)
                print(f"\n  Time: {elapsed:.1f}s")
            else:
                print(f"  [FAIL] Not enough valid predictions")

        del model
        torch.cuda.empty_cache()
        return

    # ========================================================================
    # 多步模式（原有逻辑）
    # ========================================================================
    print("=" * 80)
    print("Model Benchmark - Multi-metric Evaluation")
    print("=" * 80)
    print(f"Predict: {PRED_LEN}, Samples: {args.n_samples}, Seed: {args.seed}")
    print(f"Quantile samples: {N_QUANTILE_SAMPLES}")
    print(f"Device: {device}")

    # 加载数据集（按需）
    data_cache = {}

    def get_val_data(data_path):
        if data_path not in data_cache:
            full_path = os.path.join(project_root, data_path)
            print(f"Loading val data: {full_path}")
            with open(full_path, 'rb') as f:
                data_cache[data_path] = pickle.load(f)
            print(f"  {len(data_cache[data_path])} stocks")
        return data_cache[data_path]

    # Tokenizer 缓存
    tokenizer_cache = {}

    def get_tokenizer(tok_path):
        if tok_path not in tokenizer_cache:
            full_path = os.path.join(project_root, tok_path)
            print(f"Loading tokenizer: {full_path}")
            tokenizer_cache[tok_path] = KronosTokenizer.from_pretrained(full_path)
            tokenizer_cache[tok_path].eval().to(device)
        return tokenizer_cache[tok_path]

    # 逐个测试
    all_results = {}
    for model_name, model_path, tok_path, max_ctx, lookback, data_path in MODELS:
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name}")
        print(f"  Path: {model_path}")
        print(f"  Lookback: {lookback}, Max context: {max_ctx}")
        print(f"  Data: {data_path}")
        print(f"{'=' * 80}")

        full_model_path = os.path.join(project_root, model_path)
        if not os.path.exists(full_model_path):
            print(f"  [SKIP] Model not found: {full_model_path}")
            continue

        val_data = get_val_data(data_path)
        tokenizer = get_tokenizer(tok_path)

        print(f"Loading model...")
        model = Kronos.from_pretrained(full_model_path)
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {n_params:.1f}M")

        t0 = time.time()
        result = benchmark_model(model, tokenizer, device, val_data,
                                 lookback, PRED_LEN, CLIP, max_ctx, args.n_samples,
                                 rng, N_QUANTILE_SAMPLES)
        elapsed = time.time() - t0

        # 打印结果
        if result:
            print(f"\n  === Summary (step 3) ===")
            print(f"  IC:       {result.get('ic', np.nan):.4f}")
            print(f"  Rank IC:  {result.get('rank_ic', np.nan):.4f}")
            print(f"  ICIR:     {result.get('icir', np.nan):.4f}")
            print(f"  DA:       {result.get('da', np.nan):.4f}")
            print(f"  DDA:      {result.get('dda', np.nan):.4f}")
            print(f"  NMSE:     {result.get('nmse', np.nan):.4f}")
            print(f"  Samples:  {result['n_samples']}")
            print(f"  Time:     {elapsed:.1f}s")

            # 多步 IC
            print(f"\n  === Multi-step Metrics ===")
            print(f"  {'Step':>4s} {'IC':>8s} {'RankIC':>8s} {'DA':>8s} {'DDA':>8s} {'NMSE':>8s} {'Bias':>8s} {'VarR':>8s}")
            for s in sorted(result['step_metrics'].keys()):
                m = result['step_metrics'][s]
                dda_str = f"{m['dda']:.4f}" if not np.isnan(m['dda']) else "    N/A"
                bias_str = f"{m['pred_bias']:+.4f}"
                varr_str = f"{m['var_ratio']:.4f}" if not np.isnan(m.get('var_ratio', np.nan)) else "    N/A"
                print(f"  {s:>4d} {m['ic']:>8.4f} {m['rank_ic']:>8.4f} {m['da']:>8.4f} {dda_str:>8s} {m['nmse']:>8.4f} {bias_str:>8s} {varr_str:>8s}")

            # 分位数覆盖率
            if 'quantile_coverage' in result:
                print(f"\n  === Quantile Coverage ===")
                print(f"  {'Step':>4s} {'q10':>8s} {'q50':>8s} {'q90':>8s} {'q10-q90':>8s}")
                for s in sorted(result['quantile_coverage'].keys()):
                    cov = result['quantile_coverage'][s]
                    print(f"  {s:>4d} {cov['q0.1']:>8.4f} {cov['q0.5']:>8.4f} {cov['q0.9']:>8.4f} {cov['q10_q90']:>8.4f}")

            # === 全特征概览 ===
            feature_names = result.get('feature_names', ['open', 'high', 'low', 'close', 'vol', 'amt'])
            print(f"\n  === All Features (step 3) ===")
            print(f"  {'Feat':>5s} {'IC':>8s} {'RankIC':>8s} {'ICIR':>8s} {'DA':>8s} {'NMSE':>8s} {'Bias':>8s} {'VarR':>8s}")
            for fname in feature_names:
                f_ic = result.get(f'{fname}_ic', np.nan)
                f_ric = result.get(f'{fname}_rank_ic', np.nan)
                f_icir = result.get(f'{fname}_icir', np.nan)
                f_da = result.get(f'{fname}_da', np.nan)
                f_nmse = result.get(f'{fname}_nmse', np.nan)
                f_sm = result.get(f'{fname}_step_metrics', {})
                f_bias = f_sm.get(3, {}).get('pred_bias', np.nan) if 3 in f_sm else np.nan
                f_varr = f_sm.get(3, {}).get('var_ratio', np.nan) if 3 in f_sm else np.nan
                bias_str = f"{f_bias:+.4f}" if not np.isnan(f_bias) else "    N/A"
                varr_str = f"{f_varr:.4f}" if not np.isnan(f_varr) else "    N/A"
                print(f"  {fname:>5s} {f_ic:>8.4f} {f_ric:>8.4f} {f_icir:>8.4f} {f_da:>8.4f} {f_nmse:>8.4f} {bias_str:>8s} {varr_str:>8s}")

            # === 波动率指标 (step 3) ===
            print(f"\n  === Volatility Metrics (step 3) ===")
            print(f"  {'Feat':>5s} {'VolCorr':>8s} {'W-IC':>8s} {'RawIC':>8s} {'Delta':>8s} {'HiVol':>8s} {'LoVol':>8s} {'Ratio':>8s}")
            print(f"  {'':>5s} {'(abs corr)':>8s} {'(w cov)':>8s} {'(pearson)':>8s} {'(WIC-IC)':>8s} {'(abs mean)':>8s} {'(abs mean)':>8s} {'(Hi/Lo)':>8s}")
            for fname in feature_names:
                f_vc = result.get(f'{fname}_vol_corr', np.nan)
                f_wic = result.get(f'{fname}_vol_weighted_ic', np.nan)
                f_ic = result.get(f'{fname}_ic', np.nan)
                f_delta = f_wic - f_ic if not (np.isnan(f_wic) or np.isnan(f_ic)) else np.nan
                f_vb = result.get(f'{fname}_vol_binned', {})
                f_hi = f_vb.get('high_mean', np.nan)
                f_lo = f_vb.get('low_mean', np.nan)
                f_ratio = f_vb.get('ratio', np.nan)
                vc_str = f"{f_vc:.4f}" if not np.isnan(f_vc) else "    N/A"
                wic_str = f"{f_wic:.4f}" if not np.isnan(f_wic) else "    N/A"
                ic_str = f"{f_ic:.4f}" if not np.isnan(f_ic) else "    N/A"
                delta_str = f"{f_delta:+.4f}" if not np.isnan(f_delta) else "    N/A"
                hi_str = f"{f_hi:.4f}" if not np.isnan(f_hi) else "    N/A"
                lo_str = f"{f_lo:.4f}" if not np.isnan(f_lo) else "    N/A"
                ratio_str = f"{f_ratio:.4f}" if not np.isnan(f_ratio) else "    N/A"
                print(f"  {fname:>5s} {vc_str:>8s} {wic_str:>8s} {ic_str:>8s} {delta_str:>8s} {hi_str:>8s} {lo_str:>8s} {ratio_str:>8s}")

            # === 朴素指标 (step 3) — 100次中对了多少次 ===
            print(f"\n  === Plain Metrics (step 3) — 'out of 100, how many correct?' ===")
            print(f"  {'Feat':>5s} {'DA':>7s} {'UpHit':>7s} {'DnHit':>7s} {'Bal':>7s} {'VolDA':>7s} {'VUp':>7s} {'VDn':>7s}")
            print(f"  {'':>5s} {'(涨跌)':>7s} {'(喊涨)':>7s} {'(喊跌)':>7s} {'(平衡)':>7s} {'(波动)':>7s} {'(喊大)':>7s} {'(喊小)':>7s}")
            for fname in feature_names:
                f_da = result.get(f'{fname}_da', np.nan)
                f_up = result.get(f'{fname}_up_hit', np.nan)
                f_dn = result.get(f'{fname}_down_hit', np.nan)
                f_bal = result.get(f'{fname}_hit_balance', np.nan)
                f_vda = result.get(f'{fname}_vol_da', np.nan)
                f_vup = result.get(f'{fname}_vol_up_hit', np.nan)
                f_vdn = result.get(f'{fname}_vol_down_hit', np.nan)
                da_str = f"{f_da:.3f}" if not np.isnan(f_da) else "   N/A"
                up_str = f"{f_up:.3f}" if not np.isnan(f_up) else "   N/A"
                dn_str = f"{f_dn:.3f}" if not np.isnan(f_dn) else "   N/A"
                bal_str = f"{f_bal:.3f}" if not np.isnan(f_bal) else "   N/A"
                vda_str = f"{f_vda:.3f}" if not np.isnan(f_vda) else "   N/A"
                vup_str = f"{f_vup:.3f}" if not np.isnan(f_vup) else "   N/A"
                vdn_str = f"{f_vdn:.3f}" if not np.isnan(f_vdn) else "   N/A"
                print(f"  {fname:>5s} {da_str:>7s} {up_str:>7s} {dn_str:>7s} {bal_str:>7s} {vda_str:>7s} {vup_str:>7s} {vdn_str:>7s}")

            all_results[model_name] = result
        else:
            print(f"  [FAIL] Not enough valid predictions")

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # === 汇总表 ===
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    # 主指标表
    print(f"\n--- Primary Metrics (step 3) ---")
    print(f"{'Model':<25s} {'IC':>8s} {'RankIC':>8s} {'ICIR':>8s} {'DA':>8s} {'DDA':>8s} {'NMSE':>8s} {'N':>6s}")
    print("-" * 79)
    for model_name, _, _, _, _, _ in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            dda_str = f"{r['dda']:.4f}" if not np.isnan(r.get('dda', np.nan)) else "    N/A"
            print(f"{model_name:<25s} {r['ic']:>8.4f} {r['rank_ic']:>8.4f} {r.get('icir',np.nan):>8.4f} {r['da']:>8.4f} {dda_str:>8s} {r['nmse']:>8.4f} {r['n_samples']:>6d}")

    # 波动率指标对比 (close only)
    feature_names = ['open', 'high', 'low', 'close', 'vol', 'amt']
    for fname in ['close']:  # 核心关注 close
        print(f"\n--- {fname} Volatility Metrics (step 3) ---")
        print(f"{'Model':<25s} {'VolCorr':>8s} {'W-IC':>8s} {'RawIC':>8s} {'Delta':>8s} {'Hi/Lo':>8s}")
        print("-" * 65)
        for model_name, _, _, _, _, _ in MODELS:
            if model_name in all_results:
                r = all_results[model_name]
                f_vc = r.get(f'{fname}_vol_corr', np.nan)
                f_wic = r.get(f'{fname}_vol_weighted_ic', np.nan)
                f_ic = r.get(f'{fname}_ic', np.nan)
                f_delta = f_wic - f_ic if not (np.isnan(f_wic) or np.isnan(f_ic)) else np.nan
                f_vb = r.get(f'{fname}_vol_binned', {})
                f_ratio = f_vb.get('ratio', np.nan)
                vc_str = f"{f_vc:.4f}" if not np.isnan(f_vc) else "    N/A"
                wic_str = f"{f_wic:.4f}" if not np.isnan(f_wic) else "    N/A"
                ic_str = f"{f_ic:.4f}" if not np.isnan(f_ic) else "    N/A"
                delta_str = f"{f_delta:+.4f}" if not np.isnan(f_delta) else "    N/A"
                ratio_str = f"{f_ratio:.4f}" if not np.isnan(f_ratio) else "    N/A"
                print(f"{model_name:<25s} {vc_str:>8s} {wic_str:>8s} {ic_str:>8s} {delta_str:>8s} {ratio_str:>8s}")

    # 朴素指标对比 (close only) — 100次中对了多少次
    for fname in ['close']:
        print(f"\n--- {fname} Plain Metrics (step 3) — 'out of 100 correct' ---")
        print(f"{'Model':<25s} {'DA':>7s} {'UpHit':>7s} {'DnHit':>7s} {'Bal':>7s} {'VolDA':>7s} {'VUp':>7s} {'VDn':>7s}")
        print(f"{'':<25s} {'(涨跌)':>7s} {'(喊涨)':>7s} {'(喊跌)':>7s} {'(均衡)':>7s} {'(波大)':>7s} {'(喊大)':>7s} {'(喊小)':>7s}")
        print("-" * 81)
        for model_name, _, _, _, _, _ in MODELS:
            if model_name in all_results:
                r = all_results[model_name]
                f_da = r.get(f'{fname}_da', np.nan)
                f_up = r.get(f'{fname}_up_hit', np.nan)
                f_dn = r.get(f'{fname}_down_hit', np.nan)
                f_bal = r.get(f'{fname}_hit_balance', np.nan)
                f_vda = r.get(f'{fname}_vol_da', np.nan)
                f_vup = r.get(f'{fname}_vol_up_hit', np.nan)
                f_vdn = r.get(f'{fname}_vol_down_hit', np.nan)
                da_str = f"{f_da:.3f}" if not np.isnan(f_da) else "   N/A"
                up_str = f"{f_up:.3f}" if not np.isnan(f_up) else "   N/A"
                dn_str = f"{f_dn:.3f}" if not np.isnan(f_dn) else "   N/A"
                bal_str = f"{f_bal:.3f}" if not np.isnan(f_bal) else "   N/A"
                vda_str = f"{f_vda:.3f}" if not np.isnan(f_vda) else "   N/A"
                vup_str = f"{f_vup:.3f}" if not np.isnan(f_vup) else "   N/A"
                vdn_str = f"{f_vdn:.3f}" if not np.isnan(f_vdn) else "   N/A"
                print(f"{model_name:<25s} {da_str:>7s} {up_str:>7s} {dn_str:>7s} {bal_str:>7s} {vda_str:>7s} {vup_str:>7s} {vdn_str:>7s}")

    # 多步 IC 表
    print(f"\n--- Multi-step IC ---")
    header = f"{'Model':<25s}" + "".join(f" {'Step'+str(s):>7s}" for s in range(1, PRED_LEN + 1))
    print(header)
    print("-" * len(header))
    for model_name, _, _, _, _, _ in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            line = f"{model_name:<25s}"
            for s in range(1, PRED_LEN + 1):
                if s in r['step_metrics']:
                    line += f" {r['step_metrics'][s]['ic']:>7.4f}"
                else:
                    line += f" {'N/A':>7s}"
            print(line)

    # 多步 DA 表
    print(f"\n--- Multi-step DA ---")
    print(header)
    print("-" * len(header))
    for model_name, _, _, _, _, _ in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            line = f"{model_name:<25s}"
            for s in range(1, PRED_LEN + 1):
                if s in r['step_metrics']:
                    line += f" {r['step_metrics'][s]['da']:>7.4f}"
                else:
                    line += f" {'N/A':>7s}"
            print(line)

    # 多步 Bias 表
    print(f"\n--- Multi-step Pred Bias ---")
    print(header)
    print("-" * len(header))
    for model_name, _, _, _, _, _ in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            line = f"{model_name:<25s}"
            for s in range(1, PRED_LEN + 1):
                if s in r['step_metrics']:
                    line += f" {r['step_metrics'][s]['pred_bias']:>+7.4f}"
                else:
                    line += f" {'N/A':>7s}"
            print(line)

    # 多步 VarR 表
    print(f"\n--- Multi-step Var Ratio (close) ---")
    print(header)
    print("-" * len(header))
    for model_name, _, _, _, _, _ in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            line = f"{model_name:<25s}"
            for s in range(1, PRED_LEN + 1):
                if s in r['step_metrics']:
                    vr = r['step_metrics'][s].get('var_ratio', np.nan)
                    line += f" {vr:>7.4f}" if not np.isnan(vr) else f" {'N/A':>7s}"
                else:
                    line += f" {'N/A':>7s}"
            print(line)

    # 全特征 IC 对比表 (step 3)
    feature_names = ['open', 'high', 'low', 'close', 'vol', 'amt']
    print(f"\n--- Feature IC Comparison (step 3) ---")
    feat_header = f"{'Model':<25s}" + "".join(f" {f:>7s}" for f in feature_names)
    print(feat_header)
    print("-" * len(feat_header))
    for model_name, _, _, _, _, _ in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            line = f"{model_name:<25s}"
            for fname in feature_names:
                f_ic = r.get(f'{fname}_ic', np.nan)
                line += f" {f_ic:>7.4f}" if not np.isnan(f_ic) else f" {'N/A':>7s}"
            print(line)

    # 全特征 DA 对比表 (step 3)
    print(f"\n--- Feature DA Comparison (step 3) ---")
    print(feat_header)
    print("-" * len(feat_header))
    for model_name, _, _, _, _, _ in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            line = f"{model_name:<25s}"
            for fname in feature_names:
                f_da = r.get(f'{fname}_da', np.nan)
                line += f" {f_da:>7.4f}" if not np.isnan(f_da) else f" {'N/A':>7s}"
            print(line)

    # 全特征 NMSE 对比表 (step 3)
    print(f"\n--- Feature NMSE Comparison (step 3) ---")
    print(feat_header)
    print("-" * len(feat_header))
    for model_name, _, _, _, _, _ in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            line = f"{model_name:<25s}"
            for fname in feature_names:
                f_nmse = r.get(f'{fname}_nmse', np.nan)
                line += f" {f_nmse:>7.4f}" if not np.isnan(f_nmse) else f" {'N/A':>7s}"
            print(line)

    print(f"\nPredict={PRED_LEN}, Samples={args.n_samples}, Seed={args.seed}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
