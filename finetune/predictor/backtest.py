"""
Kronos Predictor Backtest Entry

回测入口，支持：
- 批量回测
- 可懂指标（方向胜率/振幅误差率/涨跌停命中率）
- per-stock IC 聚合（不跨股票混算）

使用：
    python finetune/predictor/backtest.py --model mini --n-samples 1000

关键：
- backtest IC 逐股票算再聚合（E8 解决）
- sigmoid 参数配置化（E7 解决）
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference
from safetensors.torch import load_file

from finetune.predictor.core.config import BacktestConfig
from finetune.predictor.core.paths import (
    get_backtest_data_path,
    get_tokenizer_path,
    get_model_path,
    get_checkpoint_path,  # §6.11 修复：拼接 checkpoints/{name} 正确路径
)
from finetune.predictor.core.normalization import get_normalizer
from finetune.predictor.core.metrics import (
    safe_corrcoef,
    safe_spearmanr,
    detrend_to_baseline,
    excess_da,
    amplitude_error_rate,
    limit_hit_rate,
    detect_limit,
)
from finetune.predictor.core.utils import get_device


# ============================================================================
# 配置
# ============================================================================


# ============================================================================
# 回测函数
# ============================================================================

def backtest(
    model,
    tokenizer,
    test_data,
    config: BacktestConfig,
    device: torch.device,
    n_samples: int = 100,
    seed: int = 42,
    norm_mode: str = 'sliding_ma60',
    model_type: str = 'mini'
):
    """
    回测（正确口径）

    关键：
    - per-stock IC（不跨股票混算）
    - sigmoid 参数从配置读取
    - 样本由 preprocess.py 生成，统一格式：{symbol: {normalized, means, stds, original, index, lookback, predict}}
    """
    model.eval()
    tokenizer.eval()

    rng = np.random.RandomState(seed)

    # 每只股票多窗口（preprocess 滑窗生成，stride=1 默认每股 13 窗口）
    symbols = list(test_data.keys())

    # 抽样：按股票抽（n_samples=股票数），每只股票全量遍历其 windows
    if n_samples > 0 and n_samples < len(symbols):
        indices = rng.choice(len(symbols), size=n_samples, replace=False)
        symbols = [symbols[i] for i in indices]

    # 结果收集（per-symbol，多窗口聚合）
    results_by_symbol = {}

    # PR4 修复：一次性校验 context 段归一化未用 target 段数据（无泄露）
    # 对第一个股票的第一个窗口重算 context 段归一化，与 preprocess 存的 normalized 对比。
    # 若 preprocess 用了 target 数据归一化 context，重算结果会不同 → 报错。
    if symbols:
        first_sym = symbols[0]
        first_d = test_data[first_sym]
        first_w = first_d['windows'][0]  # 取第一个窗口校验
        try:
            lb = first_w['lookback']
            pd_ = first_w['predict']
            full_orig = first_w['original']
            full_norm = first_w['normalized']
            seq_len = len(full_orig)
            ctx_start = seq_len - lb - pd_
            target_start = ctx_start + lb

            # 重算 context 段归一化（仅用 context 及其前 N 步历史，不含 target）
            from finetune.predictor.core.normalization import NormalizerFactory
            required = NormalizerFactory.get_required_history(norm_mode)
            hist_start = max(0, ctx_start - required)
            ctx_with_hist = full_orig[hist_start:target_start]  # 不含 target
            normalizer = get_normalizer(norm_mode)
            recon_norm, _, _ = normalizer.normalize(ctx_with_hist.astype(np.float32))
            # recon_norm 长度 = len(ctx_with_hist)，context 段在其末尾 lb 个
            recon_ctx_norm = recon_norm[-lb:]

            stored_ctx_norm = full_norm[ctx_start:target_start]
            if not np.allclose(recon_ctx_norm, stored_ctx_norm, atol=1e-5, equal_nan=True):
                raise RuntimeError(
                    f"PR4 校验失败：{first_sym} context 段归一化与重算不一致，"
                    f"preprocess 可能用了 target 段数据归一化 context（泄露）"
                )
            print(f"[INFO] PR4 校验通过：context 段归一化未用 target 数据（{first_sym} window0）")
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[WARNING] PR4 校验跳过：{e}")

    for sym in tqdm(symbols, desc="Backtesting"):
        sym_data = test_data[sym]
        windows = sym_data['windows']

        for wi, w in enumerate(windows):
            try:
                lookback = w['lookback']
                predict = w['predict']
                # 序列布局：[required_history? + lookback + predict]
                # normalized/original/index 长度 = required_history + lookback + predict
                # lookback 段起始 = len - lookback - predict
                seq_len = len(w['normalized'])
                ctx_start = seq_len - lookback - predict
                target_start = ctx_start + lookback

                x_norm = w['normalized'][ctx_start:target_start].astype(np.float32)
                means = w['means']
                stds = w['stds']
                original = w['original']
                timestamps = w['index']

                baseline = original[target_start - 1]
                baseline_close = original[target_start - 1, 3]

                # 时间戳
                x_stamp = np.stack([
                    timestamps[ctx_start:target_start].minute.values,
                    timestamps[ctx_start:target_start].hour.values,
                    timestamps[ctx_start:target_start].weekday.values,
                    timestamps[ctx_start:target_start].day.values,
                    timestamps[ctx_start:target_start].month.values,
                ], axis=1).astype(np.float32)

                y_stamp = np.stack([
                    timestamps[target_start:target_start + predict].minute.values,
                    timestamps[target_start:target_start + predict].hour.values,
                    timestamps[target_start:target_start + predict].weekday.values,
                    timestamps[target_start:target_start + predict].day.values,
                    timestamps[target_start:target_start + predict].month.values,
                ], axis=1).astype(np.float32)

                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                    x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                    y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model,
                        x_tensor, x_stamp_tensor, y_stamp_tensor,
                        max_context={'mini': 2048, 'small': 512, 'base': 512}.get(model_type, 2048),  # B11 修复：按 model_type
                        pred_len=predict,
                        clip=5.0,
                        T=1.0,
                        top_p=0.9,
                        sample_count=1,
                        verbose=False
                    )

                    pred_norm = preds[0, -predict:, :]

                    # 反归一化（用 target 段的 means/stds）
                    # preds 已是 numpy（auto_regressive_inference 返回 np.ndarray），直接用
                    pred_raw = pred_norm * stds[target_start:target_start + predict] + means[target_start:target_start + predict]
                    actual = original[target_start:target_start + predict]

                # 增益计算（用 close）
                pred_close = pred_raw[0, 3]
                actual_close = actual[0, 3]
                pred_gain = (pred_close - baseline_close) / (abs(baseline_close) + 1e-8)
                actual_gain = (actual_close - baseline_close) / (abs(baseline_close) + 1e-8)

                # sigmoid 分值
                score = config.sigmoid_score(pred_gain)

                # 方向
                pred_dir = pred_gain > 0
                actual_dir = actual_gain > 0
                direction_correct = pred_dir == actual_dir

                # 振幅
                pred_amp = pred_raw[0, 1] - pred_raw[0, 2]
                actual_amp = actual[0, 1] - actual[0, 2]
                amp_rate = amplitude_error_rate(pred_amp, actual_amp)

                # 涨跌停
                pred_limit = detect_limit(pred_raw, baseline_close, config.limit_pct)
                actual_limit = detect_limit(actual, baseline_close, config.limit_pct)

                # 按股票收集（多窗口聚合到同一 symbol）
                if sym not in results_by_symbol:
                    results_by_symbol[sym] = {
                        'pred_gains': [],
                        'actual_gains': [],
                        'scores': [],
                        'directions': [],
                        'actual_dirs': [],  # 新增：用于计算 naive DA
                        'amp_rates': [],
                        'pred_limit': [],
                        'actual_limit': [],
                    }

                results_by_symbol[sym]['pred_gains'].append(pred_gain)
                results_by_symbol[sym]['actual_gains'].append(actual_gain)
                results_by_symbol[sym]['scores'].append(score)
                results_by_symbol[sym]['directions'].append(direction_correct)
                results_by_symbol[sym]['actual_dirs'].append(actual_dir)  # 新增
                results_by_symbol[sym]['amp_rates'].append(amp_rate)
                results_by_symbol[sym]['pred_limit'].append(pred_limit.any())
                results_by_symbol[sym]['actual_limit'].append(actual_limit.any())

            except Exception as e:
                # 回测循环不应静默吞异常（曾因 except: continue 导致全 0 无法定位）
                # 打印异常供诊断，仍 continue 跳过该窗口不中断整个回测
                print(f"[WARN backtest] {sym} window{wi}: {type(e).__name__}: {e}")
                continue

    # 聚合（per-stock IC）
    per_stock_ics = []
    per_stock_rank_ics = []

    for sym, res in results_by_symbol.items():
        if len(res['pred_gains']) >= 3:
            ic = safe_corrcoef(res['pred_gains'], res['actual_gains'])
            rank_ic = safe_spearmanr(res['pred_gains'], res['actual_gains'])
            if ic is not None:
                per_stock_ics.append(ic)
            if rank_ic is not None:
                per_stock_rank_ics.append(rank_ic)

    # 统计
    all_pred_gains = []
    all_actual_gains = []
    all_directions = []
    all_actual_dirs = []  # 新增：用于计算 naive DA
    all_amp_rates = []
    all_pred_limit = []
    all_actual_limit = []

    for sym, res in results_by_symbol.items():
        all_pred_gains.extend(res['pred_gains'])
        all_actual_gains.extend(res['actual_gains'])
        all_directions.extend(res['directions'])
        all_actual_dirs.extend(res['actual_dirs'])  # 新增
        all_amp_rates.extend(res['amp_rates'])
        all_pred_limit.extend(res['pred_limit'])
        all_actual_limit.extend(res['actual_limit'])

    # 结果
    result = {
        'n_samples': len(all_pred_gains),
        'n_stocks': len(results_by_symbol),
    }

    # IC（per-stock 聚合，补 p25/p75）
    if per_stock_ics:
        ics_arr = np.array(per_stock_ics)
        n = len(per_stock_ics)
        result['backtest_ic_mean'] = float(np.mean(ics_arr))
        result['backtest_ic_std'] = float(np.std(ics_arr)) if n >= 2 else 0.0
        result['backtest_ic_p25'] = float(np.percentile(ics_arr, 25)) if n >= 4 else None
        result['backtest_ic_p50'] = float(np.percentile(ics_arr, 50))
        result['backtest_ic_p75'] = float(np.percentile(ics_arr, 75)) if n >= 4 else None
    else:
        result['backtest_ic_mean'] = 0.0
        result['backtest_ic_std'] = 0.0
        result['backtest_ic_p25'] = None
        result['backtest_ic_p50'] = None
        result['backtest_ic_p75'] = None

    if per_stock_rank_ics:
        result['backtest_rank_ic_mean'] = float(np.mean(per_stock_rank_ics))

    # 方向胜率（DA）- 真实 naive DA 计算
    if all_directions:
        model_da = float(np.mean(all_directions))
        result['direction_accuracy'] = model_da
        # 真实 naive DA = 多数方向比例（持平预测的 DA）
        if all_actual_dirs:
            up_ratio = float(np.mean(all_actual_dirs))
            naive_da = max(up_ratio, 1 - up_ratio)
        else:
            naive_da = 0.5
        result['naive_da'] = naive_da
        result['excess_da'] = excess_da(model_da, naive_da)

    # 振幅
    if all_amp_rates:
        result['amplitude_mean'] = float(np.mean(all_amp_rates))
        result['amplitude_std'] = float(np.std(all_amp_rates))
        result['amplitude_usable_pct'] = float(np.mean(np.abs(np.array(all_amp_rates) - 1.0) < 0.3))

    # 涨跌停
    limit_result = limit_hit_rate(np.array(all_pred_limit), np.array(all_actual_limit))
    result['limit_hit_rate'] = limit_result['hit_rate']
    result['limit_pred_count'] = limit_result['n_pred_limit']
    result['limit_actual_count'] = limit_result['n_actual_limit']

    # 分布
    if all_pred_gains:
        result['pred_gain_mean'] = float(np.mean(all_pred_gains))
        result['pred_gain_std'] = float(np.std(all_pred_gains))
        result['actual_gain_mean'] = float(np.mean(all_actual_gains))
        result['actual_gain_std'] = float(np.std(all_actual_gains))
        result['win_rate'] = float(np.mean(np.array(all_actual_gains) > 0))

    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos Predictor Backtest')
    parser.add_argument('--model', type=str, default='mini',
                        help='模型类型(mini/small/base)或 checkpoint 目录完整路径')
    parser.add_argument('--checkpoint', type=str, default='best_combined_model',
                        choices=['best_model', 'best_ic_model', 'best_combined_model', 'latest_model'],
                        help='Checkpoint 名称(§6.11 修复：定位 checkpoints/{name}/，与 eval 一致)')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block',
                        choices=['time', 'block'],
                        help='Split mode for locating model checkpoint (B3 修复：不再硬编码 block)')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='抽样股票数（-1 全量）。每股票全量遍历其滑窗 windows（stride=1 默认每股 13 窗口）')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-pct', type=float, default=0.10)
    parser.add_argument('--signal-center', type=float, default=0.084)
    parser.add_argument('--signal-steepness', type=float, default=21.0)
    args = parser.parse_args()

    device = get_device()

    config = BacktestConfig(
        signal_center=args.signal_center,
        signal_steepness=args.signal_steepness,
        limit_pct=args.limit_pct,
    )

    # Tokenizer（B1/B4 修复：使用 get_tokenizer_path，不再硬编码 legacy 路径）
    if args.tokenizer:
        tokenizer_path = args.tokenizer
    else:
        tokenizer_path = get_tokenizer_path(args.norm_mode, args.model)

    if not os.path.exists(tokenizer_path):
        # B3 修复：显式警告找不到 tokenizer
        print(f"[WARNING] Tokenizer not found at {tokenizer_path}")
        # fallback 到预训练
        tokenizer_path = 'pretrained/Kronos-Tokenizer-2k' if args.model == 'mini' else 'pretrained/Kronos-Tokenizer-base'
        print(f"[WARNING] Using pretrained tokenizer: {tokenizer_path}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    # 模型（B2 修复：根据 model 参数选择架构，不再硬编码 mini）
    pretrained_paths = {
        'mini': 'pretrained/Kronos-mini',
        'small': 'pretrained/Kronos-small',
        'base': 'pretrained/Kronos-base',
    }
    model_type = args.model if args.model in pretrained_paths else 'mini'
    model = Kronos.from_pretrained(pretrained_paths[model_type])
    model.eval().to(device)

    # Checkpoint（§6.11 修复：用 get_checkpoint_path 拼接 checkpoints/{name}，与 eval 一致）
    if args.model in pretrained_paths:
        # model_type 模式：按 norm_mode/lookback/predict/split_mode 定位 + checkpoints/{name}
        model_dir = get_model_path(args.norm_mode, args.lookback, args.predict, args.split_mode, args.model)
        checkpoint_dir = get_checkpoint_path(model_dir, args.checkpoint)
    else:
        # 路径模式：args.model 直接作为 checkpoint 目录
        checkpoint_dir = os.path.join(project_root, args.model)
    safetensors_path = os.path.join(checkpoint_dir, 'model.safetensors')

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        # B1 修复：checkpoint 与模型对不上直接报错（不静默用错权重）
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint 与模型不匹配，拒绝加载: missing={missing}, unexpected={unexpected}"
            )
        print(f"[INFO] Loaded checkpoint: {checkpoint_dir}")
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_dir}")
        print(f"[WARNING] Using pretrained model - backtest may not reflect fine-tuned performance")

    # 数据：回测样本（由 preprocess.py 生成）+ 样本外 raw（full_window 借用历史）
    test_path = get_backtest_data_path(args.norm_mode, args.lookback, args.predict)

    print(f"\n{'=' * 60}")
    print(f"Kronos Predictor Backtest")
    print(f"{'=' * 60}")
    print(f"model: {args.model}")
    print(f"tokenizer: {tokenizer_path}")
    print(f"norm_mode: {args.norm_mode}")
    print(f"n_samples: {args.n_samples if args.n_samples > 0 else 'FULL'}")
    print(f"signal_center: {config.signal_center}")
    print(f"signal_steepness: {config.signal_steepness}")
    print(f"{'=' * 60}")

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # 回测（样本由 preprocess.py 预先生成，含 context+target）
    result = backtest(
        model, tokenizer, test_data, config, device,
        n_samples=args.n_samples,
        seed=args.seed,
        norm_mode=args.norm_mode,
        model_type=model_type
    )

    # 输出
    print(f"\n{'=' * 60}")
    print(f"[Backtest Results]")
    print(f"{'=' * 60}")
    print(f"  Samples: {result['n_samples']}, Stocks: {result['n_stocks']}")

    print(f"\n  [Direction Accuracy]")
    print(f"    DA: {result.get('direction_accuracy', 0):.1%}")
    print(f"    Excess DA: {result.get('excess_da', 0):.1%} (vs 50% random)")

    print(f"\n  [Amplitude Error Rate]")
    print(f"    Mean: {result.get('amplitude_mean', 1):.2f}")
    print(f"    Usable (0.7-1.3): {result.get('amplitude_usable_pct', 0):.1%}")

    print(f"\n  [Limit Hit Rate]")
    hit_rate = result.get('limit_hit_rate')
    if hit_rate is not None:
        print(f"    Hit Rate: {hit_rate:.1%} (random ~1-3%, signal >10%)")
        print(f"    Predicted: {result['limit_pred_count']}, Actual: {result['limit_actual_count']}")
    else:
        print(f"    No predicted limit")

    print(f"\n  [Backtest IC] (per-stock aggregated)")
    print(f"    Mean: {result.get('backtest_ic_mean', 0):.4f}")
    print(f"    Std: {result.get('backtest_ic_std', 0):.4f}")
    print(f"    Rank IC Mean: {result.get('backtest_rank_ic_mean', 0):.4f}")

    print(f"\n  [Distribution]")
    print(f"    Pred gain: mean={result.get('pred_gain_mean', 0):+.2%}, std={result.get('pred_gain_std', 0):.2%}")
    print(f"    Actual gain: mean={result.get('actual_gain_mean', 0):+.2%}, std={result.get('actual_gain_std', 0):.2%}")
    print(f"    Win rate: {result.get('win_rate', 0):.1%}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()