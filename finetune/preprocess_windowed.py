"""
窗口化数据预处理 - 密集采样 + 时间外推 val/test

划分策略：
- Train：密集采样（stride=10），预测目标在历史区
- Val：密集采样（stride=10），预测目标在倒数第二个预留区（时间外推）
- Test：密集采样（stride=10），预测目标在最后一个预留区（时间外推）

泄漏定义：val/test 的预测目标（窗口最后11步）从未在 train 中作为预测目标出现。
窗口的输入部分（前400步）允许跨区域，因为这只是上下文。

用法：
    python -u finetune/preprocess_windowed.py [--stride 10] [--k-val 100] [--k-test 100]
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import trange

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from finetune.config import Config

feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
ma_window = 60
clip = 5.0


def sliding_ma60_normalize_full(values, window=60):
    """对整个序列计算滑动MA60归一化"""
    n = len(values)
    normalized = np.zeros_like(values, dtype=np.float32)
    means = np.zeros((n, values.shape[1]), dtype=np.float32)
    stds = np.zeros((n, values.shape[1]), dtype=np.float32)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = values[start:i + 1]
        means[i] = np.mean(window_data, axis=0)
        stds[i] = np.std(window_data, axis=0) + 1e-5
        normalized[i] = (values[i] - means[i]) / stds[i]

    normalized = np.clip(normalized, -clip, clip)
    return normalized, means, stds


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Windowed data preprocessing')
    parser.add_argument('--lookback', type=int, default=400,
                        help='Lookback window size (default: 400, use 200 for small model)')
    parser.add_argument('--stride', type=int, default=10,
                        help='Window stride for all regions (default: 10)')
    parser.add_argument('--k-val', type=int, default=100,
                        help='Val prediction target region size in steps (default: 100)')
    parser.add_argument('--k-test', type=int, default=100,
                        help='Test prediction target region size in steps (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='processed_datasets_ma60_windowed_v3',
                        help='Output directory name')
    args = parser.parse_args()

    lookback = args.lookback
    predict = 10
    window_size = lookback + predict + 1

    print("=" * 60)
    print("Windowed Data Preprocessing (Dense + Time-Extrapolation)")
    print("=" * 60)
    print(f"Window size: {window_size} (lookback={lookback}, predict={predict})")
    print(f"Stride: {args.stride}")
    print(f"Val reserve: {args.k_val} steps (prediction targets)")
    print(f"Test reserve: {args.k_test} steps (prediction targets)")
    print(f"Seed: {args.seed}")

    # === 1. 加载原始 CSV 数据 ===
    csv_dir = os.path.join(project_root, "finetune_csv", "exported_kline_data", "stocks")
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    print(f"\nLoading CSV files from: {csv_dir}")
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")

    def is_main_board(symbol):
        code = symbol.split('.')[0]
        if code.startswith(('000', '001', '002', '003')): return True
        if code.startswith(('600', '601', '603', '605')): return True
        return False

    min_amount = 50_000_000
    min_list_days = 250
    max_gap_ratio = 0.30

    # === 2. 加载、过滤、归一化、划分 ===
    train_windows = []
    val_windows = []
    test_windows = []
    stock_count = 0
    skipped_st = skipped_short = skipped_liquidity = skipped_gap = skipped_non_main = 0
    stock_data_cache = {}

    for i in trange(len(csv_files), desc="Processing"):
        csv_file = csv_files[i]
        symbol = csv_file.replace('.csv', '')

        if not is_main_board(symbol):
            skipped_non_main += 1
            continue

        try:
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.index.name = 'datetime'
            df = df.rename(columns={'volume': 'vol'})
            df = df[['open', 'high', 'low', 'close', 'vol', 'amount']]
            df = df.rename(columns={'amount': 'amt'})
            df = df.dropna()
        except Exception:
            continue

        if len(df) < min_list_days:
            skipped_short += 1
            continue

        pct_change = df['close'].pct_change().abs() * 100
        near_st_limit = (pct_change > 4.5) & (pct_change < 6.0)
        if near_st_limit.sum() / max(len(pct_change.dropna()), 1) > 0.10:
            skipped_st += 1
            continue

        if df['amt'].mean() < min_amount:
            skipped_liquidity += 1
            continue

        date_range = (df.index[-1] - df.index[0]).days
        expected = date_range * 5 / 7
        gap_ratio = 1 - len(df) / max(expected, 1)
        if gap_ratio > max_gap_ratio:
            skipped_gap += 1
            continue

        # MA60 归一化
        values = df[feature_cols].values.astype(np.float32)
        normalized, means, stds = sliding_ma60_normalize_full(values)

        # === 划分窗口 ===
        N = len(normalized)
        k_val = args.k_val
        k_test = args.k_test

        # 需要至少 window_size 步才能形成1个窗口
        # Train 的预测目标区域: [0, N-k_val-k_test)
        # Val 的预测目标区域: [N-k_val-k_test, N-k_test)
        # Test 的预测目标区域: [N-k_test, N)

        # Train 窗口: P+411 <= N-k_val-k_test → P <= N-k_val-k_test-window_size
        train_max_start = N - k_val - k_test - window_size
        # Val 窗口: P+400 >= N-k_val-k_test 且 P+411 <= N-k_test
        #         → P >= N-k_val-k_test-lookback 且 P <= N-k_test-window_size
        val_min_start = N - k_val - k_test - lookback
        val_max_start = N - k_test - window_size
        # Test 窗口: P+400 >= N-k_test 且 P+411 <= N
        #          → P >= N-k_test-lookback 且 P <= N-window_size
        test_min_start = N - k_test - lookback
        test_max_start = N - window_size

        n_train = n_val = n_test = 0

        # Train
        if train_max_start >= 0:
            for start in range(0, train_max_start + 1, args.stride):
                train_windows.append((symbol, start))
                n_train += 1

        # Val
        if val_min_start >= 0 and val_max_start >= val_min_start:
            for start in range(val_min_start, val_max_start + 1, args.stride):
                val_windows.append((symbol, start))
                n_val += 1

        # Test
        if test_min_start >= 0 and test_max_start >= test_min_start:
            for start in range(test_min_start, test_max_start + 1, args.stride):
                test_windows.append((symbol, start))
                n_test += 1

        if n_train > 0 or n_val > 0 or n_test > 0:
            stock_data_cache[symbol] = {
                'normalized': normalized,
                'means': means,
                'stds': stds,
                'original': values,
                'index': df.index,
            }
            stock_count += 1

    print(f"\n{'=' * 60}")
    print("Filtering results:")
    print(f"  Non-main board:      {skipped_non_main}")
    print(f"  Too short (<{min_list_days}d):   {skipped_short}")
    print(f"  Suspected ST:        {skipped_st}")
    print(f"  Low liquidity:       {skipped_liquidity}")
    print(f"  Too many gaps:       {skipped_gap}")
    print(f"  Passed stocks:       {stock_count}")
    print(f"  Train windows:       {len(train_windows)}")
    print(f"  Val windows:         {len(val_windows)}")
    print(f"  Test windows:        {len(test_windows)}")

    # === 3. 构建数据集 ===
    def build_dataset(window_list):
        """从 (symbol, start) 列表构建数据集"""
        symbol_windows = {}
        for symbol, start in window_list:
            if symbol not in symbol_windows:
                symbol_windows[symbol] = []
            symbol_windows[symbol].append(start)

        dataset = {}
        for symbol, starts in symbol_windows.items():
            starts.sort()
            data = stock_data_cache[symbol]
            dataset[symbol] = {
                'normalized': data['normalized'],
                'means': data['means'],
                'stds': data['stds'],
                'original': data['original'],
                'index': data['index'],
                'windows': np.array(starts, dtype=np.int64),
            }
        return dataset

    print(f"\nBuilding datasets...")
    train_data = build_dataset(train_windows)
    val_data = build_dataset(val_windows)
    test_data = build_dataset(test_windows)

    # 统计
    def count_windows(data):
        return sum(len(d.get('windows', [])) for d in data.values())

    def count_symbols(data):
        return len(data)

    def collect_timestamps(data, stock_cache):
        """收集窗口中间点的时间分布"""
        years = {}
        for sym, d in data.items():
            for start in d.get('windows', []):
                mid = start + window_size // 2
                if sym in stock_cache and mid < len(stock_cache[sym]['index']):
                    ts = stock_cache[sym]['index'][mid]
                    yr = ts.year
                    years.setdefault(yr, 0)
                    years[yr] += 1
        return years

    print(f"\n{'=' * 60}")
    print("Dataset statistics:")

    for name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        n_wins = count_windows(data)
        n_syms = count_symbols(data)
        years = collect_timestamps(data, stock_data_cache)
        print(f"\n  {name}: {n_wins} windows, {n_syms} symbols")
        print(f"    Year distribution: {dict(sorted(years.items()))}")

    # === 4. 验证无泄漏 ===
    print(f"\n{'=' * 60}")
    print("Leakage check:")

    # 检查 train 和 val 的预测目标是否有重叠
    # 收集 train 的预测目标位置集合（采样检查）
    train_targets = set()
    for sym, d in train_data.items():
        for start in d.get('windows', []):
            # 预测目标: [start+lookback, start+window_size)
            for t in range(start + lookback, start + window_size):
                train_targets.add((sym, t))

    val_leaks = 0
    for sym, d in val_data.items():
        for start in d.get('windows', []):
            for t in range(start + lookback, start + window_size):
                if (sym, t) in train_targets:
                    val_leaks += 1
                    break  # 每个窗口只计一次
            if val_leaks > 0 and val_leaks <= 3:
                print(f"  [LEAK] Val window at {sym}:{start} overlaps train target")

    test_leaks = 0
    for sym, d in test_data.items():
        for start in d.get('windows', []):
            for t in range(start + lookback, start + window_size):
                if (sym, t) in train_targets:
                    test_leaks += 1
                    break
            if test_leaks > 0 and test_leaks <= 3:
                print(f"  [LEAK] Test window at {sym}:{start} overlaps train target")

    if val_leaks == 0 and test_leaks == 0:
        print("  [OK] No leakage detected: val/test prediction targets never appear in train targets")
    else:
        print(f"  [LEAK] Leaks found: val={val_leaks}, test={test_leaks}")

    # === 5. 保存 ===
    output_path = os.path.join(project_root, "finetune", "data", args.output)
    os.makedirs(output_path, exist_ok=True)

    print(f"\nSaving to: {output_path}")

    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        filepath = os.path.join(output_path, f"{name}_data.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  {name}_data.pkl: {size_mb:.1f} MB")

    meta = {
        'stride': args.stride,
        'window_size': window_size,
        'k_val': args.k_val,
        'k_test': args.k_test,
        'seed': args.seed,
        'n_train': len(train_windows),
        'n_val': len(val_windows),
        'n_test': len(test_windows),
        'n_stocks': stock_count,
    }
    with open(os.path.join(output_path, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\n{'=' * 60}")
    print("Preprocessing completed!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
