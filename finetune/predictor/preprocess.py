"""
Kronos Predictor Preprocess Module

数据预处理入口

功能：
- 从 raw 数据生成 processed 数据
- 支持 time_split 和 block_split
- 运行 validate_no_leakage
- 生成 meta.pkl（fingerprint、边界、样本数）

使用：
    python finetune/predictor/preprocess.py \
        --norm-mode sliding_ma60 \
        --lookback 400 \
        --predict 10 \
        --split-mode block \
        --validate
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from finetune.predictor.core.config import DataConfig, parse_norm_mode
from finetune.predictor.core.paths import (
    get_raw_path,
    get_backtest_raw_path,
    get_split_data_path,
    get_backtest_data_path,
    get_meta_path,
    ensure_dir,
)
from finetune.predictor.core.schema import (
    SampleSchema,
    MetaSchema,
    sample_to_dict,
    compute_fingerprint,
    intervals_overlap,
)
from finetune.predictor.core.splitting import (
    time_split,
    block_split,
    validate_no_leakage,
    get_split_stats,
)
from finetune.predictor.core.normalization import (
    get_normalizer,
    NormalizerFactory,
)
from finetune.predictor.core.utils import safe_save_pickle, safe_save_json


def load_raw_data(raw_path: str) -> Dict[str, Any]:
    """
    加载原始数据

    Args:
        raw_path: kline_daily_raw.pkl 路径

    Returns:
        {symbol: DataFrame or dict}
    """
    print(f"Loading raw data: {raw_path}")
    with open(raw_path, 'rb') as f:
        raw_data = pickle.load(f)

    print(f"Loaded {len(raw_data)} symbols")
    return raw_data


def create_samples_from_raw(
    raw_data: Dict[str, Any],
    config: DataConfig
) -> List[SampleSchema]:
    """
    从原始数据创建样本

    Args:
        raw_data: 原始数据
        config: DataConfig

    Returns:
        样本列表（SampleSchema）
    """
    samples = []
    lookback = config.lookback
    predict = config.predict
    window_size = lookback + predict

    # block 模式：块内做 sliding_ma（块首用 expanding），块内窗口只需 window_size，无需额外历史
    # time 模式：整条 sliding_ma，窗口也只需 window_size
    use_block_generation = (config.split_mode == 'block')
    block_size = config.block_size if use_block_generation else None

    def _gen_range(seq_len):
        """生成 (窗口起点 i, 所属 block 起点) 的迭代器。block 模式按块，time 模式全序列(block_start=None)。"""
        if use_block_generation:
            b_start = 0
            while b_start + window_size <= seq_len:
                b_end = min(b_start + block_size, seq_len)
                # 块内 i：i + window_size ≤ b_end
                i_lo = b_start
                i_hi = b_end - window_size  # i+window_size ≤ b_end
                for i in range(i_lo, i_hi + 1):
                    yield i, b_start
                b_start = b_end
        else:
            for i in range(seq_len - window_size + 1):
                yield i, None  # time 模式无 block 归一化

    for symbol, data in tqdm(raw_data.items(), desc="Creating samples"):
        if hasattr(data, 'columns'):
            seq_len = len(data)
            if seq_len < window_size:
                continue
            for i, b_start in _gen_range(seq_len):
                samples.append(SampleSchema(
                    symbol=symbol,
                    window_start=i,
                    lookback_start=i,
                    lookback_end=i + lookback,
                    target_start=i + lookback,
                    target_end=i + lookback + predict,
                    split='unknown',
                    block_start=b_start,
                    values=data.values[i:i + window_size],
                    index=data.index[i:i + window_size],
                ))
        else:
            values = data.get('values', data.get('original'))
            index = data.get('index')
            if values is None:
                continue
            seq_len = len(values)
            if seq_len < window_size:
                continue
            for i, b_start in _gen_range(seq_len):
                samples.append(SampleSchema(
                    symbol=symbol,
                    window_start=i,
                    lookback_start=i,
                    lookback_end=i + lookback,
                    target_start=i + lookback,
                    target_end=i + lookback + predict,
                    split='unknown',
                    block_start=b_start,
                    values=values[i:i + window_size],
                    index=index[i:i + window_size] if index is not None else None,
                ))

    print(f"Created {len(samples)} samples")
    return samples


def apply_split(
    samples: List[SampleSchema],
    config: DataConfig,
    train_end: int = None,
    val_end: int = None
) -> Tuple[List[SampleSchema], List[SampleSchema], List[SampleSchema]]:
    """
    应用分割策略

    Args:
        samples: 样本列表
        config: DataConfig
        train_end: 时间分割的训练边界
        val_end: 时间分割的验证边界

    Returns:
        (train, val, test)
    """
    if config.split_mode == 'time':
        if train_end is None or val_end is None:
            raise ValueError("time_split requires train_end and val_end")
        return time_split(samples, train_end, val_end)

    elif config.split_mode == 'block':
        return block_split(
            samples,
            block_size=config.block_size,
            seed=config.seed
        )

    else:
        raise ValueError(f"Unknown split_mode: {config.split_mode}")


def create_backtest_samples(
    train_raw: Dict[str, Any],
    backtest_raw: Dict[str, Any],
    config: DataConfig,
    stride: int = 1
) -> Dict[str, Any]:
    """
    生成 backtest 样本（滑窗多窗口，每窗口独立归一化）

    两种归一化场景的窗口构成（由 required_history 决定）：
      - full_window：required_history=0，窗口 = lookback + predict，统一归一化（窗口内 mean/std）
      - sliding_maN：required_history=N，窗口 = N + lookback + predict，前 N 根参与归一化
        （提供滑动窗口历史）但不参与推理/测试，需从 train_raw 借用

    每窗口独立归一化（方案 A）：拼接 前置(required_history) + lookback + predict 后归一化一次。
    这样每窗口的归一化前缀都是真实历史，无统一归一化的 expanding 边界问题。

    窗口 k 的构成：
      - target = backtest[k : k+predict]（backtest 自身作为预测段）
      - context(required_history + lookback) = (train_raw 末尾 + backtest[0:k]) 末尾 context_len 根
        —— target 之前的真实历史，随窗口推进滑动，无泄露
      - k=0 时 context 全来自 train_raw 末尾；k 增大时逐渐含 backtest 前缀

    自回归语义下 stride=1 是天然口径：每个 target 起点都对应一个预测窗口。
    backtest_raw 22 根、predict=10、stride=1 → 每股 13 个窗口。

    Args:
        train_raw: kline_daily_raw（借用末尾 context 段，含归一化前置历史）
        backtest_raw: backtest_raw（自身作为预测段，滑窗形成多窗口）
        config: DataConfig
        stride: 窗口步长（默认 1）

    Returns:
        {symbol: {windows: [{normalized, means, stds, original, index, lookback, predict, target_start}], lookback, predict, n_windows}}
    """
    lookback = config.lookback
    predict = config.predict
    required_history = NormalizerFactory.get_required_history(config.norm_mode)
    normalizer = NormalizerFactory.create(config.norm_mode, clip=config.clip)
    context_len = lookback + required_history  # 每窗口 context 段长度（含归一化前置）
    window_len = context_len + predict          # 每窗口完整序列长度

    samples = {}
    skipped = 0
    total_windows = 0

    for symbol, bt_data in tqdm(backtest_raw.items(), desc="Backtest samples"):
        train_data = train_raw.get(symbol)
        if train_data is None:
            skipped += 1
            continue

        train_values = train_data['values']
        train_index = train_data['index']
        bt_values = bt_data['values']
        bt_index = bt_data['index']

        # train_raw 不足以提供首个窗口的 context（含归一化前置）
        if len(train_values) < context_len:
            skipped += 1
            continue
        # backtest_raw 不足以提供至少 1 个 target
        if len(bt_values) < predict:
            skipped += 1
            continue

        # 预取 train_raw 末尾 context_len 根作为 context 池前缀（k=0 时直接用）
        train_ctx_values = train_values[-context_len:]
        train_ctx_index = train_index[-context_len:]

        n_bt = len(bt_values)
        windows = []
        for k in range(0, n_bt - predict + 1, stride):
            # context 段 = (train_ctx + backtest[0:k]) 末尾 context_len 根
            if k == 0:
                ctx_values = train_ctx_values
                ctx_index = train_ctx_index
            else:
                combined_values = np.concatenate([train_ctx_values, bt_values[:k]], axis=0)
                combined_index = train_ctx_index.append(bt_index[:k])
                ctx_values = combined_values[-context_len:]
                ctx_index = combined_index[-context_len:]

            # 拼接完整序列：context(required_history + lookback) + predict
            tgt_values = bt_values[k:k + predict]
            tgt_index = bt_index[k:k + predict]
            full_values = np.concatenate([ctx_values, tgt_values], axis=0)
            full_index = ctx_index.append(tgt_index)

            # 每窗口独立归一化（sliding_ma 依赖 context 前缀历史，已含 required_history）
            normalized, means, stds = normalizer.normalize(full_values)

            windows.append({
                'normalized': normalized.astype(np.float32),
                'means': means.astype(np.float32),
                'stds': stds.astype(np.float32),
                'original': full_values.astype(np.float32),
                'index': full_index,
                'lookback': lookback,
                'predict': predict,
                'target_start': k,  # 在 backtest_raw 中的 target 起点（诊断用）
            })
            total_windows += 1

        if windows:
            samples[symbol] = {
                'windows': windows,
                'lookback': lookback,
                'predict': predict,
                'n_windows': len(windows),
            }

    print(f"Backtest samples: {len(samples)} stocks, {total_windows} windows (stride={stride}, skipped {skipped})")
    return samples


def save_backtest_samples(
    samples: Dict[str, Any],
    samples_path: str,
    config: DataConfig,
    backtest_raw_path: str
):
    """
    保存 backtest 样本与元数据

    输出：
      - samples.pkl：backtest 样本
      - meta.pkl：backtest 元数据（样本数、context/target 来源、时间区间、fingerprint）
    """
    ensure_dir(samples_path)
    with open(samples_path, 'wb') as f:
        pickle.dump(samples, f)
    n_stocks = len(samples)
    n_windows = sum(s.get('n_windows', len(s.get('windows', []))) for s in samples.values())
    print(f"Saved {n_stocks} stocks ({n_windows} windows) backtest samples to {samples_path}")

    # 统计 target 时间区间（所有窗口的 target 段索引范围）
    target_start_dates = []
    target_end_dates = []
    for s in samples.values():
        for w in s.get('windows', []):
            pd_ = w['predict']
            idx = w['index']
            target_start_dates.append(idx[-pd_])
            target_end_dates.append(idx[-1])

    meta_path = get_meta_path(
        config.norm_mode, config.lookback, config.predict, role='backtest'
    )
    ensure_dir(meta_path)
    safe_save_json({
        'norm_mode': config.norm_mode,
        'lookback': config.lookback,
        'predict': config.predict,
        'role': 'backtest',
        'n_stocks': n_stocks,
        'n_windows': n_windows,
        'n_samples': n_windows,  # 兼容旧字段（= 窗口数）
        'context_source': 'kline_daily_raw.pkl (末尾 lookback+required_history 根) + backtest_raw 前缀',
        'target_source': os.path.basename(backtest_raw_path),
        'target_start': str(min(target_start_dates)) if target_start_dates else None,
        'target_end': str(max(target_end_dates)) if target_end_dates else None,
        'backtest_raw_fingerprint': compute_fingerprint(backtest_raw_path),
        'created_at': datetime.now().isoformat(),
    }, meta_path)
    print(f"Saved backtest meta to {meta_path}")


def save_split_data(
    split_samples: List[SampleSchema],
    output_path: str,
    config: DataConfig,
    raw_data: Dict[str, Any]
):
    """
    保存分割数据（可直接使用的归一化数据）

    block 模式（per-block 归一化，防止跨 split 泄露）：
        {symbol: {
            'mode': 'block',
            'blocks': {b_start: {
                'normalized': (block_len, 6),  # 仅此 block 归一化（block 内 sliding_ma，块首 expanding）
                'means': (block_len, 6),
                'stds': (block_len, 6),
                'original': (block_len, 6),
                'index': (block_len,),
                'windows': (N,),  # 该 block 的窗口起点（绝对位置，在 b_start..b_end 内）
            }},
        }}
    time 模式（整条归一化，时间正序无泄露）：
        {symbol: {
            'mode': 'time',
            'normalized': (T, 6),
            'means': (T, 6),
            'stds': (T, 6),
            'original': (T, 6),
            'index': (T,),
            'windows': (N,),
        }}

    Args:
        split_samples: 分割后的样本列表
        output_path: 输出路径
        config: DataConfig（含 split_mode/block_size）
        raw_data: 原始数据
    """
    normalizer = NormalizerFactory.create(config.norm_mode, clip=config.clip)
    use_block = (config.split_mode == 'block')

    if use_block:
        # 按 (symbol, b_start) 收集窗口起点
        blocks_by_key = defaultdict(list)  # (symbol, b_start) -> [window_start, ...]
        for sample in split_samples:
            blocks_by_key[(sample.symbol, sample.block_start)].append(sample.window_start)

        data_by_symbol = {}
        for symbol in {s.symbol for s in split_samples}:
            raw = raw_data.get(symbol)
            if raw is None:
                continue
            if hasattr(raw, 'columns'):
                full_values = raw.values.astype(np.float32)
                full_index = raw.index
            else:
                full_values = np.asarray(raw['values'], dtype=np.float32)
                full_index = raw['index']

            blocks_data = {}
            for (sym, b_start), windows in blocks_by_key.items():
                if sym != symbol:
                    continue
                b_end = min(b_start + config.block_size, len(full_values))
                block_values = full_values[b_start:b_end]
                block_index = full_index[b_start:b_end]
                # block 内独立归一化（sliding_ma 仅用 block 内数据，块首 expanding）
                norm, means, stds = normalizer.normalize(block_values)
                blocks_data[b_start] = {
                    'normalized': norm.astype(np.float32),
                    'means': means.astype(np.float32),
                    'stds': stds.astype(np.float32),
                    'original': block_values,
                    'index': block_index,
                    'windows': np.array(sorted(set(windows)), dtype=np.int64),
                }
            if blocks_data:
                data_by_symbol[symbol] = {'mode': 'block', 'blocks': blocks_data}
    else:
        # time 模式：整条序列归一化
        windows_by_symbol = defaultdict(list)
        for sample in split_samples:
            windows_by_symbol[sample.symbol].append(sample.window_start)

        data_by_symbol = {}
        for symbol, windows in windows_by_symbol.items():
            raw = raw_data.get(symbol)
            if raw is None:
                continue
            if hasattr(raw, 'columns'):
                values = raw.values.astype(np.float32)
                index = raw.index
            else:
                values = np.asarray(raw['values'], dtype=np.float32)
                index = raw['index']
            normalized, means, stds = normalizer.normalize(values)
            data_by_symbol[symbol] = {
                'mode': 'time',
                'normalized': normalized.astype(np.float32),
                'means': means.astype(np.float32),
                'stds': stds.astype(np.float32),
                'original': values,
                'index': index,
                'windows': np.array(sorted(set(windows)), dtype=np.int64),
            }

    ensure_dir(output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(data_by_symbol, f)

    if use_block:
        n_samples = sum(len(b['windows']) for d in data_by_symbol.values() for b in d['blocks'].values())
        n_blocks = sum(len(d['blocks']) for d in data_by_symbol.values())
        print(f"Saved {n_samples} samples ({len(data_by_symbol)} symbols, {n_blocks} blocks, per-block normalized) to {output_path}")
    else:
        n_samples = sum(len(d['windows']) for d in data_by_symbol.values())
        print(f"Saved {n_samples} samples ({len(data_by_symbol)} symbols, full-series normalized) to {output_path}")


def save_meta(
    config: DataConfig,
    train: List[SampleSchema],
    val: List[SampleSchema],
    test: List[SampleSchema],
    raw_path: str,
    leakage_passed: bool
):
    """
    保存元数据

    Args:
        config: DataConfig
        train, val, test: 分割后的样本
        raw_path: 原始数据路径
        leakage_passed: 泄露检查结果
    """
    meta = MetaSchema(
        norm_mode=config.norm_mode,
        lookback=config.lookback,
        predict=config.predict,
        split_mode=config.split_mode,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        leakage_check_passed=leakage_passed,
        leakage_check_timestamp=datetime.now().isoformat(),
        created_at=datetime.now().isoformat(),
        data_fingerprint=compute_fingerprint(raw_path),
    )

    meta_path = get_meta_path(
        config.norm_mode,
        config.lookback,
        config.predict,
        config.split_mode
    )

    ensure_dir(meta_path)
    safe_save_json({
        'norm_mode': meta.norm_mode,
        'lookback': meta.lookback,
        'predict': meta.predict,
        'split_mode': meta.split_mode,
        'n_train': meta.n_train,
        'n_val': meta.n_val,
        'n_test': meta.n_test,
        'leakage_check_passed': meta.leakage_check_passed,
        'leakage_check_timestamp': meta.leakage_check_timestamp,
        'created_at': meta.created_at,
        'data_fingerprint': meta.data_fingerprint,
    }, meta_path)

    print(f"Saved meta to {meta_path}")


def preprocess(
    config: DataConfig,
    raw_path: str = None,
    backtest_raw_path: str = None,
    train_end: int = None,
    val_end: int = None,
    validate: bool = True,
    skip_backtest: bool = False,
    backtest_stride: int = 1
):
    """
    预处理主流程

    Args:
        config: DataConfig
        raw_path: 训练原始数据路径（kline_daily_raw）
        backtest_raw_path: 回测原始数据路径（backtest_raw）
        train_end: 时间分割训练边界
        val_end: 时间分割验证边界
        validate: 是否运行泄露检查
        skip_backtest: 是否跳过 backtest 样本生成
        backtest_stride: backtest 滑窗步长（默认 1，自回归天然口径）
    """
    if raw_path is None:
        raw_path = get_raw_path()
    if backtest_raw_path is None:
        backtest_raw_path = get_backtest_raw_path()

    # 1. 加载原始数据
    raw_data = load_raw_data(raw_path)

    # 2. 创建样本
    samples = create_samples_from_raw(raw_data, config)

    # 3. 应用分割
    train, val, test = apply_split(samples, config, train_end, val_end)

    # 4. 泄露检查
    leakage_passed = False
    if validate:
        try:
            validate_no_leakage(train, val, test)
            leakage_passed = True
        except AssertionError as e:
            print(f"Leakage check FAILED: {e}")
            leakage_passed = False

    # 5. 保存分割数据
    for split_name, split_samples in [('train', train), ('val', val), ('test', test)]:
        output_path = get_split_data_path(
            config.norm_mode,
            config.lookback,
            config.predict,
            config.split_mode,
            split_name
        )
        save_split_data(split_samples, output_path, config, raw_data)

    # 6. 保存元数据
    save_meta(config, train, val, test, raw_path, leakage_passed)

    # 7. 统计信息
    stats = get_split_stats(train, val, test)
    print(f"\nSplit stats:")
    print(f"  Train: {stats['n_train']} samples")
    print(f"  Val: {stats['n_val']} samples")
    print(f"  Test: {stats['n_test']} samples")
    print(f"  Symbols: {stats['n_symbols']}")
    print(f"  Leakage check: {'PASSED' if leakage_passed else 'FAILED'}")

    # 8. 生成 backtest 样本（context 来自 kline_daily_raw，target 来自 backtest_raw）
    if not skip_backtest:
        if os.path.exists(backtest_raw_path):
            print(f"\n[Backtest] Loading backtest raw: {backtest_raw_path}")
            with open(backtest_raw_path, 'rb') as f:
                backtest_raw = pickle.load(f)
            bt_samples = create_backtest_samples(raw_data, backtest_raw, config, stride=backtest_stride)
            bt_output_path = get_backtest_data_path(
                config.norm_mode, config.lookback, config.predict
            )
            save_backtest_samples(bt_samples, bt_output_path, config, backtest_raw_path)
        else:
            print(f"\n[Backtest] 跳过：{backtest_raw_path} 不存在")

    return leakage_passed


def main():
    parser = argparse.ArgumentParser(description='Kronos Data Preprocess')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block',
                        choices=['time', 'block'])
    parser.add_argument('--raw-data', type=str, default=None,
                        help='训练 raw 路径（默认 finetune/data/raw/kline_daily_raw.pkl）')
    parser.add_argument('--backtest-raw', type=str, default=None,
                        help='回测 raw 路径（默认 finetune/data/raw/backtest_raw.pkl）')
    parser.add_argument('--train-end', type=int, default=None,
                        help='Time split train boundary (target_end)')
    parser.add_argument('--val-end', type=int, default=None,
                        help='Time split val boundary (target_end)')
    parser.add_argument('--validate', dest='validate', action='store_true', default=True,
                        help='运行泄露检查（默认开启）')
    parser.add_argument('--no-validate', dest='validate', action='store_false',
                        help='跳过泄露检查')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='跳过 backtest 样本生成')
    parser.add_argument('--backtest-stride', type=int, default=1,
                        help='backtest 滑窗步长（默认 1，自回归天然口径。backtest_raw 22 根 predict=10 → 每股 13 窗口）')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples-per-block', type=int, default=100,
                        help='每块窗口数（stride=1）。block_size 自动计算：window_size + samples_per_block - 1')
    args = parser.parse_args()

    config = DataConfig(
        norm_mode=args.norm_mode,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
        seed=args.seed,
        samples_per_block=args.samples_per_block,
    )

    raw_path = args.raw_data or get_raw_path()
    backtest_raw_path = args.backtest_raw or get_backtest_raw_path()

    if args.split_mode == 'time' and (args.train_end is None or args.val_end is None):
        parser.error("time split requires --train-end and --val-end")

    print("=" * 60)
    print("Kronos Data Preprocess")
    print("=" * 60)
    print(f"norm_mode: {config.norm_mode}")
    print(f"lookback: {config.lookback}")
    print(f"predict: {config.predict}")
    print(f"split_mode: {config.split_mode}")
    if config.split_mode == 'block':
        print(f"samples_per_block: {config.samples_per_block}")
        print(f"block_size: {config.block_size} (= window_size {config.lookback + config.predict} + samples_per_block {config.samples_per_block} - 1)")
    else:
        print(f"train_end: {args.train_end}")
        print(f"val_end: {args.val_end}")
    print(f"raw_data: {raw_path}")
    print(f"backtest_raw: {backtest_raw_path}")
    print(f"validate: {args.validate}")
    print(f"skip_backtest: {args.skip_backtest}")
    print(f"backtest_stride: {args.backtest_stride}")
    print("=" * 60)

    passed = preprocess(
        config,
        raw_path=raw_path,
        backtest_raw_path=backtest_raw_path,
        train_end=args.train_end,
        val_end=args.val_end,
        validate=args.validate,
        skip_backtest=args.skip_backtest,
        backtest_stride=args.backtest_stride
    )

    if args.validate and not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()