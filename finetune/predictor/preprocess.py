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

    # 获取 norm_mode 需要的历史长度
    required_history = NormalizerFactory.get_required_history(config.norm_mode)

    for symbol, data in tqdm(raw_data.items(), desc="Creating samples"):
        # 检测数据格式
        if hasattr(data, 'columns'):
            # DataFrame 格式
            seq_len = len(data)
            total_required = window_size + required_history

            if seq_len < total_required:
                continue

            # 创建样本
            for i in range(seq_len - total_required + 1):
                start = i + required_history
                sample = SampleSchema(
                    symbol=symbol,
                    window_start=i,
                    lookback_start=start,
                    lookback_end=start + lookback,
                    target_start=start + lookback,
                    target_end=start + lookback + predict,
                    split='unknown',  # 后续分割时填充
                    values=data.values[i:i + window_size + required_history],
                    index=data.index[i:i + window_size + required_history],
                )
                samples.append(sample)
        else:
            # dict 格式（原始 OHLCV）
            values = data.get('values', data.get('original'))
            index = data.get('index')

            if values is None:
                continue

            seq_len = len(values)
            total_required = window_size + required_history

            if seq_len < total_required:
                continue

            for i in range(seq_len - total_required + 1):
                start = i + required_history
                sample = SampleSchema(
                    symbol=symbol,
                    window_start=i,
                    lookback_start=start,
                    lookback_end=start + lookback,
                    target_start=start + lookback,
                    target_end=start + lookback + predict,
                    split='unknown',
                    values=values[i:i + window_size + required_history],
                    index=index[i:i + window_size + required_history] if index is not None else None,
                )
                samples.append(sample)

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
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            block_size=50,
            seed=config.seed
        )

    else:
        raise ValueError(f"Unknown split_mode: {config.split_mode}")


def create_backtest_samples(
    train_raw: Dict[str, Any],
    backtest_raw: Dict[str, Any],
    config: DataConfig
) -> Dict[str, Any]:
    """
    生成 backtest 样本

    每只股票一个样本：
      - context（lookback）来自 train_raw（kline_daily_raw）末尾
      - target（predict）来自 backtest_raw 开头
      - 拼接 required_history + lookback + predict 的完整序列后归一化

    Args:
        train_raw: kline_daily_raw（提供 context）
        backtest_raw: backtest_raw（提供 target）
        config: DataConfig

    Returns:
        {symbol: {normalized, means, stds, original, index, lookback, predict}}
    """
    lookback = config.lookback
    predict = config.predict
    required_history = NormalizerFactory.get_required_history(config.norm_mode)
    normalizer = NormalizerFactory.create(config.norm_mode, clip=config.clip)
    context_len = lookback + required_history  # 从 train_raw 取的长度

    samples = {}
    skipped = 0

    for symbol, bt_data in tqdm(backtest_raw.items(), desc="Backtest samples"):
        train_data = train_raw.get(symbol)
        if train_data is None:
            skipped += 1
            continue

        train_values = train_data['values']
        train_index = train_data['index']
        bt_values = bt_data['values']
        bt_index = bt_data['index']

        # train_raw 不足以提供 context
        if len(train_values) < context_len:
            skipped += 1
            continue
        # backtest_raw 不足以提供 target
        if len(bt_values) < predict:
            skipped += 1
            continue

        # context = train_raw 末尾 context_len 根
        ctx_values = train_values[-context_len:]
        ctx_index = train_index[-context_len:]
        # target = backtest_raw 前 predict 根
        tgt_values = bt_values[:predict]
        tgt_index = bt_index[:predict]

        # 拼接完整序列：required_history + lookback + predict
        full_values = np.concatenate([ctx_values, tgt_values], axis=0)
        full_index = ctx_index.append(tgt_index)

        # 归一化整个序列（sliding_ma 依赖前缀历史）
        normalized, means, stds = normalizer.normalize(full_values)

        samples[symbol] = {
            'normalized': normalized.astype(np.float32),
            'means': means.astype(np.float32),
            'stds': stds.astype(np.float32),
            'original': full_values.astype(np.float32),
            'index': full_index,
            'lookback': lookback,
            'predict': predict,
        }

    print(f"Backtest samples: {len(samples)} (skipped {skipped})")
    return samples


def save_backtest_samples(samples: Dict[str, Any], output_path: str):
    """保存 backtest 样本"""
    ensure_dir(output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Saved {len(samples)} backtest samples to {output_path}")


def save_split_data(
    split_samples: List[SampleSchema],
    output_path: str,
    config: DataConfig
):
    """
    保存分割数据

    Args:
        split_samples: 分割后的样本列表
        output_path: 输出路径
        config: DataConfig
    """
    # 转换为存储格式（按 symbol 组织）
    data_by_symbol = {}

    for sample in split_samples:
        if sample.symbol not in data_by_symbol:
            data_by_symbol[sample.symbol] = {
                'samples': [],
                'windows': [],
            }

        data_by_symbol[sample.symbol]['samples'].append(sample_to_dict(sample))
        data_by_symbol[sample.symbol]['windows'].append(sample.window_start)

    # 存储格式：保留原始数据 + 窗口索引
    # 注意：这里只存储索引，原始数据仍在 raw 文件中
    # 实际加载时需要从 raw 读取数据并按索引切片

    ensure_dir(output_path)

    # 存储为窗口索引格式（轻量）
    window_indices = {}
    for symbol, d in data_by_symbol.items():
        window_indices[symbol] = {
            'windows': d['windows'],
            'split': split_samples[0].split if split_samples else 'unknown',
        }

    with open(output_path, 'wb') as f:
        pickle.dump(window_indices, f)

    print(f"Saved {len(split_samples)} samples to {output_path}")


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
    skip_backtest: bool = False
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
        save_split_data(split_samples, output_path, config)

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
            bt_samples = create_backtest_samples(raw_data, backtest_raw, config)
            bt_output_path = get_backtest_data_path(
                config.norm_mode, config.lookback, config.predict
            )
            save_backtest_samples(bt_samples, bt_output_path)
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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = DataConfig(
        norm_mode=args.norm_mode,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
        seed=args.seed,
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
    print(f"raw_data: {raw_path}")
    print(f"backtest_raw: {backtest_raw_path}")
    print(f"validate: {args.validate}")
    print(f"skip_backtest: {args.skip_backtest}")
    print("=" * 60)

    passed = preprocess(
        config,
        raw_path=raw_path,
        backtest_raw_path=backtest_raw_path,
        train_end=args.train_end,
        val_end=args.val_end,
        validate=args.validate,
        skip_backtest=args.skip_backtest
    )

    if args.validate and not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()