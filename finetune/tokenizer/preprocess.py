"""
Kronos Tokenizer Preprocess Module

tokenizer 专用数据预处理入口。

与 predictor 的概念彻底解耦：
- tokenizer 只依赖「归一化分布」（norm_mode），与 train/val/test 分割、
  lookback/predict、block/time 等 predictor 概念无关。
- tokenizer 做无监督重建（重建输入自身），没有「泄露」概念，故用全量数据、
  整条归一化，不分割 train/val/test。

输出：
    finetune/data/processed/{norm_mode}/tokenizer/all.pkl
    {symbol: {
        'normalized': (T, 6),   # 整条 sliding_ma 归一化
        'original':   (T, 6),
        'means':      (T, 6),
        'stds':       (T, 6),
        'index':      (T,),
    }}
    无 windows 字段——tokenizer 训练时自行用滑动窗口采样。

使用：
    python finetune/tokenizer/preprocess.py \
        --norm-mode sliding_ma60
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from finetune.predictor.core.paths import get_raw_path, ensure_dir, PROJECT_ROOT
from finetune.predictor.core.normalization import NormalizerFactory
from finetune.predictor.core.utils import safe_save_json


def get_tokenizer_data_path(norm_mode: str) -> str:
    """
    tokenizer 专用数据路径（只按 norm_mode 键控）

    Returns:
        finetune/data/processed/{norm_mode}/tokenizer/all.pkl
    """
    return os.path.join(
        PROJECT_ROOT,
        f"finetune/data/processed/{norm_mode}/tokenizer/all.pkl",
    )


def split_val_symbols(all_data: Dict[str, Any], seed: int = 42, val_holdout_ratio: float = 0.1):
    """
    随机抽 val_holdout_ratio 比例股票作 val（仅 early-stop 信号，非防泄露）。

    tokenizer 做无监督重建，无泄露概念，故 val 仅用于 early-stop / 验证信号。
    用固定 seed 划分，保证 train 与 validate 用同一 val 集（否则验证集不同步）。

    Args:
        all_data: {symbol: {'normalized', ...}} 全量数据
        seed: 划分种子（须与 train 一致）
        val_holdout_ratio: val 股票比例

    Returns:
        (train_data, val_data, val_symbols): train/val 字典 + val symbol 集合
    """
    rng_split = np.random.RandomState(seed)
    symbols = sorted(all_data.keys())
    rng_split.shuffle(symbols)
    n_val_symbols = max(1, int(len(symbols) * val_holdout_ratio))
    val_symbols = set(symbols[:n_val_symbols])
    train_data = {s: d for s, d in all_data.items() if s not in val_symbols}
    val_data = {s: d for s, d in all_data.items() if s in val_symbols}
    return train_data, val_data, val_symbols


def create_tokenizer_data(
    raw_data: Dict[str, Any],
    norm_mode: str,
    clip: float = 5.0,
) -> Dict[str, Dict[str, Any]]:
    """
    从 raw 数据生成 tokenizer 专用数据：每只股票整条归一化，无分割、无 windows。

    Args:
        raw_data: {symbol: DataFrame or {'values': (T,6), 'index': ...}}
        norm_mode: 归一化模式（full_window / sliding_ma{N}）
        clip: 归一化截断范围

    Returns:
        {symbol: {'normalized', 'original', 'means', 'stds', 'index'}}
    """
    normalizer = NormalizerFactory.create(norm_mode, clip=clip)
    data_by_symbol = {}

    for symbol, data in tqdm(raw_data.items(), desc="Normalizing"):
        if hasattr(data, 'columns'):
            # DataFrame
            values = data.values.astype(np.float32)
            index = data.index
        else:
            # dict: {'values': ..., 'index': ...}
            values = np.asarray(data.get('values', data.get('original')), dtype=np.float32)
            index = data.get('index')

        if values is None or len(values) == 0:
            continue

        normalized, means, stds = normalizer.normalize(values)
        data_by_symbol[symbol] = {
            'normalized': normalized.astype(np.float32),
            'original': values,
            'means': means.astype(np.float32),
            'stds': stds.astype(np.float32),
            'index': index,
        }

    return data_by_symbol


def save_tokenizer_data(
    data_by_symbol: Dict[str, Dict[str, Any]],
    output_path: str,
    norm_mode: str,
    raw_path: str,
):
    """保存 tokenizer 数据 + 元数据"""
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        pickle.dump(data_by_symbol, f)

    n_samples = sum(len(d['normalized']) for d in data_by_symbol.values())
    print(f"Saved {n_samples} rows ({len(data_by_symbol)} symbols, full-series normalized) to {output_path}")

    # 元数据
    import hashlib
    with open(raw_path, 'rb') as f:
        fingerprint = hashlib.sha256(f.read()).hexdigest()[:16]
    meta_path = os.path.join(os.path.dirname(output_path), 'meta.json')
    safe_save_json({
        'norm_mode': norm_mode,
        'n_symbols': len(data_by_symbol),
        'n_rows': n_samples,
        'raw_fingerprint': fingerprint,
        'created_at': datetime.now().isoformat(),
    }, meta_path)
    print(f"Saved meta to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='Kronos Tokenizer Data Preprocess')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'],
                        help='归一化模式（决定数据分布，tokenizer 唯一依赖）')
    parser.add_argument('--raw-data', type=str, default=None,
                        help='自定义 raw 路径（默认 finetune/data/raw/kline_daily_raw.pkl）')
    args = parser.parse_args()

    raw_path = args.raw_data or get_raw_path()
    output_path = get_tokenizer_data_path(args.norm_mode)

    print("=" * 60)
    print("Kronos Tokenizer Data Preprocess")
    print("=" * 60)
    print(f"norm_mode: {args.norm_mode}")
    print(f"raw_data: {raw_path}")
    print(f"output: {output_path}")
    print("=" * 60)

    # 加载 raw
    print("\n[1] Loading raw data...")
    with open(raw_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"Loaded {len(raw_data)} symbols")

    # 整条归一化（无分割、无 windows）
    print("\n[2] Normalizing (full-series, no split)...")
    data_by_symbol = create_tokenizer_data(raw_data, args.norm_mode)

    # 保存
    print("\n[3] Saving...")
    save_tokenizer_data(data_by_symbol, output_path, args.norm_mode, raw_path)

    print("\n" + "=" * 60)
    print("Done. Next: train tokenizer")
    print(f"  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) finetune/tokenizer/train.py \\")
    print(f"      --norm-mode {args.norm_mode} --model mini")
    print("=" * 60)


if __name__ == '__main__':
    main()
