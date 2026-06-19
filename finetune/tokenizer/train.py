"""
Kronos Tokenizer Training Entry

每个 norm_mode 需单独训练 tokenizer（归一化后分布不同）。

使用：
    python finetune/tokenizer/train.py \
        --norm-mode sliding_ma60 \
        --model mini \
        --sample-ratio 0.1

默认使用 finetune/data/raw/kline_daily_raw.pkl
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer
from finetune.predictor.core.paths import (
    get_raw_path,
    get_tokenizer_path,
    ensure_dir,
)
from finetune.predictor.core.normalization import get_normalizer, NormalizerFactory
from finetune.predictor.core.schema import compute_fingerprint
from finetune.predictor.core.utils import safe_save_json


# vocab_size 映射
VOCAB_SIZE_MAP = {
    'mini': 2048,
    'small': 4096,
    'base': 8192,
}


def train_tokenizer(
    norm_mode: str,
    model_type: str,
    raw_data_path: str = None,
    lookback: int = 400,
    predict: int = 10,
    sample_ratio: float = 0.1,
    seed: int = 42
):
    """
    训练 tokenizer

    步骤：
    1. 加载原始数据
    2. 按 norm_mode 归一化
    3. 抽样训练数据
    4. 训练 tokenizer
    5. 保存到 outputs/tokenizers/{norm_mode}/{model_type}/

    Args:
        norm_mode: 归一化模式（full_window/sliding_ma{N}）
        model_type: 模型类型（mini/small/base）→ vocab_size 不同
        raw_data_path: 原始数据路径（默认 kline_daily_raw.pkl）
        lookback: 回看窗口长度
        predict: 预测步数
        sample_ratio: 抽样比例（默认 10%）
        seed: 随机种子
    """
    if raw_data_path is None:
        raw_data_path = get_raw_path()

    vocab_size = VOCAB_SIZE_MAP[model_type]

    print("=" * 60)
    print("Kronos Tokenizer Training")
    print("=" * 60)
    print(f"norm_mode: {norm_mode}")
    print(f"model_type: {model_type}")
    print(f"vocab_size: {vocab_size}")
    print(f"raw_data: {raw_data_path}")
    print(f"sample_ratio: {sample_ratio}")
    print("=" * 60)

    # 1. 加载原始数据
    print("\n[1] Loading raw data...")
    with open(raw_data_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"Loaded {len(raw_data)} symbols")

    # 2. 获取归一化器
    normalizer = get_normalizer(norm_mode)

    # 3. 收集并归一化数据
    print("\n[2] Normalizing data...")
    all_normalized = []

    for symbol, data in raw_data.items():
        # 检测数据格式
        if hasattr(data, 'columns'):
            # DataFrame 格式
            values = data.values.astype(np.float32)
        else:
            # dict 格式
            values = np.asarray(data.get('values', data.get('original')), dtype=np.float32)

        if len(values) < lookback + predict:
            continue

        # 归一化
        normalized, _, _ = normalizer.normalize(values)
        all_normalized.append(normalized)

    print(f"Collected {len(all_normalized)} normalized sequences")

    # 4. 抽样
    print("\n[3] Sampling training data...")
    rng = np.random.RandomState(seed)
    n_samples = int(len(all_normalized) * sample_ratio)
    if n_samples < 100:
        n_samples = min(100, len(all_normalized))
    sampled_indices = rng.choice(len(all_normalized), size=n_samples, replace=False)
    sampled_data = [all_normalized[i] for i in sampled_indices]
    print(f"Sampled {n_samples} sequences for tokenizer training")

    # 5. 训练 tokenizer
    print("\n[4] Training tokenizer...")
    tokenizer = KronosTokenizer.train(
        data=sampled_data,
        vocab_size=vocab_size,
        max_length=lookback + predict,
    )
    print(f"Tokenizer trained with vocab_size={tokenizer.vocab_size}")

    # 6. 保存
    print("\n[5] Saving tokenizer...")
    output_path = get_tokenizer_path(norm_mode, model_type)
    ensure_dir(output_path)
    tokenizer.save_pretrained(output_path)

    # 7. 保存元数据
    meta = {
        'norm_mode': norm_mode,
        'model_type': model_type,
        'vocab_size': vocab_size,
        'lookback': lookback,
        'predict': predict,
        'sample_ratio': sample_ratio,
        'n_samples': n_samples,
        'data_fingerprint': compute_fingerprint(raw_data_path),
        'created_at': datetime.now().isoformat(),
    }
    safe_save_json(meta, os.path.join(output_path, 'meta.json'))

    print(f"\nTokenizer saved to: {output_path}")
    print("=" * 60)

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Kronos Tokenizer Training')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--model', type=str, default='mini',
                        choices=['mini', 'small', 'base'])
    parser.add_argument('--raw-data', type=str, default=None,
                        help='Raw data path (default: kline_daily_raw.pkl)')
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                        help='Sampling ratio for tokenizer training')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_tokenizer(
        norm_mode=args.norm_mode,
        model_type=args.model,
        raw_data_path=args.raw_data,
        lookback=args.lookback,
        predict=args.predict,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()