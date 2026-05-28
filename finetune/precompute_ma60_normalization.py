"""
预归一化数据脚本 - 滑动MA60

将原始数据预先计算滑动MA60归一化，保存归一化参数。
训练时直接加载，避免动态计算。

输出：
- processed_datasets_ma60/
  - train_data.pkl: {symbol: {'data': norm_data, 'means': means, 'stds': stds}}
  - val_data.pkl
  - test_data.pkl
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

feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
ma_window = 60
clip = 5.0

# 输入数据集路径
INPUT_DATASETS = {
    'processed_datasets': 'finetune/data/processed_datasets',
    'processed_datasets_mid': 'finetune/data/processed_datasets_mid',
    'processed_datasets_small': 'finetune/data/processed_datasets_small',
}

def sliding_ma60_normalize_full(values, window=60):
    """
    对整个序列计算滑动MA60归一化

    返回：
    - normalized: 归一化后的数据
    - means: 每个点的均值（用于反归一化）
    - stds: 每个点的标准差（用于反归一化）
    """
    n = len(values)
    normalized = np.zeros_like(values, dtype=np.float32)
    means = np.zeros((n, values.shape[1]), dtype=np.float32)
    stds = np.zeros((n, values.shape[1]), dtype=np.float32)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = values[start:i+1]
        means[i] = np.mean(window_data, axis=0)
        stds[i] = np.std(window_data, axis=0) + 1e-5
        normalized[i] = (values[i] - means[i]) / stds[i]

    normalized = np.clip(normalized, -clip, clip)
    return normalized, means, stds

def process_dataset(input_path, output_path, dataset_type):
    """处理单个数据集"""
    print(f"\n处理: {input_path} ({dataset_type})")

    # 加载原始数据
    data_file = f"{input_path}/{dataset_type}_data.pkl"
    if not os.path.exists(data_file):
        print(f"  文件不存在: {data_file}")
        return

    with open(data_file, 'rb') as f:
        raw_data = pickle.load(f)

    print(f"  股票数: {len(raw_data)}")

    # 处理每只股票
    processed_data = {}
    skipped = 0

    for i in trange(len(raw_data), desc=f"Processing {dataset_type}"):
        symbol = list(raw_data.keys())[i]
        df = raw_data[symbol]

        values = df[feature_cols].values.astype(np.float32)

        if len(values) < 100:  # 最小长度
            skipped += 1
            continue

        # 计算滑动MA60归一化
        normalized, means, stds = sliding_ma60_normalize_full(values)

        # 保存归一化数据 + 原始数据（用于计算收益率等）
        processed_data[symbol] = {
            'normalized': normalized,      # 归一化后的特征
            'means': means,                # 反归一化用的均值
            'stds': stds,                  # 反归一化用的标准差
            'original': values,            # 原始数据
            'index': df.index,             # 时间索引
        }

    print(f"  处理完成: {len(processed_data)} 股票, 跳过 {skipped}")

    # 保存
    os.makedirs(output_path, exist_ok=True)
    output_file = f"{output_path}/{dataset_type}_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"  保存到: {output_file}")

    # 显示文件大小
    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"  文件大小: {size_mb:.1f} MB")

def main():
    print("="*60)
    print("预归一化数据脚本 - 滑动MA60")
    print("="*60)
    print(f"MA窗口: {ma_window}")
    print(f"Clip: {clip}")

    for input_name, input_path in INPUT_DATASETS.items():
        output_path = f"finetune/data/{input_name}_ma60"

        print(f"\n{'='*60}")
        print(f"输入: {input_path}")
        print(f"输出: {output_path}")
        print("="*60)

        # 处理三个数据集
        for dataset_type in ['train', 'val', 'test']:
            process_dataset(input_path, output_path, dataset_type)

    print("\n" + "="*60)
    print("预归一化完成!")
    print("="*60)

    # 列出输出目录
    print("\n输出目录:")
    for input_name in INPUT_DATASETS.keys():
        output_path = f"finetune/data/{input_name}_ma60"
        if os.path.exists(output_path):
            files = os.listdir(output_path)
            print(f"  {output_path}: {files}")

if __name__ == '__main__':
    main()