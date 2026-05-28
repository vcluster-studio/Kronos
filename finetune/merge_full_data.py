"""
合并 train/val/test 为全量数据集
"""

import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

input_dir = os.path.join(project_root, 'finetune', 'data', 'processed_datasets_new')
output_path = os.path.join(input_dir, 'full_data.pkl')

print("合并数据集...")

# 加载三个数据集
with open(os.path.join(input_dir, 'train_data.pkl'), 'rb') as f:
    train_data = pickle.load(f)
with open(os.path.join(input_dir, 'val_data.pkl'), 'rb') as f:
    val_data = pickle.load(f)
with open(os.path.join(input_dir, 'test_data.pkl'), 'rb') as f:
    test_data = pickle.load(f)

print(f"train: {len(train_data)} stocks")
print(f"val: {len(val_data)} stocks")
print(f"test: {len(test_data)} stocks")

# 合并每只股票的数据
full_data = {}
for symbol in train_data.keys():
    # 拼接 train + val + test
    train_df = train_data[symbol]
    val_df = val_data[symbol]
    test_df = test_data[symbol]

    import pandas as pd
    full_df = pd.concat([train_df, val_df, test_df])
    full_data[symbol] = full_df

print(f"\n全量数据集: {len(full_data)} stocks")

# 显示样例
sample_symbol = list(full_data.keys())[0]
sample_df = full_data[sample_symbol]
print(f"\n样例 {sample_symbol}:")
print(f"  shape: {sample_df.shape}")
print(f"  date range: {sample_df.index.min().strftime('%Y-%m-%d')} ~ {sample_df.index.max().strftime('%Y-%m-%d')}")

# 保存
with open(output_path, 'wb') as f:
    pickle.dump(full_data, f)

print(f"\n保存: {output_path}")

# 文件大小
size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"大小: {size_mb:.1f} MB")