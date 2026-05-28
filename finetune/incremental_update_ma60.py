"""
增量更新预处理数据：将新的kline_daily SQL数据追加到已有的MA60 pkl中

1. 解析新SQL文件
2. 追加到已有pkl的各股票数据末尾
3. 重新计算MA60归一化（新数据点需要新的rolling窗口）
4. 保存更新后的pkl
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

WINDOW = 60
CLIP = 5.0
feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']


def parse_new_sql_file(sql_path):
    """解析新格式SQL文件（带id、turnover_rate等额外字段）"""
    print(f"解析SQL文件: {sql_path}")

    # 新格式: VALUES (id, 'stock_code', 'trade_date', 'open', 'high', 'low', 'close', volume, 'amount', ...)
    pattern = re.compile(
        r"VALUES \(\d+, '([^']+)', '([^']+)', "
        r"'?([^',]+)'?, '?([^',]+)'?, '?([^',]+)'?, '?([^',]+)'?, "
        r"'?([^',]+)'?, '?([^',]+)'?",
        re.DOTALL
    )

    data = {}
    line_count = 0

    with open(sql_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            match = pattern.search(line)
            if match:
                stock_code = match.group(1)
                trade_date = match.group(2)
                open_price = float(match.group(3).strip().replace("'", ""))
                high_price = float(match.group(4).strip().replace("'", ""))
                low_price = float(match.group(5).strip().replace("'", ""))
                close_price = float(match.group(6).strip().replace("'", ""))
                volume = float(match.group(7).strip().replace("'", ""))
                amount = float(match.group(8).strip().replace("'", ""))

                if stock_code not in data:
                    data[stock_code] = []

                data[stock_code].append({
                    'date': trade_date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'vol': volume,
                    'amt': amount
                })

    print(f"  解析完成: {line_count} 行, {len(data)} 股票")
    return data


def vectorized_ma60_normalize(values, window=60, clip=5.0):
    """向量化MA60归一化"""
    df_rolling = pd.DataFrame(values, columns=feature_cols)

    means = df_rolling.rolling(window=window, min_periods=1).mean().values.astype(np.float32)
    stds = df_rolling.rolling(window=window, min_periods=1).std().values.astype(np.float32)
    stds = np.where(stds < 1e-5, 1e-5, stds)

    normalized = (values - means) / stds
    normalized = np.clip(normalized, -clip, clip)

    return normalized.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


def main():
    new_sql_path = 'data/kline_daily_20260519~20260524.sql'
    pkl_path = 'finetune/data/kline_daily_ma60.pkl'

    print("=" * 60)
    print("增量更新MA60预处理数据")
    print("=" * 60)

    # 1. 加载已有pkl
    print(f"\n[1] 加载已有数据: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        norm_dict = pickle.load(f)
    print(f"  已有股票数: {len(norm_dict)}")

    # 2. 解析新SQL
    print(f"\n[2] 解析新数据: {new_sql_path}")
    new_data = parse_new_sql_file(new_sql_path)
    print(f"  新数据股票数: {len(new_data)}")

    # 3. 追加数据并重新归一化
    print("\n[3] 追加数据并重新MA60归一化...")
    updated_count = 0
    new_count = 0

    for symbol, new_rows in tqdm(new_data.items(), desc="处理股票"):
        new_df = pd.DataFrame(new_rows)
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df = new_df.sort_values('date').reset_index(drop=True)
        new_values = new_df[feature_cols].values.astype(np.float32)
        new_timestamps = new_df['date'].values

        if symbol in norm_dict:
            # 追加到已有数据
            existing = norm_dict[symbol]
            existing_original = existing['original']
            existing_index = existing['index']

            # 检查新数据是否在已有数据之后
            last_existing_date = existing_index[-1]
            new_start = pd.to_datetime(new_timestamps[0])

            if new_start <= last_existing_date:
                # 有重叠，只追加新部分
                mask = pd.DatetimeIndex(new_timestamps) > last_existing_date
                if mask.sum() == 0:
                    continue
                new_values = new_values[mask]
                new_timestamps = new_timestamps[mask]

            # 拼接
            combined_original = np.concatenate([existing_original, new_values], axis=0)
            combined_index = existing_index.append(pd.DatetimeIndex(new_timestamps))

            # 重新归一化（整个序列）
            normalized, means, stds = vectorized_ma60_normalize(combined_original, WINDOW, CLIP)

            norm_dict[symbol] = {
                'normalized': normalized,
                'means': means,
                'stds': stds,
                'original': combined_original,
                'index': combined_index
            }
            updated_count += 1
        else:
            # 新股票
            if len(new_values) < 70:  # 不够MA60窗口
                continue
            normalized, means, stds = vectorized_ma60_normalize(new_values, WINDOW, CLIP)
            norm_dict[symbol] = {
                'normalized': normalized,
                'means': means,
                'stds': stds,
                'original': new_values,
                'index': pd.DatetimeIndex(new_timestamps)
            }
            new_count += 1

    print(f"\n  更新股票: {updated_count}")
    print(f"  新增股票: {new_count}")

    # 4. 验证
    print("\n[4] 验证数据...")
    sample_symbol = list(norm_dict.keys())[0]
    sample = norm_dict[sample_symbol]
    print(f"  样本股票: {sample_symbol}")
    print(f"  index range: {sample['index'].min()} ~ {sample['index'].max()}")

    # 5. 保存
    print(f"\n[5] 保存至 {pkl_path}...")
    with open(pkl_path, 'wb') as f:
        pickle.dump(norm_dict, f)

    new_size = os.path.getsize(pkl_path) / 1024 / 1024
    print(f"  文件大小: {new_size:.1f} MB")

    print("\n" + "=" * 60)
    print("增量更新完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
