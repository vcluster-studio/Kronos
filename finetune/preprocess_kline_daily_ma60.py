"""
预处理脚本：将kline_daily.sql数据进行MA60归一化并保存

输出格式与 processed_datasets_ma60 一致，方便后续推理使用
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

# 参数
WINDOW = 60  # MA窗口
CLIP = 5.0
LOOKBACK = 400
PRED_LEN = 10
feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']


def parse_sql_file(sql_path):
    """解析SQL文件"""
    print(f"解析SQL文件: {sql_path}")
    pattern = re.compile(
        r"VALUES \('([^']+)', '([^']+)', "
        r"'?([^',]+)'?, '?([^',]+)'?, '?([^',]+)'?, '?([^',]+)'?, "
        r"'?([^',]+)'?, '?([^',]+)'?,",
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

            if line_count % 2000000 == 0:
                print(f"  已解析 {line_count} 行, {len(data)} 股票...")

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
    print("=" * 60)
    print("MA60预处理 - kline_daily.sql")
    print("=" * 60)

    sql_path = 'data/kline_daily.sql'
    output_path = 'finetune/data/kline_daily_ma60.pkl'

    # 解析
    print("\n[1] 解析SQL...")
    data = parse_sql_file(sql_path)

    # 转换并归一化
    print("\n[2] 转换DataFrame并MA60归一化...")
    processed_data = {}
    valid_count = 0
    skipped_count = 0

    for symbol, rows in tqdm(data.items(), desc="处理股票"):
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 检查长度
        if len(df) < LOOKBACK + PRED_LEN:
            skipped_count += 1
            continue

        # 原始数据
        original = df[feature_cols].values.astype(np.float32)
        timestamps = df['date'].values

        # MA60归一化
        normalized, means, stds = vectorized_ma60_normalize(original, WINDOW, CLIP)

        processed_data[symbol] = {
            'normalized': normalized,
            'means': means,
            'stds': stds,
            'original': original,
            'index': pd.DatetimeIndex(timestamps)
        }
        valid_count += 1

    print(f"\n有效股票: {valid_count}")
    print(f"跳过股票: {skipped_count}")

    # 保存
    print(f"\n[3] 保存至 {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    # 验证
    print("\n[4] 验证数据...")
    sample_symbol = list(processed_data.keys())[0]
    sample = processed_data[sample_symbol]
    print(f"  样本股票: {sample_symbol}")
    print(f"  normalized shape: {sample['normalized'].shape}")
    print(f"  means shape: {sample['means'].shape}")
    print(f"  stds shape: {sample['stds'].shape}")
    print(f"  index length: {len(sample['index'])}")
    print(f"  index range: {sample['index'].min()} ~ {sample['index'].max()}")

    print("\n" + "=" * 60)
    print("预处理完成")
    print("=" * 60)
    print(f"输出文件: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()