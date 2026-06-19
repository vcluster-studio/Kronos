"""
原始数据解析：将 kline_daily.sql 转换为 kline_daily_raw.pkl

输出：每只股票的完整原始序列（未归一化），包含：
- values: (T, 6) 原始 OHLCV 数据
- index: DatetimeIndex 时间索引

作为整个项目的输入数据源。
"""

import os
import sys
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']


def parse_sql_line(line):
    """解析 SQL INSERT 行"""
    # INSERT INTO "kline_daily" ("stock_code", "trade_date", ...) VALUES ('600000.SH', '2018-01-02', '12.6100', ...);
    match = re.search(r"VALUES \('([^']+)', '([^']+)', '([0-9.]+)', '([0-9.]+)', '([0-9.]+)', '([0-9.]+)', ([0-9]+), '([0-9.]+)'", line)
    if match:
        return {
            'symbol': match.group(1),
            'date': match.group(2),
            'open': float(match.group(3)),
            'high': float(match.group(4)),
            'low': float(match.group(5)),
            'close': float(match.group(6)),
            'vol': float(match.group(7)),
            'amt': float(match.group(8)),
        }
    return None


def main():
    parser = argparse.ArgumentParser(
        description='解析 kline_daily sql 为 kline_daily_raw.pkl（训练/验证/测试输入源）'
    )
    parser.add_argument('--sql', type=str, default=os.path.join(script_dir, 'kline_daily.sql'),
                        help='输入 sql 文件路径')
    parser.add_argument('--output', type=str,
                        default=os.path.join(project_root, 'finetune', 'data', 'raw', 'kline_daily_raw.pkl'),
                        help='输出 pkl 路径')
    parser.add_argument('--min-length', type=int, default=250,
                        help='股票最短序列长度，不足则丢弃')
    args = parser.parse_args()

    sql_path = args.sql
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("原始数据解析 - kline_daily.sql → kline_daily_raw.pkl")
    print("=" * 60)

    if not os.path.exists(sql_path):
        print(f"Error: SQL file not found: {sql_path}")
        sys.exit(1)

    print(f"\n[1] 解析 SQL...")
    print(f"SQL文件: {sql_path}")

    # 按股票分组收集数据
    stock_data = {}

    with open(sql_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if not line.startswith('INSERT INTO "kline_daily"'):
                continue

            row = parse_sql_line(line)
            if row is None:
                continue

            symbol = row['symbol']
            if symbol not in stock_data:
                stock_data[symbol] = []

            stock_data[symbol].append(row)
            count += 1

            if count % 2000000 == 0:
                print(f"  已解析 {count} 行, {len(stock_data)} 股票...")

        print(f"  总行数: {count}, {len(stock_data)} 股票")

    print(f"\n[2] 转换为 DataFrame...")
    raw_data = {}
    skipped = 0

    for symbol in tqdm(stock_data.keys(), desc="转换股票"):
        rows = stock_data[symbol]
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.index.name = 'datetime'
        df = df[feature_cols]

        # 过滤太短的股票
        if len(df) < args.min_length:
            skipped += 1
            continue

        raw_data[symbol] = {
            'values': df.values.astype(np.float32),
            'index': df.index,
        }

    print(f"有效股票: {len(raw_data)}")
    print(f"跳过（太短）: {skipped}")

    print(f"\n[3] 保存 {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(raw_data, f)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"文件大小: {size_mb:.1f} MB")

    # 验证
    print(f"\n[4] 验证数据...")
    sample_sym = list(raw_data.keys())[0]
    sample = raw_data[sample_sym]
    print(f"  样本股票: {sample_sym}")
    print(f"  values shape: {sample['values'].shape}")
    print(f"  index range: {sample['index'][0]} ~ {sample['index'][-1]}")

    print(f"\n{'=' * 60}")
    print("完成")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()