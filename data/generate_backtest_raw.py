"""
回测原始数据生成：从样本外 kline_daily sql 解析为 backtest_raw.pkl

输入：
  data/kline_daily_20260508~20260617.sql  样本外 sql（2026-05-08 ~ 2026-06-17）

输出：
  finetune/data/raw/backtest_raw.pkl      回测 raw（2026-05-19 ~ 2026-06-17）

说明:
  与 kline_daily_raw.pkl（截止 2026-05-18）时间区间不重叠。
  backtest 窗口 = kline_daily_raw 末尾 lookback 根（context）+ 本文件（target）。

  本文件只取 2026-05-19 及之后的数据，2026-05-08 ~ 2026-05-18 部分丢弃
  （该区间已在 kline_daily_raw.pkl 中）。

结构：{symbol: {'values': (T,6) [open,high,low,close,vol,amt], 'index': DatetimeIndex}}
"""

import os
import sys
import re
import pickle
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']

# 样本外 sql 列名：stock_code, trade_date, open_price, high_price, low_price, close_price, volume, amount
SQL_PATTERN = re.compile(
    r"VALUES \(\d+, '([^']+)', '([^']+)', '([0-9.]+)', '([0-9.]+)', '([0-9.]+)', '([0-9.]+)', ([0-9]+), '([0-9.]+)'"
)

# backtest 起始日期（kline_daily_raw.pkl 截止日的次日）
BACKTEST_START_DEFAULT = '2026-05-19'


def parse_sql_line(line):
    """解析样本外 sql INSERT 行"""
    match = SQL_PATTERN.search(line)
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
        description='解析样本外 kline_daily sql 为 backtest_raw.pkl（回测 target 输入源）'
    )
    parser.add_argument('--sql', type=str,
                        default=os.path.join(script_dir, 'kline_daily_20260508~20260617.sql'),
                        help='输入 sql 文件路径')
    parser.add_argument('--output', type=str,
                        default=os.path.join(project_root, 'finetune', 'data', 'raw', 'backtest_raw.pkl'),
                        help='输出 pkl 路径')
    parser.add_argument('--start', type=str, default=BACKTEST_START_DEFAULT,
                        help='backtest 起始日期（含），早于此日期的数据丢弃')
    args = parser.parse_args()

    sql_path = args.sql
    output_path = args.output
    backtest_start = pd.Timestamp(args.start)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("回测原始数据生成 - kline_daily sql → backtest_raw.pkl")
    print("=" * 60)

    if not os.path.exists(sql_path):
        print(f"Error: sql 文件不存在: {sql_path}")
        sys.exit(1)

    # 1. 解析 sql
    print(f"\n[1] 解析 sql: {sql_path}")
    stock_data = {}
    count = 0
    with open(sql_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('INSERT INTO'):
                continue
            row = parse_sql_line(line)
            if row is None:
                continue
            stock_data.setdefault(row['symbol'], []).append(row)
            count += 1
    print(f"  总行数: {count}, 股票数: {len(stock_data)}")

    # 2. 转换为 DataFrame 并截取起始日期之后
    print(f"\n[2] 转换为序列（起始 {backtest_start.date()}）...")
    backtest_data = {}
    for symbol, rows in stock_data.items():
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.index.name = 'datetime'
        df = df[feature_cols]

        df = df[df.index >= backtest_start]
        if len(df) == 0:
            continue

        backtest_data[symbol] = {
            'values': df.values.astype(np.float32),
            'index': df.index,
        }

    print(f"  有效股票: {len(backtest_data)}")

    # 3. 保存
    print(f"\n[3] 保存 {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(backtest_data, f)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  文件大小: {size_mb:.1f} MB")

    # 4. 验证
    print(f"\n[4] 验证数据...")
    sample_sym = list(backtest_data.keys())[0]
    sample = backtest_data[sample_sym]
    print(f"  样本股票: {sample_sym}")
    print(f"  values shape: {sample['values'].shape}")
    print(f"  date range: {sample['index'][0]} ~ {sample['index'][-1]}")

    print(f"\n{'=' * 60}")
    print("完成")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
