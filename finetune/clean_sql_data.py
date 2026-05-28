"""
SQL 数据清洗脚本

将 kline_daily.sql 转换为训练所需的 pickle 格式

输入: PostgreSQL INSERT 语句
输出: {stock_code: DataFrame} 格式的 pickle 文件
      DataFrame columns: ['open', 'high', 'low', 'close', 'vol', 'amt']
      DatetimeIndex with name 'datetime'
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)


def parse_sql_line(line):
    """
    解析单行 SQL INSERT 语句

    返回: (stock_code, trade_date, open, high, low, close, volume, amount) 或 None
    """
    # 使用正则提取 VALUES 部分
    pattern = r"VALUES \('([^']+)', '([^']+)', '([^']+)', '([^']+)', '([^']+)', '([^']+)', (\d+), '([^']+)'"
    match = re.search(pattern, line)

    if not match:
        return None

    stock_code = match.group(1)  # e.g., '600000.SH'
    trade_date = match.group(2)  # e.g., '2018-01-02'
    open_price = float(match.group(3))
    high_price = float(match.group(4))
    low_price = float(match.group(5))
    close_price = float(match.group(6))
    volume = int(match.group(7))
    amount = float(match.group(8))

    return stock_code, trade_date, open_price, high_price, low_price, close_price, volume, amount


def parse_sql_file_fast(sql_path, output_dir, chunk_size=100000):
    """
    高效解析大型 SQL 文件

    使用分块读取和字典累积方式处理
    """
    print(f"解析 SQL 文件: {sql_path}")
    print(f"输出目录: {output_dir}")

    # 统计信息
    total_lines = 0
    valid_lines = 0
    stock_count = 0

    # 数据字典: {stock_code: list of (date, open, high, low, close, vol, amt)}
    data_dict = {}

    # 分块读取
    with open(sql_path, 'r', encoding='utf-8') as f:
        chunk = []

        for line in tqdm(f, desc="读取SQL", unit="行"):
            total_lines += 1
            chunk.append(line)

            if len(chunk) >= chunk_size:
                # 处理当前块
                for l in chunk:
                    result = parse_sql_line(l)
                    if result:
                        valid_lines += 1
                        stock_code, trade_date, open_p, high_p, low_p, close_p, vol, amt = result

                        if stock_code not in data_dict:
                            data_dict[stock_code] = []
                        data_dict[stock_code].append((trade_date, open_p, high_p, low_p, close_p, vol, amt))

                chunk = []

        # 处理剩余行
        for l in chunk:
            result = parse_sql_line(l)
            if result:
                valid_lines += 1
                stock_code, trade_date, open_p, high_p, low_p, close_p, vol, amt = result

                if stock_code not in data_dict:
                    data_dict[stock_code] = []
                data_dict[stock_code].append((trade_date, open_p, high_p, low_p, close_p, vol, amt))

    print(f"\n读取完成: {total_lines} 行, 有效数据: {valid_lines} 行")
    print(f"股票数量: {len(data_dict)}")

    # 转换为 DataFrame
    print("\n转换为 DataFrame...")
    final_data = {}

    for stock_code, records in tqdm(data_dict.items(), desc="转换DataFrame"):
        # 按日期排序
        records.sort(key=lambda x: x[0])

        # 创建 DataFrame
        df = pd.DataFrame(records, columns=['datetime', 'open', 'high', 'low', 'close', 'vol', 'amt'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

        # 确保数据类型正确
        df['open'] = df['open'].astype(np.float32)
        df['high'] = df['high'].astype(np.float32)
        df['low'] = df['low'].astype(np.float32)
        df['close'] = df['close'].astype(np.float32)
        df['vol'] = df['vol'].astype(np.int64)
        df['amt'] = df['amt'].astype(np.float64)

        final_data[stock_code] = df

    print(f"转换完成: {len(final_data)} 只股票")

    # 显示统计信息
    show_stats(final_data)

    # 拆分 train/val/test
    split_and_save(final_data, output_dir)

    return final_data


def show_stats(data_dict):
    """显示数据统计信息"""
    print("\n数据统计:")
    print("-" * 60)

    # 总数据点
    total_points = sum(len(df) for df in data_dict.values())
    print(f"总数据点: {total_points}")

    # 每只股票数据量
    lengths = [len(df) for df in data_dict.values()]
    print(f"股票数据量范围: {min(lengths)} ~ {max(lengths)}")
    print(f"平均数据量: {np.mean(lengths):.0f}")

    # 日期范围
    all_dates = []
    for df in data_dict.values():
        all_dates.extend(df.index.tolist())

    date_min = min(all_dates)
    date_max = max(all_dates)
    print(f"日期范围: {date_min.strftime('%Y-%m-%d')} ~ {date_max.strftime('%Y-%m-%d')}")

    # 样例数据
    sample_symbol = list(data_dict.keys())[0]
    print(f"\n样例数据 ({sample_symbol}):")
    print(data_dict[sample_symbol].head(3))


def split_and_save(data_dict, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    按时间拆分数据集并保存

    拆分方式: 每只股票按时间顺序拆分
    - train: 前 70%
    - val: 中 15%
    - test: 后 15%

    同时保存全量数据集（不拆分）
    """
    os.makedirs(output_dir, exist_ok=True)

    # === 保存全量数据集 ===
    print("\n保存全量数据集...")
    full_path = os.path.join(output_dir, 'full_data.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"保存: {full_path} ({len(data_dict)} stocks)")

    train_data = {}
    val_data = {}
    test_data = {}

    print("\n拆分数据集...")
    for stock_code, df in tqdm(data_dict.items(), desc="拆分"):
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data[stock_code] = df.iloc[:train_end]
        val_data[stock_code] = df.iloc[train_end:val_end]
        test_data[stock_code] = df.iloc[val_end:]

    # 保存
    print("\n保存 pickle 文件...")
    train_path = os.path.join(output_dir, 'train_data.pkl')
    val_path = os.path.join(output_dir, 'val_data.pkl')
    test_path = os.path.join(output_dir, 'test_data.pkl')

    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"保存: {train_path} ({len(train_data)} stocks)")

    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    print(f"保存: {val_path} ({len(val_data)} stocks)")

    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"保存: {test_path} ({len(test_data)} stocks)")

    # 显示拆分统计
    print("\n拆分统计:")
    for stock_code in list(data_dict.keys())[:3]:
        df = data_dict[stock_code]
        print(f"  {stock_code}: total={len(df)}, train={len(train_data[stock_code])}, val={len(val_data[stock_code])}, test={len(test_data[stock_code])}")


def main():
    sql_path = os.path.join(project_root, 'data', 'kline_daily.sql')
    output_dir = os.path.join(project_root, 'finetune', 'data', 'processed_datasets_new')

    print("=" * 70)
    print("SQL 数据清洗脚本")
    print("=" * 70)

    # 检查输入文件
    if not os.path.exists(sql_path):
        print(f"错误: SQL 文件不存在 {sql_path}")
        return

    # 解析并转换
    data = parse_sql_file_fast(sql_path, output_dir)

    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()