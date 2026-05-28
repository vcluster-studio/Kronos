"""
合并 K线数据与股票信息，过滤 ST/停牌/退市股
"""

import os
import re
import pickle
import pandas as pd
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)


def parse_stocks_sql(sql_path):
    """解析 stocks.sql 文件"""
    print(f"解析股票信息: {sql_path}")

    stocks_info = {}

    # 字段顺序: stock_code, name, market, stock_type, ..., is_st(10), is_suspended(11), is_delisted(12), ..., is_active(15), is_deleted(16)
    # 使用更精确的正则提取所有字段值

    with open(sql_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取stocks.sql"):
            if not line.startswith('INSERT'):
                continue

            # 提取 VALUES (...) 部分
            values_match = re.search(r"VALUES \((.+)\);", line)
            if not values_match:
                continue

            values_str = values_match.group(1)

            # 分解字段值（处理 NULL 和字符串）
            # 使用简单的分割方法
            parts = []
            current = ''
            in_quote = False

            for char in values_str:
                if char == "'" and not in_quote:
                    in_quote = True
                elif char == "'" and in_quote:
                    in_quote = False
                elif char == ',' and not in_quote:
                    parts.append(current.strip())
                    current = ''
                else:
                    current += char
            parts.append(current.strip())

            # parts[0] = stock_code, parts[1] = name, parts[2] = market, parts[3] = stock_type
            # parts[9] = is_st, parts[10] = is_suspended, parts[11] = is_delisted
            # parts[14] = is_active, parts[15] = is_deleted

            if len(parts) < 16:
                continue

            stock_code = parts[0].strip("'")
            name = parts[1].strip("'")
            market = parts[2].strip("'")
            stock_type = parts[3].strip("'")

            is_st = parts[9].strip("'") == 't'
            is_suspended = parts[10].strip("'") == 't'
            is_delisted = parts[11].strip("'") == 't'
            is_active = parts[14].strip("'") == 't'
            is_deleted = parts[15].strip("'") == 't'

            stocks_info[stock_code] = {
                'name': name,
                'market': market,
                'stock_type': stock_type,
                'is_st': is_st,
                'is_suspended': is_suspended,
                'is_delisted': is_delisted,
                'is_active': is_active,
                'is_deleted': is_deleted
            }

    return stocks_info


def main():
    # 加载股票信息
    stocks_sql_path = os.path.join(project_root, 'data', 'stocks.sql')
    stocks_info = parse_stocks_sql(stocks_sql_path)

    print(f"\n股票信息统计:")
    print(f"  总记录数: {len(stocks_info)}")

    # 按类型统计
    type_counts = {}
    for info in stocks_info.values():
        t = info['stock_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  类型分布: {type_counts}")

    # ST/停牌/退市统计
    st_count = sum(1 for i in stocks_info.values() if i['is_st'])
    suspended_count = sum(1 for i in stocks_info.values() if i['is_suspended'])
    delisted_count = sum(1 for i in stocks_info.values() if i['is_delisted'])
    inactive_count = sum(1 for i in stocks_info.values() if not i['is_active'])
    print(f"  ST股: {st_count}")
    print(f"  停牌股: {suspended_count}")
    print(f"  退市股: {delisted_count}")
    print(f"  非活跃: {inactive_count}")

    # 加载 K线数据
    kline_path = os.path.join(project_root, 'finetune/data/processed_datasets_new/full_data.pkl')
    print(f"\n加载 K线数据: {kline_path}")
    with open(kline_path, 'rb') as f:
        kline_data = pickle.load(f)
    print(f"  K线股票数: {len(kline_data)}")

    # 过滤
    valid_stocks = {}
    removed = {'st': [], 'suspended': [], 'delisted': [], 'inactive': [], 'not_stock': [], 'no_info': []}

    for symbol, df in kline_data.items():
        info = stocks_info.get(symbol)

        if info is None:
            removed['no_info'].append(symbol)
            continue

        if info['stock_type'] != 'stock':
            removed['not_stock'].append(symbol)
            continue

        if info['is_st']:
            removed['st'].append(symbol)
            continue

        if info['is_suspended']:
            removed['suspended'].append(symbol)
            continue

        if info['is_delisted']:
            removed['delisted'].append(symbol)
            continue

        if not info['is_active']:
            removed['inactive'].append(symbol)
            continue

        valid_stocks[symbol] = df

    print(f"\n过滤结果:")
    for reason, symbols in removed.items():
        print(f"  {reason}: {len(symbols)} 只")
    print(f"  有效股票: {len(valid_stocks)} 只")

    # 显示被移除的样例
    if removed['st']:
        print(f"\nST股样例: {removed['st'][:5]}")
    if removed['delisted']:
        print(f"退市股样例: {removed['delisted'][:5]}")
    if removed['not_stock']:
        print(f"非股票样例: {removed['not_stock'][:5]}")

    # 保存过滤后的数据
    output_dir = os.path.join(project_root, 'finetune/data/processed_datasets_clean')
    os.makedirs(output_dir, exist_ok=True)

    # 保存全量
    with open(os.path.join(output_dir, 'full_data.pkl'), 'wb') as f:
        pickle.dump(valid_stocks, f)
    print(f"\n保存: {output_dir}/full_data.pkl")

    # 拆分 train/val/test
    train_data = {}
    val_data = {}
    test_data = {}

    for symbol, df in valid_stocks.items():
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_data[symbol] = df.iloc[:train_end]
        val_data[symbol] = df.iloc[train_end:val_end]
        test_data[symbol] = df.iloc[val_end:]

    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    print(f"保存: {output_dir}/train_data.pkl")
    print(f"保存: {output_dir}/val_data.pkl")
    print(f"保存: {output_dir}/test_data.pkl")

    # 文件大小
    for fname in ['full_data.pkl', 'train_data.pkl', 'val_data.pkl', 'test_data.pkl']:
        size = os.path.getsize(os.path.join(output_dir, fname)) / 1024 / 1024
        print(f"  {fname}: {size:.1f} MB")

    # 最终统计
    sample_symbol = list(valid_stocks.keys())[0]
    sample_df = valid_stocks[sample_symbol]
    print(f"\n最终数据统计:")
    print(f"  股票数: {len(valid_stocks)}")
    print(f"  样例 {sample_symbol}: {sample_df.index.min().strftime('%Y-%m-%d')} ~ {sample_df.index.max().strftime('%Y-%m-%d')}")


if __name__ == '__main__':
    main()