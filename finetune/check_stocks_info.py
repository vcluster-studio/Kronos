import pickle
import re

# 解析 stocks.sql 找 ST股和退市股
st_stocks = []
delisted_stocks = []
all_stock_type_stocks = []

with open('data/stocks.sql', 'r', encoding='utf-8') as f:
    for line in f:
        if 'INSERT' not in line:
            continue
        values_match = re.search(r'VALUES \((.+)\);', line)
        if not values_match:
            continue

        values_str = values_match.group(1)
        # 手动分割
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

        if len(parts) < 16:
            continue

        stock_code = parts[0].strip("'")
        stock_type = parts[3].strip("'")
        is_st = parts[9].strip("'") == 't'
        is_delisted = parts[11].strip("'") == 't'

        if stock_type == 'stock':
            all_stock_type_stocks.append(stock_code)
            if is_st:
                st_stocks.append(stock_code)
            if is_delisted:
                delisted_stocks.append(stock_code)

print(f'stocks.sql 中 stock类型数量: {len(all_stock_type_stocks)}')
print(f'ST股数量: {len(st_stocks)}')
print(f'退市股数量: {len(delisted_stocks)}')
print(f'ST股样例: {st_stocks[:15]}')
print(f'退市股样例: {delisted_stocks[:15]}')

# 检查这些是否在 kline 数据中
with open('finetune/data/processed_datasets_new/full_data.pkl', 'rb') as f:
    kline_data = pickle.load(f)

kline_symbols = set(kline_data.keys())
print(f'\nK线数据股票数: {len(kline_symbols)}')

st_in_kline = [s for s in st_stocks if s in kline_symbols]
st_not_in_kline = [s for s in st_stocks if s not in kline_symbols]
delisted_in_kline = [s for s in delisted_stocks if s in kline_symbols]

print(f'ST股在 K线中: {len(st_in_kline)}')
print(f'ST股不在 K线中: {len(st_not_in_kline)}')
print(f'退市股在 K线中: {len(delisted_in_kline)}')

if st_in_kline:
    print(f'\n需要过滤的 ST股样例: {st_in_kline[:10]}')
if delisted_in_kline:
    print(f'需要过滤的退市股样例: {delisted_in_kline[:10]}')