import pickle
import pandas as pd

# 验证新数据集格式
with open('finetune/data/processed_datasets_new/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

symbols = list(data.keys())[:3]
print(f'Total stocks: {len(data)}')
for sym in symbols:
    df = data[sym]
    print(f'\n{sym}:')
    print(f'  columns: {list(df.columns)}')
    print(f'  shape: {df.shape}')
    print(f'  date range: {df.index.min().strftime("%Y-%m-%d")} ~ {df.index.max().strftime("%Y-%m-%d")}')
    print(df.head(3))