import pickle
import pandas as pd

# Check existing dataset format
with open('finetune/data/processed_datasets/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

symbols = list(data.keys())[:3]
print(f'Total stocks: {len(data)}')
for sym in symbols:
    df = data[sym]
    print(f'\n{sym}:')
    print(f'  columns: {list(df.columns)}')
    print(f'  index_type: {type(df.index).__name__}')
    print(f'  index_name: {df.index.name}')
    print(f'  shape: {df.shape}')
    print(df.head(3))