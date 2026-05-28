"""测试推理脚本"""

import os
import sys
import pickle
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from batch_inference_sql import KronosPredictor, generate_prediction_sql_batch

# 加载模型
predictor = KronosPredictor(
    model_path='final_models/Kronos-mini',
    tokenizer_path='final_models/Kronos-Tokenizer-2k',
    max_context=2048
)

# 加载全量数据
data_path = 'finetune/data/processed_datasets_clean/full_data.pkl'
print(f"\n加载数据: {data_path}")
with open(data_path, 'rb') as f:
    full_data = pickle.load(f)

# 测试 3 只股票，每只倒推 5 次
test_symbols = ['600000.SH', '000001.SZ', '300750.SZ']
lookback = 400
pred_len = 10

all_predictions = {}

for symbol in test_symbols:
    if symbol not in full_data:
        continue
    df = full_data[symbol]
    print(f"\n{symbol}: 数据量 {len(df)}")

    preds_list = []
    # 倒推 5 次
    for end_idx in range(len(df), len(df) - 5, -1):
        hist_df = df.iloc[:end_idx]
        pred_df, last_close = predictor.predict_single(
            hist_df, lookback=lookback, pred_len=pred_len
        )
        if pred_df is not None:
            preds_list.append({
                'pred_date': hist_df.index[-1],
                'last_close': last_close,
                'predictions': pred_df
            })
            print(f"  预测基准: {hist_df.index[-1].strftime('%Y-%m-%d')}, 收盘价: {last_close:.2f}")
            print(f"  预测结果前3天:")
            print(pred_df.head(3).to_string())

    all_predictions[symbol] = preds_list

# 生成测试 SQL
output_path = 'outputs/prediction_results/test_predictions.sql'
generate_prediction_sql_batch(
    all_predictions,
    model_id='Kronos-mini',
    lookback=lookback,
    pred_len=pred_len,
    output_path=output_path
)

print("\n测试完成!")