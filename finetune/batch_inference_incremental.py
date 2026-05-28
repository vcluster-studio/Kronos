"""
增量推理脚本 - 对2026-05-19~05-24期间进行推理

使用更新后的MA60预处理数据和最优MA60-IC模型
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# 最优模型路径
TOKENIZER_PATH = 'final_models/Kronos-Tokenizer-2k-MA60'
MODEL_PATH = 'final_models/Kronos-mini-MA60'

# 参数
LOOKBACK = 400
PRED_LEN = 10
CLIP = 5.0
T = 1.0
TOP_P = 0.9
START_DATE = '2026-05-19'
END_DATE = '2026-05-24'
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']


def calc_time_stamps(timestamps):
    """计算时间戳特征"""
    return np.stack([
        timestamps.minute.values,
        timestamps.hour.values,
        timestamps.weekday.values,
        timestamps.day.values,
        timestamps.month.values
    ], axis=1).astype(np.float32)


def get_price_limit(stock_code):
    """涨跌幅限制"""
    code_num = stock_code.split('.')[0]
    if code_num.startswith('300') or code_num.startswith('301') or code_num.startswith('688'):
        return 0.20
    return 0.10


def clip_prediction(pred_values, last_close, price_limit):
    """涨跌幅剪裁"""
    upper = last_close * (1 + price_limit)
    lower = last_close * (1 - price_limit)

    pred_values[:, 1] = np.clip(pred_values[:, 1], None, upper)
    pred_values[:, 2] = np.clip(pred_values[:, 2], lower, None)

    for i in range(len(pred_values)):
        row_low, row_high = pred_values[i, 2], pred_values[i, 1]
        pred_values[i, 0] = np.clip(pred_values[i, 0], row_low, row_high)
        pred_values[i, 3] = np.clip(pred_values[i, 3], row_low, row_high)

    return pred_values


def generate_future_dates(last_date, pred_len):
    """生成未来交易日"""
    future = []
    current = last_date + timedelta(days=1)
    while len(future) < pred_len:
        if current.weekday() < 5:
            future.append(current)
        current += timedelta(days=1)
    return pd.DatetimeIndex(future)


def generate_sql(predictions, model_id, lookback, pred_len, created_at):
    """生成SQL"""
    sql_lines = []

    for symbol, pred_data in predictions.items():
        pred_date = pred_data['pred_date'].strftime('%Y-%m-%d')
        pred_values = pred_data['predictions']
        future_dates = pred_data['future_dates']
        last_close = pred_data['last_close']
        price_limit = pred_data['price_limit']

        pred_clipped = clip_prediction(pred_values, last_close, price_limit)

        pred_list = []
        for i, date in enumerate(future_dates):
            pred_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(float(pred_clipped[i, 0]), 2),
                'high': round(float(pred_clipped[i, 1]), 2),
                'low': round(float(pred_clipped[i, 2]), 2),
                'close': round(float(pred_clipped[i, 3]), 2),
                'volume': int(pred_clipped[i, 4]),
                'amount': int(pred_clipped[i, 5])
            })

        pred_json = json.dumps(pred_list, ensure_ascii=False)
        model_info = json.dumps({
            'T': T, 'name': 'Kronos (MA60-V1-IC)',
            'top_p': TOP_P, 'device': str(device), 'norm': 'sliding_ma60'
        }, ensure_ascii=False)
        extras = json.dumps({
            'limit_pct': price_limit, 'last_close': round(float(last_close), 2)
        }, ensure_ascii=False)

        sql = f"insert into prediction_results (stock_code, model_id, pred_date, created_at, pred_path, lookback, pred_len, model_info, extras) values ('{symbol}', '{model_id}', '{pred_date}', '{created_at}', '{pred_json}', {lookback}, {pred_len}, '{model_info}', '{extras}');"
        sql_lines.append(sql)

    return sql_lines


def main():
    print("=" * 60)
    print("增量推理 - 2026-05-19~05-24")
    print("=" * 60)
    print(f"模型: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"推理日期: {START_DATE} ~ {END_DATE}")
    print(f"设备: {device}")
    print(f"批次大小: {BATCH_SIZE}")

    # 加载模型
    print("\n加载模型...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model = Kronos.from_pretrained(MODEL_PATH)
    tokenizer = tokenizer.eval().to(device)
    model = model.eval().to(device)
    print("模型加载完成")

    # 加载预处理数据
    print("\n加载预处理数据...")
    data_path = 'finetune/data/kline_daily_ma60.pkl'
    with open(data_path, 'rb') as f:
        norm_dict = pickle.load(f)
    print(f"股票数: {len(norm_dict)}")

    # 获取预测日期范围
    sample_data = list(norm_dict.values())[0]
    all_dates = sample_data['index']
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    pred_dates = all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)]
    print(f"预测日期: {pred_dates.tolist()}")
    print(f"预测日期数: {len(pred_dates)}")

    # 批量推理
    print("\n开始批量推理...")
    output_path = 'outputs/prediction_results/predictions_ma60_v1ic_2026_0519_0524.sql'
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f+08')
    model_id = 'ma60-v1-ic'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w', encoding='utf-8')
    total_sql_count = 0

    for pred_date in tqdm(pred_dates, desc="预测进度"):
        symbols_batch = []
        x_norm_batch = []
        means_batch = []
        stds_batch = []
        last_closes = []
        limits_batch = []

        hist_ts = None

        for symbol, data in norm_dict.items():
            timestamps = data['index']
            if pred_date not in timestamps:
                continue

            idx = timestamps.get_loc(pred_date) + 1
            if idx < LOOKBACK + PRED_LEN:
                continue

            normalized = data['normalized'][idx-LOOKBACK:idx]
            means = data['means'][idx-LOOKBACK:idx]
            stds = data['stds'][idx-LOOKBACK:idx]

            x_norm_batch.append(normalized[-LOOKBACK:])
            means_batch.append(means[-PRED_LEN:])
            stds_batch.append(stds[-PRED_LEN:])
            symbols_batch.append(symbol)
            last_closes.append(data['original'][idx-1, 3])
            limits_batch.append(get_price_limit(symbol))

            if hist_ts is None:
                hist_ts = timestamps[idx-LOOKBACK:idx]

        if len(x_norm_batch) == 0:
            continue

        x_stamp = calc_time_stamps(hist_ts)
        future_dates = generate_future_dates(pred_date, PRED_LEN)
        y_stamp = calc_time_stamps(future_dates)

        predictions = {}

        for b_start in range(0, len(x_norm_batch), BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, len(x_norm_batch))

            batch_x = np.stack(x_norm_batch[b_start:b_end], axis=0)
            batch_x_stamp = np.tile(x_stamp, (b_end - b_start, 1, 1))
            batch_y_stamp = np.tile(y_stamp, (b_end - b_start, 1, 1))

            with torch.no_grad():
                x_tensor = torch.from_numpy(batch_x).to(device)
                x_stamp_tensor = torch.from_numpy(batch_x_stamp).to(device)
                y_stamp_tensor = torch.from_numpy(batch_y_stamp).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048, pred_len=PRED_LEN,
                    clip=CLIP, T=T, top_k=0, top_p=TOP_P,
                    sample_count=1, verbose=False
                )

                if isinstance(preds, torch.Tensor):
                    preds = preds[:, -PRED_LEN:, :].cpu().numpy()
                else:
                    preds = preds[:, -PRED_LEN:, :]

            for i, idx in enumerate(range(b_start, b_end)):
                symbol = symbols_batch[idx]
                pred_denorm = preds[i] * stds_batch[idx] + means_batch[idx]

                predictions[symbol] = {
                    'pred_date': pred_date,
                    'predictions': pred_denorm,
                    'future_dates': future_dates,
                    'last_close': last_closes[idx],
                    'price_limit': limits_batch[idx]
                }

        if predictions:
            sql_lines = generate_sql(predictions, model_id, LOOKBACK, PRED_LEN, created_at)
            output_file.write('\n'.join(sql_lines) + '\n')
            output_file.flush()
            total_sql_count += len(sql_lines)

    output_file.close()

    print("\n" + "=" * 60)
    print("推理完成")
    print("=" * 60)
    print(f"预测日期数: {len(pred_dates)}")
    print(f"SQL总数: {total_sql_count}")
    print(f"保存路径: {output_path}")


if __name__ == '__main__':
    main()
