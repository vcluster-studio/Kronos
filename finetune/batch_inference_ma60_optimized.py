"""
批量推理脚本 - 优化版：使用最优MA60模型

优化：
1. 向量化MA60归一化（pandas rolling）
2. 大批次GPU推理（batch_size=128）
3. 预计算所有日期的归一化数据
"""

import os
import sys
import re
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
TOKENIZER_PATH = 'outputs/models/ma60_tokenizer_v1/checkpoints/best_model'
MODEL_PATH = 'outputs/models/ma60_predictor_v1/checkpoints/best_ic_model'

# 参数
LOOKBACK = 400
PRED_LEN = 10
CLIP = 5.0
WINDOW = 60  # MA窗口
T = 1.0
TOP_P = 0.9
START_DATE = '2025-01-01'
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # 转换DataFrame
    df_dict = {}
    for symbol, rows in tqdm(data.items(), desc="转换DataFrame"):
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date')
        df_dict[symbol] = df

    return df_dict


def vectorized_ma60_normalize(df, window=60, clip=5.0):
    """
    向量化MA60归一化 - 使用pandas rolling

    返回: normalized DataFrame, means DataFrame, stds DataFrame
    """
    values = df[feature_cols].values.astype(np.float32)

    # 使用pandas DataFrame rolling
    df_rolling = pd.DataFrame(values, columns=feature_cols)

    # 滚动均值和标准差
    means = df_rolling.rolling(window=window, min_periods=1).mean().values.astype(np.float32)
    stds = df_rolling.rolling(window=window, min_periods=1).std().values.astype(np.float32)
    stds = np.where(stds < 1e-5, 1e-5, stds)

    # 归一化
    normalized = (values - means) / stds
    normalized = np.clip(normalized, -clip, clip)

    return normalized, means, stds


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

    pred_values[:, 1] = np.clip(pred_values[:, 1], None, upper)  # high
    pred_values[:, 2] = np.clip(pred_values[:, 2], lower, None)  # low

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
    print("MA60最优模型批量推理 - 优化版")
    print("=" * 60)
    print(f"模型: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"筛选起始: {START_DATE}")
    print(f"设备: {device}")
    print(f"批次大小: {BATCH_SIZE}")

    # 加载模型
    print("\n加载模型...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model = Kronos.from_pretrained(MODEL_PATH)
    tokenizer = tokenizer.eval().to(device)
    model = model.eval().to(device)
    print("模型加载完成")

    # 解析数据
    print("\n解析数据...")
    sql_path = 'data/kline_daily.sql'
    df_dict = parse_sql_file(sql_path)

    # 预计算归一化数据
    print("\n预计算MA60归一化...")
    norm_dict = {}
    for symbol, df in tqdm(df_dict.items(), desc="MA60归一化"):
        if len(df) >= LOOKBACK + PRED_LEN:
            normalized, means, stds = vectorized_ma60_normalize(df, WINDOW, CLIP)
            norm_dict[symbol] = {
                'df': df,
                'normalized': normalized,
                'means': means,
                'stds': stds
            }

    print(f"有效股票: {len(norm_dict)}")

    # 获取预测日期范围
    sample_df = list(norm_dict.values())[0]['df']
    all_dates = sample_df.index
    start_dt = pd.to_datetime(START_DATE)
    pred_dates = all_dates[all_dates >= start_dt]
    print(f"预测日期数: {len(pred_dates)}")

    # 批量推理
    print("\n开始批量推理...")
    output_path = 'outputs/prediction_results/predictions_2025_ma60_v1ic.sql'
    all_sql_lines = []
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f+08')
    model_id = 'ma60-v1-ic'

    for pred_date in tqdm(pred_dates, desc="预测进度"):
        # 收集数据
        symbols_batch = []
        x_norm_batch = []
        x_stamp_batch = []
        means_batch = []
        stds_batch = []
        last_closes = []
        limits_batch = []

        for symbol, data in norm_dict.items():
            df = data['df']
            if pred_date not in df.index:
                continue

            idx = df.index.get_loc(pred_date) + 1
            if idx < LOOKBACK + PRED_LEN:
                continue

            normalized = data['normalized'][idx-LOOKBACK:idx]
            means = data['means'][idx-LOOKBACK:idx]
            stds = data['stds'][idx-LOOKBACK:idx]

            x_norm_batch.append(normalized[-LOOKBACK:])
            means_batch.append(means[-PRED_LEN:])
            stds_batch.append(stds[-PRED_LEN:])
            symbols_batch.append(symbol)
            last_closes.append(df['close'].iloc[idx-1])
            limits_batch.append(get_price_limit(symbol))

            # 时间戳 - 使用最后一个股票的历史时间戳
            hist_ts = df.index[idx-LOOKBACK:idx]

        if len(x_norm_batch) == 0:
            continue

        # 计算时间戳（所有股票使用相同的）
        x_stamp = calc_time_stamps(hist_ts)
        future_dates = generate_future_dates(pred_date, PRED_LEN)
        y_stamp = calc_time_stamps(future_dates)

        # 分批推理
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

            # 反归一化
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

        # 生成SQL
        if predictions:
            sql_lines = generate_sql(predictions, model_id, LOOKBACK, PRED_LEN, created_at)
            all_sql_lines.extend(sql_lines)

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_sql_lines))

    print("\n" + "=" * 60)
    print("推理完成")
    print("=" * 60)
    print(f"SQL总数: {len(all_sql_lines)}")
    print(f"保存路径: {output_path}")


if __name__ == '__main__':
    main()