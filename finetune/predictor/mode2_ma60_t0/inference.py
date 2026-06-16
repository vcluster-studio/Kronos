"""
批量推理脚本：使用最优MA60模型对kline_daily.sql数据进行推理

模型: V1-IC (best_ic_model) + MA60 tokenizer
归一化: sliding MA60
数据源: kline_daily.sql
筛选: 2025-01-01以来的预测结果
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
T = 1.0
TOP_P = 0.9
START_DATE = '2025-01-01'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_sql_file(sql_path):
    """解析kline_daily.sql文件"""
    print(f"解析SQL文件: {sql_path}")

    # 正则匹配 - 注意：数值字段有些带引号有些不带
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
                # 去掉可能的引号和尾部空格
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

            if line_count % 1000000 == 0:
                print(f"  已解析 {line_count} 行, {len(data)} 股票...")

    print(f"  解析行数: {line_count}")
    print(f"  股票数: {len(data)}")

    # 转换为DataFrame
    df_dict = {}
    for symbol, rows in data.items():
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date')
        df_dict[symbol] = df

    return df_dict


def sliding_ma60_normalize(values, window=60, clip=5.0):
    """
    MA60滑动归一化

    每个时间点使用前60步的均值和标准差进行归一化
    """
    n = len(values)
    normalized = np.zeros_like(values, dtype=np.float32)
    means = np.zeros((n, values.shape[1]), dtype=np.float32)
    stds = np.zeros((n, values.shape[1]), dtype=np.float32)

    # 前window步使用累积统计
    for i in range(min(window, n)):
        cum_mean = np.mean(values[:i+1], axis=0)
        cum_std = np.std(values[:i+1], axis=0) + 1e-5
        means[i] = cum_mean
        stds[i] = cum_std
        normalized[i] = (values[i] - cum_mean) / cum_std

    # 后续使用滑动窗口
    for i in range(window, n):
        window_vals = values[i-window+1:i+1]
        means[i] = np.mean(window_vals, axis=0)
        stds[i] = np.std(window_vals, axis=0) + 1e-5
        normalized[i] = (values[i] - means[i]) / stds[i]

    # 裁剪
    normalized = np.clip(normalized, -clip, clip)

    return normalized, means, stds


def calc_time_stamps(timestamps):
    """计算时间戳特征"""
    if isinstance(timestamps, pd.DatetimeIndex):
        return np.stack([
            timestamps.minute.values,
            timestamps.hour.values,
            timestamps.weekday.values,
            timestamps.day.values,
            timestamps.month.values
        ], axis=1).astype(np.float32)
    else:
        return np.stack([
            timestamps.dt.minute.values,
            timestamps.dt.hour.values,
            timestamps.dt.weekday.values,
            timestamps.dt.day.values,
            timestamps.dt.month.values
        ], axis=1).astype(np.float32)


def get_price_limit(stock_code):
    """获取涨跌幅限制"""
    code_num = stock_code.split('.')[0]
    if code_num.startswith('300') or code_num.startswith('301') or code_num.startswith('688'):
        return 0.20  # 创业板/科创板
    else:
        return 0.10  # 主板


def clip_prediction(pred_values, last_close, price_limit):
    """涨跌幅剪裁"""
    pred_values = pred_values.copy()

    upper_limit = last_close * (1 + price_limit)
    lower_limit = last_close * (1 - price_limit)

    # high/low剪裁
    pred_values[:, 1] = np.clip(pred_values[:, 1], None, upper_limit)  # high
    pred_values[:, 2] = np.clip(pred_values[:, 2], lower_limit, None)  # low

    # open/close必须在[low, high]范围内
    for i in range(len(pred_values)):
        row_low = pred_values[i, 2]
        row_high = pred_values[i, 1]
        pred_values[i, 0] = np.clip(pred_values[i, 0], row_low, row_high)  # open
        pred_values[i, 3] = np.clip(pred_values[i, 3], row_low, row_high)  # close

    return pred_values


def generate_future_dates(last_date, pred_len):
    """生成未来交易日"""
    future_dates = []
    current = last_date + timedelta(days=1)
    while len(future_dates) < pred_len:
        if current.weekday() < 5:  # 周一到周五
            future_dates.append(current)
        current += timedelta(days=1)
    return pd.DatetimeIndex(future_dates)


def generate_sql(predictions, model_id, lookback, pred_len, created_at):
    """生成SQL语句"""
    sql_lines = []
    feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']

    for symbol, pred_data in predictions.items():
        pred_date = pred_data['pred_date'].strftime('%Y-%m-%d')
        pred_values = pred_data['predictions']
        future_dates = pred_data['future_dates']
        last_close = pred_data['last_close']
        price_limit = pred_data['price_limit']

        # 剪裁
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
            'T': T,
            'name': 'Kronos (MA60-V1-IC)',
            'top_p': TOP_P,
            'device': str(device),
            'norm': 'sliding_ma60'
        }, ensure_ascii=False)
        extras = json.dumps({
            'limit_pct': price_limit,
            'last_close': round(float(last_close), 2)
        }, ensure_ascii=False)

        sql = f"insert into prediction_results (stock_code, model_id, pred_date, created_at, pred_path, lookback, pred_len, model_info, extras) values ('{symbol}', '{model_id}', '{pred_date}', '{created_at}', '{pred_json}', {lookback}, {pred_len}, '{model_info}', '{extras}');"
        sql_lines.append(sql)

    return sql_lines


def main():
    print("=" * 60)
    print("MA60最优模型批量推理")
    print("=" * 60)
    print(f"模型: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"筛选起始: {START_DATE}")
    print(f"设备: {device}")

    # 加载模型
    print("\n加载模型...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model = Kronos.from_pretrained(MODEL_PATH)
    tokenizer = tokenizer.eval().to(device)
    model = model.eval().to(device)
    print("模型加载完成")

    # 解析SQL数据
    print("\n解析数据...")
    sql_path = 'data/kline_daily.sql'
    df_dict = parse_sql_file(sql_path)

    # 筛选有效股票
    feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
    valid_stocks = {}
    for symbol, df in df_dict.items():
        if len(df) >= LOOKBACK + PRED_LEN:
            # 确保列名正确
            if 'volume' in df.columns:
                df = df.rename(columns={'volume': 'vol'})
            if 'amount' in df.columns:
                df = df.rename(columns={'amount': 'amt'})
            valid_stocks[symbol] = df

    print(f"有效股票: {len(valid_stocks)}")

    # 找出所有可用日期
    sample_df = list(valid_stocks.values())[0]
    all_dates = sample_df.index

    # 筛选2025-01-01以来的日期
    start_dt = pd.to_datetime(START_DATE)
    pred_dates = all_dates[all_dates >= start_dt]
    print(f"预测日期范围: {pred_dates.min()} ~ {pred_dates.max()}")
    print(f"预测日期数: {len(pred_dates)}")

    # 批量推理
    print("\n开始批量推理...")
    output_path = 'outputs/prediction_results/predictions_2025_ma60_v1ic.sql'
    all_sql_lines = []
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f+08')
    model_id = 'ma60-v1-ic'

    batch_size = 32  # 批量大小

    for pred_date in tqdm(pred_dates, desc="预测进度"):
        # 确定end_idx
        end_idx = sample_df.index.get_loc(pred_date) + 1

        if end_idx < LOOKBACK:
            continue

        # 收集该日期可预测的股票
        symbols_to_pred = []
        data_list = []
        means_list = []
        stds_list = []
        last_closes = []
        price_limits = []

        for symbol, df in valid_stocks.items():
            if pred_date not in df.index:
                continue

            idx = df.index.get_loc(pred_date) + 1
            if idx < LOOKBACK + PRED_LEN:
                continue

            # 取历史数据
            hist_df = df.iloc[idx-LOOKBACK:idx]
            values = hist_df[feature_cols].values.astype(np.float32)

            # MA60归一化
            normalized, means, stds = sliding_ma60_normalize(values, window=60, clip=CLIP)

            # 取LOOKBACK部分
            x_norm = normalized[-LOOKBACK:]
            x_means = means[-LOOKBACK:]
            x_stds = stds[-LOOKBACK:]

            data_list.append(x_norm)
            means_list.append(x_means[-PRED_LEN:])  # 用于反归一化的means
            stds_list.append(x_stds[-PRED_LEN:])    # 用于反归一化的stds
            symbols_to_pred.append(symbol)
            last_closes.append(df['close'].iloc[idx-1])
            price_limits.append(get_price_limit(symbol))

        if len(data_list) == 0:
            continue

        # 批量推理
        predictions = {}

        for batch_start in range(0, len(data_list), batch_size):
            batch_end = min(batch_start + batch_size, len(data_list))
            batch_data = np.stack(data_list[batch_start:batch_end], axis=0)

            # 时间戳
            hist_timestamps = hist_df.index[-LOOKBACK:]
            x_stamp = calc_time_stamps(hist_timestamps)
            x_stamp = np.tile(x_stamp, (batch_end - batch_start, 1, 1))

            # 未来时间戳
            future_dates = generate_future_dates(pred_date, PRED_LEN)
            y_stamp = calc_time_stamps(future_dates)
            y_stamp = np.tile(y_stamp, (batch_end - batch_start, 1, 1))

            with torch.no_grad():
                x_tensor = torch.from_numpy(batch_data).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048, pred_len=PRED_LEN,
                    clip=CLIP, T=T, top_k=0, top_p=TOP_P,
                    sample_count=1, verbose=False
                )
                # preds可能是numpy array或torch tensor
                if isinstance(preds, torch.Tensor):
                    preds = preds[:, -PRED_LEN:, :].cpu().numpy()
                else:
                    preds = preds[:, -PRED_LEN:, :]

            # 反归一化并保存
            for i, idx in enumerate(range(batch_start, batch_end)):
                symbol = symbols_to_pred[idx]
                pred_norm = preds[i]
                pred_denorm = pred_norm * stds_list[idx] + means_list[idx]

                predictions[symbol] = {
                    'pred_date': pred_date,
                    'predictions': pred_denorm,
                    'future_dates': future_dates,
                    'last_close': last_closes[idx],
                    'price_limit': price_limits[idx]
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
    print(f"预测日期数: {len(pred_dates)}")
    print(f"SQL总数: {len(all_sql_lines)}")
    print(f"保存路径: {output_path}")


if __name__ == '__main__':
    main()