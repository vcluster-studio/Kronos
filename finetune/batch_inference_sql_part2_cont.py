"""
批量推理脚本 - 第二批续跑：从断点继续，减小batch size

- batch size 64（避免OOM）
- 追加模式写入
- 起点从2025-11-14开始（断点位置）
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference


def batch_normalize(x_batch, clip=5.0):
    """批量归一化"""
    means = np.mean(x_batch, axis=1)
    stds = np.std(x_batch, axis=1) + 1e-5
    x_norm = (x_batch - means[:, np.newaxis, :]) / stds[:, np.newaxis, :]
    x_norm = np.clip(x_norm, -clip, clip)
    return x_norm, means, stds


def calc_time_stamps_batch(timestamps_batch):
    """批量时间戳"""
    B = len(timestamps_batch)
    seq_len = len(timestamps_batch[0])
    time_features = np.zeros((B, seq_len, 5), dtype=np.float32)

    for i, ts in enumerate(timestamps_batch):
        if isinstance(ts, pd.DatetimeIndex):
            time_features[i, :, 0] = ts.minute.values
            time_features[i, :, 1] = ts.hour.values
            time_features[i, :, 2] = ts.weekday.values
            time_features[i, :, 3] = ts.day.values
            time_features[i, :, 4] = ts.month.values
        else:
            time_features[i, :, 0] = ts.dt.minute.values
            time_features[i, :, 1] = ts.dt.hour.values
            time_features[i, :, 2] = ts.dt.weekday.values
            time_features[i, :, 3] = ts.dt.day.values
            time_features[i, :, 4] = ts.dt.month.values

    return time_features


def get_price_limit(stock_code):
    """涨跌幅限制"""
    code_num = stock_code.split('.')[0]
    if code_num.startswith('300') or code_num.startswith('301') or code_num.startswith('688'):
        return 0.20
    else:
        return 0.10


def clip_prediction(pred_df, last_close, price_limit):
    """涨跌幅剪裁"""
    pred_df = pred_df.copy()
    upper_limit = round(last_close * (1 + price_limit), 2)
    lower_limit = round(last_close * (1 - price_limit), 2)
    pred_df['high'] = pred_df['high'].clip(upper=upper_limit).round(2)
    pred_df['low'] = pred_df['low'].clip(lower=lower_limit).round(2)
    for idx in pred_df.index:
        row_low = pred_df.loc[idx, 'low']
        row_high = pred_df.loc[idx, 'high']
        pred_df.loc[idx, 'open'] = round(np.clip(pred_df.loc[idx, 'open'], row_low, row_high), 2)
        pred_df.loc[idx, 'close'] = round(np.clip(pred_df.loc[idx, 'close'], row_low, row_high), 2)
    return pred_df


class BatchPredictor:
    """批量预测器"""

    def __init__(self, model_path, tokenizer_path, device=None, max_context=2048, clip=5, batch_size=64):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_context = max_context
        self.clip = clip
        self.batch_size = batch_size
        self.feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']

        print(f"加载模型: {model_path}")
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        self.model = Kronos.from_pretrained(model_path)
        self.tokenizer = self.tokenizer.eval().to(self.device)
        self.model = self.model.eval().to(self.device)
        print(f"模型已加载到 {self.device}")
        print(f"batch size: {self.batch_size}")

    def predict_batch(self, data_dict, lookback=400, pred_len=10, end_indices=None, T=1.0, top_p=0.9):
        """批量预测"""
        symbols = []
        data_list = []
        timestamps_list = []
        means_list = []
        stds_list = []
        pred_dates = []
        last_closes = []
        price_limits = []

        for symbol, df in data_dict.items():
            if len(df) < lookback:
                continue

            if end_indices and symbol in end_indices:
                end_idx = end_indices[symbol]
            else:
                end_idx = len(df)

            if end_idx < lookback:
                continue

            start_idx = end_idx - lookback
            hist_df = df.iloc[start_idx:end_idx]
            values = hist_df[self.feature_cols].values.astype(np.float32)

            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0) + 1e-5
            x_norm = (values - mean) / std
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            data_list.append(x_norm)
            timestamps_list.append(hist_df.index)
            means_list.append(mean)
            stds_list.append(std)
            symbols.append(symbol)
            pred_dates.append(hist_df.index[-1])
            last_closes.append(df['close'].iloc[end_idx - 1])
            price_limits.append(get_price_limit(symbol))

        if len(data_list) == 0:
            return {}

        B = len(data_list)
        batch_size = min(B, self.batch_size)  # 使用配置的batch size
        predictions = {}

        total_batches = (B + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, B)
            batch_symbols = symbols[batch_start:batch_end]
            batch_data = data_list[batch_start:batch_end]
            batch_timestamps = timestamps_list[batch_start:batch_end]
            batch_means = means_list[batch_start:batch_end]
            batch_stds = stds_list[batch_start:batch_end]
            batch_pred_dates = pred_dates[batch_start:batch_end]
            batch_last_closes = last_closes[batch_start:batch_end]
            batch_price_limits = price_limits[batch_start:batch_end]

            x_batch = np.stack(batch_data, axis=0)
            x_stamp = calc_time_stamps_batch(batch_timestamps)

            y_timestamps_list = []
            for ts in batch_timestamps:
                last_date = ts[-1]
                future_dates = []
                current = last_date + timedelta(days=1)
                while len(future_dates) < pred_len:
                    if current.weekday() < 5:
                        future_dates.append(current)
                    current += timedelta(days=1)
                y_timestamps_list.append(pd.DatetimeIndex(future_dates))

            y_stamp = calc_time_stamps_batch(y_timestamps_list)

            # 清理显存
            torch.cuda.empty_cache()

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_batch).to(self.device)
                x_stamp_tensor = torch.from_numpy(x_stamp).to(self.device)
                y_stamp_tensor = torch.from_numpy(y_stamp).to(self.device)

                preds = auto_regressive_inference(
                    self.tokenizer, self.model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    self.max_context, pred_len,
                    self.clip, T, top_k=0, top_p=top_p,
                    sample_count=1, verbose=False
                )
                preds = preds[:, -pred_len:, :]

            for i, symbol in enumerate(batch_symbols):
                pred_denorm = preds[i] * batch_stds[i] + batch_means[i]
                pred_df = pd.DataFrame(
                    pred_denorm,
                    columns=self.feature_cols,
                    index=y_timestamps_list[i]
                )
                pred_df = clip_prediction(pred_df, batch_last_closes[i], batch_price_limits[i])
                predictions[symbol] = {
                    'pred_date': batch_pred_dates[i],
                    'last_close': round(float(batch_last_closes[i]), 2),
                    'predictions': pred_df
                }

            # 释放tensor
            del x_tensor, x_stamp_tensor, y_stamp_tensor, preds
            torch.cuda.empty_cache()

            print(f"    批次 {batch_idx+1}/{total_batches}: {len(predictions)}/{B} 股票已完成", flush=True)

        return predictions


def generate_sql(predictions, model_id, lookback, pred_len, T, top_p, created_at):
    """生成SQL语句"""
    sql_lines = []
    for symbol, pred_data in predictions.items():
        pred_date = pred_data['pred_date'].strftime('%Y-%m-%d')
        pred_df = pred_data['predictions']
        last_close = float(pred_data['last_close'])

        pred_list = []
        for idx, row in pred_df.iterrows():
            pred_list.append({
                'date': idx.strftime('%Y-%m-%d'),
                'open': round(float(row['open']), 2),
                'high': round(float(row['high']), 2),
                'low': round(float(row['low']), 2),
                'close': round(float(row['close']), 2),
                'volume': int(float(row['vol'])),
                'amount': int(float(row['amt']))
            })

        pred_json = json.dumps(pred_list, ensure_ascii=False)
        model_info = json.dumps({'T': T, 'name': f'Kronos ({model_id})', 'top_p': top_p}, ensure_ascii=False)
        extras = json.dumps({'last_close': round(last_close, 2)}, ensure_ascii=False)

        sql = f"insert into prediction_results (stock_code, model_id, pred_date, created_at, pred_path, lookback, pred_len, model_info, extras) values ('{symbol}', '{model_id}', '{pred_date}', '{created_at}', '{pred_json}', {lookback}, {pred_len}, '{model_info}', '{extras}');"
        sql_lines.append(sql)

    return sql_lines


def main():
    lookback = 400
    pred_len = 10
    T = 1.0
    top_p = 0.9
    batch_size = 64  # 降低batch size避免OOM

    # 从断点继续：2025-11-14（往前还需约80天）
    # 已完成约20天（从2025-12-12到2025-11-15附近）
    start_from_date = '2025-11-14'
    max_days = 80  # 剩余天数

    # 加载模型
    predictor = BatchPredictor(
        model_path='final_models/Kronos-mini',
        tokenizer_path='final_models/Kronos-Tokenizer-2k',
        max_context=2048,
        batch_size=batch_size
    )

    # 加载数据
    data_path = 'finetune/data/processed_datasets_clean/full_data.pkl'
    print(f"\n加载数据: {data_path}")
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)
    print(f"股票数: {len(full_data)}")

    valid_stocks = {s: df for s, df in full_data.items() if len(df) >= lookback}
    print(f"有效股票: {len(valid_stocks)}")

    sample_df = list(valid_stocks.values())[0]
    dates = sample_df.index

    # 找到起点位置
    start_date = pd.Timestamp(start_from_date)
    start_idx = dates.get_loc(start_date)

    # 计算终点位置（往前80天）
    end_idx = start_idx - max_days
    end_date = dates[end_idx]

    print(f"\n续跑预测范围:")
    print(f"  起点: {start_date.strftime('%Y-%m-%d')} (断点)")
    print(f"  终点: {end_date.strftime('%Y-%m-%d')}")
    print(f"  倒推天数: {max_days}")

    # 追加模式：清理旧文件中可能不完整的2025-11-14数据
    # 重新开始写入part3文件
    output_path = 'outputs/prediction_results/predictions_mini_part2_cont.sql'
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f+08')

    output_file = open(output_path, 'w', encoding='utf-8')

    print(f"\n开始批量预测 (batch size={batch_size})...")

    days_predicted = 0
    total_sql = 0

    # 从 start_idx 往前倒推到 end_idx
    for current_idx in range(start_idx, end_idx, -1):
        if days_predicted >= max_days:
            break

        # 构建每只股票的 end_indices
        end_indices = {}
        target_date = dates[current_idx - 1]

        for symbol, df in valid_stocks.items():
            if target_date in df.index:
                idx = df.index.get_loc(target_date) + 1
                if idx >= lookback:
                    end_indices[symbol] = idx

        if len(end_indices) < 10:
            continue

        pred_date_str = target_date.strftime('%Y-%m-%d')
        print(f"\n预测日期: {pred_date_str} (股票数: {len(end_indices)})")

        predictions = predictor.predict_batch(
            valid_stocks,
            lookback=lookback,
            pred_len=pred_len,
            end_indices=end_indices,
            T=T,
            top_p=top_p
        )

        if predictions:
            sql_lines = generate_sql(predictions, 'criticality-mini', lookback, pred_len, T, top_p, created_at)

            # 增量写入文件
            for sql in sql_lines:
                output_file.write(sql + '\n')

            total_sql += len(sql_lines)
            days_predicted += 1
            print(f"  生成 {len(sql_lines)} 条SQL, 累计: {total_sql}")
            output_file.flush()  # 立即写入磁盘

    output_file.close()

    print(f"\n续跑完成!")
    print(f"预测天数: {days_predicted}")
    print(f"SQL总数: {total_sql}")
    print(f"保存: {output_path}")


if __name__ == '__main__':
    main()