"""
批量推理脚本：从最新K线倒推，每日预测所有股票

优化策略：
- 同一预测日期的所有股票批量处理
- GPU批量推理（批量大小8，适配MX330）
- 涨跌幅剪裁：主板±10%，创业板/科创板±20%
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
    """
    获取涨跌幅限制

    主板: ±10%
    创业板（300, 301）: ±20%
    科创板（688）: ±20%
    """
    code_num = stock_code.split('.')[0]

    if code_num.startswith('300') or code_num.startswith('301') or code_num.startswith('688'):
        return 0.20  # 创业板/科创板
    else:
        return 0.10  # 主板


def clip_prediction(pred_df, last_close, price_limit):
    """
    涨跌幅剪裁

    确保:
    - high ≤ last_close * (1 + limit)
    - low ≥ last_close * (1 - limit)
    - open/close 在 [low, high] 范围内
    - 价格保留2位小数
    """
    pred_df = pred_df.copy()

    upper_limit = round(last_close * (1 + price_limit), 2)
    lower_limit = round(last_close * (1 - price_limit), 2)

    # 剪裁 high/low
    pred_df['high'] = pred_df['high'].clip(upper=upper_limit).round(2)
    pred_df['low'] = pred_df['low'].clip(lower=lower_limit).round(2)

    # open/close 必须在 [low, high] 范围内
    for idx in pred_df.index:
        row_low = pred_df.loc[idx, 'low']
        row_high = pred_df.loc[idx, 'high']
        pred_df.loc[idx, 'open'] = round(np.clip(pred_df.loc[idx, 'open'], row_low, row_high), 2)
        pred_df.loc[idx, 'close'] = round(np.clip(pred_df.loc[idx, 'close'], row_low, row_high), 2)

    return pred_df


class BatchPredictor:
    """批量预测器"""

    def __init__(self, model_path, tokenizer_path, device=None, max_context=2048, clip=5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_context = max_context
        self.clip = clip
        self.feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']

        print(f"加载模型: {model_path}")
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        self.model = Kronos.from_pretrained(model_path)
        self.tokenizer = self.tokenizer.eval().to(self.device)
        self.model = self.model.eval().to(self.device)
        print(f"模型已加载到 {self.device}")

    def predict_batch(self, data_dict, lookback=400, pred_len=10, end_indices=None, T=1.0, top_p=0.9):
        """
        批量预测所有股票

        Args:
            data_dict: {symbol: DataFrame}
            lookback: 回看天数
            pred_len: 预测天数
            end_indices: {symbol: end_idx} 每只股票的结束索引

        Returns:
            predictions: {symbol: {'pred_date', 'last_close', 'predictions'}}
        """
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

            # 确定 end_idx
            if end_indices and symbol in end_indices:
                end_idx = end_indices[symbol]
            else:
                end_idx = len(df)

            if end_idx < lookback:
                continue

            start_idx = end_idx - lookback
            hist_df = df.iloc[start_idx:end_idx]
            values = hist_df[self.feature_cols].values.astype(np.float32)

            # 归一化
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0) + 1e-5
            x_norm = (values - mean) / std
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            data_list.append(x_norm)
            timestamps_list.append(hist_df.index)
            means_list.append(mean)
            stds_list.append(std)
            symbols.append(symbol)

            # pred_date = 基准日（hist_df最后一根K线日期）
            pred_dates.append(hist_df.index[-1])

            # last_close = 基准日收盘价（用于剪裁）
            last_closes.append(df['close'].iloc[end_idx - 1])

            # 涨跌幅限制
            price_limits.append(get_price_limit(symbol))

        if len(data_list) == 0:
            return {}

        B = len(data_list)
        batch_size = min(B, 64)  # 批量大小

        predictions = {}

        # 分批处理
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

            # 堆叠
            x_batch = np.stack(batch_data, axis=0)
            x_stamp = calc_time_stamps_batch(batch_timestamps)

            # 生成未来时间戳
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

            # GPU推理
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

            # 反归一化、剪裁、组装结果
            for i, symbol in enumerate(batch_symbols):
                pred_denorm = preds[i] * batch_stds[i] + batch_means[i]

                pred_df = pd.DataFrame(
                    pred_denorm,
                    columns=self.feature_cols,
                    index=y_timestamps_list[i]
                )

                # 涨跌幅剪裁
                pred_df = clip_prediction(pred_df, batch_last_closes[i], batch_price_limits[i])

                predictions[symbol] = {
                    'pred_date': batch_pred_dates[i],
                    'last_close': round(float(batch_last_closes[i]), 2),
                    'predictions': pred_df
                }

            # 进度显示
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
        extras = json.dumps({'last_close': round(float(last_close), 2)}, ensure_ascii=False)

        sql = f"insert into prediction_results (stock_code, model_id, pred_date, created_at, pred_path, lookback, pred_len, model_info, extras) values ('{symbol}', '{model_id}', '{pred_date}', '{created_at}', '{pred_json}', {lookback}, {pred_len}, '{model_info}', '{extras}');"
        sql_lines.append(sql)

    return sql_lines


def main():
    lookback = 400
    pred_len = 10
    T = 1.0
    top_p = 0.9
    max_days = 100  # 每只股票最多倒推天数

    # 加载模型
    predictor = BatchPredictor(
        model_path='final_models/Kronos-mini',
        tokenizer_path='final_models/Kronos-Tokenizer-2k',
        max_context=2048
    )

    # 加载数据
    data_path = 'finetune/data/processed_datasets_clean/full_data.pkl'
    print(f"\n加载数据: {data_path}")
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)
    print(f"股票数: {len(full_data)}")

    # 统计每只股票的有效天数
    valid_stocks = {s: df for s, df in full_data.items() if len(df) >= lookback}
    print(f"有效股票: {len(valid_stocks)}")

    # 计算所有股票的数据长度范围
    lengths = [len(df) for df in valid_stocks.values()]
    max_len = max(lengths)
    min_len = min(lengths)
    print(f"数据长度范围: {min_len} ~ {max_len}")

    # 从最新日期开始倒推
    output_path = 'outputs/prediction_results/predictions_mini.sql'
    all_sql_lines = []
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f+08')

    sample_df = list(valid_stocks.values())[0]
    total_trading_days = len(sample_df)

    print(f"\n开始批量预测...")
    print(f"总交易日: {total_trading_days}, 倒推天数: {min(max_days, total_trading_days - lookback)}")

    days_predicted = 0
    for end_idx in range(total_trading_days, lookback, -1):
        if days_predicted >= max_days:
            break

        # 构建每只股票的 end_indices
        end_indices = {}
        for symbol, df in valid_stocks.items():
            target_date = sample_df.index[end_idx - 1]
            if target_date in df.index:
                idx = df.index.get_loc(target_date) + 1
                if idx >= lookback:
                    end_indices[symbol] = idx

        if len(end_indices) < 10:
            continue

        pred_date_str = sample_df.index[end_idx - 1].strftime('%Y-%m-%d')
        print(f"\n预测日期: {pred_date_str} (股票数: {len(end_indices)})")

        # 批量预测
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
            all_sql_lines.extend(sql_lines)
            days_predicted += 1
            print(f"  生成 {len(sql_lines)} 条SQL, 累计: {len(all_sql_lines)}")

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_sql_lines))

    print(f"\n完成!")
    print(f"预测天数: {days_predicted}")
    print(f"SQL总数: {len(all_sql_lines)}")
    print(f"保存: {output_path}")


if __name__ == '__main__':
    main()