"""
高吞吐量批量推理脚本

优化点：
1. 批量预处理（避免for循环）
2. 预计算时间戳（相同预测时间只需计算一次）
3. 批量归一化（numpy向量化）
4. 可选返回numpy数组（跳过DataFrame开销）
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Add project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference


def batch_normalize(x_batch, clip=5.0):
    """
    批量全窗口归一化

    Args:
        x_batch: [B, seq_len, feat] 原始数据批次

    Returns:
        x_norm: 归一化后的数据
        means: [B, feat] 每个序列的mean
        stds: [B, feat] 每个序列的std
    """
    # 每个序列独立计算mean/std
    means = np.mean(x_batch, axis=1)  # [B, feat]
    stds = np.std(x_batch, axis=1) + 1e-5  # [B, feat]

    # 扩展维度用于广播
    x_norm = (x_batch - means[:, np.newaxis, :]) / stds[:, np.newaxis, :]
    x_norm = np.clip(x_norm, -clip, clip)

    return x_norm, means, stds


def calc_time_stamps_batch(timestamps_batch):
    """
    批量计算时间戳特征

    Args:
        timestamps_batch: List of DatetimeIndex

    Returns:
        time_features: [B, seq_len, 5] 时间特征
    """
    B = len(timestamps_batch)
    seq_len = len(timestamps_batch[0])

    # 预分配数组
    time_features = np.zeros((B, seq_len, 5), dtype=np.float32)

    for i, ts in enumerate(timestamps_batch):
        if isinstance(ts, pd.DatetimeIndex):
            time_features[i, :, 0] = ts.minute
            time_features[i, :, 1] = ts.hour
            time_features[i, :, 2] = ts.weekday
            time_features[i, :, 3] = ts.day
            time_features[i, :, 4] = ts.month
        else:
            time_features[i, :, 0] = ts.dt.minute
            time_features[i, :, 1] = ts.dt.hour
            time_features[i, :, 2] = ts.dt.weekday
            time_features[i, :, 3] = ts.dt.day
            time_features[i, :, 4] = ts.dt.month

    return time_features


class HighThroughputPredictor:
    """高吞吐量预测器"""

    def __init__(self, model_path, tokenizer_path, device=None, max_context=512, clip=5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_col = 'amount'

        # 加载模型
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        self.model = Kronos.from_pretrained(model_path)
        self.tokenizer = self.tokenizer.eval().to(self.device)
        self.model = self.model.eval().to(self.device)

        print(f"Model loaded on {self.device}")

    def predict_batch_fast(self, data_dict, lookback=400, pred_len=10,
                           x_timestamp=None, y_timestamp=None,
                           T=1.0, top_k=0, top_p=0.9, sample_count=1,
                           return_numpy=False, verbose=False):
        """
        高吞吐量批量推理

        Args:
            data_dict: {symbol: DataFrame} 字典格式数据
            lookback: 历史窗口长度
            pred_len: 预测步数
            x_timestamp: 所有股票共享的历史时间戳（可选，如果相同则预计算）
            y_timestamp: 所有股票共享的预测时间戳（可选）
            return_numpy: 是否返回numpy数组而非DataFrame（更快）

        Returns:
            如果 return_numpy=False: {symbol: DataFrame} 预测结果
            如果 return_numpy=True: {symbol: np.array} 预测结果数组
        """
        feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']

        # 收集有效股票
        symbols = []
        data_list = []

        for symbol, df in data_dict.items():
            if len(df) >= lookback + pred_len:
                symbols.append(symbol)
                # 提取数据（lookback + pred_len）
                values = df[feature_cols].values[-(lookback + pred_len):].astype(np.float32)
                data_list.append(values)

        if len(data_list) == 0:
            return {}

        # 批量堆叠
        full_batch = np.stack(data_list, axis=0)  # [B, lookback+pred_len, feat]
        B = full_batch.shape[0]

        # 分离历史和预测数据
        x_batch = full_batch[:, :lookback, :]  # [B, lookback, feat]
        y_batch = full_batch[:, lookback:, :]   # [B, pred_len, feat] - 用于计算实际值

        # 批量归一化
        x_norm, means, stds = batch_normalize(x_batch, clip=self.clip)

        # 时间戳处理
        if x_timestamp is None or y_timestamp is None:
            # 使用第一个股票的时间戳作为模板
            first_df = data_dict[symbols[0]]
            dates = first_df.index[-(lookback + pred_len):]
            dates = pd.to_datetime(dates)
            x_timestamp = dates[:lookback]
            y_timestamp = dates[lookback:]

        # 批量复制时间戳（所有股票共享）
        x_stamp = calc_time_stamps_batch([x_timestamp] * B)  # [B, lookback, 5]
        y_stamp = calc_time_stamps_batch([y_timestamp] * B)  # [B, pred_len, 5]

        # GPU推理
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).to(self.device)
            x_stamp_tensor = torch.from_numpy(x_stamp).to(self.device)
            y_stamp_tensor = torch.from_numpy(y_stamp).to(self.device)

            preds = auto_regressive_inference(
                self.tokenizer, self.model,
                x_tensor, x_stamp_tensor, y_stamp_tensor,
                self.max_context, pred_len,
                self.clip, T, top_k, top_p,
                sample_count, verbose
            )

            preds = preds[:, -pred_len:, :]  # [B, pred_len, feat]

        # 批量反归一化
        preds_denorm = preds * stds[:, np.newaxis, :] + means[:, np.newaxis, :]

        # 结果组装
        results = {}
        if return_numpy:
            for i, symbol in enumerate(symbols):
                results[symbol] = preds_denorm[i]
        else:
            for i, symbol in enumerate(symbols):
                pred_df = pd.DataFrame(
                    preds_denorm[i],
                    columns=self.price_cols + [self.vol_col, self.amt_col],
                    index=y_timestamp
                )
                results[symbol] = pred_df

        return results, symbols, y_batch  # 返回预测结果、股票列表、实际值


def benchmark_throughput():
    """吞吐量基准测试"""
    import time

    # 加载测试数据
    test_path = os.path.join(project_root, "finetune/data/processed_datasets/test_data.pkl")
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    print(f"测试数据: {len(test_data)} 只股票")

    # 加载模型
    predictor = HighThroughputPredictor(
        model_path='final_models/Kronos-mini',
        tokenizer_path='final_models/Kronos-Tokenizer-2k',
        max_context=2048
    )

    # 测试不同批量大小
    batch_sizes = [10, 50, 100, 200, 500]

    print("\n吞吐量测试:")
    print("-" * 50)

    for batch_size in batch_sizes:
        # 取指定数量股票
        test_subset = dict(list(test_data.items())[:batch_size])

        start_time = time.time()

        results, symbols, actuals = predictor.predict_batch_fast(
            test_subset,
            lookback=400,
            pred_len=10,
            return_numpy=True
        )

        elapsed = time.time() - start_time
        throughput = batch_size / elapsed

        print(f"批量 {batch_size:4d}: {elapsed:.2f}s, {throughput:.1f} stocks/s")

    print("-" * 50)


if __name__ == '__main__':
    benchmark_throughput()