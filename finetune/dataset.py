import os
import sys
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling financial time series data.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.
        config: Configuration object with dataset_path, batch_size, etc.
    """

    def __init__(self, data_type: str = 'train', config=None):
        if config is None:
            raise ValueError("config is required. Pass config from train_tokenizer.py or train_predictor.py")
        self.config = config
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        # === 加载股票分类信息 ===
        categories_path = f"{self.config.dataset_path}/stock_categories.pkl"
        if os.path.exists(categories_path):
            with open(categories_path, 'rb') as f:
                self.stock_categories = pickle.load(f)
            self.use_stratified_sampling = True
        else:
            self.stock_categories = {}
            self.use_stratified_sampling = False

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # === 分层索引构建 ===
        # 按股票类型分组索引
        self.category_indices = {
            'large': [],   # 大盘股
            'mid': [],     # 中盘股
            'small': []    # 小盘股
        }
        self.indices = []  # 全部索引（兼容旧逻辑）

        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features and store them directly in the dataframe.
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # Keep only necessary columns to save memory.
                self.data[symbol] = df[self.feature_list + self.time_feature_list]

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    idx_tuple = (symbol, i)
                    self.indices.append(idx_tuple)

                    # === 添加到分类索引 ===
                    category = self.stock_categories.get(symbol, 'mid')  # 默认为中盘股
                    self.category_indices[category].append(idx_tuple)

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))

        # 打印分层统计
        if self.use_stratified_sampling:
            print(f"[{data_type.upper()}] Stratified indices:")
            for cat, indices in self.category_indices.items():
                print(f"  - {cat}: {len(indices)} samples ({len(indices)/len(self.indices)*100:.1f}%)")

        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        # === 分层采样：每类股票均衡采样 ===
        if self.use_stratified_sampling:
            # 随机选择一个分类（均匀选择）
            categories = ['large', 'mid', 'small']
            # 检查每个分类是否有样本
            valid_categories = [cat for cat in categories if len(self.category_indices[cat]) > 0]

            if len(valid_categories) > 0:
                # 均匀选择分类
                selected_category = self.py_rng.choice(valid_categories)

                # 从选中的分类中随机选择一个样本
                cat_indices = self.category_indices[selected_category]
                random_idx = self.py_rng.randint(0, len(cat_indices) - 1)
                symbol, start_idx = cat_indices[random_idx]
            else:
                # 降级为普通随机采样
                random_idx = self.py_rng.randint(0, len(self.indices) - 1)
                symbol, start_idx = self.indices[random_idx]
        else:
            # 不分层时，普通随机采样
            random_idx = self.py_rng.randint(0, len(self.indices) - 1)
            symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # === 归一化模式选择 ===
        # norm_mode: 'full_window' - 全窗口归一化（pretrained原始方式）
        #            'sliding_ma60' - 滑动MA60归一化（每个点看自己的前60步）
        #            'sliding_ma20' - 滑动MA20归一化（每个点看自己的前20步）
        past_len = self.config.lookback_window
        norm_mode = getattr(self.config, 'norm_mode', 'full_window')

        if norm_mode == 'full_window':
            # 全窗口归一化：使用整个lookback窗口的mean/std
            past_x = x[:past_len]
            x_mean = np.mean(past_x, axis=0)
            x_std = np.std(past_x, axis=0) + 1e-5
            x = (x - x_mean) / x_std
            x = np.clip(x, -self.config.clip, self.config.clip)

        elif norm_mode.startswith('sliding_ma'):
            # 滑动MA归一化（向量化实现）：每个点根据自己的前N步计算mean/std
            # 从norm_mode中提取窗口大小，如 'sliding_ma60' -> 60
            ma_window = int(norm_mode.split('_ma')[-1])

            # 使用 pandas rolling 实现向量化滑动计算
            # 注意：要看当前点"之前"的数据，用 shift(1) 排除当前点
            import pandas as pd
            df = pd.DataFrame(x)

            # shift(1) 让 rolling 只看前面N步（不包括当前点）
            df_shifted = df.shift(1)
            # ddof=0 与 numpy.std 默认行为一致
            rolling_mean = df_shifted.rolling(window=ma_window, min_periods=1).mean().values.copy()
            rolling_std = df_shifted.rolling(window=ma_window, min_periods=1).std(ddof=0).values.copy()

            # 第一个点没有历史数据，用自身作为mean，std=1e-5
            rolling_mean[0] = x[0]
            rolling_std[0] = 1e-5

            # 处理 std=0 或 NaN 的情况
            rolling_std = np.where(np.isnan(rolling_std), 1e-5, rolling_std)
            rolling_std = np.where(rolling_std < 1e-5, 1e-5, rolling_std)

            # 确保数据类型一致（float32）
            rolling_mean = rolling_mean.astype(np.float32)
            rolling_std = rolling_std.astype(np.float32)

            x_norm = (x - rolling_mean) / rolling_std
            x = np.clip(x_norm, -self.config.clip, self.config.clip)

        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}. Use 'full_window' or 'sliding_ma{N}'")

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        # === 返回用于方向损失的信息 ===
        # 原始 close 价格的涨跌方向
        # close 列是第 3 列（open, high, low, close, vol, amt）
        # window = lookback + predict + 1 = 411 个时间步
        # 索引: 0-399 (lookback), 400-409 (predict), 410 (extra for next token pred)
        #
        # 模型预测 pred_tokens[-10:] 对应 token_out 的位置 400-409
        # 所以模型预测的终点是时间步 410
        original_close = win_df['close'].values
        baseline_close = original_close[past_len - 1]  # 索引 399：lookback 窗口最后一点
        pred_close_end = original_close[past_len + self.config.predict_window]  # 索引 410：模型预测的终点

        # 涨跌方向：True = 涨，False = 跌（相对于买入价）
        direction = pred_close_end > baseline_close
        direction_tensor = torch.tensor(direction, dtype=torch.float32)

        return x_tensor, x_stamp_tensor, direction_tensor


if __name__ == '__main__':
    # Example usage and verification.
    from config import Config

    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train', config=Config())

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp, try_dir = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
        print(f"Sample direction: {try_dir}")
    else:
        print("Dataset is empty.")
