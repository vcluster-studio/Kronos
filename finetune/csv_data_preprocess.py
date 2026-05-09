"""
CSV 数据预处理脚本

将 finetune_csv/exported_kline_data/stocks/ 目录下的 CSV 文件
直接转换为训练所需的 pkl 格式，无需转 Qlib 格式。

用法：
    python finetune/csv_data_preprocess.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import trange

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from finetune.config import Config


class CSVDataPreprocessor:
    """
    将 CSV 格式的 K 线数据转换为训练所需的 pkl 格式。

    过滤策略（策略 A：纯主板模型）：
    1. 只保留主板股票（000/002/600/601/603 开头）
    2. 排除 ST 股票（通过涨跌幅推断）
    3. 排除次新股（上市不足 1 年）
    4. 排除流动性差的股票（日均成交额 < 5000 万）
    5. 排除停牌过多的股票（缺失数据 > 30%）
    """

    def __init__(self):
        self.config = Config()
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']

        # 过滤参数
        self.min_amount = 50_000_000  # 日均成交额阈值：5000 万
        self.min_list_days = 250       # 最小上市天数：约 1 年
        self.max_gap_ratio = 0.30      # 最大允许缺失比例

    def is_main_board(self, symbol: str) -> bool:
        """
        判断是否为主板股票。

        主板代码规则：
        - 000xxx, 001xxx, 002xxx, 003xxx: 深市主板（含原中小板）
        - 600xxx, 601xxx, 603xxx, 605xxx: 沪市主板

        排除：
        - 300xxx, 301xxx: 创业板（±20% 涨跌停）
        - 688xxx: 科创板（±20% 涨跌停）
        - 8xxxxx: 北交所
        """
        code = symbol.split('.')[0]  # 提取代码部分，如 '000001'

        # 深市主板
        if code.startswith('000') or code.startswith('001') or \
           code.startswith('002') or code.startswith('003'):
            return True

        # 沪市主板
        if code.startswith('600') or code.startswith('601') or \
           code.startswith('603') or code.startswith('605'):
            return True

        return False

    def is_likely_st(self, df: pd.DataFrame) -> bool:
        """
        通过涨跌幅推断是否为 ST 股票。

        ST 股票特征：涨跌停限制为 ±5%
        如果连续出现多日涨跌幅在 ±5%~±6% 之间，很可能是 ST
        """
        # 计算日涨跌幅
        pct_change = df['close'].pct_change().abs() * 100

        # 检查是否有频繁触及 ±5% 边界的情况
        # 正常股票涨跌停是 ±10%，ST 是 ±5%
        # 如果涨跌幅多次在 4.5%~5.5% 区间，疑似 ST
        near_st_limit = (pct_change > 4.5) & (pct_change < 6.0)
        st_suspicion_rate = near_st_limit.sum() / max(len(pct_change.dropna()), 1)

        # 如果超过 10% 的交易日疑似触及 ST 涨跌停，判定为 ST
        return st_suspicion_rate > 0.10

    def load_csv_data(self, csv_dir: str) -> dict:
        """
        加载 CSV 目录下的所有股票数据。
        """
        print(f"Loading CSV files from: {csv_dir}")

        data = {}
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

        # 先按主板过滤
        main_board_files = []
        excluded_board = []
        for csv_file in csv_files:
            symbol = csv_file.replace('.csv', '')
            if self.is_main_board(symbol):
                main_board_files.append(csv_file)
            else:
                excluded_board.append(symbol[:3])

        print(f"Found {len(csv_files)} CSV files")
        print(f"  - Main board: {len(main_board_files)}")
        print(f"  - Excluded (non-main board): {len(csv_files) - len(main_board_files)}")

        for i in trange(len(main_board_files), desc="Loading CSVs"):
            csv_file = main_board_files[i]
            symbol = csv_file.replace('.csv', '')
            file_path = os.path.join(csv_dir, csv_file)

            try:
                df = pd.read_csv(file_path)

                # 转换日期列
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df.index.name = 'datetime'

                # 重命名列
                df = df.rename(columns={'volume': 'vol'})

                # 选择并重排列
                df = df[['open', 'high', 'low', 'close', 'vol', 'amount']]
                df = df.rename(columns={'amount': 'amt'})

                # 删除缺失值
                df = df.dropna()

                if len(df) < self.config.lookback_window + self.config.predict_window + 1:
                    continue

                data[symbol] = df

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue

        print(f"Successfully loaded {len(data)} main board stocks")
        return data

    def filter_by_quality(self, data: dict) -> dict:
        """
        数据质量过滤。
        """
        print("\n" + "=" * 60)
        print("Filtering data by quality...")
        print("=" * 60)
        print(f"  - Min average daily amount: {self.min_amount/1e8:.2f} 亿")
        print(f"  - Min listing days: {self.min_list_days}")
        print(f"  - Max gap ratio: {self.max_gap_ratio * 100:.0f}%")

        filtered_data = {}
        stats = {'st': 0, 'new': 0, 'low_liquidity': 0, 'gap': 0, 'passed': 0}

        symbols = list(data.keys())
        for i in trange(len(symbols), desc="Quality filtering"):
            symbol = symbols[i]
            df = data[symbol]

            # 1. 检测 ST 股票
            if self.is_likely_st(df):
                stats['st'] += 1
                continue

            # 2. 检查上市时间（次新股过滤）
            # 使用数据时间跨度作为代理
            data_days = len(df)
            if data_days < self.min_list_days:
                stats['new'] += 1
                continue

            # 3. 检查流动性（日均成交额）
            avg_amount = df['amt'].mean()
            if avg_amount < self.min_amount:
                stats['low_liquidity'] += 1
                continue

            # 4. 检查数据缺失（停牌）
            date_range = (df.index[-1] - df.index[0]).days
            expected_trading_days = date_range * 5 / 7  # 粗略估算
            actual_trading_days = len(df)
            gap_ratio = 1 - actual_trading_days / max(expected_trading_days, 1)

            if gap_ratio > self.max_gap_ratio:
                stats['gap'] += 1
                continue

            filtered_data[symbol] = df
            stats['passed'] += 1

        print("\n" + "-" * 60)
        print("Filtering results:")
        print(f"  - Excluded (suspected ST):          {stats['st']:5d}")
        print(f"  - Excluded (data < {self.min_list_days} days):       {stats['new']:5d}")
        print(f"  - Excluded (amount < {self.min_amount/1e8:.1f}亿):    {stats['low_liquidity']:5d}")
        print(f"  - Excluded (gap > {self.max_gap_ratio*100:.0f}%):            {stats['gap']:5d}")
        print(f"  - Passed:                           {stats['passed']:5d}")
        print("-" * 60)

        return filtered_data

    def prepare_dataset(self, data: dict):
        """
        按时间范围划分数据集并保存。
        同时计算股票分类（大盘/中盘/小盘）用于分层采样。
        """
        print("\n" + "=" * 60)
        print("Splitting data into train, validation, and test sets...")
        print("=" * 60)

        train_data, val_data, test_data = {}, {}, {}

        # 时间范围
        train_start, train_end = self.config.train_time_range
        val_start, val_end = self.config.val_time_range
        test_start, test_end = self.config.test_time_range

        print(f"  Train: {train_start} ~ {train_end}")
        print(f"  Val:   {val_start} ~ {val_end}")
        print(f"  Test:  {test_start} ~ {test_end}")

        min_samples = self.config.lookback_window + self.config.predict_window + 1

        # === 计算股票分类（按成交额） ===
        stock_categories = {}  # {symbol: 'large'/'mid'/'small'}
        category_stats = {'large': 0, 'mid': 0, 'small': 0}

        symbols = list(data.keys())
        for i in trange(len(symbols), desc="Computing stock categories"):
            symbol = symbols[i]
            df = data[symbol]

            # 计算平均日成交额
            avg_amount = df['amt'].mean()

            # 分类标准
            if avg_amount > 1e9:  # > 10亿
                stock_categories[symbol] = 'large'
                category_stats['large'] += 1
            elif avg_amount >= 1e8:  # 1-10亿
                stock_categories[symbol] = 'mid'
                category_stats['mid'] += 1
            else:  # < 1亿
                stock_categories[symbol] = 'small'
                category_stats['small'] += 1

        print("\n" + "-" * 60)
        print("Stock category distribution (by avg daily amount):")
        print(f"  Large cap (> 10亿):  {category_stats['large']:5d} stocks")
        print(f"  Mid cap (1-10亿):    {category_stats['mid']:5d} stocks")
        print(f"  Small cap (< 1亿):   {category_stats['small']:5d} stocks")
        print("-" * 60)

        # 划分数据集
        for i in trange(len(symbols), desc="Splitting datasets"):
            symbol = symbols[i]
            df = data[symbol]

            # 创建时间掩码
            train_mask = (df.index >= train_start) & (df.index <= train_end)
            val_mask = (df.index >= val_start) & (df.index <= val_end)
            test_mask = (df.index >= test_start) & (df.index <= test_end)

            # 应用掩码
            train_df = df[train_mask]
            val_df = df[val_mask]
            test_df = df[test_mask]

            # 只保留有足够数据的股票
            if len(train_df) >= min_samples:
                train_data[symbol] = train_df
            if len(val_df) >= min_samples:
                val_data[symbol] = val_df
            if len(test_df) >= min_samples:
                test_data[symbol] = test_df

        # 统计信息
        train_samples = sum(len(df) for df in train_data.values())
        val_samples = sum(len(df) for df in val_data.values())
        test_samples = sum(len(df) for df in test_data.values())

        print("\n" + "-" * 60)
        print("Dataset statistics:")
        print(f"  Train: {len(train_data):5d} stocks, {train_samples:10d} samples")
        print(f"  Val:   {len(val_data):5d} stocks, {val_samples:10d} samples")
        print(f"  Test:  {len(test_data):5d} stocks, {test_samples:10d} samples")
        print("-" * 60)

        # 保存数据集
        os.makedirs(self.config.dataset_path, exist_ok=True)

        with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        # === 保存股票分类信息 ===
        with open(f"{self.config.dataset_path}/stock_categories.pkl", 'wb') as f:
            pickle.dump(stock_categories, f)
        print(f"Stock categories saved to: {self.config.dataset_path}/stock_categories.pkl")

        print(f"\nDatasets saved to: {self.config.dataset_path}")

    def run(self, csv_dir: str):
        """
        运行完整的数据预处理流程。
        """
        print("\n" + "=" * 60)
        print("Kronos A-Share Data Preprocessing")
        print("Strategy: Pure Main Board Model")
        print("=" * 60)

        # 1. 加载 CSV 数据
        data = self.load_csv_data(csv_dir)

        # 2. 数据质量过滤
        data = self.filter_by_quality(data)

        # 3. 划分数据集并保存
        self.prepare_dataset(data)

        print("\n" + "=" * 60)
        print("Data preprocessing completed!")
        print("=" * 60)


if __name__ == '__main__':
    # CSV 数据目录（从项目根目录计算）
    # 获取项目根目录（finetune 的上级目录）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = os.path.join(project_root, "finetune_csv", "exported_kline_data", "stocks")

    print(f"Project root: {project_root}")
    print(f"CSV directory: {csv_dir}")

    # 检查目录是否存在
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    # 运行预处理
    preprocessor = CSVDataPreprocessor()
    preprocessor.run(csv_dir)
