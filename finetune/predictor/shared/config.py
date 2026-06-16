"""
Kronos 数据处理配置

时间划分策略：
- 训练集：2018-01-01 ~ 2023-12-31（约6年）
- 验证集：2024-01-01 ~ 2025-12-31（约2年，约500步）
- 测试集：2026-01-01 ~ 2026-05-07（约4个月）

验证集需要足够长（>=411步），以支持 lookback=400 的训练。
"""

import os


class Config:
    """数据处理配置"""

    # 数据集路径
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "finetune", "data", "processed_datasets"
    )

    # 时间范围划分（调整为更长验证集）
    train_time_range = ('2018-01-01', '2023-06-30')  # 约5.5年
    val_time_range = ('2023-07-01', '2024-12-31')    # 约1.5年，约370步
    test_time_range = ('2025-01-01', '2026-05-07')   # 约1.3年

    # 窗口参数
    lookback_window = 400  # 2k tokenizer 需要更长上下文
    predict_window = 10

    # 最小样本数
    min_samples = lookback_window + predict_window  # 410（仅 lookback+pred，无 +1 泄漏）

    # 归一化模式
    # 'full_window' - 全窗口归一化（pretrained原始方式）：使用整个lookback窗口的mean/std
    # 'sliding_ma60' - 滑动MA60归一化：每个点根据自己前60步计算MA
    norm_mode = 'full_window'  # 默认使用全窗口归一化（与pretrained一致）

    # 特征列表
    feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
    time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

    # 训练参数
    clip = 5.0  # 归一化裁剪值
    seed = 42   # 随机种子