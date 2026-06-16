"""
Kronos 数据处理配置 - mode1_original

使用原始数据（未预先归一化），训练时动态计算 full_window 归一化。
数据来源：finetune/data/global_norm/full_series/
"""

import os


class Config:
    """数据处理配置"""

    # 数据集路径
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "finetune", "data", "global_norm", "full_series"
    )

    # 窗口参数
    lookback_window = 200  # mini 模型使用较短 lookback
    predict_window = 10

    # 最小样本数
    min_samples = lookback_window + predict_window  # 210

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