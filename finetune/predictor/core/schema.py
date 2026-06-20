"""
Kronos Predictor Core Schema Module

数据 Schema 定义

定义样本、回测、元数据的数据结构。
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


# ============================================================================
# 样本 Schema
# ============================================================================

@dataclass
class SampleSchema:
    """
    样本数据结构

    核心：target 区间定义（用于分割和泄露检查）

    字段说明：
    - symbol: 股票代码
    - window_start: 窗口在原始序列中的起始位置
    - lookback_start/end: lookback 区间边界
    - target_start/end: target 区间边界（关键）
    - split: 所属分割（train/val/test）
    - values: 原始 OHLCV 数据窗口
    - index: 时间戳序列
    """
    symbol: str
    window_start: int
    lookback_start: int
    lookback_end: int  # = window_start + lookback
    target_start: int  # = lookback_end
    target_end: int    # = target_start + predict
    split: str  # 'train' | 'val' | 'test'

    # block 归一化用：记录所属 block 在原始序列的起点（block 模式填，time 模式 None）
    # block 内归一化时，窗口的归一化数据从 block_start 起的段取，不跨 block/split
    block_start: Optional[int] = None

    # 数据（可选，加载时填充）
    values: Optional[np.ndarray] = None  # (window_size, n_features)
    index: Optional[Any] = None  # DatetimeIndex

    def validate(self):
        """校验样本结构"""
        # 区间边界校验
        assert self.lookback_end == self.window_start + (self.lookback_end - self.lookback_start), \
            "lookback_end 计算错误"
        assert self.target_start == self.lookback_end, \
            "target_start 应等于 lookback_end"
        assert self.target_end > self.target_start, \
            "target_end 应大于 target_start"

        # 禁止 +1（防止泄露）
        window_size = self.target_end - self.window_start
        lookback = self.lookback_end - self.lookback_start
        predict = self.target_end - self.target_start
        assert window_size == lookback + predict, \
            f"window_size ({window_size}) != lookback ({lookback}) + predict ({predict})"

    @property
    def window_size(self) -> int:
        """完整窗口大小"""
        return self.target_end - self.window_start

    @property
    def lookback(self) -> int:
        """lookback 长度"""
        return self.lookback_end - self.lookback_start

    @property
    def predict(self) -> int:
        """predict 长度"""
        return self.target_end - self.target_start


def sample_to_dict(sample: SampleSchema) -> Dict[str, Any]:
    """将 SampleSchema 转为字典（用于 pickle）"""
    return {
        'symbol': sample.symbol,
        'window_start': sample.window_start,
        'lookback_start': sample.lookback_start,
        'lookback_end': sample.lookback_end,
        'target_start': sample.target_start,
        'target_end': sample.target_end,
        'split': sample.split,
        'block_start': sample.block_start,
        'values': sample.values,
        'index': sample.index,
    }


def dict_to_sample(d: Dict[str, Any]) -> SampleSchema:
    """将字典转为 SampleSchema"""
    return SampleSchema(**d)


# ============================================================================
# Backtest Schema
# ============================================================================

@dataclass
class BacktestSchema:
    """
    回测样本数据结构

    核心：context/target 分离，actual_target 不入模型

    字段说明：
    - symbol: 股票代码
    - context_start/end: lookback 区间边界
    - target_start/end: predict 区间边界
    - x_context: lookback 原始数据
    - y_timestamp: predict 时间戳
    - actual_target: predict 原始数据（只用于评估，不进入模型）
    - normalization_meta: 归一化元数据

    断言：
    - context_end < target_start（lookback/target 边界正确）
    - normalizer.fit_range <= context_end（normalizer 不使用未来数据）
    """
    symbol: str
    context_start: int
    context_end: int
    target_start: int
    target_end: int

    # 数据
    x_context: np.ndarray  # (lookback, n_features) 原始数据
    y_timestamp: Any       # DatetimeIndex，predict 时间戳
    actual_target: np.ndarray  # (predict, n_features) 原始数据（不入模型）
    normalization_meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        """校验回测结构"""
        # 边界校验
        assert self.context_end < self.target_start, \
            f"context_end ({self.context_end}) >= target_start ({self.target_start})"

        # normalizer 校验
        fit_range = self.normalization_meta.get('fit_range', self.context_end)
        assert fit_range <= self.context_end, \
            f"normalizer fit_range ({fit_range}) > context_end ({self.context_end})，使用了未来数据"


# ============================================================================
# Meta Schema
# ============================================================================

@dataclass
class MetaSchema:
    """
    元数据结构

    用于记录分割边界、数据指纹、泄露检查结果
    """
    norm_mode: str
    lookback: int
    predict: int
    split_mode: str  # 'time' | 'block'

    # 分割边界（时间分割）
    train_end: Optional[int] = None  # target_end <= train_end
    val_end: Optional[int] = None    # target_end <= val_end

    # 数据指纹
    data_fingerprint: Optional[str] = None  # sha256 hash

    # 样本数
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    # 泄露检查结果
    leakage_check_passed: bool = False
    leakage_check_timestamp: Optional[str] = None

    # 创建时间
    created_at: Optional[str] = None


@dataclass
class TrainingInfoSchema:
    """
    训练信息结构（training_info.json）

    实时更新，每个 epoch 完成时写入
    """
    status: str  # 'running' | 'completed' | 'early_stopped'
    start_time: str

    # 配置
    config: Dict[str, Any]

    # 数据
    data: Dict[str, Any] = field(default_factory=dict)

    # Tokenizer
    tokenizer: Dict[str, Any] = field(default_factory=dict)

    # Epoch 记录
    epochs: List[Dict[str, Any]] = field(default_factory=list)

    # 最佳结果
    best: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于 JSON 序列化）"""
        return {
            'status': self.status,
            'start_time': self.start_time,
            'config': self.config,
            'data': self.data,
            'tokenizer': self.tokenizer,
            'epochs': self.epochs,
            'best': self.best,
        }


# ============================================================================
# 区间辅助函数
# ============================================================================

def intervals_overlap(intervals_a: List[Tuple[int, int]],
                      intervals_b: List[Tuple[int, int]]) -> bool:
    """
    检查两组区间是否有相交

    区间 [s, e) 不相交: e1 <= s2 或 e2 <= s1

    Args:
        intervals_a: [(start, end)] 区间列表
        intervals_b: [(start, end)] 区间列表

    Returns:
        True 如果有相交，False 如果不相交
    """
    for s1, e1 in intervals_a:
        for s2, e2 in intervals_b:
            # 区间相交: NOT (e1 <= s2 OR e2 <= s1)
            if not (e1 <= s2 or e2 <= s1):
                return True
    return False


def compute_fingerprint(data: Any) -> str:
    """
    计算数据指纹（用于一致性校验）

    Args:
        data: 可序列化的数据

    Returns:
        sha256 fingerprint
    """
    import hashlib
    import pickle

    data_bytes = pickle.dumps(data)
    return hashlib.sha256(data_bytes).hexdigest()