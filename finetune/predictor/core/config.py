"""
Kronos Predictor Core Config Module

配置对象定义与校验

拆分为三类：
- DataConfig: 数据相关配置（不变量）
- TrainConfig: 训练超参
- ArtifactConfig: 模型/Tokenizer 路径
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """
    数据配置 - 不变量

    不变量约束：
    - min_samples == lookback + predict（禁止 +1，防止泄露）
    - norm_mode 必须在白名单内
    """
    norm_mode: str = 'sliding_ma60'
    lookback: int = 400
    predict: int = 10
    split_mode: str = 'block'  # 'time' | 'block'
    samples_per_block: int = 100  # 每块窗口数（stride=1）
    # block_size 由 __post_init__ 自动计算：window_size + samples_per_block - 1
    block_size: int = field(init=False)
    min_samples: int = field(init=False)
    features: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'vol', 'amt'])
    time_features: List[str] = field(default_factory=lambda: ['minute', 'hour', 'weekday', 'day', 'month'])
    clip: float = 5.0
    seed: int = 42

    # norm_mode 白名单
    VALID_NORM_MODES = ['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120']

    def __post_init__(self):
        """初始化后自动计算 block_size 和 min_samples 并校验"""
        window_size = self.lookback + self.predict
        self.block_size = window_size + self.samples_per_block - 1  # 块内正好 samples_per_block 个窗口
        self.min_samples = window_size  # 禁止 +1
        self.validate()

    def validate(self):
        """校验配置约束"""
        # 不变量校验
        assert self.min_samples == self.lookback + self.predict, \
            f"min_samples ({self.min_samples}) != lookback + predict ({self.lookback + self.predict})"

        # norm_mode 白名单
        assert self.norm_mode in self.VALID_NORM_MODES, \
            f"norm_mode '{self.norm_mode}' not in {self.VALID_NORM_MODES}"

        # lookback/predict 正数约束
        assert self.lookback > 0, "lookback must be positive"
        assert self.predict > 0, "predict must be positive"

        # block_size 约束：必须 ≥ window_size，否则块内放不下一个完整窗口
        # block_size = window_size + samples_per_block - 1，由 __post_init__ 计算
        window_size = self.lookback + self.predict
        assert self.block_size >= window_size, \
            f"block_size ({self.block_size}) < window_size ({window_size})，块内放不下一个完整窗口"
        # 验证窗口数
        actual_windows = self.block_size - window_size + 1
        assert actual_windows == self.samples_per_block, \
            f"块内窗口数 {actual_windows} != samples_per_block {self.samples_per_block}"

    def get_window_size(self) -> int:
        """获取完整窗口大小（不含泄露）"""
        return self.lookback + self.predict

    def get_norm_window(self) -> Optional[int]:
        """获取归一化窗口大小（仅 sliding_ma 模式）"""
        if self.norm_mode == 'full_window':
            return None
        elif self.norm_mode.startswith('sliding_ma'):
            return int(self.norm_mode.split('_ma')[-1])
        return None


@dataclass
class TrainConfig:
    """
    训练超参

    Early stopping patience审视（§1.6.7）：
    - 当前 patience=12，归零条件包含「val_loss 改进」和「IC 改进」两条途径
    - 去趋势口径下 IC 量级和波动结构变化，patience 需重新审视
    - Phase -1 用正确口径跑 IC 曲线后确认 patience 值
    - Phase 4 不要求停止 epoch 与旧训练一致
    """
    model_type: str = 'mini'  # 'mini' | 'small' | 'base'
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01  # 从大值起步（feedback_learning_rate）
    weight_decay: float = 0.01
    early_stopping_patience: int = 12  # 需在去趋势口径 IC 曲线上重新确认
    early_stopping_grace_period: int = 8
    warmup_epochs: int = 2
    lr_min: float = 1e-5

    # IC 归零条件（§1.6.7）
    # patience_counter 归零有两种途径：
    # 1. val_loss 改进 → patience 归零
    # 2. IC 改进 → patience 归零
    # Phase -1 需决定是否保留第二条途径
    ic_patience_reset: bool = True  # 是否允许 IC 改进归零 patience

    # IC 权重（用于 checkpoint 选择）
    # 当前 combined_score = IC_norm * 0.6 + DA * 0.4
    # 去趋势后 IC 量级变化，需在 Phase -1 用正确口径重新标定
    combined_ic_weight: float = 0.6
    combined_da_weight: float = 0.4

    # 模型路径映射
    MODEL_PATHS = {
        'mini': 'pretrained/Kronos-mini',
        'small': 'pretrained/Kronos-small',
        'base': 'pretrained/Kronos-base',
    }

    MAX_CONTEXT = {
        'mini': 2048,
        'small': 512,
        'base': 512,
    }

    def get_model_path(self) -> str:
        """获取模型路径"""
        return self.MODEL_PATHS[self.model_type]

    def get_max_context(self) -> int:
        """获取最大上下文长度"""
        return self.MAX_CONTEXT[self.model_type]


@dataclass
class ArtifactConfig:
    """
    模型/Tokenizer 路径配置

    S2 修复：model_type 字段必须在类中声明
    """
    model_type: str = 'mini'  # 新增（S2 修复）
    pretrained_model_path: str = 'pretrained/Kronos-mini'
    tokenizer_path: Optional[str] = None
    output_dir: Optional[str] = None

    def resolve_tokenizer_path(self, data_config: DataConfig) -> str:
        """
        根据 norm_mode 和 model_type 自动选择 tokenizer

        路径格式：outputs/tokenizers/{norm_mode}/{model_type}

        T3 修复：架构映射（非 vocab 映射）
        - mini → Kronos-Tokenizer-2k（group_size=5, context=2048）
        - small/base → Kronos-Tokenizer-base（group_size=4, context=512）
        vocab_size 由预训练架构决定，不是配置参数
        """
        if self.tokenizer_path:
            return self.tokenizer_path
        # 统一使用 model_type 作为键
        return f"outputs/tokenizers/{data_config.norm_mode}/{self.model_type}"

    def resolve_pretrained_path(self) -> str:
        """根据 model_type 自动选择预训练模型"""
        paths = {
            'mini': 'pretrained/Kronos-mini',
            'small': 'pretrained/Kronos-small',
            'base': 'pretrained/Kronos-base',
        }
        return paths.get(self.model_type, 'pretrained/Kronos-mini')


@dataclass
class BacktestConfig:
    """
    回测配置

    E7 修复：sigmoid 魔数移入配置并注释来源
    """
    signal_center: float = 0.084   # sigmoid 中心：涨幅 8.4% 处分值=0.5（经验标定）
    signal_steepness: float = 21.0  # sigmoid 陡度：控制分值对涨幅的敏感度（经验标定）
    limit_pct: float = 0.10         # 涨跌停阈值：主板 10%，创业板 20%

    def sigmoid_score(self, gain: float) -> float:
        """将涨幅转换为 sigmoid 分值"""
        import numpy as np
        return 1 / (1 + np.exp(-self.signal_steepness * (gain - self.signal_center)))


def parse_norm_mode(norm_mode: str) -> dict:
    """
    解析 norm_mode 返回归一化配置

    Args:
        norm_mode: 'full_window' | 'sliding_ma{N}'

    Returns:
        {'method': 'full_window' | 'sliding_ma', 'window': None | N}
    """
    if norm_mode == 'full_window':
        return {'method': 'full_window', 'window': None}
    elif norm_mode.startswith('sliding_ma'):
        window = int(norm_mode.split('_ma')[-1])
        return {'method': 'sliding_ma', 'window': window}
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")