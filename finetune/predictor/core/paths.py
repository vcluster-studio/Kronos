"""
Kronos Predictor Core Paths Module

路径构建函数

统一路径管理，避免路径语义混乱。

三类路径函数：
1. split_data: 训练/验证/测试数据
2. backtest: 回测数据
3. meta: 元数据
"""

import os
from typing import Optional

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_raw_path() -> str:
    """
    获取原始数据路径（统一位置）

    Returns:
        "data/kline_daily_raw.pkl"
    """
    return os.path.join(PROJECT_ROOT, "data/kline_daily_raw.pkl")


def get_split_data_path(
    norm_mode: str,
    lookback: int,
    predict: int,
    split_mode: str,
    split_name: str
) -> str:
    """
    获取分割数据路径

    Args:
        norm_mode: 归一化模式（full_window/sliding_ma{N}）
        lookback: 回看窗口
        predict: 预测步数
        split_mode: 分割模式（time/block）
        split_name: 分割名称（train/val/test）

    Returns:
        finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{split_name}.pkl

    说明:
        processed/ 按 norm_mode 键控的原因：
        - sliding_ma{N} 需要窗口起点前有 N 步历史
        - 不同 N 的可用样本集不同
        - full_window 不需要额外历史，但为统一结构也保留 norm_mode 键
    """
    return os.path.join(
        PROJECT_ROOT,
        f"finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{split_name}.pkl"
    )


def get_backtest_data_path(
    norm_mode: str,
    lookback: int,
    predict: int
) -> str:
    """
    获取回测数据路径

    Args:
        norm_mode: 归一化模式
        lookback: 回看窗口
        predict: 预测步数

    Returns:
        finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}/backtest/samples.pkl
    """
    return os.path.join(
        PROJECT_ROOT,
        f"finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}/backtest/samples.pkl"
    )


def get_meta_path(
    norm_mode: str,
    lookback: int,
    predict: int,
    split_mode: Optional[str] = None,
    role: str = "train"
) -> str:
    """
    获取元数据路径

    Args:
        norm_mode: 归一化模式
        lookback: 回看窗口
        predict: 预测步数
        split_mode: 分割模式（可选）
        role: 角色（train/backtest）

    Returns:
        meta.pkl 路径
    """
    base = os.path.join(
        PROJECT_ROOT,
        f"finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}"
    )
    if split_mode:
        return os.path.join(base, split_mode, "meta.pkl")
    elif role == "backtest":
        return os.path.join(base, "backtest", "meta.pkl")
    return os.path.join(base, "meta.pkl")


def get_model_path(
    norm_mode: str,
    lookback: int,
    predict: int,
    split_mode: str,
    model_type: str
) -> str:
    """
    获取模型保存路径

    Args:
        norm_mode: 归一化模式
        lookback: 回看窗口
        predict: 预测步数
        split_mode: 分割模式
        model_type: 模型类型（mini/small/base）

    Returns:
        outputs/models/{norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{model_type}
    """
    return os.path.join(
        PROJECT_ROOT,
        f"outputs/models/{norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{model_type}"
    )


def get_tokenizer_path(
    norm_mode: str,
    model_type: str
) -> str:
    """
    获取 tokenizer 路径

    Args:
        norm_mode: 归一化模式
        model_type: 模型类型（mini/small/base）

    Returns:
        outputs/tokenizers/{norm_mode}/{model_type}

    说明:
        vocab_size 映射：mini→2048, small→4096, base→8192
    """
    return os.path.join(
        PROJECT_ROOT,
        f"outputs/tokenizers/{norm_mode}/{model_type}"
    )


def get_checkpoint_path(
    model_path: str,
    checkpoint_name: str = "latest_model"
) -> str:
    """
    获取 checkpoint 路径

    Args:
        model_path: 模型基础路径
        checkpoint_name: checkpoint 名称（latest_model/best_model/best_ic_model/best_combined_model）

    Returns:
        checkpoint 目录路径
    """
    return os.path.join(model_path, "checkpoints", checkpoint_name)


def get_training_info_path(model_path: str) -> str:
    """
    获取 training_info.json 路径

    Args:
        model_path: 模型基础路径

    Returns:
        training_info.json 文件路径
    """
    return os.path.join(model_path, "training_info.json")


def get_summary_path(model_path: str) -> str:
    """
    获取 summary.json 路径

    Args:
        model_path: 模型基础路径

    Returns:
        summary.json 文件路径
    """
    return os.path.join(model_path, "summary.json")


def ensure_dir(path: str) -> str:
    """
    确保目录存在

    Args:
        path: 目录路径

    Returns:
        目录路径（已创建）
    """
    dir_path = path if not os.path.splitext(path)[1] else os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return path


# ============================================================================
# 兼容旧路径（临时，重构完成后废弃）
# ============================================================================

def get_legacy_data_path(
    norm_mode: str,
    lookback: int,
    split_name: str
) -> str:
    """
    获取旧格式数据路径（兼容现有数据）

    Args:
        norm_mode: 'ma60' | 'global_norm'
        lookback: 回看窗口
        split_name: train/val/test/final_test

    Returns:
        旧格式路径
    """
    if norm_mode == 'ma60':
        # MA60 预归一化数据
        return os.path.join(
            PROJECT_ROOT,
            f"finetune/data/ma60_norm/block_lb{lookback}_pd10/{split_name}_data.pkl"
        )
    elif norm_mode == 'global_norm':
        # Full window 归一化数据
        return os.path.join(
            PROJECT_ROOT,
            f"finetune/data/global_norm/full_series/{split_name}_data.pkl"
        )
    else:
        raise ValueError(f"Unknown legacy norm_mode: {norm_mode}")