"""
Kronos Predictor Core Utils Module

通用工具函数

包含：
- seed 设置
- device 获取
- 时间格式化
- 文件操作
"""

import os
import random
import numpy as np
import torch
from datetime import datetime
from typing import Optional, Any


def set_seed(seed: int):
    """
    设置随机种子（全局）

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    获取计算设备

    Args:
        device_id: GPU ID（可选）

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        return torch.device('cuda')
    return torch.device('cpu')


def format_time(seconds: float) -> str:
    """
    格式化时间（秒 → HH:MM:SS）

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(ts: Any) -> str:
    """
    格式化时间戳

    Args:
        ts: datetime 或 pandas Timestamp

    Returns:
        ISO 格式字符串
    """
    if hasattr(ts, 'isoformat'):
        return ts.isoformat()
    return str(ts)


def get_model_size(model: torch.nn.Module) -> float:
    """
    获取模型参数量（百万）

    Args:
        model: PyTorch 模型

    Returns:
        参数量（M）
    """
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    计算模型参数数量

    Args:
        model: PyTorch 模型
        trainable_only: 仅计算可训练参数

    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# 文件操作
# ============================================================================

def ensure_dir(path: str) -> str:
    """
    确保目录存在

    Args:
        path: 目录路径或文件路径

    Returns:
        目录路径
    """
    dir_path = path if not os.path.splitext(path)[1] else os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


def safe_save_json(data: dict, path: str):
    """
    安全保存 JSON 文件（M6 修复：原子写入）

    写入临时文件，然后 rename，避免 crash 时损坏文件。

    Args:
        data: 数据字典
        path: 文件路径
    """
    import json
    import tempfile

    ensure_dir(path)

    # 写入临时文件
    temp_path = path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

    # Rename（原子操作）
    os.replace(temp_path, path)


def safe_load_json(path: str) -> dict:
    """
    安全加载 JSON 文件

    Args:
        path: 文件路径

    Returns:
        数据字典，文件不存在时返回空 dict
    """
    import json

    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_save_pickle(data: Any, path: str):
    """
    安全保存 pickle 文件（UT3 修复：原子写入）

    写入临时文件，然后 rename，避免 crash 时损坏文件。

    Args:
        data: 数据对象
        path: 文件路径
    """
    import pickle

    ensure_dir(path)

    # 写入临时文件
    temp_path = path + '.tmp'
    with open(temp_path, 'wb') as f:
        pickle.dump(data, f)

    # Rename（原子操作）
    os.replace(temp_path, path)


def safe_load_pickle(path: str) -> Any:
    """
    安全加载 pickle 文件

    Args:
        path: 文件路径

    Returns:
        数据对象，文件不存在时返回 None
    """
    import pickle

    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# 时间戳处理
# ============================================================================

def extract_time_features(timestamps: Any) -> np.ndarray:
    """
    从时间戳提取特征

    Args:
        timestamps: DatetimeIndex 或 datetime 列表

    Returns:
        (n, 5) 数组：[minute, hour, weekday, day, month]
    """
    import pandas as pd

    if not hasattr(timestamps, 'minute'):
        timestamps = pd.DatetimeIndex(timestamps)

    return np.stack([
        timestamps.minute.values.astype(np.float32),
        timestamps.hour.values.astype(np.float32),
        timestamps.weekday.values.astype(np.float32),
        timestamps.day.values.astype(np.float32),
        timestamps.month.values.astype(np.float32),
    ], axis=1)


# ============================================================================
# 进程信息
# ============================================================================

def get_rank_info() -> tuple:
    """
    获取 DDP rank 信息

    Returns:
        (rank, local_rank, world_size, use_ddp)
    """
    import torch.distributed as dist

    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # 单卡时不启用 DDP
        if world_size == 1:
            return 0, 0, 1, False

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_ddp():
    """
    清理 DDP
    """
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# 调试辅助
# ============================================================================

def debug_print(msg: str, rank: int = 0, is_main: bool = True):
    """
    仅主进程打印

    Args:
        msg: 消息
        rank: 当前 rank
        is_main: 是否主进程（可传入 rank == 0）
    """
    if is_main:
        print(msg)


def verbose_print(msg: str, verbose: bool = True):
    """
    条件打印

    Args:
        msg: 消息
        verbose: 是否打印
    """
    if verbose:
        print(msg)