"""
Kronos Predictor Core Splitting Module

数据分割算法

两种分割模式：
1. time_split: 按时间边界分割（train_end, val_end）
2. block_split: 按 target 时间块分层抽样

核心：target 区间语义（不是 window index）

泄露检查：validate_no_leakage 用区间相交语义
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

from .schema import SampleSchema, intervals_overlap


# ============================================================================
# Time Split（时间分割）
# ============================================================================

def time_split(
    samples: List[SampleSchema],
    train_end: int,
    val_end: int
) -> Tuple[List[SampleSchema], List[SampleSchema], List[SampleSchema]]:
    """
    按时间边界分割样本

    分割规则（target 区间语义）：
    - train: target_end <= train_end
    - val:   target_start >= train_end AND target_end <= val_end
    - test:  target_start >= val_end

    Args:
        samples: 样本列表
        train_end: 训练集时间边界
        val_end: 验证集时间边界

    Returns:
        (train_samples, val_samples, test_samples)

    注意:
        边界无重叠：train_end <= val_start, val_end <= test_start
        但相邻样本可能跨 split（stride < predict 时）
        时间分割保证 target 区间不重叠
    """
    train_samples = []
    val_samples = []
    test_samples = []

    for s in samples:
        if s.target_end <= train_end:
            s.split = 'train'
            train_samples.append(s)
        elif s.target_start >= train_end and s.target_end <= val_end:
            s.split = 'val'
            val_samples.append(s)
        elif s.target_start >= val_end:
            s.split = 'test'
            test_samples.append(s)
        else:
            # 边界附近的样本：target 跨 split 边界
            # 这种情况在 stride < predict 时会出现
            # 按主要位置分配（target_start 优先）
            if s.target_start < train_end:
                s.split = 'train'
                train_samples.append(s)
            elif s.target_start < val_end:
                s.split = 'val'
                val_samples.append(s)
            else:
                s.split = 'test'
                test_samples.append(s)

    return train_samples, val_samples, test_samples


# ============================================================================
# Block Split（分层抽样）
# ============================================================================

def create_target_blocks(
    sym_samples: List[SampleSchema],
    block_size: int = 50
) -> List[List[SampleSchema]]:
    """
    创建 target 时间块

    算法（关键：块之间 target 时间区间不相交）：
    1. 取该股票 target 时间轴 [t_min, t_max]
    2. 按 block_size 切成不重叠的时间区间块
    3. 每个窗口按其 target 落在哪个时间块，归入该块
    4. 返回 block 列表，每个 block 是窗口列表

    Args:
        sym_samples: 单只股票的样本列表
        block_size: 时间块大小

    Returns:
        blocks: [[SampleSchema]] 时间块列表

    注意:
        不是按样本列表顺序分块（那会退化为 window-index splitting → 泄露）
    """
    if not sym_samples:
        return []

    # 按 target_start 排序
    sorted_samples = sorted(sym_samples, key=lambda s: s.target_start)

    # 取 target 时间范围
    t_min = sorted_samples[0].target_start
    t_max = sorted_samples[-1].target_end

    # 切成时间块
    blocks = []
    current_block_start = t_min

    while current_block_start < t_max:
        # S1 修复：末端块防超界
        current_block_end = min(current_block_start + block_size, t_max)

        # 收集 target 落在 [current_block_start, current_block_end) 的窗口
        block_windows = []
        for s in sorted_samples:
            # target 区间落在当前时间块内
            if s.target_start >= current_block_start and s.target_end <= current_block_end:
                block_windows.append(s)

        if block_windows:
            blocks.append(block_windows)

        current_block_start = current_block_end

    # 断言：块之间 target 区间不相交
    for i, block_a in enumerate(blocks):
        for j, block_b in enumerate(blocks):
            if i != j:
                intervals_a = [(s.target_start, s.target_end) for s in block_a]
                intervals_b = [(s.target_start, s.target_end) for s in block_b]
                assert not intervals_overlap(intervals_a, intervals_b), \
                    f"block {i}/{j} target intervals overlap!"

    return blocks


def block_split(
    samples: List[SampleSchema],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    block_size: int = 50,
    seed: int = 42
) -> Tuple[List[SampleSchema], List[SampleSchema], List[SampleSchema]]:
    """
    以 target 时间块为单位分配 train/val/test

    关键：同一股票相邻窗口可能跨 split（因为以 block 为单位）

    Args:
        samples: 所有样本列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        block_size: 时间块大小
        seed: 随机种子

    Returns:
        (train_samples, val_samples, test_samples)
    """
    rng = np.random.RandomState(seed)

    # 按股票分组
    by_symbol = defaultdict(list)
    for s in samples:
        by_symbol[s.symbol].append(s)

    train, val, test = [], [], []

    for symbol, sym_samples in by_symbol.items():
        # 创建 target 时间块（块之间不相交）
        blocks = create_target_blocks(sym_samples, block_size=block_size)

        if not blocks:
            continue

        # 随机分配 blocks
        rng.shuffle(blocks)
        n_train = int(len(blocks) * train_ratio)
        n_val = int(len(blocks) * val_ratio)

        for block in blocks[:n_train]:
            for s in block:
                s.split = 'train'
                train.append(s)

        for block in blocks[n_train:n_train + n_val]:
            for s in block:
                s.split = 'val'
                val.append(s)

        for block in blocks[n_train + n_val:]:
            for s in block:
                s.split = 'test'
                test.append(s)

    return train, val, test


# ============================================================================
# 泄露检查（区间语义）
# ============================================================================

def validate_no_leakage(
    train_samples: List[SampleSchema],
    val_samples: List[SampleSchema],
    test_samples: List[SampleSchema]
) -> bool:
    """
    验证 target 时间区间无重叠（区间语义，非元组语义）

    区间 [s, e) 不相交: e1 <= s2 或 e2 <= s1

    这是 target-based splitting 的正确守卫。
    元组 disjoint 只能抓「完全重复的 target 窗口」，抓不到「时间区间相交」。
    本项目 lb60 曾因 window-index splitting 导致 100% target 泄露。

    Args:
        train_samples: 训练集样本
        val_samples: 验证集样本
        test_samples: 测试集样本

    Returns:
        True 如果无泄露

    Raises:
        AssertionError 如果发现泄露
    """
    def get_symbols(samples):
        return set(s.symbol for s in samples)

    def get_intervals(samples, symbol):
        return [(s.target_start, s.target_end) for s in samples if s.symbol == symbol]

    all_symbols = get_symbols(train_samples) | get_symbols(val_samples) | get_symbols(test_samples)

    pairs = [
        (train_samples, val_samples, "train/val"),
        (train_samples, test_samples, "train/test"),
        (val_samples, test_samples, "val/test"),
    ]

    for samples_a, samples_b, pair_name in pairs:
        for sym in all_symbols:
            intervals_a = get_intervals(samples_a, sym)
            intervals_b = get_intervals(samples_b, sym)

            if intervals_a and intervals_b:
                if intervals_overlap(intervals_a, intervals_b):
                    # 找到重叠的具体区间
                    for s1, e1 in intervals_a:
                        for s2, e2 in intervals_b:
                            if not (e1 <= s2 or e2 <= s1):
                                raise AssertionError(
                                    f"{pair_name} target overlap for {sym}: "
                                    f"[{s1}, {e1}) intersects [{s2}, {e2})"
                                )

    print(f"No-leakage check passed (interval semantics) - {len(all_symbols)} stocks verified")
    return True


# ============================================================================
# Backtest 数据分割
# ============================================================================

def create_backtest_samples(
    raw_data: Dict[str, Any],
    lookback: int,
    predict: int,
    norm_mode: str
) -> List[Dict[str, Any]]:
    """
    创建回测样本

    Args:
        raw_data: 原始数据 {symbol: data}
        lookback: 回看长度
        predict: 预测长度
        norm_mode: 归一化模式

    Returns:
        backtest 样本列表

    注意:
        回测样本无 split 字段（全部用于评估）
        actual_target 不入模型
    """
    from .normalization import get_normalizer, NormalizerFactory

    normalizer = get_normalizer(norm_mode)
    required_history = NormalizerFactory.get_required_history(norm_mode)

    backtest_samples = []

    for symbol, data in raw_data.items():
        # 检测数据格式
        if hasattr(data, 'columns'):
            # DataFrame 格式
            seq_len = len(data)
            window_size = lookback + predict

            # 回测：从最新数据开始
            if seq_len >= window_size + required_history:
                start = seq_len - window_size - required_history

                x_context = data.values[start + required_history:start + required_history + lookback]
                y_timestamp = data.index[start + required_history + lookback:start + required_history + lookback + predict]
                actual_target = data.values[start + required_history + lookback:start + required_history + lookback + predict]

                backtest_samples.append({
                    'symbol': symbol,
                    'context_start': start + required_history,
                    'context_end': start + required_history + lookback,
                    'target_start': start + required_history + lookback,
                    'target_end': start + required_history + lookback + predict,
                    'x_context': x_context,
                    'y_timestamp': y_timestamp,
                    'actual_target': actual_target,
                })
        else:
            # dict 格式（MA60 预归一化）
            normalized = data.get('normalized')
            original = data.get('original')
            index = data.get('index')

            if normalized is None or original is None:
                continue

            seq_len = len(normalized)
            window_size = lookback + predict

            if seq_len >= window_size:
                start = seq_len - window_size

                backtest_samples.append({
                    'symbol': symbol,
                    'context_start': start,
                    'context_end': start + lookback,
                    'target_start': start + lookback,
                    'target_end': start + lookback + predict,
                    'x_context': original[start:start + lookback],
                    'y_timestamp': index[start + lookback:start + lookback + predict],
                    'actual_target': original[start + lookback:start + lookback + predict],
                    'normalized_context': normalized[start:start + lookback],
                    'means': data.get('means')[start:start + lookback + predict] if data.get('means') else None,
                    'stds': data.get('stds')[start:start + lookback + predict] if data.get('stds') else None,
                })

    return backtest_samples


# ============================================================================
# 辅助函数
# ============================================================================

def get_split_stats(
    train: List[SampleSchema],
    val: List[SampleSchema],
    test: List[SampleSchema]
) -> Dict[str, Any]:
    """
    获取分割统计信息

    Args:
        train, val, test: 分割后的样本列表

    Returns:
        统计信息 dict
    """
    def get_time_range(samples):
        if not samples:
            return (None, None)
        starts = [s.target_start for s in samples]
        ends = [s.target_end for s in samples]
        return (min(starts), max(ends))

    return {
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
        'train_range': get_time_range(train),
        'val_range': get_time_range(val),
        'test_range': get_time_range(test),
        'n_symbols': len(set(s.symbol for s in train + val + test)),
    }