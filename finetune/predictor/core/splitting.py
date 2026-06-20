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
            # I7 修复：丢弃跨 split 边界的样本（而非强分）
            # 跨边界样本的 target 与两个 split 相交，无法保证无泄露
            # 强分会触发 validate_no_leakage 崩溃
            continue  # 丢弃

    return train_samples, val_samples, test_samples


# ============================================================================
# Block Split（分层抽样）
# ============================================================================

def create_target_blocks(
    sym_samples: List[SampleSchema],
    block_size: int = 600
) -> List[List[SampleSchema]]:
    """
    创建 target 时间块

    设计（2026-06-20 重设计）：样本由 preprocess.create_samples 按**块**生成——
    先按 block_size 切不重叠时间段，块内 stride=1 生成窗口（target 不超块边界）。
    故样本天然属于某个块，块间 target 不相交自动成立。

    块边界与 create_samples 一致：从 0 起，block_size 步进。
    block_id = window_start // block_size。

    Args:
        sym_samples: 单只股票的样本列表（已按块生成）
        block_size: 时间块大小（须与 create_samples 一致）

    Returns:
        blocks: [[SampleSchema]] 时间块列表
    """
    if not sym_samples:
        return []

    # 按 block_id（window_start // block_size）分组
    blocks_by_id = defaultdict(list)
    for s in sym_samples:
        block_id = s.window_start // block_size
        blocks_by_id[block_id].append(s)

    # 按 block_id 排序
    blocks = [blocks_by_id[k] for k in sorted(blocks_by_id.keys())]

    # Sanity 检查：块间 target 不相交（应永不触发，因样本按块生成 target 不跨块）
    all_intervals = []
    for block_idx, block in enumerate(blocks):
        for s in block:
            all_intervals.append((s.target_start, s.target_end, block_idx))
    all_intervals.sort(key=lambda x: x[0])
    for i in range(len(all_intervals) - 1):
        s1, e1, b1 = all_intervals[i]
        s2, e2, b2 = all_intervals[i + 1]
        if e1 > s2 and b1 != b2:
            raise AssertionError(
                f"Block {b1} and {b2} target intervals overlap: [{s1},{e1}) vs [{s2},{e2})。"
                f"样本应按块生成（target 不跨块），检查 create_samples。"
            )

    return blocks


def block_split(
    samples: List[SampleSchema],
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    block_size: int = 600,
    seed: int = 42
) -> Tuple[List[SampleSchema], List[SampleSchema], List[SampleSchema]]:
    """
    以 target 时间块为单位分配 train/val/test

    分配规则：
    1. 6个指标：个股train/val/test缺度 + 全局train/val/test缺度
    2. 选择最大缺度对应的split（谁最缺给谁）
    3. 同等缺度时按优先级：train>val>test
    4. 使用优先权后，该split优先级降为最低，依次循环

    Args:
        samples: 所有样本列表
        train_ratio: 训练集比例（默认 0.75）
        val_ratio: 验证集比例（默认 0.15）
        test_ratio: 测试集比例（默认 0.15）
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

    # 目标比例
    target_ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    split_names = ['train', 'val', 'test']

    # 全局统计（用 block 数量）
    global_assigned = {name: 0 for name in split_names}
    global_total = 0

    # 每只股票的分配结果
    stock_assignments = {symbol: {name: [] for name in split_names} for symbol in by_symbol}

    # 优先级轮换状态：记录每个split的当前优先级顺序
    # 初始优先级：train(0) > val(1) > test(2)
    priority_order = {'train': 0, 'val': 1, 'test': 2}

    # 收集所有股票的块
    all_stock_blocks = {}
    for symbol, sym_samples in by_symbol.items():
        blocks = create_target_blocks(sym_samples, block_size=block_size)
        if blocks:
            rng.shuffle(blocks)
            all_stock_blocks[symbol] = blocks
            global_total += len(blocks)

    # 分配算法
    for symbol, blocks in all_stock_blocks.items():
        n_blocks = len(blocks)

        for block in blocks:
            # 计算个股当前比例和缺度
            stock_assigned = {name: len(stock_assignments[symbol][name]) for name in split_names}
            stock_total = sum(stock_assigned.values())
            if stock_total > 0:
                stock_ratios = {name: stock_assigned[name] / stock_total for name in split_names}
            else:
                stock_ratios = {name: 0.0 for name in split_names}

            # 缺度 = 目标比例 - 当前比例（正数表示缺，负数表示超）
            stock_gaps = {name: target_ratios[name] - stock_ratios[name] for name in split_names}
            stock_gap_rates = {name: stock_gaps[name] / target_ratios[name] if target_ratios[name] > 0 else 0 for name in split_names}

            # 计算全局当前比例和缺度
            global_total_assigned = sum(global_assigned.values())
            if global_total_assigned > 0:
                global_ratios = {name: global_assigned[name] / global_total_assigned for name in split_names}
            else:
                global_ratios = {name: 0.0 for name in split_names}

            global_gaps = {name: target_ratios[name] - global_ratios[name] for name in split_names}
            global_gap_rates = {name: global_gaps[name] / target_ratios[name] if target_ratios[name] > 0 else 0 for name in split_names}

            # 6个指标：缺度（正数表示缺，负数表示超）
            indicators = {
                'stock_train': stock_gap_rates['train'],
                'stock_val': stock_gap_rates['val'],
                'stock_test': stock_gap_rates['test'],
                'global_train': global_gap_rates['train'],
                'global_val': global_gap_rates['val'],
                'global_test': global_gap_rates['test'],
            }

            # 找最大缺度（按固定顺序遍历，避免字典顺序不确定）
            max_gap = -float('inf')
            for name in split_names:
                split_gap = max(indicators[f'stock_{name}'], indicators[f'global_{name}'])
                if split_gap > max_gap:
                    max_gap = split_gap
                    best_split = name

            # 找出所有缺度等于最大缺度的 split
            splits_with_max_gap = []
            for name in split_names:
                split_gap = max(indicators[f'stock_{name}'], indicators[f'global_{name}'])
                if split_gap == max_gap:
                    splits_with_max_gap.append(name)

            # 如果多个split缺度相等，按优先级选择并轮换
            if len(splits_with_max_gap) > 1:
                # 按当前优先级顺序排序（数字越小优先级越高）
                splits_with_max_gap.sort(key=lambda s: priority_order[s])
                best_split = splits_with_max_gap[0]

                # 使用优先权后，轮换优先级（循环队列）
                # 例如：[0,1,2] → 选中0 → [2,0,1]；选中1 → [1,2,0]；选中2 → [0,1,2]
                # 实现：选中的 split 优先级设为2，其他按相对顺序递减
                current_order = sorted(split_names, key=lambda s: priority_order[s])
                new_order = []
                for name in current_order:
                    if name != best_split:
                        new_order.append(name)
                new_order.append(best_split)  # 选中的放最后
                # 重新分配优先级：0=最高，1=次之，2=最低
                for i, name in enumerate(new_order):
                    priority_order[name] = i

            # 分配块
            stock_assignments[symbol][best_split].append(block)
            global_assigned[best_split] += 1

    # 收集结果
    train, val, test = [], [], []
    for symbol, assignments in stock_assignments.items():
        for block in assignments['train']:
            for s in block:
                s.split = 'train'
                train.append(s)
        for block in assignments['val']:
            for s in block:
                s.split = 'val'
                val.append(s)
        for block in assignments['test']:
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
    验证 target 时间区间无跨 split 重叠（区间语义）

    区间 [s, e) 不相交: e1 <= s2 或 e2 <= s1
    同 split 内相邻样本 target 重叠是允许的（block 内滑动样本本就重叠），
    只检查跨 split（train/val、train/test、val/test）的 target 相交。

    这是 target-based splitting 的正确守卫。
    本项目 lb60 曾因 window-index splitting 导致 100% target 泄露。

    算法（扫描线，O(N log N)）：
    1. 按股票分组，每只股票把三个 split 的区间合并为 (start, end, split)
    2. 按 start 排序
    3. 扫描时维护每个 split 已见区间的最大 end
    4. 新区间进来，若其它 split 的 max_end > 新区间 start，则相交（泄露）

    Args:
        train_samples: 训练集样本
        val_samples: 验证集样本
        test_samples: 测试集样本

    Returns:
        True 如果无泄露

    Raises:
        AssertionError 如果发现泄露
    """
    from collections import defaultdict

    # 按股票收集 (start, end, split)
    by_symbol = defaultdict(list)
    for s in train_samples:
        by_symbol[s.symbol].append((s.target_start, s.target_end, 'train'))
    for s in val_samples:
        by_symbol[s.symbol].append((s.target_start, s.target_end, 'val'))
    for s in test_samples:
        by_symbol[s.symbol].append((s.target_start, s.target_end, 'test'))

    split_names = ('train', 'val', 'test')

    for sym, intervals in by_symbol.items():
        if not intervals:
            continue

        # 按 start 排序（start 相同按 end 排序）
        intervals.sort(key=lambda x: (x[0], x[1]))

        # 每个 split 已见区间的最大 end
        max_end = {name: -1 for name in split_names}

        for start, end, split in intervals:
            # 检查其它 split 是否有区间 end > start（即与新区间相交）
            for other in split_names:
                if other == split:
                    continue
                if max_end[other] > start:
                    raise AssertionError(
                        f"{split}/{other} target overlap for {sym}: "
                        f"new [{start}, {end}) intersects existing end={max_end[other]}"
                    )
            # 更新本 split 的 max_end
            if end > max_end[split]:
                max_end[split] = end

    print(f"No-leakage check passed (interval semantics) - {len(by_symbol)} stocks verified")
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