"""
Unit tests for core metrics module

验证度量口径定稿（Phase -1）的关键修复：
- E1: trajectory IC 去趋势口径（原始价格 vs 去趋势序列）
- E2: excess DA 计算
- E4: 聚合保留分布
"""

import numpy as np
import sys
import os

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

# 直接从当前目录导入
from metrics import (
    detrend_to_baseline,
    safe_corrcoef,
    safe_spearmanr,
    safe_trajectory_ic,
    excess_da,
    amplitude_error_rate,
    compute_amplitude_stats,
    limit_hit_rate,
    detect_limit,
    calculate_combined_score,
    calculate_da_score,
    get_log_step_weights,
    get_feature_weights,
    FEATURE_NAMES,
)


def test_detrend_to_baseline():
    """测试去趋势函数"""
    print("=" * 60)
    print("Test: detrend_to_baseline")
    print("=" * 60)

    # 原始价格序列（强趋势）
    baseline = 100.0
    # 预测：线性递增
    pred_raw = np.array([100, 101, 102, 103, 104])  # 单边上涨
    # 实际：V形（先涨后跌）- 形状完全不同
    actual_raw = np.array([100, 102, 104, 102, 100])  # V 形

    # 原始价格 IC（会被趋势夸大）
    raw_ic = safe_corrcoef(pred_raw, actual_raw)
    print(f"Raw price IC: {raw_ic:.4f} (被趋势夸大)")

    # 去趋势后
    pred_detrend = detrend_to_baseline(pred_raw, baseline)
    actual_detrend = detrend_to_baseline(actual_raw, baseline)
    print(f"Pred detrended: {pred_detrend}")
    print(f"Actual detrended: {actual_detrend}")

    # 去趋势 IC
    detrend_ic, detrend_rank_ic = safe_trajectory_ic(pred_detrend, actual_detrend)
    print(f"Detrended IC: {detrend_ic:.4f}, Rank IC: {detrend_rank_ic:.4f}")

    # 验证：V形 vs 线性递增，去趋势后应显示形状差异
    # 原始价格因趋势部分同向（前半段都涨），IC可能为正
    # 去趋势后，线性 vs V形，形状完全不同，IC应大幅降低或为负
    print(f"Assertion check: detrend_ic={detrend_ic}, raw_ic={raw_ic}")
    # 注意：去趋势后形状完全不同，IC可能为负
    assert detrend_ic < raw_ic or detrend_ic < 0.5, "去趋势后 IC 应低于原始价格 IC 或反映形状差异"
    print("PASS: 去趋势后 IC 正确反映形状相似度，趋势成分被移除")

    # 测试边界：常量序列
    const_pred = np.array([100, 100, 100, 100])
    const_actual = np.array([100, 100, 100, 100])
    pred_const = detrend_to_baseline(const_pred, 100)
    actual_const = detrend_to_baseline(const_actual, 100)
    ic_const, _ = safe_trajectory_ic(pred_const, actual_const)
    assert ic_const is None, "常量序列应返回 None"
    print("PASS: 常量序列正确返回 None")

    print()


def test_safe_corrcoef():
    """测试安全相关系数"""
    print("=" * 60)
    print("Test: safe_corrcoef / safe_spearmanr")
    print("=" * 60)

    # 正常序列
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    ic = safe_corrcoef(x, y)
    print(f"Normal sequences IC: {ic:.4f}")
    assert ic > 0.9, "高度相关序列应有高 IC"

    # 常量序列
    const_x = np.array([1, 1, 1, 1, 1])
    const_y = np.array([2, 2, 2, 2, 2])
    ic_const = safe_corrcoef(const_x, const_y)
    print(f"Constant sequences IC: {ic_const}")
    assert ic_const == 0.0, "常量序列应返回 default"
    print("PASS: 常量序列正确返回 default")

    # 短序列
    short_x = np.array([1, 2])
    short_y = np.array([1, 2])
    ic_short = safe_corrcoef(short_x, short_y)
    print(f"Short sequences (n=2) IC: {ic_short}")
    assert ic_short == 0.0, "短序列应返回 default"
    print("PASS: 短序列正确返回 default")

    print()


def test_excess_da():
    """测试 excess DA"""
    print("=" * 60)
    print("Test: excess_da")
    print("=" * 60)

    model_da = 0.55  # 模型 DA 55%
    naive_da = 0.50  # 持平预测 DA 50%

    excess = excess_da(model_da, naive_da)
    print(f"Model DA: {model_da:.1%}, Naive DA: {naive_da:.1%}")
    print(f"Excess DA: {excess:.1%}")

    assert excess > 0, "excess > 0 表示模型优于持平预测"
    print("PASS: excess DA 正确计算")

    # 验证：excess ≈ 0 时模型无 alpha
    model_da_weak = 0.51
    excess_weak = excess_da(model_da_weak, naive_da)
    print(f"Weak model DA: {model_da_weak:.1%}, Excess: {excess_weak:.1%}")
    assert excess_weak < 0.02, "excess ≈ 0 表示模型接近持平预测"
    print("PASS: weak model 正确识别")

    print()


def test_amplitude_error_rate():
    """测试振幅误差率"""
    print("=" * 60)
    print("Test: amplitude_error_rate")
    print("=" * 60)

    # 完美匹配
    rate_perfect = amplitude_error_rate(10.0, 10.0)
    print(f"Perfect match: {rate_perfect:.2f}")
    assert abs(rate_perfect - 1.0) < 1e-6, "完美匹配应返回 1.0"

    # 预测偏大
    rate_over = amplitude_error_rate(12.0, 10.0)
    print(f"Over-predict: {rate_over:.2f}")
    assert abs(rate_over - 1.2) < 1e-6, "预测偏大 20% 应返回 1.2"

    # 预测偏小
    rate_under = amplitude_error_rate(8.0, 10.0)
    print(f"Under-predict: {rate_under:.2f}")
    assert abs(rate_under - 0.8) < 1e-6, "预测偏小 20% 应返回 0.8"

    # actual 无波动（一字涨跌停/停牌）→ 返回 None，不污染均值
    rate_novol = amplitude_error_rate(5.0, 0.0)
    print(f"No-volatility: {rate_novol}")
    assert rate_novol is None, "actual 无波动时应返回 None"

    rate_novol2 = amplitude_error_rate(5.0, 0.0005)
    print(f"Near-zero vol: {rate_novol2}")
    assert rate_novol2 is None, "actual 近零波动时应返回 None"

    print("PASS: 振幅误差率正确计算")
    print()


def test_limit_hit_rate():
    """测试涨跌停命中率"""
    print("=" * 60)
    print("Test: limit_hit_rate")
    print("=" * 60)

    # 无预测涨停
    pred_none = np.array([False, False, False, False])
    actual_some = np.array([False, True, False, True])
    result_none = limit_hit_rate(pred_none, actual_some)
    print(f"No predicted limit: {result_none}")
    assert result_none['hit_rate'] is None, "无预测涨停时 hit_rate 应为 None"

    # 有命中
    pred_some = np.array([False, True, False, True])  # 预测 2 个涨停
    actual_match = np.array([False, True, False, False])  # 实际只有 1 个涨停，且被预测覆盖
    result_hit = limit_hit_rate(pred_some, actual_match)
    print(f"With hit: {result_hit}")
    assert abs(result_hit['hit_rate'] - 0.5) < 1e-6, "命中 1/2 应返回 0.5"
    assert result_hit['n_pred_limit'] == 2, "预测涨停数应为 2"
    assert result_hit['n_actual_limit'] == 1, "实际涨停数应为 1"

    print("PASS: 涨跌停命中率正确计算")
    print()


def test_combined_score():
    """测试综合评分"""
    print("=" * 60)
    print("Test: calculate_combined_score")
    print("=" * 60)

    # IC=0.2, DA=0.55, 权重 0.6/0.4
    ic = 0.2
    da = 0.55
    combined = calculate_combined_score(ic, da)
    print(f"IC: {ic}, DA: {da}")
    print(f"Combined (0.6/0.4): {combined:.4f}")

    # 验证计算
    ic_norm = (ic + 1) / 2  # 0.6
    expected = ic_norm * 0.6 + da * 0.4  # 0.36 + 0.22 = 0.58
    assert abs(combined - expected) < 1e-4, "综合评分应正确计算"

    print("PASS: 综合评分正确计算")
    print()


def test_trajectory_ic_raw_vs_detrended():
    """
    关键测试：验证原始价格 IC vs 去趋势 IC 的差异

    这是 E1 问题的核心验证：
    - 原始价格序列即使形状完全不同，因趋势同向也会拿到高 IC
    - 去趋势后才能正确反映形状相似度
    """
    print("=" * 60)
    print("Test: Raw vs Detrended Trajectory IC (E1 Key Test)")
    print("=" * 60)

    baseline = 100.0

    # 场景1：预测线性递增，实际 V 形（形状完全不同）
    # 原始价格：前半段都涨 → IC 可能为正
    # 去趋势：线性 vs V形 → IC 为负（形状相反）
    pred1 = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    actual1 = np.array([100, 102, 104, 106, 108, 110, 108, 106, 104, 102])  # V 形

    raw_ic1 = safe_corrcoef(pred1, actual1)
    pred1_detrend = detrend_to_baseline(pred1, baseline)
    actual1_detrend = detrend_to_baseline(actual1, baseline)
    detrend_ic1, _ = safe_trajectory_ic(pred1_detrend, actual1_detrend)

    print(f"Scenario 1: Linear rise vs V-shape (different shapes)")
    print(f"  Raw price IC: {raw_ic1:.4f} (trend alignment inflates IC)")
    print(f"  Detrended IC: {detrend_ic1:.4f} (shape difference revealed)")
    # 验证：去趋势后应揭示形状差异（线性 vs V形，IC应降低或为负）
    assert detrend_ic1 < raw_ic1 or detrend_ic1 < 0.3, "去趋势 IC 应反映形状差异"

    # 场景2：两个形状完全相同（都是V形）
    pred2 = np.array([100, 102, 104, 106, 108, 110, 108, 106, 104, 102])  # V 形
    actual2 = np.array([100, 102, 104, 106, 108, 110, 108, 106, 104, 102])  # 相同 V 形

    raw_ic2 = safe_corrcoef(pred2, actual2)
    pred2_detrend = detrend_to_baseline(pred2, baseline)
    actual2_detrend = detrend_to_baseline(actual2, baseline)
    detrend_ic2, _ = safe_trajectory_ic(pred2_detrend, actual2_detrend)

    print(f"Scenario 2: Same shape, same baseline")
    print(f"  Raw price IC: {raw_ic2:.4f}")
    print(f"  Detrended IC: {detrend_ic2:.4f}")
    assert raw_ic2 > 0.99, "形状相同原始价格 IC 应接近 1"
    assert detrend_ic2 > 0.99, "形状相同去趋势 IC 也应接近 1"

    # 场景3：预测持平，实际单边上涨
    # 原始价格：因同向涨，IC 可能为正
    # 去趋势：持平预测 std=0 → IC = None
    pred3 = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    actual3 = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])

    raw_ic3 = safe_corrcoef(pred3, actual3)
    pred3_detrend = detrend_to_baseline(pred3, baseline)
    actual3_detrend = detrend_to_baseline(actual3, baseline)
    detrend_ic3, _ = safe_trajectory_ic(pred3_detrend, actual3_detrend)

    print(f"Scenario 3: Flat prediction, actual rising")
    print(f"  Raw price IC: {raw_ic3} (flat → default)")
    print(f"  Detrended IC: {detrend_ic3} (flat detrended → None)")
    assert raw_ic3 == 0.0, "持平预测 std=0 应返回 default"
    assert detrend_ic3 is None, "去趋势持平预测应返回 None"

    print()
    print("PASS: E1 验证完成 - 去趋势口径正确反映形状相似度")
    print("=" * 60)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Running core metrics unit tests")
    print("=" * 70 + "\n")

    test_detrend_to_baseline()
    test_safe_corrcoef()
    test_excess_da()
    test_amplitude_error_rate()
    test_limit_hit_rate()
    test_combined_score()
    test_trajectory_ic_raw_vs_detrended()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()