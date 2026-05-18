"""
验证脚本：预测价格路径 vs 真实价格路径对比

目标：直观对比模型预测能力和真实走势
"""

import os
import sys
import pickle
import numpy as np
import torch
import pandas as pd

# Add project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference


def load_model():
    """加载模型和tokenizer - 使用原始pretrained版本"""
    # 使用原始pretrained tokenizer（未微调）
    tokenizer = KronosTokenizer.from_pretrained("pretrained/Kronos-Tokenizer-2k")
    # 使用原始pretrained model
    model = Kronos.from_pretrained("pretrained/Kronos-mini")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tokenizer.to(device)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_one_sample(tokenizer, model, device, df, lookback=400, pred_len=10):
    """预测一个样本"""
    feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
    values = df[feature_cols].values.astype(np.float32)

    if len(values) < lookback + pred_len + 1:
        return None

    # 输入窗口
    x = values[-(lookback + pred_len):-pred_len]

    # 全窗口归一化（pretrained原始方式）
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0) + 1e-5

    x_norm = (x - x_mean) / x_std
    x_norm = np.clip(x_norm, -5.0, 5.0)

    # 时间戳
    timestamps = df.index[-(lookback + pred_len):]
    x_timestamp = timestamps[:lookback]
    y_timestamp = timestamps[lookback:]

    time_df_x = pd.DataFrame()
    time_df_x['minute'] = x_timestamp.minute if hasattr(x_timestamp, 'minute') else 0
    time_df_x['hour'] = x_timestamp.hour if hasattr(x_timestamp, 'hour') else 0
    time_df_x['weekday'] = x_timestamp.weekday if hasattr(x_timestamp, 'weekday') else 0
    time_df_x['day'] = x_timestamp.day if hasattr(x_timestamp, 'day') else 0
    time_df_x['month'] = x_timestamp.month if hasattr(x_timestamp, 'month') else 0

    time_df_y = pd.DataFrame()
    time_df_y['minute'] = y_timestamp.minute if hasattr(y_timestamp, 'minute') else 0
    time_df_y['hour'] = y_timestamp.hour if hasattr(y_timestamp, 'hour') else 0
    time_df_y['weekday'] = y_timestamp.weekday if hasattr(y_timestamp, 'weekday') else 0
    time_df_y['day'] = y_timestamp.day if hasattr(y_timestamp, 'day') else 0
    time_df_y['month'] = y_timestamp.month if hasattr(y_timestamp, 'month') else 0

    # 基准价格（输入窗口最后一点）
    baseline_close = values[-pred_len - 1, 3]

    # 预测
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).float().to(device)
        x_stamp_tensor = torch.from_numpy(time_df_x.values.astype(np.float32)).unsqueeze(0).to(device)
        y_stamp_tensor = torch.from_numpy(time_df_y.values.astype(np.float32)).unsqueeze(0).to(device)

        preds = auto_regressive_inference(
            tokenizer, model,
            x_tensor, x_stamp_tensor, y_stamp_tensor,
            max_context=512, pred_len=pred_len,
            clip=5, T=1.0, top_k=0, top_p=0.9,
            sample_count=1, verbose=False
        )

        # 取预测窗口的所有特征
        pred_all_norm = preds[0, -pred_len:, :]  # [pred_len, 6] numpy array

        # Denormalize 所有特征
        pred_all_raw = pred_all_norm * x_std + x_mean

        # 实际值
        actual_all_raw = values[-pred_len:, :]

        # 实际值的normalized版本（用同样的MA20）
        actual_all_norm = (actual_all_raw - x_mean) / x_std
        actual_all_norm = np.clip(actual_all_norm, -5.0, 5.0)

        # 输入窗口最后20点（用于观察输入趋势）
        input_close = values[-(lookback + pred_len):-pred_len, 3][-20:]

    return {
        'baseline_close': baseline_close,
        'input_close_last20': input_close,
        'pred_close': pred_all_raw[:, 3],  # 预测close路径
        'actual_close': actual_all_raw[:, 3],  # 实际close路径
        'pred_all': pred_all_raw,  # 预测所有特征
        'actual_all': actual_all_raw,  # 实际所有特征
        'pred_norm': pred_all_norm,  # 预测normalized值
        'actual_norm': actual_all_norm,  # 实际normalized值
        'x_mean': x_mean,
        'x_std': x_std,
    }


def analyze_result(result):
    """分析预测结果"""
    pred = result['pred_close']
    actual = result['actual_close']
    baseline = result['baseline_close']

    # 计算收益率
    pred_return = (pred - baseline) / baseline
    actual_return = (actual - baseline) / baseline

    # 计算误差
    abs_error = np.abs(pred - actual)
    rel_error = np.abs(pred - actual) / actual

    # 方向准确性
    pred_direction = pred_return > 0
    actual_direction = actual_return > 0
    direction_match = pred_direction == actual_direction

    print("\n" + "="*60)
    print("预测路径 vs 真实路径对比")
    print("="*60)

    print(f"\n基准价格 (P0): {baseline:.2f}")
    print(f"\n输入窗口最后5点 close:")
    for i, c in enumerate(result['input_close_last20'][-5:]):
        print(f"  P-{5-i}: {c:.2f}")

    print(f"\n{'Point':<8} {'预测Close':<12} {'实际Close':<12} {'误差':<10} {'误差%':<10} {'方向':<8}")
    print("-"*60)
    for i in range(len(pred)):
        dir_str = "OK" if direction_match[i] else "X"
        print(f"P+{i+1:<5} {pred[i]:<12.2f} {actual[i]:<12.2f} {abs_error[i]:<10.2f} {rel_error[i]*100:<10.2f}% {dir_str}")

    print("-"*60)
    print(f"平均误差: {np.mean(abs_error):.2f}")
    print(f"平均误差%: {np.mean(rel_error)*100:.2f}%")
    print(f"方向准确率: {np.mean(direction_match)*100:.1f}%")

    # 收益率对比
    print(f"\n收益率对比 (相对于P0):")
    print(f"{'Point':<8} {'预测收益%':<12} {'实际收益%':<12} {'差异%':<10}")
    print("-"*50)
    for i in range(len(pred_return)):
        diff = pred_return[i] - actual_return[i]
        print(f"P+{i+1:<5} {pred_return[i]*100:<12.2f} {actual_return[i]*100:<12.2f} {diff*100:<10.2f}")

    # 关键点分析 (P+3)
    print(f"\n=== P+3 关键点分析 ===")
    print(f"预测: {pred[2]:.2f} (收益 {pred_return[2]*100:.2f}%)")
    print(f"实际: {actual[2]:.2f} (收益 {actual_return[2]*100:.2f}%)")
    print(f"方向判断: {'正确 OK' if direction_match[2] else '错误 X'}")


def main():
    print("加载模型...")
    tokenizer, model, device = load_model()
    print(f"Device: {device}")

    print("\n加载测试数据...")
    test_path = "finetune/data/processed_datasets/test_data.pkl"
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    print(f"测试数据: {len(test_data)} stocks")

    # 随机选取3个样本对比
    np.random.seed(42)
    symbols = np.random.choice(list(test_data.keys()), size=3, replace=False)

    for symbol in symbols:
        df = test_data[symbol]
        print(f"\n{'='*60}")
        print(f"股票: {symbol}")
        print(f"数据点数: {len(df)}")

        result = predict_one_sample(tokenizer, model, device, df)
        if result:
            # 先输出normalized空间对比
            print("\n=== Normalized空间对比 (close列) ===")
            print(f"MA20 mean: {result['x_mean'][3]:.2f}, std: {result['x_std'][3]:.2f}")
            print(f"\n{'Point':<8} {'预测norm':<12} {'实际norm':<12} {'norm误差':<10}")
            print("-"*50)
            for i in range(len(result['pred_close'])):
                pred_n = result['pred_norm'][i, 3]  # 预测的normalized close
                act_n = result['actual_norm'][i, 3]  # 实际的normalized close
                print(f"P+{i+1:<5} {pred_n:<12.3f} {act_n:<12.3f} {abs(pred_n - act_n):<10.3f}")

            print("\n=== 原始价格空间对比 ===")
            analyze_result(result)
        else:
            print("数据不足，跳过")


if __name__ == "__main__":
    main()