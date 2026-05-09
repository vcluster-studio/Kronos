"""
测试原始预训练 Kronos 模型（未微调）在 A 股数据上的预测效果

用法：
    python finetune/test_original_model.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from finetune.config import Config
from model.kronos import KronosTokenizer, Kronos, KronosPredictor


def load_test_data(config):
    """加载测试数据"""
    test_path = os.path.join(config.dataset_path, "test_data.pkl")
    print(f"Loading test data from: {test_path}")

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    print(f"Test data: {len(test_data)} stocks")
    return test_data


def predict_single_stock(predictor, df, lookback=90, pred_len=10):
    """对单只股票进行预测"""
    min_required = lookback + pred_len
    if len(df) < min_required:
        return None

    try:
        # 截取历史数据用于输入
        df_history = df.iloc[-min_required:-pred_len].copy()

        # 时间戳需要是 Series 格式
        x_timestamp = pd.Series(df_history.index)
        y_timestamp = pd.Series(df.index[-pred_len:])

        pred_df = predictor.predict(df_history, x_timestamp, y_timestamp, pred_len=pred_len)
        return pred_df
    except Exception as e:
        return None


def calculate_metrics(test_data, predictor, lookback=90, pred_len=10, n_stocks=100):
    """计算预测指标"""
    print(f"\nPredicting on {n_stocks} stocks...")
    print(f"  Lookback: {lookback}, Predict length: {pred_len}")
    print(f"  Minimum data required: {lookback + pred_len}")

    predictions = []
    actuals = []
    errors = []

    symbols = list(test_data.keys())[:n_stocks]
    min_required = lookback + pred_len

    for i, symbol in enumerate(symbols):
        df = test_data[symbol]

        if len(df) < min_required:
            errors.append(f"{symbol}: Not enough data (len={len(df)}, need={min_required})")
            continue

        try:
            pred_df = predict_single_stock(predictor, df, lookback=lookback, pred_len=pred_len)

            if pred_df is None:
                errors.append(f"{symbol}: Prediction failed")
                continue

            # 计算预测收益率 vs 实际收益率
            pred_return = (pred_df['close'].iloc[-1] - pred_df['close'].iloc[0]) / pred_df['close'].iloc[0]

            actual_close = df['close'].iloc[-pred_len:]
            actual_return = (actual_close.iloc[-1] - actual_close.iloc[0]) / actual_close.iloc[0]

            predictions.append(pred_return)
            actuals.append(actual_return)

        except Exception as e:
            errors.append(f"{symbol}: {str(e)}")

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(symbols)} stocks, {len(predictions)} valid predictions")

    if errors:
        print(f"\n  Errors: {len(errors)}")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    if len(predictions) > 0:
        # IC (Pearson correlation)
        ic = np.corrcoef(predictions, actuals)[0, 1]

        # Rank IC (Spearman)
        from scipy.stats import spearmanr
        rank_ic, _ = spearmanr(predictions, actuals)

        # Direction accuracy
        direction_acc = np.mean((predictions > 0) == (actuals > 0))

        return {
            'n_samples': len(predictions),
            'IC': ic,
            'Rank IC': rank_ic,
            'Direction Accuracy': direction_acc,
        }
    else:
        return None


def main():
    print("=" * 60)
    print("Testing ORIGINAL Kronos Model (Pretrained, NOT Finetuned)")
    print("=" * 60)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 加载原始预训练模型（未微调）
    print("\nLoading ORIGINAL pretrained tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    tokenizer.eval().to(device)

    print("Loading ORIGINAL pretrained predictor...")
    model = Kronos.from_pretrained(config.pretrained_predictor_path)
    model.eval().to(device)

    # 创建预测器
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print("Predictor created successfully!")

    # 加载测试数据
    test_data = load_test_data(config)

    # 计算指标
    print("\n" + "=" * 60)
    print("Calculating Prediction Metrics")
    print("=" * 60)

    metrics = calculate_metrics(test_data, predictor, lookback=90, pred_len=10, n_stocks=100)

    if metrics:
        print("\n" + "-" * 60)
        print("Results (ORIGINAL Model):")
        print("-" * 60)
        print(f"  Samples tested:     {metrics['n_samples']}")
        print(f"  IC (Pearson):       {metrics['IC']:.4f}")
        print(f"  Rank IC (Spearman): {metrics['Rank IC']:.4f}")
        print(f"  Direction Accuracy: {metrics['Direction Accuracy']:.2%}")
        print("-" * 60)

        print("\nInterpretation:")
        if abs(metrics['IC']) > 0.03:
            print("  [OK] IC > 0.03: Model has predictive value")
        else:
            print("  [WARN] IC < 0.03: Limited predictive value")

        if metrics['Direction Accuracy'] > 0.55:
            print("  [OK] Direction accuracy > 55%: Better than random")
        else:
            print("  [WARN] Direction accuracy < 55%: Not better than random")
    else:
        print("Failed to calculate metrics")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()