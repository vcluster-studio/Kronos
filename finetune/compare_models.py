"""
对比测试三个版本的 Kronos 模型：
1. Original: 预训练模型（未微调）
2. V1: 第一次微调版本
3. V2: 优化后微调版本

用法：
    python finetune/compare_models.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

# 设置多线程并行
torch.set_num_threads(8)

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
        df_history = df.iloc[-min_required:-pred_len].copy()
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

    predictions = []
    actuals = []

    symbols = list(test_data.keys())[:n_stocks]
    min_required = lookback + pred_len

    for i, symbol in enumerate(symbols):
        df = test_data[symbol]

        if len(df) < min_required:
            continue

        try:
            pred_df = predict_single_stock(predictor, df, lookback=lookback, pred_len=pred_len)

            if pred_df is None:
                continue

            pred_return = (pred_df['close'].iloc[-1] - pred_df['close'].iloc[0]) / pred_df['close'].iloc[0]
            actual_close = df['close'].iloc[-pred_len:]
            actual_return = (actual_close.iloc[-1] - actual_close.iloc[0]) / actual_close.iloc[0]

            predictions.append(pred_return)
            actuals.append(actual_return)

        except Exception as e:
            continue

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{n_stocks} stocks, {len(predictions)} valid predictions")

    print(f"  Errors: {n_stocks - len(predictions)}")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    if len(predictions) > 0:
        from scipy.stats import spearmanr

        ic = np.corrcoef(predictions, actuals)[0, 1]
        rank_ic, _ = spearmanr(predictions, actuals)
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
    print("=" * 70)
    print("Comparing Three Kronos Model Versions on A-Share Data")
    print("=" * 70)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")

    # 加载测试数据
    test_data = load_test_data(config)

    results = {}

    # === 测试原始预训练模型 ===
    print("\n" + "=" * 70)
    print("Testing Original Pretrained Model (Not Finetuned)")
    print("=" * 70)

    tokenizer_original = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    tokenizer_original.eval().to(device)
    model_original = Kronos.from_pretrained(config.pretrained_predictor_path)
    model_original.eval().to(device)
    predictor_original = KronosPredictor(model_original, tokenizer_original, max_context=512)

    results['Original'] = calculate_metrics(test_data, predictor_original, n_stocks=100)

    # === 测试 V1 版本 ===
    print("\n" + "=" * 70)
    print("Testing V1 Model (First Finetune)")
    print("=" * 70)

    v1_tokenizer_path = os.path.join(config.save_path, "a_share_tokenizer", "checkpoints", "best_model")
    v1_predictor_path = os.path.join(config.save_path, "a_share_predictor", "checkpoints", "best_model")

    print(f"V1 tokenizer: {v1_tokenizer_path}")
    print(f"V1 predictor: {v1_predictor_path}")

    tokenizer_v1 = KronosTokenizer.from_pretrained(v1_tokenizer_path)
    tokenizer_v1.eval().to(device)
    model_v1 = Kronos.from_pretrained(v1_predictor_path)
    model_v1.eval().to(device)
    predictor_v1 = KronosPredictor(model_v1, tokenizer_v1, max_context=512)

    results['V1'] = calculate_metrics(test_data, predictor_v1, n_stocks=100)

    # === 测试 V2 版本 ===
    print("\n" + "=" * 70)
    print("Testing V2 Model (Optimized)")
    print("=" * 70)

    v2_predictor_path = os.path.join(config.save_path, "a_share_predictor_v2", "checkpoints", "best_model")
    print(f"V2 predictor: {v2_predictor_path}")

    model_v2 = Kronos.from_pretrained(v2_predictor_path)
    model_v2.eval().to(device)
    predictor_v2 = KronosPredictor(model_v2, tokenizer_v1, max_context=512)  # 使用 V1 tokenizer

    results['V2'] = calculate_metrics(test_data, predictor_v2, n_stocks=100)

    # === 综合对比 ===
    print("\n" + "=" * 70)
    print("Final Comparison")
    print("=" * 70)

    if all(results.values()):
        print("\n" + "-" * 70)
        print(f"{'Metric':<20} {'Original':>12} {'V1':>12} {'V2':>12}")
        print("-" * 70)
        print(f"{'Samples':<20} {results['Original']['n_samples']:>12} {results['V1']['n_samples']:>12} {results['V2']['n_samples']:>12}")
        print(f"{'IC (Pearson)':<20} {results['Original']['IC']:>12.4f} {results['V1']['IC']:>12.4f} {results['V2']['IC']:>12.4f}")
        print(f"{'Rank IC':<20} {results['Original']['Rank IC']:>12.4f} {results['V1']['Rank IC']:>12.4f} {results['V2']['Rank IC']:>12.4f}")
        print(f"{'Direction Acc':<20} {results['Original']['Direction Accuracy']:>12.2%} {results['V1']['Direction Accuracy']:>12.2%} {results['V2']['Direction Accuracy']:>12.2%}")
        print("-" * 70)

        print("\nAnalysis:")
        best_ic = max(results['Original']['IC'], results['V1']['IC'], results['V2']['IC'])
        best_model = 'Original' if results['Original']['IC'] == best_ic else ('V1' if results['V1']['IC'] == best_ic else 'V2')
        print(f"  Best IC: {best_model} ({best_ic:.4f})")

        best_rank_ic = max(results['Original']['Rank IC'], results['V1']['Rank IC'], results['V2']['Rank IC'])
        best_model_rank = 'Original' if results['Original']['Rank IC'] == best_rank_ic else ('V1' if results['V1']['Rank IC'] == best_rank_ic else 'V2')
        print(f"  Best Rank IC: {best_model_rank} ({best_rank_ic:.4f})")

        print("\nConclusions:")
        if results['V1']['IC'] > results['Original']['IC']:
            print("  [OK] V1 improves over Original")
        else:
            print("  [WARN] V1 does not improve over Original")

        if results['V2']['IC'] > results['V1']['IC']:
            print("  [OK] V2 improves over V1")
        else:
            print("  [WARN] V2 does not improve over V1")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()