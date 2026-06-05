"""Time extrapolation IC test - evaluate model on out-of-sample test data"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
import torch
from scipy.stats import spearmanr
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

TOKENIZER_PATH = 'outputs/models/ma60_tokenizer_v1/checkpoints/best_model'
TEST_DATA_V3 = 'finetune/data/processed_datasets_ma60_windowed_v3/test_data.pkl'
LOOKBACK = 400
PRED_LEN = 10
IC_POINT = 3


def evaluate_model_on_test(model_dir, tokenizer, device, test_data, n_samples=None):
    """Evaluate a model on time-extrapolated test data."""
    model = Kronos.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    predictions = []
    actuals = []

    has_windows = 'windows' in list(test_data.values())[0]

    if has_windows:
        all_windows = []
        for sym in test_data:
            for start in test_data[sym]['windows']:
                all_windows.append((sym, int(start)))

        if n_samples and n_samples < len(all_windows):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(all_windows), n_samples, replace=False)
            sampled = [all_windows[i] for i in indices]
        else:
            sampled = all_windows
    else:
        sampled = []
        for sym in test_data:
            seq_len = len(test_data[sym]['normalized'])
            if seq_len >= LOOKBACK + PRED_LEN:
                sampled.append((sym, seq_len - LOOKBACK - PRED_LEN))

    print(f"  Evaluating {len(sampled)} windows...")

    for sym, start_idx in sampled:
        data = test_data[sym]
        try:
            x_norm = data['normalized'][start_idx:start_idx + LOOKBACK].astype(np.float32)
            means_full = data['means'][start_idx:start_idx + LOOKBACK + PRED_LEN]
            stds_full = data['stds'][start_idx:start_idx + LOOKBACK + PRED_LEN]

            x_ts = data['index'][start_idx:start_idx + LOOKBACK]
            y_ts = data['index'][start_idx + LOOKBACK:start_idx + LOOKBACK + PRED_LEN]

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            original_close = data['original'][:, 3]
            baseline_close = original_close[start_idx + LOOKBACK - 1]

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=2048, pred_len=PRED_LEN,
                    clip=5.0, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                pred_close_norm = preds[0, -PRED_LEN:, 3]
                pred_close_raw = pred_close_norm * stds_full[LOOKBACK:, 3] + means_full[LOOKBACK:, 3]
                pred_return = (pred_close_raw[IC_POINT - 1] - baseline_close) / baseline_close

            actual_close = original_close[start_idx + LOOKBACK:]
            actual_return = (actual_close[IC_POINT - 1] - baseline_close) / baseline_close

            predictions.append(pred_return)
            actuals.append(actual_return)
        except Exception:
            continue

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    ic = np.corrcoef(predictions, actuals)[0, 1]
    rank_ic, _ = spearmanr(predictions, actuals)
    direction_acc = np.mean((predictions > 0) == (actuals > 0))

    return {
        'ic': ic, 'rank_ic': rank_ic, 'direction_acc': direction_acc,
        'n_samples': len(predictions),
        'pred_mean': predictions.mean(), 'pred_std': predictions.std(),
        'actual_mean': actuals.mean(), 'actual_std': actuals.std(),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = KronosTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), '..', TOKENIZER_PATH))
    tokenizer.to(device)
    tokenizer.eval()

    with open(TEST_DATA_V3, 'rb') as f:
        test_data = pickle.load(f)

    models = {
        'V6c_best_model (Ep5 VL, pure CE)': 'outputs/models/ma60_predictor_mini_v6c/checkpoints/best_model',
        'V6c_best_ic (Ep5)': 'outputs/models/ma60_predictor_mini_v6c/checkpoints/best_ic_model',
        'V6b_best_model (Ep2 VL, close_loss=0)': 'outputs/models/ma60_predictor_mini_v6b/checkpoints/best_model',
        'V5_best_model (Ep26 VL, CE+direction)': 'outputs/models/ma60_predictor_mini_v5/checkpoints/best_model',
        'V5_best_ic (Ep2)': 'outputs/models/ma60_predictor_mini_v5/checkpoints/best_ic_model',
        'Pretrained (baseline)': 'pretrained/Kronos-mini',
    }

    print("=" * 70)
    print("Time Extrapolation IC Test (2025 data, 24858 windows)")
    print("=" * 70)

    # Use 500 samples for speed, can increase later
    n_samples = 500

    for name, path in models.items():
        full_path = os.path.join(os.path.dirname(__file__), '..', path)
        if not os.path.exists(full_path):
            print(f"\n  {name}: SKIPPED (not found: {path})")
            continue

        print(f"\n  {name}:")
        result = evaluate_model_on_test(full_path, tokenizer, device, test_data, n_samples=n_samples)
        print(f"    IC:            {result['ic']:.4f}")
        print(f"    Rank IC:       {result['rank_ic']:.4f}")
        print(f"    Direction Acc: {result['direction_acc']:.4f}")
        print(f"    N samples:     {result['n_samples']}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()