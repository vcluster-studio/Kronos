"""
Mode2 模型评估 - MA60 t0 归一化
使用测试集计算 Trajectory IC / MAE / DA

用法:
    python eval.py --model outputs/models/ma60_predictor_lb60_pd1/checkpoints/best_model
"""

import os
import sys
import pickle
import argparse
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos
from finetune.predictor.shared.eval import evaluate_ma60, print_evaluation_result

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ma60_data(data_path, lookback, predict, train_ratio=0.70, val_ratio=0.15):
    """加载 MA60 数据并划分"""
    with open(data_path, 'rb') as f:
        raw = pickle.load(f)

    all_data = {}
    train_indices, val_indices, test_indices = [], [], []
    window = lookback + predict

    for sym in sorted(raw.keys()):
        d = raw[sym]
        seq_len = len(d['normalized'])
        if seq_len < window + 1:
            continue

        all_data[sym] = {
            'normalized': d['normalized'].astype('float32'),
            'original': d['original'].astype('float32'),
            'means': d['means'].astype('float32'),
            'stds': d['stds'].astype('float32'),
            'index': d['index'],
        }

        n_windows = seq_len - window
        tr_end = int(n_windows * train_ratio)
        val_end = int(n_windows * (train_ratio + val_ratio))

        for i in range(n_windows):
            idx = (sym, i)
            if i < tr_end:
                train_indices.append(idx)
            elif i < val_end:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

    return all_data, train_indices, val_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description='Mode2 Model Evaluation')
    parser.add_argument('--model', type=str,
                        default='outputs/models/ma60_predictor_lb60_pd1/checkpoints/best_model')
    parser.add_argument('--tokenizer', type=str,
                        default='outputs/models/ma60_tokenizer_v1/checkpoints/best_model')
    parser.add_argument('--data', type=str, default='finetune/data/kline_daily_ma60.pkl')
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--predict', type=int, default=5)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Mode2 Model Evaluation (MA60 t0)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Lookback: {args.lookback}, Predict: {args.predict}")
    print(f"Test samples: {args.n_samples}")

    # 加载 tokenizer
    tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, args.tokenizer))
    tokenizer.eval().to(DEVICE)

    # 加载模型
    model = Kronos.from_pretrained(os.path.join(project_root, "pretrained/Kronos-mini"))
    model.eval().to(DEVICE)

    # 加载 checkpoint
    checkpoint_path = os.path.join(project_root, args.model)
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded safetensors from {safetensors_path}")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {bin_path}")
    else:
        print("Warning: No checkpoint found, using pretrained model")

    # 加载 MA60 数据
    all_data, _, _, test_indices = load_ma60_data(
        os.path.join(project_root, args.data),
        args.lookback, args.predict
    )
    print(f"Test indices: {len(test_indices)} windows")

    # 评估
    print("\nEvaluating...")
    result = evaluate_ma60(
        model, tokenizer, all_data, test_indices,
        lookback=args.lookback, predict=args.predict,
        n_samples=args.n_samples, seed=args.seed
    )

    # 打印结果
    print_evaluation_result(result, predict=args.predict, title="Mode2 Evaluation Results")


if __name__ == '__main__':
    main()