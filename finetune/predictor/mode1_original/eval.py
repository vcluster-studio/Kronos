"""
Mode1 模型评估 - 使用测试集计算 Trajectory IC / MAE / DA

用法:
    python eval.py --model outputs/models/global_predictor_v1/checkpoints/best_ic_model
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
from finetune.predictor.shared.eval import evaluate_full_window, print_evaluation_result

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='Mode1 Model Evaluation')
    parser.add_argument('--model', type=str,
                        default='outputs/models/global_predictor_v1/checkpoints/best_ic_model')
    parser.add_argument('--tokenizer', type=str, default='pretrained/Kronos-Tokenizer-2k')
    parser.add_argument('--data', type=str,
                        default='finetune/data/global_norm/full_series/test_data.pkl')
    parser.add_argument('--lookback', type=int, default=200)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Mode1 Model Evaluation")
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

    # 加载测试数据
    with open(os.path.join(project_root, args.data), 'rb') as f:
        test_data = pickle.load(f)
    print(f"Test data: {len(test_data)} stocks")

    # 评估
    print("\nEvaluating...")
    result = evaluate_full_window(
        model, tokenizer, test_data,
        lookback=args.lookback, predict=args.predict,
        n_samples=args.n_samples, seed=args.seed
    )

    # 打印结果
    print_evaluation_result(result, predict=args.predict, title="Mode1 Evaluation Results")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()