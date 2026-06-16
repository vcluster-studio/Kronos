"""
Model Evaluation Script - 测试集完整评估

评估训练好的模型在测试集上的表现。

Usage:
    python eval_trained_models.py --models mode2_mini_lb400 mode7_base_lb400 ...
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer
from finetune.predictor.shared.eval import evaluate_ma60, print_evaluation_result, FEATURE_NAMES

# 默认配置
TOKENIZER_PATH = 'outputs/tokenizers/ma60_tokenizer_base_v1/checkpoints/best_model'
DATA_DIR_TEMPLATE = 'finetune/data/ma60_norm/windowed_lb{lookback}_pd10'
MODELS_DIR = 'outputs/models'

# 模型配置映射
MODEL_CONFIGS = {
    'mode2_mini_lb400': {'model_type': 'mini', 'lookback': 400, 'max_context': 2048},
    'mode2_lb400_pd10': {'model_type': 'mini', 'lookback': 400, 'max_context': 2048},
    'mode7_base_lb400': {'model_type': 'base', 'lookback': 400, 'max_context': 512},
    'mode8_small_lb400': {'model_type': 'small', 'lookback': 400, 'max_context': 512},
    'mode9_small_lb60': {'model_type': 'small', 'lookback': 60, 'max_context': 512},
    'mode10_small_lb246': {'model_type': 'small', 'lookback': 246, 'max_context': 512},
}

# 预训练模型路径
PRETRAINED_PATHS = {
    'mini': 'pretrained/Kronos-mini',
    'small': 'pretrained/Kronos-small',
    'base': 'pretrained/Kronos-base',
}


def get_test_indices(test_data, lookback, predict):
    """获取测试集窗口索引"""
    indices = []
    window = lookback + predict

    for symbol in test_data.keys():
        d = test_data[symbol]
        # 检查是否有预分配的windows
        if 'windows' in d:
            for start in d['windows']:
                indices.append((symbol, int(start)))
        else:
            seq_len = len(d['normalized'])
            if seq_len >= window:
                for i in range(seq_len - window + 1):
                    indices.append((symbol, i))

    return indices


def evaluate_model(model_name, checkpoint_type='best_ic_model', n_samples=500, seed=42):
    """评估单个模型"""

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return None

    config = MODEL_CONFIGS[model_name]
    lookback = config['lookback']
    predict = 10
    max_context = config['max_context']

    # 数据路径
    data_dir = DATA_DIR_TEMPLATE.format(lookback=lookback)
    test_path = os.path.join(project_root, data_dir, 'test_data.pkl')

    # 模型路径
    model_path = os.path.join(project_root, MODELS_DIR, model_name, 'checkpoints', checkpoint_type)

    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {checkpoint_type}")
    print(f"Lookback: {lookback}, Max context: {max_context}")
    print(f"Test data: {test_path}")
    print(f"{'='*80}")

    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载tokenizer
    tokenizer = KronosTokenizer.from_pretrained(os.path.join(project_root, TOKENIZER_PATH))
    tokenizer.to(device)
    tokenizer.eval()

    # 加载模型
    model = Kronos.from_pretrained(model_path)
    model.to(device)
    model.eval()

    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model size: {model_size:.2f}M")

    # 加载测试数据
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # 获取测试窗口索引
    indices = get_test_indices(test_data, lookback, predict)
    print(f"Test windows: {len(indices)}")

    # 评估
    result = evaluate_ma60(
        model, tokenizer, test_data, indices,
        lookback=lookback, predict=predict,
        n_samples=n_samples, seed=seed, clip=5.0
    )

    # 打印结果
    print_evaluation_result(result, predict=predict, title=f"{model_name} Test Results")

    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on test set')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['mode2_mini_lb400', 'mode8_small_lb400', 'mode10_small_lb246'],
                        help='Models to evaluate')
    parser.add_argument('--checkpoint', type=str, default='best_ic_model',
                        help='Checkpoint type: best_ic_model, best_model, latest_model')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    for model_name in args.models:
        result = evaluate_model(
            model_name,
            checkpoint_type=args.checkpoint,
            n_samples=args.n_samples,
            seed=args.seed
        )
        if result:
            results[model_name] = result

    # 汇总对比
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'close_IC':>10} {'close_DA1':>10} {'open_IC':>10}")
    print("-" * 50)
    for model_name, r in results.items():
        close_ic = r.get('close_trajectory_ic', 0)
        close_da1 = r.get('close_da_step1', 0)
        open_ic = r.get('open_trajectory_ic', 0)
        print(f"{model_name:<20} {close_ic:>10.4f} {close_da1:>10.1%} {open_ic:>10.4f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()