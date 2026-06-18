"""
全量测试集评估脚本 - 支持所有模型类型

自动扫描 outputs/models/ 下所有模型目录，进行全量测试集评估。

Usage:
    python finetune/predictor/shared/eval_all_models.py              # 全量评估所有模型
    python finetune/predictor/shared/eval_all_models.py --quick 1000 # 抽样1000窗口
    python finetune/predictor/shared/eval_all_models.py --test       # 最小验证(10窗口)
    python finetune/predictor/shared/eval_all_models.py --models final/mini final/small
"""

import os
import sys
import argparse
import json
import pickle
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference

# ============================================================================
# 配置
# ============================================================================

DEFAULT_TOKENIZER = 'outputs/tokenizers/ma60_tokenizer_base_v1/checkpoints/best_model'
DATA_DIR_TEMPLATE = 'finetune/data/ma60_norm/windowed_lb{lookback}_pd10'
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']
PREDICT = 10

# ============================================================================
# 模型扫描
# ============================================================================

def scan_models(models_dir='outputs/models'):
    """
    扫描所有模型目录，返回模型信息列表

    Returns:
        list of dict: [{'name', 'path', 'category', 'model_type', 'lookback', 'norm_mode'}, ...]
    """
    models = []
    categories = ['final', 'archived', 'deprecated', 'experiments']

    for category in categories:
        category_path = os.path.join(project_root, models_dir, category)
        if not os.path.exists(category_path):
            continue

        for model_name in os.listdir(category_path):
            model_path = os.path.join(category_path, model_name)
            if not os.path.isdir(model_path):
                continue

            # 查找模型checkpoint
            checkpoint_path = find_checkpoint(model_path)
            if not checkpoint_path:
                continue

            # 解析模型信息
            info = parse_model_info(model_name, model_path, checkpoint_path, category)
            if info:
                models.append(info)

    return models


def find_checkpoint(model_path):
    """查找模型checkpoint路径"""
    # 尝试多种可能的路径结构
    candidates = [
        os.path.join(model_path, 'best_ic_model'),
        os.path.join(model_path, 'checkpoints', 'best_ic_model'),
        os.path.join(model_path, 'checkpoints', 'best_model'),
    ]

    for cand in candidates:
        if os.path.exists(cand) and os.path.exists(os.path.join(cand, 'config.json')):
            return cand

    return None


def parse_model_info(model_name, model_path, checkpoint_path, category):
    """解析模型信息 - 仅读取model_config.json"""
    config_path = os.path.join(model_path, 'model_config.json')
    if not os.path.exists(config_path):
        print(f"Warning: No model_config.json found for {category}/{model_name}, skipping")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read model_config.json for {model_name}: {e}")
        return None

    return {
        'name': model_name,
        'path': checkpoint_path,
        'category': category,
        'model_type': config.get('model_type', 'mini'),
        'norm_mode': config.get('norm_mode', 'ma60'),
        'lookback': config.get('lookback', 400),
        'predict': config.get('predict', 10),
        'tokenizer_path': config.get('tokenizer', DEFAULT_TOKENIZER),
        'test_data_path': config.get('test_data', None),
        'max_context': config.get('max_context', 2048),
        'training_mode': config.get('training_mode', 'sampled'),
        'val_ic': config.get('val_ic', None),
        'notes': config.get('notes', ''),
    }


# ============================================================================
# 数据加载
# ============================================================================

def load_test_data(lookback, norm_mode='ma60', test_data_path=None):
    """加载测试数据"""
    # 优先使用配置文件指定的路径
    if test_data_path:
        data_path = os.path.join(project_root, test_data_path)
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        # 否则回退到自动匹配

    if norm_mode == 'ma60':
        data_path = os.path.join(project_root, DATA_DIR_TEMPLATE.format(lookback=lookback), 'test_data.pkl')
        if not os.path.exists(data_path):
            data_path = os.path.join(project_root, DATA_DIR_TEMPLATE.format(lookback=400), 'test_data.pkl')
    elif norm_mode == 'ma20':
        data_path = os.path.join(project_root, 'finetune/data/ma20_norm/windowed_lb{lookback}_pd10/test_data.pkl'.format(lookback=lookback))
    elif norm_mode == 'full_window':
        data_path = os.path.join(project_root, 'finetune/data/global_norm/full_series/test_data.pkl')
    else:
        data_path = os.path.join(project_root, DATA_DIR_TEMPLATE.format(lookback=400), 'test_data.pkl')

    if not os.path.exists(data_path):
        print(f"Warning: Test data not found at {data_path}")
        return None

    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    return test_data


def get_test_indices(test_data, lookback, predict):
    """获取测试集窗口索引"""
    indices = []
    window = lookback + predict

    sample_val = next(iter(test_data.values()))

    # DataFrame 格式（full_window 归一化）
    if hasattr(sample_val, 'columns'):
        for symbol in test_data.keys():
            df = test_data[symbol]
            if len(df) >= window:
                indices.append((symbol, len(df) - window))

    # Dict 格式（MA windowed）
    elif 'windows' in sample_val:
        for symbol in test_data.keys():
            d = test_data[symbol]
            for start in d['windows']:
                indices.append((symbol, int(start)))

    else:
        for symbol in test_data.keys():
            d = test_data[symbol]
            seq_len = len(d['normalized'])
            if seq_len >= window:
                for i in range(seq_len - window + 1):
                    indices.append((symbol, i))

    return indices


# ============================================================================
# 评估
# ============================================================================

def evaluate_model(model_info, tokenizer, test_data, indices, device='cuda',
                   sample_size=None, seed=42, desc=None):
    """
    评估单个模型

    Args:
        model_info: 模型信息字典
        tokenizer: 分词器
        test_data: 测试数据
        indices: 测试窗口索引
        sample_size: 抽样数量（None表示全量）
        desc: 进度条描述

    Returns:
        dict: 评估结果
    """
    model_path = model_info['path']
    lookback = model_info['lookback']
    norm_mode = model_info['norm_mode']
    predict = PREDICT

    # 抽样
    if sample_size and sample_size < len(indices):
        random.seed(seed)
        eval_indices = random.sample(indices, sample_size)
    else:
        eval_indices = indices

    if desc is None:
        desc = model_info['name']

    # 加载模型
    model = Kronos.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 评估
    trajectory_ics = {f: [] for f in FEATURE_NAMES}
    trajectory_rics = {f: [] for f in FEATURE_NAMES}
    da_by_step = [{f: [] for f in FEATURE_NAMES} for _ in range(predict)]

    for (sym, start) in tqdm(eval_indices, desc=desc, disable=len(eval_indices) < 100):
        d = test_data[sym]
        window = lookback + predict
        end = start + window

        try:
            # MA60 预归一化数据
            if norm_mode in ['ma60', 'ma20']:
                if end > len(d['normalized']):
                    continue

                norm = d['normalized'][start:end]
                orig = d['original'][start:end]
                means = d['means'][start:end]
                stds = d['stds'][start:end]
                ts = d['index'][start:end]

                stamp = np.stack([
                    ts.minute.values.astype(np.float32),
                    ts.hour.values.astype(np.float32),
                    ts.weekday.values.astype(np.float32),
                    ts.day.values.astype(np.float32),
                    ts.month.values.astype(np.float32),
                ], axis=1)

                x_norm = norm[:lookback]
                baseline = orig[lookback - 1]

                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                    x_stamp = torch.from_numpy(stamp[:lookback]).unsqueeze(0).to(device)
                    y_stamp = torch.from_numpy(stamp[lookback:lookback+predict]).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model, x_tensor, x_stamp, y_stamp,
                        max_context=model_info.get('max_context', 2048),
                        pred_len=predict, clip=5.0,
                        T=1.0, top_p=0.9, sample_count=1
                    )

                pred_norm = preds[0, lookback:lookback+predict]
                pred_raw = pred_norm * stds[lookback:lookback+predict] + means[lookback:lookback+predict]
                actual = orig[lookback:lookback+predict]

            # full_window 归一化
            elif norm_mode == 'full_window':
                df = d
                slice_df = df.iloc[start:end]
                x = slice_df.values[:lookback]
                y = slice_df.values[lookback:]
                baseline = slice_df.values[lookback - 1]

                x_mean = np.mean(x, axis=0)
                x_std = np.std(x, axis=0) + 1e-5
                x_norm_raw = (x - x_mean) / x_std
                x_norm_raw = np.clip(x_norm_raw, -5.0, 5.0)

                ts = df.index[start:end]
                x_norm = torch.from_numpy(x_norm_raw).unsqueeze(0).to(device)
                stamp = np.stack([
                    ts.minute.values.astype(np.float32),
                    ts.hour.values.astype(np.float32),
                    ts.weekday.values.astype(np.float32),
                    ts.day.values.astype(np.float32),
                    ts.month.values.astype(np.float32),
                ], axis=1)

                with torch.no_grad():
                    x_stamp = torch.from_numpy(stamp[:lookback]).unsqueeze(0).to(device)
                    y_stamp = torch.from_numpy(stamp[lookback:lookback+predict]).unsqueeze(0).to(device)

                    preds = auto_regressive_inference(
                        tokenizer, model, x_norm, x_stamp, y_stamp,
                        max_context=model_info.get('max_context', 2048),
                        pred_len=predict, clip=5.0,
                        T=1.0, top_p=0.9, sample_count=1
                    )

                pred_norm = preds[0, lookback:lookback+predict]
                pred_raw = pred_norm * x_std + x_mean
                actual = y

            # Trajectory IC
            for fi, fn in enumerate(FEATURE_NAMES):
                pred_traj = pred_raw[:, fi]
                actual_traj = actual[:, fi]

                if len(pred_traj) >= 3:
                    # 检查轨迹方差，避免除零警告
                    pred_std = np.std(pred_traj)
                    actual_std = np.std(actual_traj)
                    if pred_std > 1e-8 and actual_std > 1e-8:
                        traj_ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
                        if np.isfinite(traj_ic):
                            trajectory_ics[fn].append(traj_ic)

                        traj_ric, _ = spearmanr(pred_traj, actual_traj)
                        if np.isfinite(traj_ric):
                            trajectory_rics[fn].append(traj_ric)

            # DA by step
            for step_idx in range(predict):
                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                    actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                    da_by_step[step_idx][fn].append(pred_dir == actual_dir)

        except Exception as e:
            continue

    # 计算结果
    results = {}
    for fn in FEATURE_NAMES:
        results[f'{fn}_trajectory_ic'] = np.mean(trajectory_ics[fn]) if trajectory_ics[fn] else 0.0
        results[f'{fn}_trajectory_rank_ic'] = np.mean(trajectory_rics[fn]) if trajectory_rics[fn] else 0.0

    for step_idx in range(predict):
        for fn in FEATURE_NAMES:
            results[f'{fn}_da_step{step_idx+1}'] = np.mean(da_by_step[step_idx][fn]) if da_by_step[step_idx][fn] else 0.0

    results['n_samples'] = len(eval_indices)

    # 清理
    del model
    torch.cuda.empty_cache()

    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full test set evaluation for all models')
    parser.add_argument('--quick', type=int, default=None,
                        help='Quick evaluation with sampled windows (e.g., 1000)')
    parser.add_argument('--test', action='store_true',
                        help='Minimal test with 10 windows for script validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--models', type=str, nargs='*',
                        help='Specific models to evaluate (e.g., final/mini deprecated/mini_ma60_lb400)')
    parser.add_argument('--output', type=str, default='outputs/models/full_test_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # --test 模式使用最小抽样
    if args.test:
        args.quick = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 扫描模型
    if args.models:
        # 指定模型
        models = []
        for model_spec in args.models:
            parts = model_spec.split('/')
            if len(parts) == 2:
                category, name = parts
                model_path = os.path.join(project_root, 'outputs/models', category, name)
                checkpoint_path = find_checkpoint(model_path)
                if checkpoint_path:
                    info = parse_model_info(name, model_path, checkpoint_path, category)
                    if info:
                        models.append(info)
    else:
        # 扫描所有模型
        models = scan_models()

    if not models:
        print("No models found!")
        return

    print(f"\nFound {len(models)} models:")
    for m in models:
        print(f"  - {m['category']}/{m['name']} (type={m['model_type']}, lb={m['lookback']}, norm={m['norm_mode']})")

    # 评估结果
    all_results = {}

    for model_info in models:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_info['category']}/{model_info['name']}")
        print(f"Model type: {model_info['model_type']}")
        print(f"Lookback: {model_info['lookback']}")
        print(f"Norm mode: {model_info['norm_mode']}")
        print(f"Tokenizer: {model_info.get('tokenizer_path', DEFAULT_TOKENIZER)}")
        print(f"{'='*80}")

        # 加载tokenizer（根据模型norm_mode）
        tokenizer_path = model_info.get('tokenizer_path', DEFAULT_TOKENIZER)
        full_tokenizer_path = os.path.join(project_root, tokenizer_path)

        # 检查tokenizer是否存在
        if not os.path.exists(full_tokenizer_path):
            # 尝试HuggingFace远程加载（需要网络）
            try:
                print(f"Tokenizer not found locally, trying HuggingFace: {tokenizer_path}")
                tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                print(f"Warning: Tokenizer not available for {model_info['name']}")
                print(f"  Path: {tokenizer_path}")
                print(f"  Error: {e}")
                print(f"  Skipping this model")
                continue
        else:
            tokenizer = KronosTokenizer.from_pretrained(full_tokenizer_path)

        tokenizer.to(device)
        tokenizer.eval()
        print(f"Tokenizer loaded")

        # 加载测试数据
        test_data = load_test_data(
            model_info['lookback'],
            model_info['norm_mode'],
            model_info.get('test_data_path')
        )
        if test_data is None:
            print(f"Skipping - test data not found")
            continue

        # 获取测试索引
        indices = get_test_indices(test_data, model_info['lookback'], PREDICT)
        print(f"Total test windows: {len(indices)}")

        # 评估
        results = evaluate_model(
            model_info, tokenizer, test_data, indices,
            device=device, sample_size=args.quick, seed=args.seed
        )

        # 保存结果
        key = f"{model_info['category']}/{model_info['name']}"
        all_results[key] = {
            'model_type': model_info['model_type'],
            'lookback': model_info['lookback'],
            'norm_mode': model_info['norm_mode'],
            **results
        }

        # 打印结果
        print(f"\n[Trajectory IC]")
        print(f"{'Feature':<8} {'IC':>10} {'RankIC':>10}")
        print("-" * 28)
        for fn in FEATURE_NAMES:
            ic = results.get(f'{fn}_trajectory_ic', 0)
            ric = results.get(f'{fn}_trajectory_rank_ic', 0)
            print(f"{fn:<8} {ic:>10.4f} {ric:>10.4f}")

        print(f"\n[Per-step DA for close]")
        for step_idx in range(PREDICT):
            da = results.get(f'close_da_step{step_idx+1}', 0)
            print(f"  +{step_idx+1}: {da:.1%}")

        print(f"\nN Samples: {results['n_samples']}")

    # 汇总
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<40} {'close IC':>10} {'close DA+1':>10}")
    print("-" * 60)
    for key, r in all_results.items():
        close_ic = r.get('close_trajectory_ic', 0)
        close_da1 = r.get('close_da_step1', 0)
        print(f"{key:<40} {close_ic:>10.4f} {close_da1:>9.1%}")

    # 保存JSON
    output_path = os.path.join(project_root, args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()