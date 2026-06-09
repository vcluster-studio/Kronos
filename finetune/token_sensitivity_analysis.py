"""
Token敏感性分析 — 诊断 Tokenizer 主要编码什么信息

实验设计：
1. 原始数据 → token_a
2. close 扰动数据 → token_b
3. vol 扰动数据 → token_c

统计 token distance，判断 Tokenizer 对各特征的敏感度。

Usage:
    python -u finetune/token_sensitivity_analysis.py
"""

import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
import pickle
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']

def main():
    # 加载 tokenizer
    from model.kronos import KronosTokenizer
    tokenizer = KronosTokenizer.from_pretrained('outputs/models/ma60_tokenizer_v1/checkpoints/best_model')
    tokenizer.eval().to(DEVICE)
    print(f"Tokenizer loaded")

    # 加载验证数据
    with open('finetune/data/processed_datasets_ma60_windowed_v3/val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    print(f"Val data: {len(val_data)} stocks")

    # 采样数据
    rng = np.random.RandomState(42)
    symbols = rng.choice(list(val_data.keys()), 100, replace=False)

    # 扰动方式
    def shuffle_feature(data, feature_idx):
        """对指定特征进行随机扰动（保持其他特征不变）"""
        perturbed = data.copy()
        # 随机打乱该特征的顺序
        shuffled_values = perturbed[:, feature_idx].copy()
        np.random.shuffle(shuffled_values)
        perturbed[:, feature_idx] = shuffled_values
        return perturbed

    def add_noise_feature(data, feature_idx, noise_scale=0.1):
        """对指定特征添加噪声"""
        perturbed = data.copy()
        noise = np.random.randn(len(data)) * noise_scale * np.std(data[:, feature_idx])
        perturbed[:, feature_idx] = perturbed[:, feature_idx] + noise
        return perturbed

    def get_tokens(tokenizer, data):
        """编码数据得到 tokens"""
        with torch.no_grad():
            x = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).to(DEVICE)
            x = torch.clamp(x, -5.0, 5.0)  # clip
            s1, s2 = tokenizer.encode(x, half=True)
        return s1.cpu().numpy()[0], s2.cpu().numpy()[0]

    def token_distance(tokens_a, tokens_b):
        """计算 token 序列的距离"""
        s1_a, s2_a = tokens_a
        s1_b, s2_b = tokens_b
        # Hamming distance: 不同 token 的比例
        s1_diff = np.mean(s1_a != s1_b)
        s2_diff = np.mean(s2_a != s2_b)
        return s1_diff, s2_diff, (s1_diff + s2_diff) / 2

    def decode_distance(tokenizer, tokens_a, tokens_b):
        """计算 decode 后的实际值距离"""
        with torch.no_grad():
            s1_a = torch.from_numpy(tokens_a[0]).unsqueeze(0).to(DEVICE)
            s2_a = torch.from_numpy(tokens_a[1]).unsqueeze(0).to(DEVICE)
            s1_b = torch.from_numpy(tokens_b[0]).unsqueeze(0).to(DEVICE)
            s2_b = torch.from_numpy(tokens_b[1]).unsqueeze(0).to(DEVICE)
            z_a = tokenizer.decode([s1_a, s2_a], half=True).cpu().numpy()[0]
            z_b = tokenizer.decode([s1_b, s2_b], half=True).cpu().numpy()[0]
        # 各特征的 MAE
        mae_per_feature = np.mean(np.abs(z_a - z_b), axis=0)
        return mae_per_feature

    # 收集统计
    results = {
        'shuffle': {f: {'hamming': [], 'decode_mae': []} for f in FEATURE_NAMES},
        'noise': {f: {'hamming': [], 'decode_mae': []} for f in FEATURE_NAMES},
    }

    print("\n" + "="*60)
    print("Token Sensitivity Analysis")
    print("="*60)

    for sym in tqdm(symbols, desc="Processing"):
        d = val_data[sym]
        original = d['normalized'].astype(np.float32)

        # 原始 tokens
        tokens_orig = get_tokens(tokenizer, original)

        # 对每个特征进行扰动测试
        for fi, fn in enumerate(FEATURE_NAMES):
            # Shuffle 测试
            np.random.seed(42 + fi)  # 固定扰动种子
            perturbed_shuffle = shuffle_feature(original, fi)
            tokens_shuffle = get_tokens(tokenizer, perturbed_shuffle)
            hamming = token_distance(tokens_orig, tokens_shuffle)
            decode_mae = decode_distance(tokenizer, tokens_orig, tokens_shuffle)

            results['shuffle'][fn]['hamming'].append(hamming[2])  # avg hamming
            results['shuffle'][fn]['decode_mae'].append(decode_mae)

            # Noise 测试
            np.random.seed(42 + fi + 6)
            perturbed_noise = add_noise_feature(original, fi, noise_scale=0.1)
            tokens_noise = get_tokens(tokenizer, perturbed_noise)
            hamming_noise = token_distance(tokens_orig, tokens_noise)
            decode_mae_noise = decode_distance(tokenizer, tokens_orig, tokens_noise)

            results['noise'][fn]['hamming'].append(hamming_noise[2])
            results['noise'][fn]['decode_mae'].append(decode_mae_noise)

    # 输出结果
    print("\n" + "="*60)
    print("Results: Shuffle Test (随机打乱某特征)")
    print("="*60)
    print(f"{'Feature':<8} {'Avg Hamming':>12} {'Decode MAE':>12}")
    print("-"*60)
    for fn in FEATURE_NAMES:
        avg_hamming = np.mean(results['shuffle'][fn]['hamming'])
        avg_decode_mae = np.mean(results['shuffle'][fn]['decode_mae'])
        print(f"{fn:<8} {avg_hamming:>12.4f} {avg_decode_mae:>12.4f}")

    print("\n" + "="*60)
    print("Results: Noise Test (添加10%噪声)")
    print("="*60)
    print(f"{'Feature':<8} {'Avg Hamming':>12} {'Decode MAE':>12}")
    print("-"*60)
    for fn in FEATURE_NAMES:
        avg_hamming = np.mean(results['noise'][fn]['hamming'])
        avg_decode_mae = np.mean(results['noise'][fn]['decode_mae'])
        print(f"{fn:<8} {avg_hamming:>12.4f} {avg_decode_mae:>12.4f}")

    # 解读
    print("\n" + "="*60)
    print("Diagnosis")
    print("="*60)

    shuffle_hamming = {fn: np.mean(results['shuffle'][fn]['hamming']) for fn in FEATURE_NAMES}
    price_avg = np.mean([shuffle_hamming[f] for f in ['open', 'high', 'low', 'close']])
    vol_avg = np.mean([shuffle_hamming[f] for f in ['vol', 'amt']])

    print(f"Price features avg Hamming: {price_avg:.4f}")
    print(f"Volume features avg Hamming: {vol_avg:.4f}")
    print(f"Ratio (vol/price): {vol_avg/price_avg:.2f}")

    if vol_avg > price_avg * 1.5:
        print("\n>>> 结论: Tokenizer 对 vol/amt 更敏感，可能容量偏向量能编码")
    elif price_avg > vol_avg * 1.5:
        print("\n>>> 结论: Tokenizer 对价格更敏感，问题可能在 Predictor")
    else:
        print("\n>>> 结论: Tokenizer 对两类特征敏感度相近，问题可能在 Predictor CE 目标")

    # 保存详细结果
    output = {
        'shuffle_hamming': shuffle_hamming,
        'price_avg': price_avg,
        'vol_avg': vol_avg,
        'diagnosis': 'tokenizer_vol_bias' if vol_avg > price_avg * 1.5 else 'predictor_ce_bias'
    }
    with open('finetune/token_sensitivity_results.pkl', 'wb') as f:
        pickle.dump(output, f)
    print(f"\nResults saved to: finetune/token_sensitivity_results.pkl")

if __name__ == '__main__':
    main()