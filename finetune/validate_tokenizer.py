"""
Tokenizer 重建误差验证脚本

功能：验证训练后的 Tokenizer 重建精度
用法：python -u finetune/validate_tokenizer.py
"""

import os
import sys
import pickle
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer

def load_sample_data(data_path, num_samples=100):
    """加载验证样本"""
    val_path = f"{data_path}/val_data.pkl"
    with open(val_path, 'rb') as f:
        data = pickle.load(f)

    samples = []
    symbols = list(data.keys())[:20]
    window = 60

    for symbol in symbols:
        df = data[symbol]
        values = df[['open', 'high', 'low', 'close', 'vol', 'amt']].values.astype(np.float32)

        if len(values) < window:
            continue

        for i in range(0, min(len(values) - window, 3)):
            window_data = values[i:i+window]
            mean = np.mean(window_data[:30], axis=0)
            std = np.std(window_data[:30], axis=0) + 1e-5
            window_data = (window_data - mean) / std
            window_data = np.clip(window_data, -3, 3)
            samples.append(window_data)

    return samples

def evaluate_tokenizer(tokenizer, samples, device):
    """评估 tokenizer 的重建误差"""
    tokenizer.to(device)
    tokenizer.eval()

    recon_errors = []

    with torch.no_grad():
        for sample in samples:
            x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
            zs, _, _, _ = tokenizer(x)
            z_pre, z = zs

            recon_loss = torch.mean((z - x) ** 2).item()
            recon_errors.append(recon_loss)

    return np.mean(recon_errors), np.std(recon_errors)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 数据集配置
    datasets = {
        'mid': 'processed_datasets_mid',
        'full': 'processed_datasets',
        'small': 'processed_datasets_small',
    }

    # Tokenizer 配置
    tokenizers = {
        '原始': 'pretrained/Kronos-Tokenizer-base',
        'mid训练后': 'outputs/models/mid_tokenizer_v1/checkpoints/best_model',
    }

    print("\n" + "="*60)
    print("Tokenizer Reconstruction Error Validation")
    print("="*60)

    for ds_name, ds_path in datasets.items():
        data_path = os.path.join(project_root, "finetune", "data", ds_path)
        if not os.path.exists(data_path):
            continue

        print(f"\n--- {ds_name} Dataset ---")
        samples = load_sample_data(data_path)
        print(f"Samples: {len(samples)}")

        for tok_name, tok_path in tokenizers.items():
            full_path = os.path.join(project_root, tok_path)
            if not os.path.exists(full_path):
                print(f"  {tok_name}: Not found")
                continue

            tokenizer = KronosTokenizer.from_pretrained(full_path)
            mean_error, std_error = evaluate_tokenizer(tokenizer, samples, device)
            print(f"  {tok_name}: {mean_error:.6f} +/- {std_error:.6f}")

if __name__ == '__main__':
    main()