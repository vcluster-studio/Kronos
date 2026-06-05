"""分析 6 维特征的数据分布，验证 vol/amt 是否因为量级/波动异常而主导模型"""
import pickle
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

feature_names = ['open', 'high', 'low', 'close', 'vol', 'amt']

data_path = os.path.join(project_root, 'finetune/data/processed_datasets_ma60_windowed_v3_small/val_data.pkl')
with open(data_path, 'rb') as f:
    val_data = pickle.load(f)

stats = {f: {'norm_vals': [], 'raw_vals': [], 'returns': []} for f in range(6)}
n_stocks = 0

for sym, data in val_data.items():
    if n_stocks >= 200:
        break
    n_stocks += 1
    norm = data['normalized']
    orig = data['original']

    for f in range(6):
        stats[f]['norm_vals'].extend(norm[-200:, f].tolist())
        stats[f]['raw_vals'].extend(orig[-200:, f].tolist())
        raw_f = orig[-200:, f]
        ret = np.diff(raw_f) / (raw_f[:-1] + 1e-8)
        stats[f]['returns'].extend(ret.tolist())

print('=== Normalized Values Stats ===')
header = f"{'Feat':>5s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'kurt':>8s} {'|>5|%':>8s} {'|>3|%':>8s}"
print(header)
for f in range(6):
    vals = np.array(stats[f]['norm_vals'])
    kurt = np.mean((vals - np.mean(vals))**4) / (np.std(vals)**4 + 1e-8)
    print(f'{feature_names[f]:>5s} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f} '
          f'{kurt:>8.2f} {np.mean(np.abs(vals) > 5)*100:>7.2f}% {np.mean(np.abs(vals) > 3)*100:>7.2f}%')

print()
print('=== Raw Value Returns Stats ===')
header = f"{'Feat':>5s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'kurt':>8s}"
print(header)
for f in range(6):
    rets = np.array(stats[f]['returns'])
    rets = rets[np.isfinite(rets)]
    rets = np.clip(rets, -1, 1)
    kurt = np.mean((rets - np.mean(rets))**4) / (np.std(rets)**4 + 1e-8)
    print(f'{feature_names[f]:>5s} {np.mean(rets):>8.4f} {np.std(rets):>8.4f} {np.min(rets):>8.4f} {np.max(rets):>8.4f} {kurt:>8.2f}')

print()
print('=== Raw Value Magnitude ===')
header = f"{'Feat':>5s} {'mean':>12s} {'std':>12s} {'cv':>8s}"
print(header)
for f in range(6):
    vals = np.array(stats[f]['raw_vals'])
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    print(f'{feature_names[f]:>5s} {mean_v:>12.2f} {std_v:>12.2f} {std_v/(abs(mean_v)+1e-8):>8.4f}')

# Token 分布分析：看 tokenizer 对不同特征的 token 分布
print()
print('=== MA60 Local Window Stats (means/stds per feature) ===')
for sym, data in list(val_data.items())[:5]:
    means = data['means'][-200:]
    stds = data['stds'][-200:]
    print(f"  {sym}:")
    for f in range(6):
        m_mean = np.mean(means[:, f])
        s_mean = np.mean(stds[:, f])
        s_std = np.std(stds[:, f])
        print(f"    {feature_names[f]:>5s}: local_mean={m_mean:.4f}, local_std_mean={s_mean:.4f}, local_std_std={s_std:.4f}")

# 关键：归一化后各特征的 token 覆盖范围
print()
print('=== Token Coverage: how many unique value bins per feature ===')
# 模拟 tokenizer 的分桶：clip 到 [-5,5]，然后离散化
for f in range(6):
    vals = np.array(stats[f]['norm_vals'])
    clipped = np.clip(vals, -5, 5)
    # 计算有效信息量 (熵)
    hist, _ = np.histogram(clipped, bins=100, range=(-5, 5))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    print(f'{feature_names[f]:>5s}: entropy={entropy:.2f} bits, range=[{np.min(clipped):.2f}, {np.max(clipped):.2f}]')
