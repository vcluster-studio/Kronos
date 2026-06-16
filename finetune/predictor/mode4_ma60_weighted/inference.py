"""
按真实收益幅度分桶评估
分析不同波动区间内的预测质量
"""

import os, sys, json, pickle, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOOKBACK = 60
PREDICT = 5


def load_data(data_path, lookback, predict, train_ratio=0.70, val_ratio=0.15):
    """加载 MA60 数据

    返回:
        all_data: 所有数据
        train_indices: 训练集索引
        val_indices: 验证集索引 (用于训练时模型选择)
        test_indices: 测试集索引 (用于最终评估)
    """
    with open(data_path, 'rb') as f:
        raw = pickle.load(f)

    all_data = {}
    train_indices = []
    val_indices = []
    test_indices = []
    window = lookback + predict

    for sym in sorted(raw.keys()):
        d = raw[sym]
        seq_len = len(d['normalized'])
        if seq_len < window + 1:
            continue

        all_data[sym] = {
            'normalized': d['normalized'].astype(np.float32),
            'original': d['original'].astype(np.float32),
            'means': d['means'].astype(np.float32),
            'stds': d['stds'].astype(np.float32),
            'index': d['index'],
        }

        n_windows = seq_len - window
        tr_end = int(n_windows * train_ratio)
        val_end = int(n_windows * (train_ratio + val_ratio))

        for i in range(n_windows):
            if i < tr_end:
                train_indices.append((sym, i))
            elif tr_end <= i < val_end:
                val_indices.append((sym, i))
            else:
                test_indices.append((sym, i))

    return all_data, train_indices, val_indices, test_indices


def precompute_bit_mask(vocab_size, n_bits, device):
    indices = torch.arange(vocab_size, device=device, dtype=torch.long)
    bit_mask = torch.zeros(vocab_size, n_bits, device=device, dtype=torch.float32)
    for b in range(n_bits):
        bit_mask[:, b] = ((indices >> b) & 1).float()
    return bit_mask


class BucketEvaluator:
    def __init__(self, model_path, tokenizer_path="outputs/models/ma60_tokenizer_v1/checkpoints/best_model"):
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        self.model = Kronos.from_pretrained("pretrained/Kronos-mini")

        # Load finetuned weights - model_path can be a checkpoint dir directly
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
            # Try safetensors first (newer format)
        safetensors_path = checkpoint_path / "model.safetensors"
        bin_path = checkpoint_path / "pytorch_model.bin"

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded safetensors from {safetensors_path}")
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {bin_path}")
        else:
            print(f"Warning: no model weights found in {checkpoint_path}")

        self.model = self.model.to(DEVICE)
        self.model.eval()

        # Move tokenizer to same device
        self.tokenizer = self.tokenizer.to(DEVICE)
        self.tokenizer.eval()

        # Freeze tokenizer
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        # Precompute bit_masks
        s1_bits = 10
        s2_bits = 10
        vocab_s1 = 2 ** s1_bits
        vocab_s2 = 2 ** s2_bits
        self.s1_bit_mask = precompute_bit_mask(vocab_s1, s1_bits, DEVICE)
        self.s2_bit_mask = precompute_bit_mask(vocab_s2, s2_bits, DEVICE)
        self.q_scale = 1.0 / math.sqrt(s1_bits + s2_bits)

        # Bucket thresholds (absolute return)
        self.buckets = {
            "<1%": (0, 0.01),
            "1%~2%": (0.01, 0.02),
            "2%~5%": (0.02, 0.05),
            ">5%": (0.05, float("inf"))
        }

        # Storage
        self.bucket_data = {k: {"pred_delta": [], "true_delta": [],
                                "pred_close": [], "true_close": [],
                                "pred_open": [], "true_open": [],
                                "pred_vol": [], "true_vol": [],
                                "actual_return": []}
                          for k in self.buckets.keys()}

    def soft_decode(self, s1_logits, s2_logits):
        """Soft decode: logits → probs @ bit_mask → bipolar → decode"""
        s1_probs = F.softmax(s1_logits, dim=-1)
        s2_probs = F.softmax(s2_logits, dim=-1)

        s1_soft_bits = s1_probs @ self.s1_bit_mask
        s2_soft_bits = s2_probs @ self.s2_bit_mask

        soft_bits = torch.cat([s1_soft_bits, s2_soft_bits], dim=-1)
        soft_bits = soft_bits * 2 - 1  # → bipolar
        soft_bits = soft_bits * self.q_scale

        decoded = self.tokenizer.decode_from_bits(soft_bits)
        return decoded

    def evaluate_sample(self, all_data, sym, start):
        """评估单个样本"""
        d = all_data[sym]
        lb = LOOKBACK
        p = PREDICT
        window = lb + p
        end = start + window

        norm = d['normalized'][start:end]
        orig = d['original'][start:end]
        means = d['means'][start:end]
        stds = d['stds'][start:end]

        # Timestamp
        ts = d['index'][start:end]
        stamp = np.stack([
            ts.minute.values.astype(np.float32),
            ts.hour.values.astype(np.float32),
            ts.weekday.values.astype(np.float32),
            ts.day.values.astype(np.float32),
            ts.month.values.astype(np.float32),
        ], axis=1)

        # Tokenize
        with torch.no_grad():
            norm_t = torch.from_numpy(norm).unsqueeze(0).to(DEVICE)
            stamp_t = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
            t0, t1 = self.tokenizer.encode(norm_t, half=True)

        # Forward: 输入 lookback 步，预测 predict 步
        with torch.no_grad():
            s1_logits, s2_logits = self.model(t0[:, :-p], t1[:, :-p], stamp=stamp_t[:, :-p, :])

        # Soft decode 预测
        pred_norm = self.soft_decode(s1_logits[:, -p:, :], s2_logits[:, -p:, :])  # (1, p, 6)

        # Denormalize
        means_t = torch.from_numpy(means[lb:lb+p]).unsqueeze(0).to(DEVICE)
        stds_t = torch.from_numpy(stds[lb:lb+p]).unsqueeze(0).to(DEVICE)
        pred_raw = pred_norm * stds_t + means_t  # (1, p, 6)

        # Baseline (最后一个历史 close)
        baseline_close = orig[lb - 1, 3]

        # Ground truth
        actual_close = orig[lb:lb + p, 3]
        actual_open = orig[lb:lb + p, 0]
        actual_vol = orig[lb:lb + p, 4]

        # Predicted
        pred_close = pred_raw[0, :, 3].cpu().numpy()
        pred_open = pred_raw[0, :, 0].cpu().numpy()
        pred_vol = pred_raw[0, :, 4].cpu().numpy()

        # Delta (t+1 作为主要指标)
        pred_delta = (pred_close[0] - baseline_close) / (abs(baseline_close) + 1e-8)
        true_delta = (actual_close[0] - baseline_close) / (abs(baseline_close) + 1e-8)

        return {
            "pred_delta": pred_delta,
            "true_delta": true_delta,
            "pred_close": pred_close[0],
            "true_close": actual_close[0],
            "pred_open": pred_open[0],
            "true_open": actual_open[0],
            "pred_vol": pred_vol[0],
            "true_vol": actual_vol[0],
            "baseline": baseline_close,
        }

    def run(self, all_data, indices, n_samples=5000):
        """批量评估"""
        if n_samples > 0 and n_samples < len(indices):
            sample_indices = np.random.choice(len(indices), n_samples, replace=False)
            indices = [indices[i] for i in sample_indices]

        for idx in tqdm(indices, desc="Evaluating"):
            sym, start = idx
            result = self.evaluate_sample(all_data, sym, start)

            # 分桶
            abs_return = abs(result["true_delta"])
            for bucket_name, (low, high) in self.buckets.items():
                if low <= abs_return < high:
                    self.bucket_data[bucket_name]["pred_delta"].append(result["pred_delta"])
                    self.bucket_data[bucket_name]["true_delta"].append(result["true_delta"])
                    self.bucket_data[bucket_name]["pred_close"].append(result["pred_close"])
                    self.bucket_data[bucket_name]["true_close"].append(result["true_close"])
                    self.bucket_data[bucket_name]["pred_open"].append(result["pred_open"])
                    self.bucket_data[bucket_name]["true_open"].append(result["true_open"])
                    self.bucket_data[bucket_name]["pred_vol"].append(result["pred_vol"])
                    self.bucket_data[bucket_name]["true_vol"].append(result["true_vol"])
                    self.bucket_data[bucket_name]["actual_return"].append(result["true_delta"])
                    break

        return self.bucket_data

    def compute_metrics(self):
        """计算各桶指标"""
        results = {}
        total_samples = sum(len(b["true_delta"]) for b in self.bucket_data.values())

        for bucket_name, data in self.bucket_data.items():
            if len(data["true_delta"]) == 0:
                results[bucket_name] = {
                    "n_samples": 0, "pct": 0, "MAE": None, "DA": None,
                    "IC": None, "bias": None, "close_MAE": None
                }
                continue

            pred_delta = np.array(data["pred_delta"])
            true_delta = np.array(data["true_delta"])
            pred_close = np.array(data["pred_close"])
            true_close = np.array(data["true_close"])

            n = len(pred_delta)
            pct = n / total_samples * 100 if total_samples > 0 else 0

            # MAE (百分比)
            mae = np.mean(np.abs(pred_delta - true_delta)) * 100

            # Direction Accuracy
            pred_sign = np.sign(pred_delta)
            true_sign = np.sign(true_delta)
            pred_sign[pred_sign == 0] = 1
            true_sign[true_sign == 0] = 1
            da = np.mean(pred_sign == true_sign) * 100

            # IC (Pearson)
            if n > 10:
                ic = np.corrcoef(pred_delta, true_delta)[0, 1]
                if np.isnan(ic):
                    ic = None
            else:
                ic = None

            # Bias (系统性偏移)
            bias = np.mean(pred_delta - true_delta) * 100

            # Close MAE (绝对价格误差 %)
            close_mae = np.mean(np.abs(pred_close - true_close) / (np.abs(true_close) + 1e-8)) * 100

            results[bucket_name] = {
                "n_samples": n,
                "pct": pct,
                "MAE": mae,
                "DA": da,
                "IC": ic,
                "bias": bias,
                "close_MAE": close_mae
            }

        return results, total_samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/models/ma60_predictor_lb60_pd5_v3_l10.05_best/checkpoints/best_model")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="评估哪个数据集: train/val/test (默认: test)")
    args = parser.parse_args()

    # Paths
    model_path = args.model
    data_path = "finetune/data/kline_daily_ma60.pkl"

    # Load data
    print("Loading data...")
    all_data, train_indices, val_indices, test_indices = load_data(data_path, LOOKBACK, PREDICT)

    # Select split
    split_map = {
        "train": ("训练集", train_indices),
        "val": ("验证集", val_indices),
        "test": ("测试集", test_indices)
    }
    split_name, indices = split_map[args.split]

    print(f"数据集划分:")
    print(f"  训练集: {len(train_indices)} 窗口 ({len(train_indices)/(len(train_indices)+len(val_indices)+len(test_indices))*100:.1f}%)")
    print(f"  验证集: {len(val_indices)} 窗口 ({len(val_indices)/(len(train_indices)+len(val_indices)+len(test_indices))*100:.1f}%)")
    print(f"  测试集: {len(test_indices)} 窗口 ({len(test_indices)/(len(train_indices)+len(val_indices)+len(test_indices))*100:.1f}%)")
    print(f"\n使用 {split_name} 进行评估: {len(indices)} 窗口")

    # Initialize evaluator
    print("Loading model...")
    evaluator = BucketEvaluator(model_path)

    # Run evaluation
    print(f"Running bucket evaluation on {split_name}...")
    bucket_data = evaluator.run(all_data, indices, n_samples=args.samples)

    # Compute metrics
    results, total = evaluator.compute_metrics()

    # Print results
    print("\n" + "="*80)
    print("分桶评估结果 (按真实收益幅度 |actual_return|)")
    print("="*80)
    print(f"\n{'Bucket':<10} {'N':>8} {'占比%':>8} {'MAE%':>8} {'DA%':>8} {'IC':>8} {'Bias%':>8}")
    print("-"*70)

    for bucket_name in ["<1%", "1%~2%", "2%~5%", ">5%"]:
        metrics = results[bucket_name]
        ic_str = f"{metrics['IC']:.3f}" if metrics['IC'] is not None else "N/A"
        mae_str = f"{metrics['MAE']:.2f}" if metrics['MAE'] is not None else "N/A"
        da_str = f"{metrics['DA']:.1f}" if metrics['DA'] is not None else "N/A"
        bias_str = f"{metrics['bias']:.2f}" if metrics['bias'] is not None else "N/A"

        print(f"{bucket_name:<10} {metrics['n_samples']:>8} {metrics['pct']:>7.1f}% "
              f"{mae_str:>8} {da_str:>8} {ic_str:>8} {bias_str:>8}")

    print("\n" + "="*80)
    print("关键洞察")
    print("="*80)

    # <1% 占比分析
    low_pct = results["<1%"]["pct"]
    if low_pct > 50:
        print(f"\n[!] 低波动样本 (<1%) 占比 {low_pct:.1f}%，主导数据集")
        print(f"   该桶 DA={results['<1%']['DA']:.1f}%（接近随机 50%）")
        print(f"   该桶 MAE={results['<1%']['MAE']:.2f}%")
        print(f"   建议: 对低波动样本降权或 mask")

    # >2% 桶质量
    mid_bucket = results["2%~5%"]
    if mid_bucket["n_samples"] > 0:
        print(f"\n[+] 中等波动 (2%~5%): {mid_bucket['n_samples']} 样本 ({mid_bucket['pct']:.1f}%)")
        print(f"   MAE={mid_bucket['MAE']:.2f}%, DA={mid_bucket['DA']:.1f}%, IC={mid_bucket['IC']:.3f}")
        if mid_bucket["DA"] > 60:
            print(f"   方向判断有预测价值!")
        else:
            print(f"   方向判断接近随机，但 MAE 较低")

    high_bucket = results[">5%"]
    if high_bucket["n_samples"] > 0:
        print(f"\n[+] 高波动 (>5%): {high_bucket['n_samples']} 样本 ({high_bucket['pct']:.1f}%)")
        print(f"   MAE={high_bucket['MAE']:.2f}%, DA={high_bucket['DA']:.1f}%, IC={high_bucket['IC']:.3f}")
    else:
        print(f"\n[!] >5% 样本太少 (可能需要更长预测窗口)")

    # 整体建议
    print("\n" + "="*80)
    print("优化建议")
    print("="*80)

    avg_mae = np.mean([r['MAE'] for r in results.values() if r['MAE'] is not None])
    print(f"\n当前整体 MAE ≈ {avg_mae:.2f}%")
    print(f"目标: 压到 2%~3%")

    # Sample weight 建议
    print("\n建议样本权重配置:")
    print("  sample_weight = clamp(abs(actual_return) / 0.02, 0.2, 2.0)")
    print(f"  <1% 桶权重: 0.2~0.5 (降权)")
    print(f"  2%~5% 桶权重: 1.0~2.5 (正常~加权)")
    print(f"  >5% 桶权重: 2.0 (最高权重)")

    # Save results
    output_dir = Path(model_path)
    output_path = output_dir / f"bucket_eval_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()