"""
Kronos 预测训练 — HuggingFace Trainer 版本
- 全序列滑动窗口 stride=1，不重叠不泄漏
- CE loss 支持单步/多步预测 + horizon 衰减（--predict > 1 启用）
- TensorBoard + checkpoint + 自动日志
"""

import os, sys, json, time, argparse, pickle, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from scipy.stats import spearmanr
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'vol', 'amt']


# ============================================================================
# 数据加载 & 切分
# ============================================================================

def load_data(data_path, lookback, predict, train_ratio=0.70, val_ratio=0.15):
    """加载 MA60 数据，返回全量数据和 (stock, start) 窗口索引"""
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
            idx = (sym, i)
            if i < tr_end:
                train_indices.append(idx)
            elif i < val_end:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

    return all_data, train_indices, val_indices, test_indices


# ============================================================================
# Dataset
# ============================================================================

class KronosWindowDataset(Dataset):
    """每个样本: (stock, start) → 一个 61 步窗口"""

    def __init__(self, all_data, indices, lookback, predict):
        self.all_data = all_data
        self.indices = indices
        self.window = lookback + predict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sym, start = self.indices[idx]
        d = self.all_data[sym]
        end = start + self.window

        norm = d['normalized'][start:end]      # (61, 6)
        orig = d['original'][start:end]         # (61, 6)
        means = d['means'][start:end]           # (61, 6)
        stds = d['stds'][start:end]             # (61, 6)

        ts = d['index'][start:end]
        stamp = np.stack([
            ts.minute.values.astype(np.float32),
            ts.hour.values.astype(np.float32),
            ts.weekday.values.astype(np.float32),
            ts.day.values.astype(np.float32),
            ts.month.values.astype(np.float32),
        ], axis=1)  # (61, 5)

        return {
            'normalized': torch.from_numpy(norm),
            'stamp': torch.from_numpy(stamp),
            'original': torch.from_numpy(orig),
            'means': torch.from_numpy(means),
            'stds': torch.from_numpy(stds),
            'symbol': sym,
        }


# ============================================================================
# Trainer
# ============================================================================

class KronosTrainer(Trainer):
    """重写 compute_loss：tokenize + forward + 预测步数 CE（支持单步/多步）"""

    def __init__(self, tokenizer, predict, horizon_gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kronos_tokenizer = tokenizer
        self.kronos_tokenizer.eval()
        for p in self.kronos_tokenizer.parameters():
            p.requires_grad = False
        self.predict = predict
        self.horizon_gamma = horizon_gamma
        # 预计算 horizon 衰减权重: [1.0, γ, γ², ...]
        if predict > 1:
            self.h_weights = torch.tensor(
                [horizon_gamma ** i for i in range(predict)]
            )
        else:
            self.h_weights = torch.tensor([1.0])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        norm = inputs['normalized']
        stamp = inputs['stamp']

        # Tokenize（tokenizer 冻结）
        with torch.no_grad():
            t0, t1 = self.kronos_tokenizer.encode(norm, half=True)

        # Forward: lookback 步输入 → 预测最后 predict 步
        s1_logits, s2_logits = model(t0[:, :-self.predict], t1[:, :-self.predict],
                                     stamp=stamp[:, :-self.predict, :])

        # CE loss 最后 predict 个位置，horizon 衰减
        weights = self.h_weights.to(s1_logits.device)
        s1_loss = 0.0
        s2_loss = 0.0
        for i in range(self.predict):
            w = weights[i]
            s1_loss = s1_loss + w * F.cross_entropy(
                s1_logits[:, -(self.predict - i), :], t0[:, -(self.predict - i)])
            s2_loss = s2_loss + w * F.cross_entropy(
                s2_logits[:, -(self.predict - i), :], t1[:, -(self.predict - i)])
        loss = (s1_loss + s2_loss) / weights.sum()

        return loss


# ============================================================================
# 评估 Callback
# ============================================================================

class EvalCallback(TrainerCallback):
    """每个 epoch 结束后跑 IC/DA 评估，结果写入 JSONL + TensorBoard
    使用 auto_regressive_inference 进行真实评估（无 ground truth leakage）"""

    def __init__(self, trainer, val_dataset, tokenizer, output_dir, lookback, max_context=2048, clip=5.0, eval_samples=2000):
        self.trainer = trainer
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.lookback = lookback
        self.max_context = max_context
        self.clip = clip
        self.eval_samples = eval_samples
        self.val_losses = []
        self.close_ics = []
        self.results = []
        self.predict = trainer.predict  # pred_len

    def on_epoch_end(self, args, state, control, **kwargs):
        model = self.trainer.model
        model.eval()
        p = self.predict  # pred_len
        lb = self.lookback

        rng = np.random.RandomState(42)
        indices = rng.choice(len(self.val_dataset), min(self.eval_samples, len(self.val_dataset)), replace=False)

        # 多步评估：all_preds[step][feature]
        all_preds = [{f: [] for f in FEATURE_NAMES} for _ in range(p)]
        all_actuals = [{f: [] for f in FEATURE_NAMES} for _ in range(p)]
        n = 0

        with torch.no_grad():
            for idx in indices:
                item = self.val_dataset[idx]
                # item['normalized'] shape: (lookback + predict, 6)
                norm = item['normalized'].numpy()
                stamp = item['stamp'].numpy()
                orig = item['original'].numpy()
                means = item['means'].numpy()
                stds = item['stds'].numpy()

                # 只用 lookback 部分作为输入（无 ground truth leakage）
                x_tensor = torch.from_numpy(norm[:lb]).unsqueeze(0).to(DEVICE)
                x_stamp = torch.from_numpy(stamp[:lb]).unsqueeze(0).to(DEVICE)
                y_stamp = torch.from_numpy(stamp[lb:lb+p]).unsqueeze(0).to(DEVICE)

                # auto_regressive inference
                pred = auto_regressive_inference(
                    self.tokenizer, model,
                    x_tensor, x_stamp, y_stamp,
                    max_context=self.max_context,
                    pred_len=p,
                    clip=self.clip,
                    T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                # pred shape: (1, lb+p, 6) -> 取预测部分
                pred_norm = pred[0, lb:lb+p, :]  # (p, 6)

                # Denormalize 每一步
                for step_idx in range(p):
                    pred_raw = pred_norm[step_idx] * stds[lb + step_idx] + means[lb + step_idx]
                    baseline = orig[lb - 1]  # 最后一个历史位置作为 baseline
                    actual = orig[lb + step_idx]

                    for fi, fn in enumerate(FEATURE_NAMES):
                        pred_ret = (pred_raw[fi] - baseline[fi]) / (abs(baseline[fi]) + 1e-8)
                        actual_ret = (actual[fi] - baseline[fi]) / (abs(baseline[fi]) + 1e-8)
                        if np.isfinite(pred_ret) and np.isfinite(actual_ret):
                            all_preds[step_idx][fn].append(pred_ret)
                            all_actuals[step_idx][fn].append(actual_ret)

                n += 1

        # 计算指标
        result = {
            'epoch': int(state.epoch),
            'global_step': state.global_step,
            'n_samples': n,
            '_note': 'auto_regressive inference (no ground truth leakage)',
        }

        for step_idx in range(p):
            suffix = f'_step{step_idx+1}' if p > 1 else ''
            step_results = []
            for f in FEATURE_NAMES:
                preds = np.array(all_preds[step_idx][f])
                actuals = np.array(all_actuals[step_idx][f])
                if len(preds) >= 10:
                    ic = np.corrcoef(preds, actuals)[0, 1]
                    ric = spearmanr(preds, actuals)[0]
                    da = np.mean(np.sign(preds) == np.sign(actuals))
                    key_ic = f'{f}_ic{suffix}'
                    key_ric = f'{f}_rank_ic{suffix}'
                    key_da = f'{f}_da{suffix}'
                    result[key_ic] = round(float(ic), 6)
                    result[key_ric] = round(float(ric), 6)
                    result[key_da] = round(float(da), 6)
                    step_results.append((f, ic, ric, da))

            if step_idx == p - 1:  # 记录最后一步的 close_ic（用于 tracking）
                if f'close_ic{suffix}' in result:
                    self.close_ics.append(result[f'close_ic{suffix}'])

        self.results.append(result)
        self.val_losses.append(eval_loss)

        # 写入 JSONL
        log_path = os.path.join(self.output_dir, 'eval_results.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(result) + '\n')

        # 写入 TensorBoard
        if self.trainer.is_world_process_zero():
            writer = getattr(self.trainer, 'tb_writer', None)
            if writer is not None:
                writer.add_scalar('eval/loss', eval_loss, state.global_step)
                for key in result:
                    if key.endswith('_ic') or key.endswith('_rank_ic') or key.endswith('_da'):
                        writer.add_scalar(f'eval/{key}', result[key], state.global_step)

        # 打印
        suffix = f' (gamma={self.trainer.horizon_gamma})' if p > 1 else ''
        print(f"\n  [Eval] Epoch {int(state.epoch)} | Loss: {eval_loss:.4f}{suffix} | Samples: {n}")
        if p > 1:
            # 多步：打印每步 close IC 汇总
            print(f"  {'Step':<8} {'close_IC':>8} {'close_RIC':>8} {'close_DA':>8} {'open_IC':>8}")
            for step_idx in range(p):
                sfx = f'_step{step_idx+1}'
                ci = result.get(f'close_ic{sfx}', '-')
                cri = result.get(f'close_rank_ic{sfx}', '-')
                cda = result.get(f'close_da{sfx}', '-')
                oi = result.get(f'open_ic{sfx}', '-')
                cis = f'{ci:>8.4f}' if isinstance(ci, float) else f'{ci:>8}'
                cris = f'{cri:>8.4f}' if isinstance(cri, float) else f'{cri:>8}'
                cdas = f'{cda:>8.4f}' if isinstance(cda, float) else f'{cda:>8}'
                ois = f'{oi:>8.4f}' if isinstance(oi, float) else f'{oi:>8}'
                print(f"  +{step_idx+1:<7} {cis} {cris} {cdas} {ois}")
        else:
            # 单步：打印全特征
            print(f"  {'F':<8} {'IC':>8} {'RankIC':>8} {'DA':>8}")
            for f in FEATURE_NAMES:
                key = f'{f}_ic'
                if key in result:
                    print(f"  {f:<8} {result[key]:>8.4f} {result[f'{f}_rank_ic']:>8.4f} {result[f'{f}_da']:>8.4f}")

        model.train()


class SaveConfigCallback(TrainerCallback):
    """复制 config.json 到 checkpoint 目录（Kronos save_pretrained 不保存 config）"""

    def __init__(self, config_src):
        self.config_src = config_src

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            shutil.copy(self.config_src, checkpoint_dir)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pretrained/Kronos-mini')
    parser.add_argument('--tokenizer', default='outputs/models/ma60_tokenizer_v1/checkpoints/best_model')
    parser.add_argument('--data', default='finetune/data/kline_daily_ma60.pkl')
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--predict', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--horizon-gamma', type=float, default=0.75,
                        help='Horizon decay factor: w_i = gamma^i (ignored when predict=1)')
    parser.add_argument('--train-samples', type=int, default=None,
                        help='Limit training windows to N samples (subsample for fast prototyping)')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    lookback, predict = args.lookback, args.predict
    tag = f'lb{lookback}_pd{predict}'
    if args.train_samples:
        tag += f'_samp{args.train_samples}'
    output_dir = args.output_dir or os.path.join(
        project_root, 'outputs', 'models', f'ma60_predictor_{tag}')

    print("=" * 60)
    print(f"Kronos Trainer ({lookback}+{predict}, HF Trainer)")
    print("=" * 60)
    print(f"Model: {args.model}"); print(f"Tokenizer: {args.tokenizer}")
    print(f"Data: {args.data}"); print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")

    # 加载 tokenizer（冻结）
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(DEVICE)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # 加载模型
    model = Kronos.from_pretrained(args.model).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 数据
    print("\nLoading & splitting data...")
    all_data, train_idx, val_idx, test_idx = load_data(
        args.data, lookback, predict, train_ratio=0.70, val_ratio=0.15)
    print(f"  Train: {len(train_idx)} windows, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # 子采样训练集（快速原型验证）
    if args.train_samples and args.train_samples < len(train_idx):
        rng = np.random.RandomState(42)
        idx_arr = np.arange(len(train_idx))
        sampled = rng.choice(idx_arr, args.train_samples, replace=False)
        train_idx = [train_idx[i] for i in sampled]
        print(f"  → Subsampled to {len(train_idx)} training windows")
    train_ds = KronosWindowDataset(all_data, train_idx, lookback, predict)
    val_ds = KronosWindowDataset(all_data, val_idx, lookback, predict)

    # Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.0,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        save_strategy='epoch',
        save_total_limit=3,
        report_to=['tensorboard'],
        bf16=False, fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # 获取模型的 max_context
    max_context = getattr(model, 'max_context', 2048) or 2048

    trainer = KronosTrainer(
        tokenizer=tokenizer,
        predict=predict,
        horizon_gamma=args.horizon_gamma,
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    eval_callback = EvalCallback(
        trainer, val_ds, tokenizer, output_dir,
        lookback=lookback,
        max_context=max_context,
        clip=5.0
    )
    trainer.add_callback(eval_callback)

    # 复制 config.json（Kronos save_pretrained 不保存 config）
    pretrained_config = os.path.join(project_root, args.model, 'config.json')
    if os.path.exists(pretrained_config):
        shutil.copy(pretrained_config, output_dir)
        trainer.add_callback(SaveConfigCallback(pretrained_config))

    # Train（自动检测 checkpoint 恢复）
    print("\nStarting training...")
    checkpoint_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith('checkpoint-')])
    resume_ckpt = os.path.join(output_dir, checkpoint_dirs[-1]) if checkpoint_dirs else None
    if resume_ckpt:
        print(f"Resuming from: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 保存
    model.save_pretrained(os.path.join(output_dir, 'checkpoints', 'final_model'))
    print(f"\nDone. Output: {output_dir}")
    print(f"TensorBoard: tensorboard --logdir {os.path.join(output_dir, 'logs')}")


if __name__ == '__main__':
    main()
