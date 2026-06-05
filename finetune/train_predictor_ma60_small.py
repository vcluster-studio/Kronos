"""
MA60 Predictor Training Script - for small model (group_size=4)

基于 MA60 base tokenizer (group_size=4) 微调 Kronos-small predictor。
使用预计算好的 MA60 归一化数据（lookback=200，适配 small 模型 max_context=512）。

目标：纯 token CE loss 训练，不添加额外 head。IC/DA 仅用于评估监控。

Usage:
    python -u finetune/train_predictor_ma60_small.py --pcgrad
    python -u finetune/train_predictor_ma60_small.py  # 原始模式（无 PCGrad）

数据格式：
    processed_datasets_ma60_windowed_v3_small/{train,val,test}_data.pkl
    每个股票: {'normalized': (T,6), 'means': (T,6), 'stds': (T,6), 'original': (T,6), 'index': DatetimeIndex, 'windows': array}
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# ============================================================================
# 配置
# ============================================================================

# MA60 base tokenizer 路径（group_size=4，需先运行 train_tokenizer_ma60_base.py）
TOKENIZER_MA60_BASE = 'outputs/models/ma60_tokenizer_base_v1/checkpoints/best_model'

# 预归一化数据路径（窗口化 v3_small，lookback=200 适配 small 模型）
DATA_PATHS = {
    'train': 'finetune/data/processed_datasets_ma60_windowed_v3_small/train_data.pkl',
    'val': 'finetune/data/processed_datasets_ma60_windowed_v3_small/val_data.pkl',
    'test': 'finetune/data/processed_datasets_ma60_windowed_v3_small/test_data.pkl',
}

# Predictor 预训练路径（small 模型）
PREDICTOR_PRETRAINED = 'pretrained/Kronos-small'

# 训练参数 — small 模型 Cosine Annealing + Layer Freezing
TRAINING_PARAMS = {
    'epochs': 50,
    'batch_size': 8,            # small 模型更大，减小 batch
    'learning_rate': 0.003,     # 保守起始 LR
    'weight_decay': 0.01,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'seed': 100,
    'lookback': 200,              # small 模型 max_context=512，lookback=200 (400 tokens) + predict=10 (20 tokens) = 420 tokens
    'predict': 10,
    'clip': 5.0,
    'max_context': 512,         # small 模型 context 限制
    'early_stopping_patience': 12,
    'early_stopping_grace_period': 8,
    # IC 测试
    'ic_test_samples': 500,
    'ic_point': 3,
    # Cosine Annealing LR
    'lr_scheduler': 'cosine',
    'lr_min': 1e-5,
    'warmup_epochs': 2,
    # 冻结层（V5b: 冻结前4层+embedding，验证容量过大导致IC崩塌的假设）
    'freeze_layers': 4,
    'freeze_embedding': True,
}

# 保存目录
SAVE_FOLDER = 'ma60_predictor_small_v6'


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


class MA60Dataset(Dataset):
    """MA60 预归一化数据集"""

    def __init__(self, data_type='train', config=None):
        self.config = config
        self.data_type = data_type
        self.py_rng = np.random.RandomState(config.seed)

        data_path = DATA_PATHS[data_type]
        print(f"[{data_type.upper()}] Loading: {data_path}")
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.symbols = list(self.raw_data.keys())
        self.window = config.lookback + config.predict + 1

        # 预计算索引
        print(f"[{data_type.upper()}] Pre-computing indices...")
        self.indices = []

        # 检测数据格式：有 windows 字段则为窗口化格式
        sample_data = self.raw_data[self.symbols[0]]
        has_windows = 'windows' in sample_data

        if has_windows:
            # 窗口化格式：使用预分配的窗口列表
            for symbol in self.symbols:
                data = self.raw_data[symbol]
                if 'windows' in data:
                    for start in data['windows']:
                        self.indices.append((symbol, int(start)))
            print(f"[{data_type.upper()}] Windowed format: using pre-assigned {len(self.indices)} windows")
        else:
            # 传统格式：所有滑动窗口
            for symbol in self.symbols:
                data = self.raw_data[symbol]
                seq_len = len(data['normalized'])
                if seq_len >= self.window:
                    for i in range(seq_len - self.window + 1):
                        self.indices.append((symbol, i))

        n_iter = config.n_train_iter if data_type == 'train' else config.n_val_iter
        self.n_samples = min(n_iter, len(self.indices))
        print(f"[{data_type.upper()}] {len(self.indices)} windows, using {self.n_samples} samples/epoch")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rand_idx = self.py_rng.randint(0, len(self.indices))
        symbol, start_idx = self.indices[rand_idx]
        end_idx = start_idx + self.window

        data = self.raw_data[symbol]

        x_norm = data['normalized'][start_idx:end_idx].astype(np.float32)

        timestamps = data['index'][start_idx:end_idx]
        x_stamp = np.stack([
            timestamps.minute.values,
            timestamps.hour.values,
            timestamps.weekday.values,
            timestamps.day.values,
            timestamps.month.values,
        ], axis=1).astype(np.float32)

        return torch.from_numpy(x_norm), torch.from_numpy(x_stamp)


def compute_vol_corr(predictions, actuals):
    """波动率预测准确性：预测幅度与真实幅度的相关性"""
    pred_abs = np.abs(predictions)
    actual_abs = np.abs(actuals)
    if len(pred_abs) < 10:
        return np.nan
    return np.corrcoef(pred_abs, actual_abs)[0, 1]


def compute_vol_binned(predictions, actuals, n_bins=2):
    """分档波动率验证：高/低波动预测组的实际波动差异"""
    if len(predictions) < 20:
        return {'high_mean': np.nan, 'low_mean': np.nan, 'ratio': np.nan, 'n': len(predictions)}
    pred_abs = np.abs(predictions)
    actual_abs = np.abs(actuals)
    median = np.median(pred_abs)
    high_mask = pred_abs >= median
    low_mask = pred_abs < median
    high_mean = np.mean(actual_abs[high_mask]) if high_mask.sum() > 0 else np.nan
    low_mean = np.mean(actual_abs[low_mask]) if low_mask.sum() > 0 else np.nan
    ratio = high_mean / low_mean if (low_mean and low_mean > 1e-12) else np.nan
    return {'high_mean': float(high_mean), 'low_mean': float(low_mean),
            'ratio': float(ratio), 'n': len(predictions)}


def compute_vol_weighted_ic(predictions, actuals):
    """波动率加权 IC：以预测幅度为权重计算加权 Pearson 相关"""
    if len(predictions) < 10:
        return np.nan
    weights = np.abs(predictions)
    w_sum = weights.sum()
    if w_sum < 1e-12:
        return np.nan
    w_mean_pred = np.average(predictions, weights=weights)
    w_mean_actual = np.average(actuals, weights=weights)
    w_cov = np.average((predictions - w_mean_pred) * (actuals - w_mean_actual), weights=weights)
    w_var_pred = np.average((predictions - w_mean_pred) ** 2, weights=weights)
    w_var_actual = np.average((actuals - w_mean_actual) ** 2, weights=weights)
    denom = np.sqrt(w_var_pred * w_var_actual)
    if denom < 1e-12:
        return np.nan
    return float(w_cov / denom)


def compute_vol_da(predictions, actuals):
    """波动率方向准确率：100次中波动大小判断对了几次"""
    if len(predictions) < 20:
        return np.nan, np.nan, np.nan
    pred_abs = np.abs(predictions)
    actual_abs = np.abs(actuals)
    pred_median = np.median(pred_abs)
    actual_median = np.median(actual_abs)
    pred_high = pred_abs >= pred_median
    actual_high = actual_abs >= actual_median
    accuracy = np.mean(pred_high == actual_high)
    up_hit = np.mean(actual_high[pred_high]) if pred_high.sum() > 0 else np.nan
    down_hit = np.mean(~actual_high[~pred_high]) if (~pred_high).sum() > 0 else np.nan
    return float(accuracy), float(up_hit), float(down_hit)


def compute_up_down_hits(predictions, actuals):
    """涨跌命中率：喊涨对了多少，喊跌对了多少"""
    if len(predictions) < 20:
        return np.nan, np.nan, np.nan
    pred_up = predictions > 0
    pred_down = predictions < 0
    actual_up = actuals > 0
    actual_down = actuals < 0
    up_hit = np.mean(actual_up[pred_up]) if pred_up.sum() > 0 else np.nan
    down_hit = np.mean(actual_down[pred_down]) if pred_down.sum() > 0 else np.nan
    if not np.isnan(up_hit) and not np.isnan(down_hit) and up_hit + down_hit > 1e-12:
        balance = 2 * up_hit * down_hit / (up_hit + down_hit)
    else:
        balance = np.nan
    return float(up_hit), float(down_hit), float(balance)


def quick_ic_test_ma60(model, tokenizer, device, val_data, n_samples=500,
                       lookback=200, pred_len=10, ic_point=3, clip=5.0, max_context=512,
                       rng=None):
    """多维度 IC 测试（使用 val_data，随机采样股票）

    返回指标：IC, Rank IC, DA, DDA, NMSE, ICIR, 多步 IC（全部 6 维特征）
    """
    model.eval()

    feature_names = ['open', 'high', 'low', 'close', 'vol', 'amt']

    all_symbols = list(val_data.keys())
    if rng is not None:
        symbols = rng.choice(all_symbols, size=min(n_samples, len(all_symbols)), replace=False).tolist()
    else:
        symbols = all_symbols[:n_samples]

    # 多步多特征收集: step_predictions[step][feat_idx] = list of floats
    step_preds_all = {s: {f: [] for f in range(6)} for s in range(1, pred_len + 1)}
    step_actuals_all = {s: {f: [] for f in range(6)} for s in range(1, pred_len + 1)}
    step_actuals_prev_all = {s: {f: [] for f in range(6)} for s in range(1, pred_len + 1)}

    for symbol in symbols:
        data = val_data[symbol]
        seq_len = len(data['normalized'])

        if seq_len < lookback + pred_len:
            continue

        try:
            full_len = lookback + pred_len
            end_idx = seq_len

            x_norm_full = data['normalized'][end_idx - full_len:end_idx].astype(np.float32)
            means_full = data['means'][end_idx - full_len:end_idx]
            stds_full = data['stds'][end_idx - full_len:end_idx]
            timestamps_full = data['index'][end_idx - full_len:end_idx]

            x_norm = x_norm_full[:lookback]
            x_ts = timestamps_full[:lookback]
            y_ts = timestamps_full[lookback:]

            x_stamp = np.stack([
                x_ts.minute.values, x_ts.hour.values, x_ts.weekday.values,
                x_ts.day.values, x_ts.month.values
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                y_ts.minute.values, y_ts.hour.values, y_ts.weekday.values,
                y_ts.day.values, y_ts.month.values
            ], axis=1).astype(np.float32)

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm[:lookback]).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context=max_context, pred_len=pred_len,
                    clip=clip, T=1.0, top_k=0, top_p=0.9,
                    sample_count=1, verbose=False
                )

                # 6 维 denormalize
                pred_raw = preds[0, -pred_len:, :] * stds_full[lookback:, :] + means_full[lookback:, :]

            actual_raw = data['original'][end_idx - pred_len:, :]
            baseline_vals = data['original'][end_idx - pred_len - 1, :]

            for s in range(1, pred_len + 1):
                for f in range(6):
                    base = float(baseline_vals[f])
                    pred_return = (float(pred_raw[s - 1, f]) - base) / base if abs(base) > 1e-8 else 0.0
                    actual_return = (float(actual_raw[s - 1, f]) - base) / base if abs(base) > 1e-8 else 0.0
                    if s == 1:
                        actual_prev = 0.0
                    else:
                        prev_val = float(actual_raw[s - 2, f])
                        actual_prev = (prev_val - base) / base if abs(base) > 1e-8 else 0.0

                    step_preds_all[s][f].append(pred_return)
                    step_actuals_all[s][f].append(actual_return)
                    step_actuals_prev_all[s][f].append(actual_prev)

        except Exception as e:
            if not any(step_preds_all[1][f] for f in range(6)):
                print(f"[IC TEST] First error: {e}")
            continue

    if len(step_preds_all[1][3]) > 5:
        result = {}

        # === 全特征指标 ===
        for f in range(6):
            fname = feature_names[f]
            step_ics_f = []
            for s in range(1, pred_len + 1):
                preds_arr = np.array(step_preds_all[s][f])
                actuals_arr = np.array(step_actuals_all[s][f])
                prevs_arr = np.array(step_actuals_prev_all[s][f])

                ic = np.corrcoef(preds_arr, actuals_arr)[0, 1]
                rank_ic, _ = spearmanr(preds_arr, actuals_arr)
                da = np.mean(np.sign(preds_arr) == np.sign(actuals_arr))
                # DDA
                actual_change = np.sign(actuals_arr) != np.sign(prevs_arr)
                pred_change = np.sign(preds_arr) != np.sign(prevs_arr)
                if actual_change.sum() > 0:
                    dda = np.mean(pred_change[actual_change] == actual_change[actual_change])
                else:
                    dda = np.nan
                # NMSE
                var_actual = np.var(actuals_arr)
                nmse = np.mean((preds_arr - actuals_arr) ** 2) / var_actual if var_actual > 1e-8 else np.nan
                # Pred bias & var ratio
                pred_bias = float(np.mean(preds_arr) - np.mean(actuals_arr))
                var_ratio = float(np.var(preds_arr) / var_actual) if var_actual > 1e-8 else np.nan

                step_ics_f.append(ic)
                result[f'{fname}_step{s}_ic'] = ic
                result[f'{fname}_step{s}_rank_ic'] = rank_ic
                result[f'{fname}_step{s}_da'] = da
                if not np.isnan(dda):
                    result[f'{fname}_step{s}_dda'] = dda
                result[f'{fname}_step{s}_nmse'] = nmse
                result[f'{fname}_step{s}_pred_bias'] = pred_bias
                result[f'{fname}_step{s}_var_ratio'] = var_ratio

            # 代表步 (step 3)
            rep = ic_point
            result[f'{fname}_ic'] = result.get(f'{fname}_step{rep}_ic', step_ics_f[0])
            result[f'{fname}_rank_ic'] = result.get(f'{fname}_step{rep}_rank_ic', 0)
            result[f'{fname}_da'] = result.get(f'{fname}_step{rep}_da', 0)
            result[f'{fname}_dda'] = result.get(f'{fname}_step{rep}_dda', np.nan)
            result[f'{fname}_nmse'] = result.get(f'{fname}_step{rep}_nmse', np.nan)

            # ICIR per feature
            ic_mean = np.mean(step_ics_f)
            ic_std = np.std(step_ics_f)
            result[f'{fname}_icir'] = ic_mean / ic_std if ic_std > 1e-8 else np.nan

            # 波动率指标 (step 3)
            step_preds_f = np.array(step_preds_all[rep][f])
            step_actuals_f = np.array(step_actuals_all[rep][f])
            result[f'{fname}_vol_corr'] = compute_vol_corr(step_preds_f, step_actuals_f)
            result[f'{fname}_vol_binned'] = compute_vol_binned(step_preds_f, step_actuals_f)
            result[f'{fname}_vol_weighted_ic'] = compute_vol_weighted_ic(step_preds_f, step_actuals_f)
            # 朴素指标
            vol_da, vol_up, vol_down = compute_vol_da(step_preds_f, step_actuals_f)
            result[f'{fname}_vol_da'] = vol_da
            result[f'{fname}_vol_up_hit'] = vol_up
            result[f'{fname}_vol_down_hit'] = vol_down
            up_hit, down_hit, balance = compute_up_down_hits(step_preds_f, step_actuals_f)
            result[f'{fname}_up_hit'] = up_hit
            result[f'{fname}_down_hit'] = down_hit
            result[f'{fname}_hit_balance'] = balance

        # 兼容：close 的指标也赋给顶层 key
        result['ic'] = result.get('close_ic', 0)
        result['rank_ic'] = result.get('close_rank_ic', 0)
        result['da'] = result.get('close_da', 0)
        result['dda'] = result.get('close_dda', np.nan)
        result['nmse'] = result.get('close_nmse', np.nan)
        result['icir'] = result.get('close_icir', np.nan)

        # 兼容：close 多步 key
        for s in range(1, pred_len + 1):
            result[f'step{s}_ic'] = result.get(f'close_step{s}_ic', 0)
            result[f'step{s}_rank_ic'] = result.get(f'close_step{s}_rank_ic', 0)
            result[f'step{s}_da'] = result.get(f'close_step{s}_da', 0)
            result[f'step{s}_nmse'] = result.get(f'close_step{s}_nmse', np.nan)
            result[f'step{s}_pred_bias'] = result.get(f'close_step{s}_pred_bias', np.nan)
            result[f'step{s}_var_ratio'] = result.get(f'close_step{s}_var_ratio', np.nan)

        result['n_samples'] = len(step_preds_all[1][3])
        result['feature_names'] = feature_names
        return result

    return None


def freeze_model_layers(model, freeze_layers=6, freeze_embedding=True):
    """冻结模型前 N 层 transformer 和 embedding"""
    if freeze_embedding:
        for param in model.module.embedding.parameters():
            param.requires_grad = False
        for param in model.module.time_emb.parameters():
            param.requires_grad = False
        print(f"[FREEZE] Embedding + TemporalEmb frozen")

    for i in range(min(freeze_layers, len(model.module.transformer))):
        for param in model.module.transformer[i].parameters():
            param.requires_grad = False
        print(f"[FREEZE] Transformer layer {i} frozen")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FREEZE] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)")


class PCGrad:
    """PCGrad: Projecting Conflicting Gradients (Yu et al., NeurIPS 2020)

    直接使用模型已计算的 s1_loss 和 s2_loss 作为两个独立任务:
      - Task 1: s1_loss — token_seq_0 的 CE（codebook 前半比特位）
      - Task 2: s2_loss — token_seq_1 的 CE（codebook 后半比特位）

    compute_loss 已返回独立的 s1_loss 和 s2_loss，无需任何转换。
    各任务独立 backward → PCGrad 梯度投影 → 合并更新。

    参考: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def project_conflicting_gradients(self, grad_s1, grad_s2):
        """若 g_s1 · g_s2 < 0，将各自投影到对方的法平面。"""
        dot = sum(torch.sum(g1 * g2) for g1, g2 in zip(grad_s1, grad_s2)
                  if g1 is not None and g2 is not None)

        if dot < 0:
            norm_s2_sq = sum(torch.sum(g * g) for g in grad_s2 if g is not None)
            norm_s1_sq = sum(torch.sum(g * g) for g in grad_s1 if g is not None)

            if norm_s2_sq > 1e-12:
                coeff = dot / norm_s2_sq
                grad_s1 = [g - coeff * g2 if g is not None and g2 is not None else g
                           for g, g2 in zip(grad_s1, grad_s2)]

            if norm_s1_sq > 1e-12:
                coeff = dot / norm_s1_sq
                grad_s2 = [g - coeff * g1 if g is not None and g1 is not None else g
                           for g, g1 in zip(grad_s2, grad_s1)]

        merged = []
        for g1, g2 in zip(grad_s1, grad_s2):
            if g1 is not None and g2 is not None:
                merged.append(g1 + g2)
            elif g1 is not None:
                merged.append(g1)
            elif g2 is not None:
                merged.append(g2)
            else:
                merged.append(None)

        return merged, float(dot)

    def pcgrad_step(self, model, tokenizer, batch_x, batch_stamp,
                    device, model_params, max_norm=3.0):
        """执行一步 PCGrad 训练。

        使用 compute_loss 已返回的 s1_loss 和 s2_loss 作为两个独立任务，
        各任务独立 backward → PCGrad 梯度投影 → 合并更新。

        Returns:
            total_loss:  总 loss 值（用于日志）
            grad_dot:    s1·s2 梯度点积（负值 = 冲突）
        """
        with torch.no_grad():
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

        token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
        token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

        s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
            token_in[0], token_in[1], batch_stamp[:, :-1, :]
        )

        # compute_loss 返回 (recon_loss, s1_loss, s2_loss)
        _, s1_loss, s2_loss = model.module.head.compute_loss(
            s1_logits, s2_logits, token_out[0], token_out[1]
        )

        # ---- Backward: Task 1 (s1_loss) ----
        self.optimizer.zero_grad()
        s1_loss.backward(retain_graph=True)
        grad_s1 = [p.grad.clone() if p.grad is not None else None
                   for p in model_params]

        # ---- Backward: Task 2 (s2_loss) ----
        self.optimizer.zero_grad()
        s2_loss.backward()
        grad_s2 = [p.grad.clone() if p.grad is not None else None
                   for p in model_params]

        # ---- PCGrad projection ----
        merged_grads, grad_dot = self.project_conflicting_gradients(grad_s1, grad_s2)

        # ---- 应用合并后的梯度 ----
        for p, mg in zip(model_params, merged_grads):
            if mg is not None:
                p.grad = mg

        torch.nn.utils.clip_grad_norm_(model_params, max_norm=max_norm)
        self.optimizer.step()

        total_loss = s1_loss.item() + s2_loss.item()
        return total_loss, grad_dot


def train_model(model, tokenizer, device, config, save_dir, val_data=None):
    """训练 MA60 predictor (small) - Cosine Annealing + Layer Freezing"""
    start_time = time.time()

    print(f"BATCHSIZE: {config.batch_size}")
    print(f"LR: {config.learning_rate}")
    print(f"LR Scheduler: {config.lr_scheduler}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"Lookback: {config.lookback}")
    print(f"Predict: {config.predict}")
    print(f"Max Context: {config.max_context}")

    # 冻结层
    if config.freeze_layers > 0 or config.freeze_embedding:
        freeze_model_layers(model, config.freeze_layers, config.freeze_embedding)

    train_dataset = MA60Dataset('train', config=config)
    val_dataset = MA60Dataset('val', config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               sampler=RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset)),
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                              sampler=SequentialSampler(val_dataset),
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 优化器（只优化可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate,
                                   weight_decay=config.weight_decay,
                                   betas=(config.adam_beta1, config.adam_beta2))

    # Cosine Annealing LR（无 warmup）
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.lr_min
    )

    best_val_loss = float('inf')
    best_ic = -999
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'ic': [], 'lr': []}
    if config.pcgrad:
        history['grad_dot'] = []

    # PCGrad 优化器
    pcgrad_optimizer = PCGrad(optimizer) if config.pcgrad else None

    for epoch_idx in range(config.epochs):
        epoch_start = time.time()
        model.train()

        train_dataset.py_rng.seed(config.seed + epoch_idx * 10000)

        epoch_losses = []
        epoch_grad_dots = []
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n=== Epoch {epoch_idx+1}/{config.epochs} ===")
        print(f"LR: {current_lr:.6f}")
        if config.pcgrad:
            print(f"PCGrad: ON (price vs volume gradient surgery)")

        for i, (batch_x, batch_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_stamp = batch_stamp.to(device, non_blocking=True)

            if config.pcgrad:
                # PCGrad 模式：s1_loss vs s2_loss 独立 backward，梯度投影
                total_loss, grad_dot = pcgrad_optimizer.pcgrad_step(
                    model, tokenizer, batch_x, batch_stamp,
                    device, trainable_params, max_norm=3.0
                )
                epoch_grad_dots.append(grad_dot)
            else:
                # 原始模式：统一 backward
                with torch.no_grad():
                    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
                    token_in[0], token_in[1], batch_stamp[:, :-1, :]
                )
                recon_loss, s1_loss, s2_loss = model.module.head.compute_loss(
                    s1_logits, s2_logits, token_out[0], token_out[1]
                )

                total_loss = recon_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
                optimizer.step()

            scheduler.step()

            epoch_losses.append(total_loss)

            if (i + 1) % 50 == 0 or i == 0:
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (i + 1) / len(train_loader) * 100
                extra = ""
                if config.pcgrad and epoch_grad_dots:
                    avg_dot = sum(epoch_grad_dots[-50:]) / min(len(epoch_grad_dots[-50:]), 50)
                    extra = f", GradDot: {avg_dot:.4f}"
                print(f"  Batch {i+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {total_loss:.4f}, Avg: {avg_loss:.4f}{extra}")

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # 记录梯度冲突
        if config.pcgrad and epoch_grad_dots:
            avg_grad_dot = sum(epoch_grad_dots) / len(epoch_grad_dots)
            n_conflict = sum(1 for d in epoch_grad_dots if d < 0)
            history['grad_dot'].append(avg_grad_dot)
            print(f"  GradDot avg: {avg_grad_dot:.4f}, conflicts: {n_conflict}/{len(epoch_grad_dots)} ({100*n_conflict/len(epoch_grad_dots):.1f}%)")

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_stamp = batch_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                s1_logits, s2_logits, hidden = model.module.forward_with_hidden(
                    token_in[0], token_in[1], batch_stamp[:, :-1, :]
                )
                recon_loss, s1_loss, s2_loss = model.module.head.compute_loss(
                    s1_logits, s2_logits, token_out[0], token_out[1]
                )

                val_loss_sum += recon_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # IC test (使用 val_data，随机采样)
        ic_rng = np.random.RandomState(config.seed + epoch_idx * 9999)
        ic_result = quick_ic_test_ma60(model.module, tokenizer, device, val_data,
                                        n_samples=config.ic_test_samples, ic_point=config.ic_point,
                                        max_context=config.max_context, rng=ic_rng)
        current_ic = 0
        if ic_result:
            current_ic = ic_result['ic']
            history['ic'].append(current_ic)

        # IC 滑动均值（用于决策，减少噪声）
        ic_window = 3
        if len(history['ic']) >= ic_window:
            ic_smoothed = np.mean(history['ic'][-ic_window:])
        else:
            ic_smoothed = current_ic

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        if ic_result:
            # === 全特征概览 (step 3) ===
            feature_names = ic_result.get('feature_names', ['open', 'high', 'low', 'close', 'vol', 'amt'])
            print(f"  {'Feat':>5s} {'IC':>7s} {'RankIC':>7s} {'ICIR':>7s} {'DA':>7s} {'DDA':>7s} {'NMSE':>7s} {'Bias':>7s} {'VarR':>7s}")
            for fname in feature_names:
                f_ic = ic_result.get(f'{fname}_ic', np.nan)
                f_ric = ic_result.get(f'{fname}_rank_ic', np.nan)
                f_icir = ic_result.get(f'{fname}_icir', np.nan)
                f_da = ic_result.get(f'{fname}_da', np.nan)
                f_dda = ic_result.get(f'{fname}_dda', np.nan)
                f_nmse = ic_result.get(f'{fname}_nmse', np.nan)
                f_bias = ic_result.get(f'{fname}_step{config.ic_point}_pred_bias', np.nan)
                f_varr = ic_result.get(f'{fname}_step{config.ic_point}_var_ratio', np.nan)
                dda_str = f"{f_dda:.4f}" if not np.isnan(f_dda) else "    N/A"
                bias_str = f"{f_bias:+.4f}" if not np.isnan(f_bias) else "    N/A"
                varr_str = f"{f_varr:.4f}" if not np.isnan(f_varr) else "    N/A"
                print(f"  {fname:>5s} {f_ic:>7.4f} {f_ric:>7.4f} {f_icir:>7.4f} {f_da:>7.4f} {dda_str:>7s} {f_nmse:>7.4f} {bias_str:>7s} {varr_str:>7s}")

            # === 波动率指标 (step 3) ===
            print(f"  {'Feat':>5s} {'VolCorr':>7s} {'W-IC':>7s} {'RawIC':>7s} {'Delta':>7s} {'Hi/Lo':>7s}")
            for fname in feature_names:
                f_vc = ic_result.get(f'{fname}_vol_corr', np.nan)
                f_wic = ic_result.get(f'{fname}_vol_weighted_ic', np.nan)
                f_ic = ic_result.get(f'{fname}_ic', np.nan)
                f_delta = f_wic - f_ic if not (np.isnan(f_wic) or np.isnan(f_ic)) else np.nan
                f_vb = ic_result.get(f'{fname}_vol_binned', {})
                f_ratio = f_vb.get('ratio', np.nan)
                vc_str = f"{f_vc:.4f}" if not np.isnan(f_vc) else "   N/A"
                wic_str = f"{f_wic:.4f}" if not np.isnan(f_wic) else "   N/A"
                ic_str = f"{f_ic:.4f}" if not np.isnan(f_ic) else "   N/A"
                delta_str = f"{f_delta:+.4f}" if not np.isnan(f_delta) else "   N/A"
                ratio_str = f"{f_ratio:.4f}" if not np.isnan(f_ratio) else "   N/A"
                print(f"  {fname:>5s} {vc_str:>7s} {wic_str:>7s} {ic_str:>7s} {delta_str:>7s} {ratio_str:>7s}")

            # === 朴素指标 (step 3) — 100次中对了多少次 ===
            print(f"  {'Feat':>5s} {'DA':>7s} {'UpHit':>7s} {'DnHit':>7s} {'Bal':>7s} {'VolDA':>7s} {'VUp':>7s} {'VDn':>7s}")
            for fname in feature_names:
                f_da = ic_result.get(f'{fname}_da', np.nan)
                f_up = ic_result.get(f'{fname}_up_hit', np.nan)
                f_dn = ic_result.get(f'{fname}_down_hit', np.nan)
                f_bal = ic_result.get(f'{fname}_hit_balance', np.nan)
                f_vda = ic_result.get(f'{fname}_vol_da', np.nan)
                f_vup = ic_result.get(f'{fname}_vol_up_hit', np.nan)
                f_vdn = ic_result.get(f'{fname}_vol_down_hit', np.nan)
                da_str = f"{f_da:.3f}" if not np.isnan(f_da) else "   N/A"
                up_str = f"{f_up:.3f}" if not np.isnan(f_up) else "   N/A"
                dn_str = f"{f_dn:.3f}" if not np.isnan(f_dn) else "   N/A"
                bal_str = f"{f_bal:.3f}" if not np.isnan(f_bal) else "   N/A"
                vda_str = f"{f_vda:.3f}" if not np.isnan(f_vda) else "   N/A"
                vup_str = f"{f_vup:.3f}" if not np.isnan(f_vup) else "   N/A"
                vdn_str = f"{f_vdn:.3f}" if not np.isnan(f_vdn) else "   N/A"
                print(f"  {fname:>5s} {da_str:>7s} {up_str:>7s} {dn_str:>7s} {bal_str:>7s} {vda_str:>7s} {vup_str:>7s} {vdn_str:>7s}")

            # === 各特征多步 IC ===
            print(f"  {'Feat':>5s}", end="")
            for s in range(1, min(config.predict + 1, 6)):
                print(f" {'S'+str(s):>6s}", end="")
            print()
            for fname in feature_names:
                print(f"  {fname:>5s}", end="")
                for s in range(1, min(config.predict + 1, 6)):
                    s_ic = ic_result.get(f'{fname}_step{s}_ic', np.nan)
                    print(f" {s_ic:>6.3f}", end="")
                print()
        else:
            print(f"IC: {current_ic:.4f}")
        print(f"IC_smoothed: {ic_smoothed:.4f} (best: {best_ic:.4f}) — monitor only, VL decides")
        if ic_smoothed > best_ic:
            best_ic = ic_smoothed
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}")

        print(f"[LR] {current_lr:.6f}")

        # Save latest
        latest_path = f"{save_dir}/checkpoints/latest_model"
        model.module.save_pretrained(latest_path)

        # Check improvement — 仅通过 val loss 判定（PCGrad 下回归模型自身）
        improved = False

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            improved = True
            save_path = f"{save_dir}/checkpoints/best_model"
            model.module.save_pretrained(save_path)
            print(f"[VL SAVED] {best_val_loss:.4f}")

        if not improved:
            patience_counter += 1

        # Early stopping
        if epoch_idx >= config.early_stopping_grace_period:
            if patience_counter >= config.early_stopping_patience:
                print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                final_path = f"{save_dir}/checkpoints/final_model"
                model.module.save_pretrained(final_path)
                break

        print(flush=True)

    # Final save
    final_path = f"{save_dir}/checkpoints/final_model"
    model.module.save_pretrained(final_path)

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'epochs_trained': epoch_idx + 1,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='MA60 Predictor Training (small, group_size=4)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--save-folder', type=str, default=SAVE_FOLDER,
                        help='Save folder name')
    parser.add_argument('--pcgrad', action='store_true',
                        help='Enable PCGrad gradient surgery (price vs volume)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("MA60 Predictor Training (small, group_size=4)")
    print("=" * 60)
    print(f"Tokenizer: {TOKENIZER_MA60_BASE}")
    print(f"Data: {DATA_PATHS['train']}")
    print(f"Pretrained: {PREDICTOR_PRETRAINED}")
    print(f"Device: {device}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("=" * 60)

    set_seed(TRAINING_PARAMS['seed'])

    save_dir = os.path.join(project_root, "outputs/models", args.save_folder)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # 加载 tokenizer
    tokenizer_path = os.path.join(project_root, TOKENIZER_MA60_BASE)
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    # 加载 predictor
    if args.resume:
        predictor_path = args.resume
        print(f"Predictor loaded from (resume): {predictor_path}")
    else:
        predictor_path = os.path.join(project_root, PREDICTOR_PRETRAINED)
        print(f"Predictor loaded from: {PREDICTOR_PRETRAINED}")
    model = Kronos.from_pretrained(predictor_path)
    model.to(device)
    print(f"Model size: {get_model_size(model):.2f}M")

    # Wrap model
    class ModelWrapper:
        def __init__(self, model):
            self.module = model
        def train(self):
            self.module.train()
        def eval(self):
            self.module.eval()
        def parameters(self):
            return self.module.parameters()
        def __getattr__(self, name):
            return getattr(self.module, name)

    model = ModelWrapper(model)

    # 配置
    class Config:
        pass

    config = Config()
    config.seed = TRAINING_PARAMS['seed']
    config.lookback = TRAINING_PARAMS['lookback']
    config.predict = TRAINING_PARAMS['predict']
    config.clip = TRAINING_PARAMS['clip']
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = TRAINING_PARAMS['weight_decay']
    config.adam_beta1 = TRAINING_PARAMS['adam_beta1']
    config.adam_beta2 = TRAINING_PARAMS['adam_beta2']
    config.n_train_iter = 2000 * config.batch_size
    config.n_val_iter = 400 * config.batch_size
    config.early_stopping_patience = TRAINING_PARAMS['early_stopping_patience']
    config.early_stopping_grace_period = TRAINING_PARAMS['early_stopping_grace_period']
    config.ic_test_samples = args.n_samples
    config.ic_point = TRAINING_PARAMS['ic_point']
    config.lr_scheduler = TRAINING_PARAMS['lr_scheduler']
    config.lr_min = TRAINING_PARAMS['lr_min']
    config.warmup_epochs = TRAINING_PARAMS['warmup_epochs']
    config.freeze_layers = TRAINING_PARAMS['freeze_layers']
    config.freeze_embedding = TRAINING_PARAMS['freeze_embedding']
    config.max_context = TRAINING_PARAMS['max_context']
    config.pcgrad = args.pcgrad

    # 加载验证数据（IC 评估用）
    val_path = DATA_PATHS['val']
    print(f"Loading val data for IC evaluation: {val_path}")
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    print(f"Val: {len(val_data)} stocks")

    # 训练
    result = train_model(model, tokenizer, device, config, save_dir, val_data)

    # 保存结果
    summary = {
        'tokenizer': TOKENIZER_MA60_BASE,
        'data_path': DATA_PATHS['train'],
        'pretrained': PREDICTOR_PRETRAINED,
        'group_size': 4,
        'config': TRAINING_PARAMS,
        'result': {
            'best_val_loss': result['best_val_loss'],
            'best_ic': result['best_ic'],
            'epochs_trained': result['epochs_trained'],
        }
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"Best IC: {result['best_ic']:.4f}")
    print(f"Epochs: {result['epochs_trained']}")
    print(f"Saved to: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
