"""
Kronos-mini 预测器训练脚本 - 支持早停和 IC 监控

用法：
    python -u finetune/train_predictor_mini.py
"""

import os
import sys
import json
import time
from time import gmtime, strftime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
from scipy.stats import spearmanr

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

print("Starting mini predictor training...", flush=True)


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


def quick_ic_test(model, tokenizer, device, test_data, n_samples=50, lookback=90, pred_len=10):
    """
    快速 IC 测试（在训练过程中）

    Returns:
        dict: { 'ic': float, 'rank_ic': float, 'direction_acc': float }
    """
    model.eval()

    predictions = []
    actuals = []

    import pandas as pd
    symbols = list(test_data.keys())[:n_samples]

    for symbol in symbols:
        df = test_data[symbol]
        if len(df) < lookback + pred_len:
            continue

        try:
            df_history = df.iloc[-(lookback + pred_len):-pred_len].copy()

            # Prepare data
            feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
            x = df_history[feature_cols].values.astype(np.float32)

            # Normalize
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -5.0, 5.0)

            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)

            # Tokenize
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(x_tensor, half=True)

                # Predict (simple forward, no sampling)
                # Use model directly
                logits = model(token_seq_0[:, :-1], token_seq_1[:, :-1], None)

                # Get predicted tokens for prediction window
                pred_tokens_0 = logits[0].argmax(dim=-1)[:, -pred_len:]  # [1, pred_len]
                pred_tokens_1 = logits[1].argmax(dim=-1)[:, -pred_len:]  # [1, pred_len]

                # Build full token sequences for decoding
                full_tokens_0 = torch.cat([token_seq_0, pred_tokens_0], dim=1)  # [1, lookback+pred_len]
                full_tokens_1 = torch.cat([token_seq_1, pred_tokens_1], dim=1)  # [1, lookback+pred_len]

                # Decode to get actual feature values (normalized)
                decoded = tokenizer.decode((full_tokens_0, full_tokens_1), half=True)  # [1, seq_len, d_in]

                # Extract close prices (column 3: open, high, low, close, vol, amt)
                pred_close_norm = decoded[0, :, 3].cpu().numpy()  # [seq_len]

                # Denormalize using saved parameters
                close_mean = x_mean[3]
                close_std = x_std[3]
                pred_close_raw = pred_close_norm * close_std + close_mean

                # Calculate predicted return
                lookback_close_mean = pred_close_raw[:lookback].mean()
                pred_close_end = pred_close_raw[-pred_len:].mean()
                pred_close_change = (pred_close_end - lookback_close_mean) / lookback_close_mean

            actual_close = df['close'].iloc[-pred_len:]
            actual_return = (actual_close.iloc[-1] - actual_close.iloc[0]) / actual_close.iloc[0]

            predictions.append(pred_close_change)
            actuals.append(actual_return)

        except Exception as e:
            continue

    if len(predictions) > 5:
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        ic = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        rank_ic, _ = spearmanr(predictions, actuals) if len(predictions) > 1 else (0, 0)
        direction_acc = np.mean((predictions > 0) == (actuals > 0))

        return {
            'ic': ic,
            'rank_ic': rank_ic,
            'direction_acc': direction_acc,
            'n_samples': len(predictions)
        }

    return None


def create_dataloader(dataset, config, data_type):
    if data_type == 'train':
        sampler = RandomSampler(dataset, replacement=True, num_samples=config.n_train_iter)
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=(data_type == 'train')
    )
    return loader


def train_model(model, tokenizer, device, config, save_dir, test_data=None):
    """训练模型，支持动态早停和周期性学习率"""
    start_time = time.time()
    print(f"BATCHSIZE: {config.batch_size}", flush=True)
    print(f"LR: {config.predictor_learning_rate}", flush=True)
    print(f"LR Scheduler: {config.lr_scheduler}", flush=True)

    # Import dataset
    from finetune.dataset import QlibDataset

    train_dataset = QlibDataset('train', config=config)
    val_dataset = QlibDataset('val', config=config)

    train_loader = create_dataloader(train_dataset, config, 'train')
    val_loader = create_dataloader(val_dataset, config, 'val')

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
    print(f"Steps/epoch: {len(train_loader)}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        weight_decay=config.adam_weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    # === 学习率调度器 ===
    if config.lr_scheduler == 'cosine_warmup':
        # 周期性学习率，能跳出局部最优
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.lr_T_0,      # 第一个周期 5 epochs
            T_mult=config.lr_T_mult, # 每次周期翻倍
            eta_min=config.lr_eta_min
        )
        print(f"CosineAnnealingWarmRestarts: T_0={config.lr_T_0}, T_mult={config.lr_T_mult}", flush=True)
    elif config.lr_scheduler == 'plateau':
        # 当 val_loss 不下降时降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=config.lr_eta_min
        )
        print(f"ReduceLROnPlateau: factor=0.5, patience=2", flush=True)
    else:
        # 默认 OneCycle
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.predictor_learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=config.epochs,
            pct_start=0.03,
            div_factor=10
        )
        print(f"OneCycleLR", flush=True)

    best_val_loss = float('inf')
    best_ic = -999
    patience_counter = 0

    # 动态早停：记录历史用于趋势判断
    val_loss_history = []
    ic_history = []

    history = {
        'train_loss': [],
        'val_loss': [],
        'ic': [],
        'lr': []
    }

    for epoch_idx in range(config.epochs):
        epoch_start = time.time()
        model.train()

        # Set dataset epoch
        train_dataset.py_rng.seed(config.seed + epoch_idx * 10000)

        epoch_losses = []
        current_lr = optimizer.param_groups[0]['lr']

        for i, (batch_x, batch_stamp, batch_direction) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_stamp = batch_stamp.to(device, non_blocking=True)
            batch_direction = batch_direction.to(device, non_blocking=True)

            # Tokenize
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward
            logits = model(token_in[0], token_in[1], batch_stamp[:, :-1, :])
            recon_loss, s1_loss, s2_loss = model.module.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            # === 方向损失 ===
            # 从预测 logits 得到预测 token（两个序列）
            pred_tokens_0 = logits[0].argmax(dim=-1)  # [batch, seq_len]
            pred_tokens_1 = logits[1].argmax(dim=-1)  # [batch, seq_len]

            # 构建完整预测 token 序列（两部分）
            full_pred_tokens_0 = torch.cat([token_in[0], pred_tokens_0[:, -config.predict_window:]], dim=1)
            full_pred_tokens_1 = torch.cat([token_in[1], pred_tokens_1[:, -config.predict_window:]], dim=1)

            # 解码预测价格（传入元组）
            with torch.no_grad():
                decoded_pred = tokenizer.decode((full_pred_tokens_0, full_pred_tokens_1), half=True)  # [batch, seq_len, d_in]

            # 计算预测方向：close 列是第 3 列
            pred_close = decoded_pred[:, :, 3]  # [batch, seq_len]
            lookback_close_mean = pred_close[:, :config.lookback_window].mean(dim=1)
            pred_close_mean = pred_close[:, -config.predict_window:].mean(dim=1)
            pred_direction_logits = pred_close_mean - lookback_close_mean

            # 方向预测概率
            pred_direction_prob = torch.sigmoid(pred_direction_logits)

            # BCE 损失
            direction_loss = F.binary_cross_entropy(pred_direction_prob, batch_direction)

            # === 组合损失 ===
            direction_loss_weight = 0.1  # 方向损失权重
            total_loss = recon_loss + direction_loss_weight * direction_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            # Step scheduler (for OneCycle and Cosine)
            if config.lr_scheduler in ['cosine_warmup', 'onecycle']:
                scheduler.step()

            epoch_losses.append(total_loss.item())

        # Record LR
        history['lr'].append(current_lr)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_stamp, batch_direction in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_stamp = batch_stamp.to(device, non_blocking=True)
                batch_direction = batch_direction.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_stamp[:, :-1, :])
                val_loss, _, _ = model.module.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )

                val_loss_sum += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        val_loss_history.append(avg_val_loss)

        # Plateau scheduler step
        if config.lr_scheduler == 'plateau':
            scheduler.step(avg_val_loss)

        # Quick IC test
        ic_result = quick_ic_test(model.module, tokenizer, device, test_data, n_samples=100)
        current_ic = 0
        if ic_result:
            current_ic = ic_result['ic']
            history['ic'].append(current_ic)
            ic_history.append(current_ic)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"\n=== Epoch {epoch_idx+1}/{config.epochs} ===", flush=True)
        print(f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}", flush=True)
        print(f"LR: {current_lr:.6f}, IC: {current_ic:.4f}", flush=True)
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}", flush=True)

        # === 每轮结束都保存最新模型 ===
        latest_path = f"{save_dir}/checkpoints/latest_model"
        model.module.save_pretrained(latest_path)

        # === 动态早停判断 ===
        improved = False

        # 1. Val Loss 改善
        if avg_val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            improved = True
            save_path = f"{save_dir}/checkpoints/best_model"
            model.module.save_pretrained(save_path)
            print(f"[VAL LOSS SAVED] {best_val_loss:.4f}", flush=True)

        # 2. IC 改善（如果启用）
        if config.early_stopping_check_ic and current_ic > best_ic + 0.01:
            best_ic = current_ic
            patience_counter = 0
            improved = True
            ic_save_path = f"{save_dir}/checkpoints/best_ic_model"
            model.module.save_pretrained(ic_save_path)
            print(f"[IC SAVED] {best_ic:.4f}", flush=True)

        if not improved:
            patience_counter += 1

        # === 结合 LR 状态的动态早停策略 ===
        # 判断是否处于"爬山过程"（LR 较高，正在探索）
        lr_high = current_lr > 0.001  # LR > 0.001 视为探索期

        # 判断是否刚重启（CosineAnnealingWarmRestarts 周期起点附近）
        # T_0=5, T_mult=2 → 重启点: epoch 5, 15, 35, ...
        restart_epochs = [5, 15, 35, 75]  # 预计算的重启点
        near_restart = any(abs(epoch_idx + 1 - e) <= 2 for e in restart_epochs)

        # 探索期状态判断
        is_exploring = lr_high or near_restart

        # Grace period（前几轮不早停）
        if epoch_idx < config.early_stopping_grace_period:
            print(f"[GRACE] Epoch {epoch_idx+1} < {config.early_stopping_grace_period}", flush=True)
        else:
            # 动态 patience：探索期宽松，精细期严格
            if is_exploring:
                dynamic_patience = config.early_stopping_patience * 2  # 爬山过程，允许更多波动
                state_desc = "EXPLORING (LR high/near restart)"
            else:
                dynamic_patience = config.early_stopping_patience  # 精细收敛期，严格判断
                state_desc = "CONVERGING (LR low)"

            print(f"[STATE] {state_desc}, Dynamic patience: {dynamic_patience}", flush=True)

            # 窗口趋势判断：看最近 window 轮是否有改善
            window = config.early_stopping_window
            if len(val_loss_history) >= window:
                window_min = min(val_loss_history[-window:])
                window_best_ic = max(ic_history[-window:]) if len(ic_history) >= window else 0

                # 窗口内有改善 → 继续
                if window_min < best_val_loss or window_best_ic > best_ic:
                    print(f"[WINDOW TREND] Recent improvement detected", flush=True)
                elif patience_counter >= dynamic_patience:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs (dynamic_patience={dynamic_patience})", flush=True)
                    # 早停时也保存最终模型
                    final_path = f"{save_dir}/checkpoints/final_model"
                    model.module.save_pretrained(final_path)
                    print(f"[FINAL SAVED] Val Loss: {avg_val_loss:.4f}, IC: {current_ic:.4f}", flush=True)
                    break

            print(f"[PATIENCE] {patience_counter}/{dynamic_patience}", flush=True)

        print(flush=True)

    # 训练正常结束时保存最终模型
    final_path = f"{save_dir}/checkpoints/final_model"
    model.module.save_pretrained(final_path)
    print(f"[FINAL SAVED] Val Loss: {avg_val_loss:.4f}, IC: {current_ic:.4f}", flush=True)

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'epochs_trained': epoch_idx + 1,
        'history': history,
        'final_val_loss': avg_val_loss,
        'final_ic': current_ic
    }


def main():
    from finetune.config_mini import ConfigMini
    from finetune.dataset import QlibDataset
    from model.kronos import KronosTokenizer, Kronos

    config = ConfigMini()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}", flush=True)
    print(f"Kronos-mini Predictor Training", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Device: {device}", flush=True)

    set_seed(config.seed)

    save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Load tokenizer (finetuned)
    tokenizer_path = config.finetuned_tokenizer_path
    if not os.path.exists(tokenizer_path):
        tokenizer_path = config.pretrained_tokenizer_path
        print(f"Warning: Using original tokenizer", flush=True)

    print(f"\nLoading tokenizer: {tokenizer_path}", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    # Load predictor (pretrained mini)
    print(f"Loading predictor: {config.pretrained_predictor_path}", flush=True)
    model = Kronos.from_pretrained(config.pretrained_predictor_path)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None) if False else model

    # Wrap for DDP-like access
    class ModelWrapper:
        def __init__(self, model):
            self.module = model
            self.device = model.device if hasattr(model, 'device') else device

        def train(self):
            self.module.train()

        def eval(self):
            self.module.eval()

        def parameters(self):
            return self.module.parameters()

        def __call__(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.module, name)

    model = ModelWrapper(model)
    print(f"Model size: {get_model_size(model.module):.2f}M", flush=True)

    # Load test data for IC monitoring
    import pickle
    test_path = os.path.join(config.dataset_path, "test_data.pkl")
    if os.path.exists(test_path):
        print(f"Loading test data for IC monitoring...", flush=True)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        print(f"Test data: {len(test_data)} stocks", flush=True)
    else:
        test_data = None
        print("Warning: No test data for IC monitoring", flush=True)

    # Train
    result = train_model(model, tokenizer, device, config, save_dir, test_data)

    # Save summary
    summary = {
        'model': 'mini_predictor',
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'config': {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.predictor_learning_rate,
            'patience': config.early_stopping_patience,
        },
        'result': {
            'best_val_loss': result['best_val_loss'],
            'best_ic': result['best_ic'],
            'epochs_trained': result['epochs_trained'],
        }
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print(f"\n{'='*60}", flush=True)
    print(f"Training completed!", flush=True)
    print(f"Best Val Loss: {result['best_val_loss']:.4f}", flush=True)
    print(f"Best IC: {result['best_ic']:.4f}", flush=True)
    print(f"Epochs: {result['epochs_trained']}", flush=True)
    print(f"Saved to: {save_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    # Patch dataset
    import finetune.dataset as ds_module
    original_init = ds_module.QlibDataset.__init__
    def patched_init(self, data_type='train', config=None):
        if config is None:
            from finetune.config_mini import ConfigMini
            config = ConfigMini()
        import finetune.config as global_config
        old_config = global_config.Config
        global_config.Config = lambda: config
        try:
            original_init(self, data_type)
        finally:
            global_config.Config = old_config
    ds_module.QlibDataset.__init__ = patched_init

    main()