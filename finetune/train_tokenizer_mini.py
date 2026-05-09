"""
Kronos-mini 分词器训练脚本 - 支持早停

用法：
    python -u finetune/train_tokenizer_mini.py
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

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

print("Starting mini tokenizer training...", flush=True)

import torch
print("Torch imported", flush=True)

from finetune.config_mini import ConfigMini
from finetune.dataset import QlibDataset
from model.kronos import KronosTokenizer


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_dataloader(config, data_type):
    dataset = QlibDataset(data_type, config=config)

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
    return loader, dataset


def train_model(model, device, config, save_dir):
    """训练模型，支持动态早停和周期性学习率"""
    start_time = time.time()
    print(f"BATCHSIZE: {config.batch_size}", flush=True)
    print(f"LR: {config.tokenizer_learning_rate}", flush=True)
    print(f"LR Scheduler: {config.lr_scheduler}", flush=True)

    train_loader, train_dataset = create_dataloader(config, 'train')
    val_loader, val_dataset = create_dataloader(config, 'val')

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
    print(f"Steps/epoch: {len(train_loader)}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    # === 学习率调度器 ===
    if config.lr_scheduler == 'cosine_warmup':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.lr_T_0,
            T_mult=config.lr_T_mult,
            eta_min=config.lr_eta_min
        )
        print(f"CosineAnnealingWarmRestarts: T_0={config.lr_T_0}, T_mult={config.lr_T_mult}", flush=True)
    elif config.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=config.lr_eta_min
        )
        print(f"ReduceLROnPlateau: factor=0.5, patience=2", flush=True)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.tokenizer_learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=config.epochs,
            pct_start=0.03,
            div_factor=10
        )
        print(f"OneCycleLR", flush=True)

    best_val_loss = float('inf')
    patience_counter = 0
    batch_idx = 0

    # 动态早停：记录历史用于趋势判断
    val_loss_history = []

    # 训练记录
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch_idx in range(config.epochs):
        epoch_start = time.time()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_losses = []

        for i, (batch_x, _, _) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)

            zs, bsq_loss, _, _ = model(batch_x)
            z_pre, z = zs

            recon_loss = F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x)
            loss = (recon_loss + bsq_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            # Step scheduler (for OneCycle and Cosine)
            if config.lr_scheduler in ['cosine_warmup', 'onecycle']:
                scheduler.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"[E{epoch_idx+1}/{config.epochs} S{i+1}/{len(train_loader)}] LR {lr:.6f} Loss {loss.item():.4f}", flush=True)

            batch_idx += 1

        # Record LR
        history['lr'].append(current_lr)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_x, _, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                zs, _, _, _ = model(batch_x)
                _, z = zs
                val_loss = F.mse_loss(z, batch_x)
                val_loss_sum += val_loss.item() * batch_x.size(0)
                val_count += batch_x.size(0)

        avg_val_loss = val_loss_sum / val_count
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        val_loss_history.append(avg_val_loss)

        # Plateau scheduler step
        if config.lr_scheduler == 'plateau':
            scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"\n=== Epoch {epoch_idx+1}/{config.epochs} ===", flush=True)
        print(f"Train Loss: {avg_train_loss:.4f}", flush=True)
        print(f"Val Loss:   {avg_val_loss:.4f}", flush=True)
        print(f"LR: {current_lr:.6f}", flush=True)
        print(f"Time: {format_time(epoch_time)}, Total: {format_time(total_time)}", flush=True)

        # === 每轮结束都保存最新模型 ===
        latest_path = f"{save_dir}/checkpoints/latest_model"
        model.save_pretrained(latest_path)

        # === 动态早停判断 ===
        improved = False

        if avg_val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            improved = True
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"[VAL LOSS SAVED] {best_val_loss:.4f}", flush=True)

        if not improved:
            patience_counter += 1

        # Grace period（前几轮不早停）
        if epoch_idx < config.early_stopping_grace_period:
            print(f"[GRACE] Epoch {epoch_idx+1} < {config.early_stopping_grace_period}", flush=True)
        else:
            # 窗口趋势判断：看最近 window 轮是否有改善
            window = config.early_stopping_window
            if len(val_loss_history) >= window:
                window_min = min(val_loss_history[-window:])

                # 窗口内有改善 → 继续
                if window_min < best_val_loss:
                    print(f"[WINDOW TREND] Recent improvement detected", flush=True)
                elif patience_counter >= config.early_stopping_patience:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs (patience={config.early_stopping_patience})", flush=True)
                    # 早停时也保存最终模型
                    final_path = f"{save_dir}/checkpoints/final_model"
                    model.save_pretrained(final_path)
                    print(f"[FINAL SAVED] Val Loss: {avg_val_loss:.4f}", flush=True)
                    break

            print(f"[PATIENCE] {patience_counter}/{config.early_stopping_patience}", flush=True)

        print(flush=True)

    # 训练正常结束时保存最终模型
    final_path = f"{save_dir}/checkpoints/final_model"
    model.save_pretrained(final_path)
    print(f"[FINAL SAVED] Val Loss: {avg_val_loss:.4f}", flush=True)

    return {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch_idx + 1,
        'history': history,
        'final_val_loss': avg_val_loss
    }


def main():
    config = ConfigMini()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}", flush=True)
    print(f"Kronos-mini Tokenizer Training", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Device: {device}", flush=True)

    set_seed(config.seed)

    save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    print(f"\nLoading tokenizer from: {config.pretrained_tokenizer_path}", flush=True)
    model = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    model.to(device)

    print(f"Model size: {get_model_size(model):.2f}M parameters", flush=True)

    # Train
    result = train_model(model, device, config, save_dir)

    # Save summary
    summary = {
        'model': 'mini_tokenizer',
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'config': {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.tokenizer_learning_rate,
            'patience': config.early_stopping_patience,
        },
        'result': result
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print(f"\n{'='*60}", flush=True)
    print(f"Training completed!", flush=True)
    print(f"Best Val Loss: {result['best_val_loss']:.4f}", flush=True)
    print(f"Epochs trained: {result['epochs_trained']}", flush=True)
    print(f"Saved to: {save_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    # Patch dataset to accept config
    import finetune.dataset as ds_module
    original_init = ds_module.QlibDataset.__init__
    def patched_init(self, data_type='train', config=None):
        if config is None:
            from finetune.config_mini import ConfigMini
            config = ConfigMini()
        # Temporarily modify the global Config
        import finetune.config as global_config
        old_config = global_config.Config
        global_config.Config = lambda: config
        try:
            original_init(self, data_type)
        finally:
            global_config.Config = old_config
    ds_module.QlibDataset.__init__ = patched_init

    main()
