"""
单卡训练脚本 - Windows 兼容版

用法：
    python train_tokenizer_single.py

不需要 torchrun，直接运行即可。
"""

import os
import sys
import json
import time
from time import gmtime, strftime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path for model imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from finetune.config import Config
from finetune.dataset import QlibDataset
from model.kronos import KronosTokenizer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_size(model):
    """Calculate model size in millions of parameters."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def format_time(seconds: float) -> str:
    """Format time in HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_dataloader(config: dict, data_type: str):
    """Create dataloader for training or validation."""
    dataset = QlibDataset(data_type)

    # Use simple random sampler for single GPU
    from torch.utils.data import RandomSampler, SequentialSampler
    if data_type == 'train':
        sampler = RandomSampler(dataset, replacement=True, num_samples=config['n_train_iter'])
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=0,  # Windows compatible
        pin_memory=True,
        drop_last=(data_type == 'train')
    )
    return loader, dataset


def train_model(model, device, config, save_dir):
    """Main training loop for tokenizer."""
    start_time = time.time()
    print(f"BATCHSIZE: {config['batch_size']}")
    print(f"Device: {device}")

    train_loader, train_dataset = create_dataloader(config, 'train')
    val_loader, val_dataset = create_dataloader(config, 'val')

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    print(f"Train steps/epoch: {len(train_loader)}, Val steps: {len(val_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['tokenizer_learning_rate'],
        weight_decay=config['adam_weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config['tokenizer_learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.03,
        div_factor=10
    )

    best_val_loss = float('inf')
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()

        for i, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.to(device, non_blocking=True)

            # Forward pass
            zs, bsq_loss, _, _ = model(ori_batch_x)
            z_pre, z = zs

            # Loss calculation
            recon_loss_pre = F.mse_loss(z_pre, ori_batch_x)
            recon_loss_all = F.mse_loss(z, ori_batch_x)
            recon_loss = recon_loss_pre + recon_loss_all
            loss = (recon_loss + bsq_loss) / 2

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            # Logging
            if (batch_idx_global + 1) % config['log_interval'] == 0:
                print(
                    f"[Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {loss.item():.4f}"
                )

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                zs, _, _, _ = model(ori_batch_x)
                _, z = zs
                val_loss = F.mse_loss(z, ori_batch_x)

                tot_val_loss += val_loss.item() * ori_batch_x.size(0)
                val_count += ori_batch_x.size(0)

        avg_val_loss = tot_val_loss / val_count if val_count > 0 else 0

        # --- End of Epoch Summary ---
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time

        print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Time This Epoch: {format_time(epoch_time)}")
        print(f"Total Time Elapsed: {format_time(total_time)}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

    return {'best_val_loss': best_val_loss}


def main():
    config = Config().__dict__

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    set_seed(config['seed'])

    save_dir = os.path.join(config['save_path'], config['tokenizer_save_folder_name'])
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Model Initialization
    print(f"Loading pretrained tokenizer from: {config['pretrained_tokenizer_path']}")
    model = KronosTokenizer.from_pretrained(config['pretrained_tokenizer_path'])
    model.to(device)

    print(f"Model Size: {get_model_size(model):.2f}M parameters")

    # Start Training
    result = train_model(model, device, config, save_dir)

    # Save summary
    summary = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'save_directory': save_dir,
        'final_result': result
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print('Training finished. Summary file saved.')


if __name__ == '__main__':
    main()
