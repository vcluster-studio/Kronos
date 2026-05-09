"""
单卡训练脚本 - Windows 兼容版

用法：
    python -u train_predictor_single.py

不需要 torchrun，直接运行即可。
"""

import os
import sys
import json
import time
from time import gmtime, strftime

print("Starting script...", flush=True)

import torch
from torch.utils.data import DataLoader

print("Torch imported", flush=True)

# Add project root to path for model imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

print(f"Project root: {project_root}", flush=True)

from finetune.config import Config
print("Config imported", flush=True)

from finetune.dataset import QlibDataset
print("Dataset imported", flush=True)

from model.kronos import KronosTokenizer, Kronos
print("Model imported", flush=True)


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


def train_model(model, tokenizer, device, config, save_dir):
    """Main training loop for predictor."""
    start_time = time.time()
    print(f"BATCHSIZE: {config['batch_size']}")
    print(f"Device: {device}")

    train_loader, train_dataset = create_dataloader(config, 'train')
    val_loader, val_dataset = create_dataloader(config, 'val')

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    print(f"Train steps/epoch: {len(train_loader)}, Val steps: {len(val_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['predictor_learning_rate'],
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

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # Logging
            if (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )

                tot_val_loss += val_loss.item()
                val_batches += 1

        avg_val_loss = tot_val_loss / val_batches if val_batches > 0 else 0

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    set_seed(config['seed'])

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Load tokenizer (fine-tuned)
    print(f"Loading fine-tuned tokenizer from: {config['finetuned_tokenizer_path']}", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    print("Tokenizer loaded", flush=True)
    tokenizer.eval().to(device)

    # Load predictor (pretrained)
    print(f"Loading pretrained predictor from: {config['pretrained_predictor_path']}", flush=True)
    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    print("Predictor loaded", flush=True)
    model.to(device)

    print(f"Predictor Model Size: {get_model_size(model):.2f}M parameters", flush=True)

    # Start Training
    result = train_model(model, tokenizer, device, config, save_dir)

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
