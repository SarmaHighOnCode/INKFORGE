"""
INKFORGE — Training Entrypoint

Trains the LSTM+MDN handwriting synthesis model on the IAM dataset.

Usage:
    python train.py --config configs/lstm_mdn_base.yaml
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

import yaml

from backend.app.ml.model import HandwritingLSTM
from backend.app.ml.dataset import IAMStrokeDataset, collate_fn
from backend.app.ml.utils import compute_mdn_loss, build_vocab


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Inkforge LSTM+MDN handwriting model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lstm_mdn_base.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training (auto will use CUDA if available).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_scheduler(optimizer, config: dict, num_training_steps: int):
    """Create learning rate scheduler."""
    scheduler_type = config["training"].get("lr_scheduler", "cosine")

    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=1e-7,
        )
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=config["training"].get("lr_step_size", 10),
            gamma=config["training"].get("lr_gamma", 0.5),
        )
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )
    else:
        return None


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    epoch: int,
    writer=None,
) -> float:
    """
    Train for one epoch.

    Args:
        model: The LSTM+MDN model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        device: PyTorch device.
        config: Training configuration.
        epoch: Current epoch number.
        writer: TensorBoard writer (optional).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    grad_clip = config["training"].get("gradient_clip_norm", 10.0)
    log_interval = config["logging"].get("log_every_n_steps", 50)

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        strokes = batch["strokes"].to(device)  # [B, seq_len, 5]
        text_indices = batch["text_indices"].to(device)  # [B, text_len]
        pen_states = batch["pen_states"].to(device)  # [B, seq_len]

        batch_size = strokes.shape[0]

        # Create style embeddings (random for now, would be learned/provided)
        style_dim = config["model"].get("style_dim", 128)
        style_z = torch.randn(batch_size, style_dim, device=device)

        # Prepare input: previous strokes (shifted by 1)
        prev_strokes = torch.zeros_like(strokes)
        prev_strokes[:, 1:] = strokes[:, :-1]

        # Expand text indices to match stroke sequence length
        seq_len = strokes.shape[1]
        text_len = text_indices.shape[1]

        # Simple approach: repeat each char for seq_len/text_len strokes
        strokes_per_char = max(1, seq_len // text_len)
        expanded_text = text_indices.repeat_interleave(strokes_per_char, dim=1)
        expanded_text = expanded_text[:, :seq_len]  # Truncate to exact length

        # Pad if needed
        if expanded_text.shape[1] < seq_len:
            pad = torch.zeros(batch_size, seq_len - expanded_text.shape[1], dtype=torch.long, device=device)
            expanded_text = torch.cat([expanded_text, pad], dim=1)

        # Forward pass
        optimizer.zero_grad()
        mdn_params, pen_logits, _ = model(expanded_text, prev_strokes, style_z)

        # Compute loss
        target_strokes = strokes[:, :, :2]  # Just Δx, Δy
        loss = compute_mdn_loss(
            mdn_params,
            pen_logits,
            target_strokes,
            pen_states,
            num_mixtures=config["model"].get("num_mixtures", 20),
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Logging
        if batch_idx % log_interval == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("train/loss", loss.item(), global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
) -> float:
    """
    Validate the model.

    Args:
        model: The LSTM+MDN model.
        dataloader: Validation data loader.
        device: PyTorch device.
        config: Training configuration.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Use deterministic style embedding for reproducible validation (seeded once per epoch)
    val_gen = torch.Generator(device=device).manual_seed(42)

    with torch.no_grad():
        for batch in dataloader:
            strokes = batch["strokes"].to(device)
            text_indices = batch["text_indices"].to(device)
            pen_states = batch["pen_states"].to(device)

            batch_size = strokes.shape[0]
            style_dim = config["model"].get("style_dim", 128)
            style_z = torch.randn(batch_size, style_dim, device=device, generator=val_gen)

            prev_strokes = torch.zeros_like(strokes)
            prev_strokes[:, 1:] = strokes[:, :-1]

            seq_len = strokes.shape[1]
            text_len = text_indices.shape[1]
            strokes_per_char = max(1, seq_len // text_len)
            expanded_text = text_indices.repeat_interleave(strokes_per_char, dim=1)[:, :seq_len]

            if expanded_text.shape[1] < seq_len:
                pad = torch.zeros(batch_size, seq_len - expanded_text.shape[1], dtype=torch.long, device=device)
                expanded_text = torch.cat([expanded_text, pad], dim=1)

            mdn_params, pen_logits, _ = model(expanded_text, prev_strokes, style_z)

            target_strokes = strokes[:, :, :2]
            loss = compute_mdn_loss(
                mdn_params,
                pen_logits,
                target_strokes,
                pen_states,
                num_mixtures=config["model"].get("num_mixtures", 20),
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: dict,
    save_path: Path,
    vocab: dict,
    stroke_mean: torch.Tensor,
    stroke_std: torch.Tensor,
    scheduler=None,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": config["model"],
        "vocab": vocab,
        "stroke_mean": stroke_mean,
        "stroke_std": stroke_std,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def main() -> None:
    """Main training loop."""
    args = parse_args()

    print("=" * 60)
    print("INKFORGE — LSTM+MDN Handwriting Model Training")
    print("=" * 60)
    print(f"Config: {args.config}")

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Get device
    device = get_device(args.device if args.device != "auto" else config.get("device", "auto"))
    print(f"Device: {device}")

    # Build vocabulary
    vocab = build_vocab()
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Initialize datasets
    data_config = config["data"]
    print(f"\nLoading data from: {data_config['data_dir']}")

    train_dataset = IAMStrokeDataset(
        data_dir=data_config["data_dir"],
        split="train",
        max_seq_len=data_config.get("max_seq_len", 700),
        augment=config["augmentation"].get("enabled", True),
        vocab=vocab,
    )

    val_dataset = IAMStrokeDataset(
        data_dir=data_config["data_dir"],
        split="val",
        max_seq_len=data_config.get("max_seq_len", 700),
        augment=False,
        vocab=vocab,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Check if we have data
    if len(train_dataset) == 0:
        print("\nWARNING: No training data found!")
        print("Please run the data preprocessing script first:")
        print("  python scripts/download_iam.py --output data/iam/")
        print("  python scripts/preprocess.py --input data/iam/ --output data/processed/")
        print("\nCreating dummy data for testing...")

        # Create minimal dummy data for testing the pipeline
        dummy_data_dir = Path("data/processed/train")
        dummy_data_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            dummy_strokes = torch.randn(100, 5).tolist()
            dummy_strokes = [[s[0], s[1], 1, 0, 0] for s in dummy_strokes[:-1]]
            dummy_strokes.append([0, 0, 0, 0, 1])  # End token

            dummy_sample = {
                "strokes": dummy_strokes,
                "text": f"Sample text {i}",
                "writer_id": i % 5,
            }
            torch.save(dummy_sample, dummy_data_dir / f"sample_{i}.pt")

        # Reload dataset
        train_dataset = IAMStrokeDataset(
            data_dir=data_config["data_dir"],
            split="train",
            max_seq_len=data_config.get("max_seq_len", 700),
            augment=config["augmentation"].get("enabled", True),
            vocab=vocab,
        )
        print(f"Created {len(train_dataset)} dummy samples for testing")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_config.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=data_config.get("pin_memory", False) and device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=data_config.get("pin_memory", False) and device.type == "cuda",
    ) if len(val_dataset) > 0 else None

    # Initialize model
    model_config = config["model"]
    model = HandwritingLSTM(
        vocab_size=model_config.get("vocab_size", vocab_size),
        char_embed_dim=model_config.get("char_embed_dim", 256),
        style_dim=model_config.get("style_dim", 128),
        hidden_dim=model_config.get("hidden_dim", 512),
        num_layers=model_config.get("num_layers", 3),
        dropout=model_config.get("dropout", 0.2),
        num_mixtures=model_config.get("num_mixtures", 20),
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0001),
    )

    # Initialize scheduler
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = get_scheduler(optimizer, config, num_training_steps)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("loss", float("inf"))
        print(f"Resuming from epoch {start_epoch}")

    # Setup TensorBoard
    writer = None
    if config["logging"].get("tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(config["logging"]["log_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging: {log_dir}")
        except ImportError:
            print("TensorBoard not available, skipping logging")

    # Create checkpoint directory
    checkpoint_dir = Path(config["checkpointing"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    patience_counter = 0
    early_stopping_patience = config["training"].get("early_stopping_patience", 15)
    val_loss = None  # Initialized here so scheduler.step can reference safely

    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, writer)
        print(f"Training Loss: {train_loss:.4f}")

        # Validate
        if val_loader is not None and (epoch + 1) % config["logging"].get("val_every_n_epochs", 1) == 0:
            val_loss = validate(model, val_loader, device, config)
            print(f"Validation Loss: {val_loss:.4f}")

            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if config["checkpointing"].get("save_best_only", True):
                    save_checkpoint(
                        model, optimizer, epoch, val_loss, config,
                        checkpoint_dir / f"{config['checkpointing']['checkpoint_name']}_best.pt",
                        vocab, train_dataset.stroke_mean, train_dataset.stroke_std,
                    )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        else:
            # No validation, save periodically
            if (epoch + 1) % config["checkpointing"].get("save_every_n_epochs", 5) == 0:
                save_checkpoint(
                    model, optimizer, epoch, train_loss, config,
                    checkpoint_dir / f"{config['checkpointing']['checkpoint_name']}_epoch{epoch + 1}.pt",
                    vocab, train_dataset.stroke_mean, train_dataset.stroke_std,
                )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                # Use val_loss when available, otherwise fall back to train_loss
                step_loss = val_loss if val_loss is not None else train_loss
                scheduler.step(step_loss)
            else:
                scheduler.step()

        if writer is not None:
            writer.add_scalar("train/epoch_loss", train_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, train_loss, config,
        checkpoint_dir / f"{config['checkpointing']['checkpoint_name']}_final.pt",
        vocab, train_dataset.stroke_mean, train_dataset.stroke_std,
    )

    # Export TorchScript model for inference
    print("\nExporting TorchScript model...")
    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save(checkpoint_dir / f"{config['checkpointing']['checkpoint_name']}_final.pts")
        print("TorchScript export successful")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    if writer is not None:
        writer.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
