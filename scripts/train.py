"""
Training script for fire detection models.

This script provides configuration-driven training with:
- Multi-epoch training loop
- Validation at each epoch
- Checkpoint saving
- Early stopping
- Tensorboard logging
- Reproducibility (seed management)
"""

import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets import FireDetectionDataset, collate_fn
from src.models import create_baseline_model, create_improved_model
from src.losses import CombinedLoss, LossScheduler
from src.utils import DetectionMetrics
from src.augmentation import get_training_augmentation, get_validation_augmentation


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict):
    """Create training and validation data loaders."""
    # Training dataset
    train_dataset = FireDetectionDataset(
        img_dir=config['data']['train_img_dir'],
        label_dir=config['data']['train_label_dir'],
        img_size=config['model']['img_size'],
        transforms=get_training_augmentation(
            img_size=config['model']['img_size'],
            mode=config['augmentation']['mode']
        ),
        augment=True,
        class_names=config['data']['class_names']
    )
    
    # Validation dataset
    val_dataset = FireDetectionDataset(
        img_dir=config['data']['val_img_dir'],
        label_dir=config['data']['val_label_dir'],
        img_size=config['model']['img_size'],
        transforms=get_validation_augmentation(img_size=config['model']['img_size']),
        augment=False,
        class_names=config['data']['class_names']
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config: dict, device: str):
    """Create model based on configuration."""
    model_type = config['model'].get('type', 'baseline')
    
    if model_type == 'baseline':
        model = create_baseline_model(
            num_classes=config['model']['num_classes'],
            pretrained=config['model'].get('pretrained', True),
            device=device
        )
    else:
        version = config['model'].get('version', 'v1')
        model = create_improved_model(
            version=version,
            num_classes=config['model']['num_classes'],
            model_size=config['model'].get('size', 'n'),
            device=device
        )
    
    return model


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    writer,
    config
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Use Ultralytics YOLO training
        # This is a simplified version - actual implementation would use YOLO's train method
        loss = model(images)
        
        if isinstance(loss, dict):
            loss = sum(loss.values())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if batch_idx % config['training'].get('log_interval', 10) == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), step)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device, epoch, writer, config):
    """Validate the model."""
    model.eval()
    metrics = DetectionMetrics(
        num_classes=config['model']['num_classes'],
        class_names=config['data']['class_names']
    )
    
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            
            # Compute loss (simplified)
            if isinstance(outputs, dict):
                loss = sum(outputs.values())
                total_loss += loss.item()
            
            # Update metrics (would need to extract predictions from outputs)
            # This is simplified - actual implementation would parse YOLO outputs
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    # Compute metrics
    summary = metrics.get_summary()
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    for key, value in summary.items():
        writer.add_scalar(f'Val/{key}', value, epoch)
    
    return avg_loss, summary


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_metric,
    config,
    save_dir,
    is_best=False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f'Saved best model to {best_path}')


def main():
    parser = argparse.ArgumentParser(description='Train fire detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config.get('experiment_name', 'fire_detection')
    save_dir = os.path.join('runs', f'{exp_name}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    
    # Create data loaders
    print('Creating data loaders...')
    train_loader, val_loader = create_data_loaders(config)
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # Create model
    print('Creating model...')
    model = create_model(config, device)
    print(f'Model created: {type(model).__name__}')
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.0005)
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training'].get('min_lr', 1e-6)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        print(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
    
    # Training loop
    print('Starting training...')
    patience = config['training'].get('patience', 50)
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["training"]["epochs"]}')
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, None, device, epoch, writer, config
        )
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device, epoch, writer, config)
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Metrics: {val_metrics}')
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        # Check if best model
        current_metric = val_metrics.get('mAP@0.50', 0.0)
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % config['training'].get('save_interval', 10) == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, best_metric, config, save_dir, is_best
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    
    print('Training completed!')
    writer.close()


if __name__ == '__main__':
    main()
