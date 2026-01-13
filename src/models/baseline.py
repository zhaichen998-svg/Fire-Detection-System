"""
Baseline YOLOv8 model for fire detection.

This module implements a YOLOv8n-based baseline model for fire detection.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import yaml


class BaselineFireDetector(nn.Module):
    """
    Baseline fire detection model using YOLOv8n architecture.
    
    This is a simple wrapper around the Ultralytics YOLOv8n model
    with minimal modifications for fire detection (flame + smoke).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_size: str = 'n',
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize baseline YOLOv8 model.
        
        Args:
            num_classes: Number of classes (default: 2 for flame and smoke)
            pretrained: Whether to use pretrained COCO weights
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            img_size: Input image size
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize YOLOv8 model
        model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
        self.model = YOLO(model_name)
        
        # Update number of classes if different from pretrained
        if pretrained and num_classes != 80:  # COCO has 80 classes
            self._update_num_classes(num_classes)
    
    def _update_num_classes(self, num_classes: int):
        """Update the number of output classes in the model."""
        # This modifies the model's head to match the desired number of classes
        # The actual implementation depends on YOLOv8 structure
        # Ultralytics YOLO handles this automatically during training
        pass
    
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, img_size, img_size)
            
        Returns:
            Dictionary containing predictions
        """
        # During training, YOLO model returns loss
        # During inference, returns predictions
        if self.training:
            return self.model(x)
        else:
            results = self.model(x, conf=self.conf_threshold, iou=self.iou_threshold)
            return results
    
    def train_step(
        self,
        images: torch.Tensor,
        targets: List[Dict],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Perform a single training step.
        
        Args:
            images: Batch of images
            targets: List of target dictionaries
            optimizer: Optimizer instance
            
        Returns:
            Loss value
        """
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        loss = self.forward(images)
        
        # Backward pass
        if isinstance(loss, dict):
            total_loss = sum(loss.values()) if isinstance(loss, dict) else loss
        else:
            total_loss = loss
        
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def predict(
        self,
        image: torch.Tensor,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Make predictions on an image.
        
        Args:
            image: Input image tensor
            conf_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold
            
        Returns:
            List of prediction dictionaries
        """
        self.eval()
        with torch.no_grad():
            conf = conf_threshold if conf_threshold is not None else self.conf_threshold
            iou = iou_threshold if iou_threshold is not None else self.iou_threshold
            
            results = self.model(image, conf=conf, iou=iou)
            return results
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optional optimizer state dict
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, load_optimizer: bool = False) -> Optional[Dict]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to return optimizer state
            
        Returns:
            Optimizer state dict if load_optimizer=True, else None
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Update model parameters
        self.num_classes = checkpoint.get('num_classes', self.num_classes)
        self.img_size = checkpoint.get('img_size', self.img_size)
        self.conf_threshold = checkpoint.get('conf_threshold', self.conf_threshold)
        self.iou_threshold = checkpoint.get('iou_threshold', self.iou_threshold)
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            return checkpoint['optimizer_state_dict']
        
        return None
    
    def get_model_info(self) -> Dict:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': f'YOLOv8n-baseline',
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }


class BaselineConfig:
    """Configuration class for baseline model."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._default_config()
        
        if config_path:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config)
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'model': {
                'num_classes': 2,
                'model_size': 'n',
                'pretrained': True,
                'img_size': 640,
                'conf_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'lr': 0.01,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'warmup_epochs': 3,
                'patience': 50
            },
            'data': {
                'train_path': 'data/train',
                'val_path': 'data/val',
                'test_path': 'data/test',
                'num_workers': 4
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.config


def create_baseline_model(
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = 'cuda'
) -> BaselineFireDetector:
    """
    Factory function to create a baseline model.
    
    Args:
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
        device: Device to load model on
        
    Returns:
        BaselineFireDetector instance
    """
    model = BaselineFireDetector(
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    model = model.to(device)
    return model
