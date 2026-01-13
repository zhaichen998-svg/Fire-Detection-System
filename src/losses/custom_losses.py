"""
Custom loss functions for fire detection.

This module implements various loss functions including:
- Focal Loss for handling class imbalance
- Weighted IoU (WIoU) for better localization
- Combined loss functions
- Loss weight scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in range (0,1) to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedIoULoss(nn.Module):
    """
    Weighted IoU Loss for better bounding box regression.
    
    This loss adds weights based on the quality of predictions to emphasize
    hard examples during training.
    """
    
    def __init__(
        self,
        beta: float = 0.5,
        eps: float = 1e-7,
        reduction: str = 'mean'
    ):
        """
        Initialize Weighted IoU Loss.
        
        Args:
            beta: Weight factor for IoU weighting
            eps: Small constant for numerical stability
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted IoU loss.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in format [x1, y1, x2, y2]
            target_boxes: Target boxes (N, 4) in format [x1, y1, x2, y2]
            
        Returns:
            Weighted IoU loss value
        """
        # Compute intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Compute areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Compute union
        union = pred_area + target_area - intersection + self.eps
        
        # Compute IoU
        iou = intersection / union
        
        # Compute weights based on IoU quality
        # Lower IoU -> Higher weight
        weights = torch.exp(-self.beta * iou)
        
        # Weighted IoU loss
        loss = weights * (1 - iou)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GIoULoss(nn.Module):
    """
    Generalized Intersection over Union Loss.
    
    Reference: Rezatofighi et al. "Generalized Intersection over Union"
    """
    
    def __init__(self, eps: float = 1e-7, reduction: str = 'mean'):
        """
        Initialize GIoU Loss.
        
        Args:
            eps: Small constant for numerical stability
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GIoU loss.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in format [x1, y1, x2, y2]
            target_boxes: Target boxes (N, 4) in format [x1, y1, x2, y2]
            
        Returns:
            GIoU loss value
        """
        # Intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Union
        union = pred_area + target_area - intersection + self.eps
        
        # IoU
        iou = intersection / union
        
        # Smallest enclosing box
        c_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        c_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        c_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        c_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + self.eps
        
        # GIoU
        giou = iou - (c_area - union) / c_area
        
        # Loss
        loss = 1 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that weighs multiple losses.
    
    Combines classification loss, localization loss, and objectness loss.
    """
    
    def __init__(
        self,
        cls_loss_weight: float = 1.0,
        box_loss_weight: float = 5.0,
        obj_loss_weight: float = 1.0,
        use_focal: bool = True,
        use_giou: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Initialize combined loss.
        
        Args:
            cls_loss_weight: Weight for classification loss
            box_loss_weight: Weight for box regression loss
            obj_loss_weight: Weight for objectness loss
            use_focal: Whether to use focal loss for classification
            use_giou: Whether to use GIoU for box regression
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_weight = box_loss_weight
        self.obj_loss_weight = obj_loss_weight
        
        # Classification loss
        if use_focal:
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss()
        
        # Box regression loss
        if use_giou:
            self.box_loss = GIoULoss()
        else:
            self.box_loss = WeightedIoULoss()
        
        # Objectness loss
        self.obj_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        cls_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        box_preds: torch.Tensor,
        box_targets: torch.Tensor,
        obj_preds: torch.Tensor,
        obj_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            cls_preds: Classification predictions
            cls_targets: Classification targets
            box_preds: Box predictions
            box_targets: Box targets
            obj_preds: Objectness predictions
            obj_targets: Objectness targets
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Compute individual losses
        cls_loss = self.cls_loss(cls_preds, cls_targets) * self.cls_loss_weight
        box_loss = self.box_loss(box_preds, box_targets) * self.box_loss_weight
        obj_loss = self.obj_loss(obj_preds, obj_targets) * self.obj_loss_weight
        
        # Total loss
        total_loss = cls_loss + box_loss + obj_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'cls': cls_loss.item(),
            'box': box_loss.item(),
            'obj': obj_loss.item()
        }
        
        return total_loss, loss_dict


class LossScheduler:
    """
    Dynamic loss weight scheduler.
    
    Adjusts loss weights during training based on performance and epoch.
    """
    
    def __init__(
        self,
        initial_cls_weight: float = 1.0,
        initial_box_weight: float = 5.0,
        initial_obj_weight: float = 1.0,
        warmup_epochs: int = 5
    ):
        """
        Initialize loss scheduler.
        
        Args:
            initial_cls_weight: Initial classification loss weight
            initial_box_weight: Initial box loss weight
            initial_obj_weight: Initial objectness loss weight
            warmup_epochs: Number of warmup epochs
        """
        self.initial_cls_weight = initial_cls_weight
        self.initial_box_weight = initial_box_weight
        self.initial_obj_weight = initial_obj_weight
        self.warmup_epochs = warmup_epochs
    
    def get_weights(self, epoch: int, max_epochs: int) -> Tuple[float, float, float]:
        """
        Get loss weights for current epoch.
        
        Args:
            epoch: Current epoch
            max_epochs: Maximum number of epochs
            
        Returns:
            Tuple of (cls_weight, box_weight, obj_weight)
        """
        # During warmup, gradually increase box loss weight
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            box_weight = self.initial_box_weight * progress
        else:
            box_weight = self.initial_box_weight
        
        # Increase classification weight in later epochs
        progress = epoch / max_epochs
        cls_weight = self.initial_cls_weight * (1.0 + progress)
        
        return cls_weight, box_weight, self.initial_obj_weight


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross entropy loss.
    
    Prevents the model from becoming over-confident in its predictions.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0 = no smoothing, 1 = uniform distribution)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            pred: Predictions of shape (N, C)
            target: Targets of shape (N,)
            
        Returns:
            Loss value
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
