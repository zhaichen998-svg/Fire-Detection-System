"""Losses package initialization."""
from .custom_losses import (
    FocalLoss,
    WeightedIoULoss,
    GIoULoss,
    CombinedLoss,
    LossScheduler,
    LabelSmoothingLoss
)

__all__ = [
    'FocalLoss',
    'WeightedIoULoss',
    'GIoULoss',
    'CombinedLoss',
    'LossScheduler',
    'LabelSmoothingLoss'
]
