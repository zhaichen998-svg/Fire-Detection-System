"""Fire Detection System package initialization."""

__version__ = "1.0.0"
__author__ = "Fire Detection Team"
__description__ = "PyTorch-based fire detection system with baseline and improvements"

from . import datasets
from . import models
from . import losses
from . import augmentation
from . import utils

__all__ = [
    'datasets',
    'models',
    'losses',
    'augmentation',
    'utils'
]
