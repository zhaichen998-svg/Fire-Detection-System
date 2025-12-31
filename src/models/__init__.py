"""Models package initialization."""
from .baseline import BaselineFireDetector, BaselineConfig, create_baseline_model
from .improved import (
    ImprovedFireDetectorV1,
    ImprovedFireDetectorV2,
    ImprovedFireDetectorV3,
    ImprovedFireDetectorV4,
    ImprovedFireDetectorV5,
    KnowledgeDistillation,
    create_improved_model,
    SEBlock,
    CBAM
)

__all__ = [
    'BaselineFireDetector',
    'BaselineConfig',
    'create_baseline_model',
    'ImprovedFireDetectorV1',
    'ImprovedFireDetectorV2',
    'ImprovedFireDetectorV3',
    'ImprovedFireDetectorV4',
    'ImprovedFireDetectorV5',
    'KnowledgeDistillation',
    'create_improved_model',
    'SEBlock',
    'CBAM'
]
