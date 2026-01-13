"""
Data augmentation strategies for fire detection.

This module implements various augmentation techniques:
- Mosaic augmentation
- CutMix and MixUp
- Adaptive augmentation
- Color jittering and HSV adjustments
- Geometric transformations
"""

import cv2
import numpy as np
import torch
import random
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MosaicAugmentation:
    """
    Mosaic augmentation that combines 4 images into one.
    
    Reference: YOLOv4 paper
    """
    
    def __init__(self, img_size: int = 640):
        """
        Initialize Mosaic augmentation.
        
        Args:
            img_size: Target image size
        """
        self.img_size = img_size
    
    def __call__(
        self,
        images: List[np.ndarray],
        boxes_list: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation.
        
        Args:
            images: List of 4 images
            boxes_list: List of 4 box arrays (YOLO format)
            labels_list: List of 4 label arrays
            
        Returns:
            Tuple of (mosaic_image, mosaic_boxes, mosaic_labels)
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        
        # Create output image
        mosaic_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        
        # Random center point
        cx = random.randint(self.img_size // 4, 3 * self.img_size // 4)
        cy = random.randint(self.img_size // 4, 3 * self.img_size // 4)
        
        mosaic_boxes = []
        mosaic_labels = []
        
        # Place each image in a quadrant
        for i, (img, boxes, labels) in enumerate(zip(images, boxes_list, labels_list)):
            h, w = img.shape[:2]
            
            # Determine placement
            if i == 0:  # Top-left
                x1a, y1a, x2a, y2a = 0, 0, cx, cy
                x1b, y1b = w - cx, h - cy
                x2b, y2b = w, h
            elif i == 1:  # Top-right
                x1a, y1a, x2a, y2a = cx, 0, self.img_size, cy
                x1b, y1b = 0, h - cy
                x2b, y2b = self.img_size - cx, h
            elif i == 2:  # Bottom-left
                x1a, y1a, x2a, y2a = 0, cy, cx, self.img_size
                x1b, y1b = w - cx, 0
                x2b, y2b = w, self.img_size - cy
            else:  # Bottom-right
                x1a, y1a, x2a, y2a = cx, cy, self.img_size, self.img_size
                x1b, y1b = 0, 0
                x2b, y2b = self.img_size - cx, self.img_size - cy
            
            # Place image
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust boxes
            if len(boxes) > 0:
                # Convert YOLO to absolute coordinates
                abs_boxes = boxes.copy()
                abs_boxes[:, 0] = boxes[:, 0] * w  # cx
                abs_boxes[:, 1] = boxes[:, 1] * h  # cy
                abs_boxes[:, 2] = boxes[:, 2] * w  # width
                abs_boxes[:, 3] = boxes[:, 3] * h  # height
                
                # Adjust based on placement
                abs_boxes[:, 0] = abs_boxes[:, 0] - x1b + x1a
                abs_boxes[:, 1] = abs_boxes[:, 1] - y1b + y1a
                
                # Convert back to YOLO format
                abs_boxes[:, 0] = abs_boxes[:, 0] / self.img_size
                abs_boxes[:, 1] = abs_boxes[:, 1] / self.img_size
                abs_boxes[:, 2] = abs_boxes[:, 2] / self.img_size
                abs_boxes[:, 3] = abs_boxes[:, 3] / self.img_size
                
                # Clip boxes
                abs_boxes[:, [0, 2]] = np.clip(abs_boxes[:, [0, 2]], 0, 1)
                abs_boxes[:, [1, 3]] = np.clip(abs_boxes[:, [1, 3]], 0, 1)
                
                mosaic_boxes.append(abs_boxes)
                mosaic_labels.append(labels)
        
        # Concatenate all boxes and labels
        if mosaic_boxes:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            mosaic_boxes = np.zeros((0, 4))
            mosaic_labels = np.array([])
        
        return mosaic_img, mosaic_boxes, mosaic_labels


class MixUpAugmentation:
    """
    MixUp augmentation that blends two images.
    
    Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization"
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        boxes1: np.ndarray,
        boxes2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply MixUp augmentation.
        
        Args:
            img1: First image
            img2: Second image
            boxes1: First box array
            boxes2: Second box array
            labels1: First label array
            labels2: Second label array
            
        Returns:
            Tuple of (mixed_image, mixed_boxes, mixed_labels)
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
        
        # Combine boxes and labels
        mixed_boxes = np.concatenate([boxes1, boxes2], axis=0)
        mixed_labels = np.concatenate([labels1, labels2], axis=0)
        
        return mixed_img, mixed_boxes, mixed_labels


class CutMixAugmentation:
    """
    CutMix augmentation that replaces a region with another image.
    
    Reference: Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers"
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix augmentation.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        boxes1: np.ndarray,
        boxes2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply CutMix augmentation.
        
        Args:
            img1: First image
            img2: Second image
            boxes1: First box array
            boxes2: Second box array
            labels1: First label array
            labels2: Second label array
            
        Returns:
            Tuple of (cutmix_image, cutmix_boxes, cutmix_labels)
        """
        h, w = img1.shape[:2]
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Sample random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        cutmix_img = img1.copy()
        cutmix_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
        
        # Combine boxes and labels
        cutmix_boxes = np.concatenate([boxes1, boxes2], axis=0)
        cutmix_labels = np.concatenate([labels1, labels2], axis=0)
        
        return cutmix_img, cutmix_boxes, cutmix_labels


class AdaptiveAugmentation:
    """
    Adaptive augmentation that adjusts based on training difficulty.
    """
    
    def __init__(
        self,
        img_size: int = 640,
        initial_strength: float = 1.0,
        min_strength: float = 0.3
    ):
        """
        Initialize adaptive augmentation.
        
        Args:
            img_size: Target image size
            initial_strength: Initial augmentation strength
            min_strength: Minimum augmentation strength
        """
        self.img_size = img_size
        self.initial_strength = initial_strength
        self.min_strength = min_strength
        self.current_strength = initial_strength
    
    def update_strength(self, epoch: int, max_epochs: int, val_loss: float = None):
        """
        Update augmentation strength based on training progress.
        
        Args:
            epoch: Current epoch
            max_epochs: Maximum epochs
            val_loss: Validation loss (optional)
        """
        # Reduce strength as training progresses
        progress = epoch / max_epochs
        self.current_strength = self.initial_strength * (1.0 - progress * (1.0 - self.min_strength))
    
    def get_transforms(self) -> A.Compose:
        """
        Get augmentation transforms based on current strength.
        
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2 * self.current_strength,
                contrast_limit=0.2 * self.current_strength,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(20 * self.current_strength),
                sat_shift_limit=int(30 * self.current_strength),
                val_shift_limit=int(20 * self.current_strength),
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0 * self.current_strength), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1 * self.current_strength,
                scale_limit=0.2 * self.current_strength,
                rotate_limit=15 * self.current_strength,
                p=0.5
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_training_augmentation(
    img_size: int = 640,
    mode: str = 'strong'
) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        img_size: Target image size
        mode: Augmentation mode ('light', 'medium', 'strong')
        
    Returns:
        Albumentations compose object
    """
    if mode == 'light':
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ]
    elif mode == 'medium':
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
        ]
    else:  # strong
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        ]
    
    transforms.extend([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_validation_augmentation(img_size: int = 640) -> A.Compose:
    """
    Get validation augmentation pipeline.
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
