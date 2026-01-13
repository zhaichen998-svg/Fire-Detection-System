"""
Evaluation metrics for fire detection.

This module implements various metrics:
- Precision, Recall, F1-score
- Average Precision (AP) at different IoU thresholds
- mAP calculation
- Per-class performance analysis
- Confusion matrix
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: Box in format [x1, y1, x2, y2]
        box2: Box in format [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
    mode: str = 'interp'
) -> float:
    """
    Compute Average Precision (AP).
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        mode: 'interp' for 11-point interpolation, 'all' for all points
        
    Returns:
        Average Precision value
    """
    if len(recalls) == 0:
        return 0.0
    
    # Sort by recall
    indices = np.argsort(recalls)
    recalls = recalls[indices]
    precisions = precisions[indices]
    
    if mode == 'interp':
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
    else:
        # All point interpolation
        # Add sentinel values
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # Compute monotonically decreasing
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        # Integrate
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


class DetectionMetrics:
    """
    Compute detection metrics for object detection.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        iou_threshold: float = 0.5
    ):
        """
        Initialize detection metrics.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.iou_threshold = iou_threshold
        
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = defaultdict(list)  # class_id -> list of (confidence, box)
        self.ground_truths = defaultdict(list)  # class_id -> list of boxes
        self.matched = defaultdict(list)  # class_id -> list of matched flags
    
    def update(
        self,
        pred_boxes: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ):
        """
        Update metrics with predictions and ground truths.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in format [x1, y1, x2, y2]
            pred_labels: Predicted labels (N,)
            pred_scores: Prediction scores (N,)
            gt_boxes: Ground truth boxes (M, 4) in format [x1, y1, x2, y2]
            gt_labels: Ground truth labels (M,)
        """
        # Store predictions
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            self.predictions[label].append((score, box))
        
        # Store ground truths
        for box, label in zip(gt_boxes, gt_labels):
            self.ground_truths[label].append(box)
            self.matched[label].append(False)
    
    def compute_precision_recall(self, class_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute precision-recall curve for a class.
        
        Args:
            class_id: Class ID
            
        Returns:
            Tuple of (recalls, precisions, ap)
        """
        # Get predictions and ground truths for this class
        preds = self.predictions[class_id]
        gts = self.ground_truths[class_id]
        
        if len(preds) == 0:
            return np.array([]), np.array([]), 0.0
        
        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x[0], reverse=True)
        
        # Match predictions to ground truths
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched_gt = [False] * len(gts)
        
        for i, (score, pred_box) in enumerate(preds):
            max_iou = 0
            max_idx = -1
            
            for j, gt_box in enumerate(gts):
                if not matched_gt[j]:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j
            
            if max_iou >= self.iou_threshold:
                tp[i] = 1
                matched_gt[max_idx] = True
            else:
                fp[i] = 1
        
        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Compute precision and recall
        recalls = tp_cumsum / len(gts) if len(gts) > 0 else tp_cumsum
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        
        return recalls, precisions, ap
    
    def compute_map(self, iou_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Compute mAP across all classes.
        
        Args:
            iou_thresholds: List of IoU thresholds (default: [0.5])
            
        Returns:
            Dictionary with mAP values
        """
        if iou_thresholds is None:
            iou_thresholds = [self.iou_threshold]
        
        results = {}
        
        for iou_thresh in iou_thresholds:
            self.iou_threshold = iou_thresh
            aps = []
            
            for class_id in range(self.num_classes):
                _, _, ap = self.compute_precision_recall(class_id)
                aps.append(ap)
            
            results[f'mAP@{iou_thresh:.2f}'] = np.mean(aps)
        
        return results
    
    def compute_f1_score(self, class_id: int, threshold: float = 0.5) -> float:
        """
        Compute F1 score for a class.
        
        Args:
            class_id: Class ID
            threshold: Confidence threshold
            
        Returns:
            F1 score
        """
        recalls, precisions, _ = self.compute_precision_recall(class_id)
        
        if len(recalls) == 0:
            return 0.0
        
        # Find precision and recall at threshold
        # For simplicity, use maximum F1
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        return np.max(f1_scores) if len(f1_scores) > 0 else 0.0
    
    def get_confusion_matrix(self, num_classes: int) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            num_classes: Number of classes
            
        Returns:
            Confusion matrix of shape (num_classes, num_classes)
        """
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        # This is a simplified version
        # Full implementation would track all predictions vs ground truths
        
        return confusion_matrix
    
    def plot_precision_recall_curve(self, class_id: int, save_path: Optional[str] = None):
        """
        Plot precision-recall curve for a class.
        
        Args:
            class_id: Class ID
            save_path: Path to save plot
        """
        recalls, precisions, ap = self.compute_precision_recall(class_id)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.class_names[class_id]}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {}
        
        # Per-class AP
        for class_id in range(self.num_classes):
            _, _, ap = self.compute_precision_recall(class_id)
            summary[f'AP_{self.class_names[class_id]}'] = ap
        
        # mAP
        map_results = self.compute_map()
        summary.update(map_results)
        
        # Per-class F1
        for class_id in range(self.num_classes):
            f1 = self.compute_f1_score(class_id)
            summary[f'F1_{self.class_names[class_id]}'] = f1
        
        return summary


def compute_mAP_at_multiple_iou(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float] = [0.5, 0.75, 0.95],
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute mAP at multiple IoU thresholds (COCO-style).
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        iou_thresholds: List of IoU thresholds
        num_classes: Number of classes
        
    Returns:
        Dictionary with mAP values at different IoU thresholds
    """
    results = {}
    
    for iou_thresh in iou_thresholds:
        metrics = DetectionMetrics(num_classes=num_classes, iou_threshold=iou_thresh)
        
        for pred, gt in zip(predictions, ground_truths):
            metrics.update(
                pred['boxes'], pred['labels'], pred['scores'],
                gt['boxes'], gt['labels']
            )
        
        map_result = metrics.compute_map([iou_thresh])
        results.update(map_result)
    
    # Compute mAP@[.5:.95]
    map_values = [results[f'mAP@{t:.2f}'] for t in iou_thresholds]
    results['mAP@[.5:.95]'] = np.mean(map_values)
    
    return results
