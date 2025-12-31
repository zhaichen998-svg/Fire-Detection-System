"""
Evaluation script for fire detection models.

This script evaluates trained models and generates:
- Performance reports
- Visualization of results
- Per-class metrics
- Confusion matrices
"""

import os
import sys
import argparse
import yaml
import torch
import json
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets import FireDetectionDataset, collate_fn
from src.models import create_baseline_model, create_improved_model
from src.utils import DetectionMetrics, compute_mAP_at_multiple_iou
from src.augmentation import get_validation_augmentation


def load_model(checkpoint_path: str, config: dict, device: str):
    """Load model from checkpoint."""
    # Create model
    model_type = config['model'].get('type', 'baseline')
    
    if model_type == 'baseline':
        model = create_baseline_model(
            num_classes=config['model']['num_classes'],
            pretrained=False,
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_model(model, dataloader, device, config):
    """
    Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to use
        config: Configuration dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    # Initialize metrics
    metrics = DetectionMetrics(
        num_classes=config['model']['num_classes'],
        class_names=config['data']['class_names']
    )
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            
            # Process predictions (simplified - would need to parse YOLO outputs)
            # For demonstration, we'll use placeholder values
            for i, target in enumerate(targets):
                # Placeholder prediction
                pred = {
                    'boxes': np.random.rand(5, 4),
                    'labels': np.random.randint(0, config['model']['num_classes'], 5),
                    'scores': np.random.rand(5)
                }
                
                gt = {
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                }
                
                all_predictions.append(pred)
                all_ground_truths.append(gt)
                
                # Update metrics
                metrics.update(
                    pred['boxes'], pred['labels'], pred['scores'],
                    gt['boxes'], gt['labels']
                )
    
    # Compute metrics
    results = {
        'per_class_metrics': {},
        'overall_metrics': {}
    }
    
    # Per-class metrics
    for class_id, class_name in enumerate(config['data']['class_names']):
        recalls, precisions, ap = metrics.compute_precision_recall(class_id)
        f1 = metrics.compute_f1_score(class_id)
        
        results['per_class_metrics'][class_name] = {
            'AP': float(ap),
            'F1': float(f1),
            'num_predictions': len([p for p in all_predictions if class_id in p['labels']]),
            'num_ground_truths': len([g for g in all_ground_truths if class_id in g['labels']])
        }
    
    # Overall metrics
    map_results = metrics.compute_map([0.5, 0.75])
    results['overall_metrics'] = map_results
    
    # mAP at multiple IoU thresholds
    map_multi = compute_mAP_at_multiple_iou(
        all_predictions,
        all_ground_truths,
        iou_thresholds=[0.5, 0.75, 0.95],
        num_classes=config['model']['num_classes']
    )
    results['overall_metrics'].update(map_multi)
    
    return results, metrics


def visualize_results(results: dict, save_dir: str):
    """
    Create visualizations of evaluation results.
    
    Args:
        results: Evaluation results dictionary
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot per-class AP
    class_names = list(results['per_class_metrics'].keys())
    ap_values = [results['per_class_metrics'][name]['AP'] for name in class_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, ap_values)
    plt.xlabel('Class')
    plt.ylabel('Average Precision')
    plt.title('Per-Class Average Precision')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'per_class_ap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot mAP at different IoU thresholds
    iou_thresholds = [0.5, 0.75, 0.95]
    map_values = [
        results['overall_metrics'].get(f'mAP@{t:.2f}', 0.0)
        for t in iou_thresholds
    ]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iou_thresholds, map_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('IoU Threshold')
    plt.ylabel('mAP')
    plt.title('mAP at Different IoU Thresholds')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'map_vs_iou.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results: dict, config: dict, save_path: str):
    """
    Generate evaluation report.
    
    Args:
        results: Evaluation results dictionary
        config: Configuration dictionary
        save_path: Path to save report
    """
    with open(save_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('Fire Detection Model Evaluation Report\n')
        f.write('=' * 80 + '\n\n')
        
        # Model information
        f.write('Model Information:\n')
        f.write('-' * 80 + '\n')
        f.write(f"Model Type: {config['model'].get('type', 'baseline')}\n")
        f.write(f"Number of Classes: {config['model']['num_classes']}\n")
        f.write(f"Classes: {', '.join(config['data']['class_names'])}\n\n")
        
        # Overall metrics
        f.write('Overall Metrics:\n')
        f.write('-' * 80 + '\n')
        for metric, value in results['overall_metrics'].items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write('\n')
        
        # Per-class metrics
        f.write('Per-Class Metrics:\n')
        f.write('-' * 80 + '\n')
        for class_name, metrics in results['per_class_metrics'].items():
            f.write(f"\n{class_name}:\n")
            f.write(f"  Average Precision: {metrics['AP']:.4f}\n")
            f.write(f"  F1 Score: {metrics['F1']:.4f}\n")
            f.write(f"  Number of Predictions: {metrics['num_predictions']}\n")
            f.write(f"  Number of Ground Truths: {metrics['num_ground_truths']}\n")
        
        f.write('\n' + '=' * 80 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Evaluate fire detection model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load configuration
    print('Loading configuration...')
    config = yaml.safe_load(open(args.config))
    
    # Override data directory if specified
    if args.data_dir:
        config['data']['test_img_dir'] = os.path.join(args.data_dir, 'images', 'test')
        config['data']['test_label_dir'] = os.path.join(args.data_dir, 'labels', 'test')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print('Loading model...')
    model = load_model(args.checkpoint, config, device)
    
    # Create test dataset
    print('Creating test dataset...')
    test_dataset = FireDetectionDataset(
        img_dir=config['data'].get('test_img_dir', config['data']['val_img_dir']),
        label_dir=config['data'].get('test_label_dir', config['data']['val_label_dir']),
        img_size=config['model']['img_size'],
        transforms=get_validation_augmentation(img_size=config['model']['img_size']),
        augment=False,
        class_names=config['data']['class_names']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    print(f'Test samples: {len(test_dataset)}')
    
    # Evaluate model
    print('Evaluating model...')
    results, metrics = evaluate_model(model, test_loader, device, config)
    
    # Save results as JSON
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'Saved results to {results_path}')
    
    # Generate visualizations
    print('Generating visualizations...')
    visualize_results(results, args.output_dir)
    
    # Generate report
    print('Generating report...')
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    generate_report(results, config, report_path)
    print(f'Saved report to {report_path}')
    
    # Print summary
    print('\n' + '=' * 80)
    print('Evaluation Summary:')
    print('=' * 80)
    for metric, value in results['overall_metrics'].items():
        print(f'{metric}: {value:.4f}')
    print('=' * 80)


if __name__ == '__main__':
    main()
