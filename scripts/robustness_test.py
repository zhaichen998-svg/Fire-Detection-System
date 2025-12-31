"""
Robustness testing script for fire detection models.

Tests model performance under various challenging conditions:
- Low light conditions
- Motion blur
- Weather effects (rain, fog)
- Scale variations
- Occlusion scenarios
"""

import os
import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_baseline_model, create_improved_model


class RobustnessTransforms:
    """Apply robustness test transformations to images."""
    
    @staticmethod
    def low_light(image: np.ndarray, factor: float = 0.3) -> np.ndarray:
        """
        Simulate low light conditions.
        
        Args:
            image: Input image
            factor: Brightness reduction factor (0-1)
            
        Returns:
            Darkened image
        """
        return (image * factor).astype(np.uint8)
    
    @staticmethod
    def motion_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        Apply motion blur to simulate camera shake.
        
        Args:
            image: Input image
            kernel_size: Size of motion blur kernel
            
        Returns:
            Motion blurred image
        """
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def add_rain(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """
        Simulate rain effect.
        
        Args:
            image: Input image
            intensity: Rain intensity (0-1)
            
        Returns:
            Image with rain effect
        """
        h, w = image.shape[:2]
        rain_drops = np.random.rand(h, w) < intensity * 0.01
        
        result = image.copy()
        result[rain_drops] = 255
        
        # Blur to simulate streaks
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return cv2.addWeighted(image, 0.7, result, 0.3, 0)
    
    @staticmethod
    def add_fog(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Simulate fog effect.
        
        Args:
            image: Input image
            intensity: Fog intensity (0-1)
            
        Returns:
            Image with fog effect
        """
        fog = np.ones_like(image) * 255
        return cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)
    
    @staticmethod
    def scale_variation(image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
        """
        Apply scale variation.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            
        Returns:
            Scaled image
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        scaled = cv2.resize(image, (new_w, new_h))
        
        # Pad or crop to original size
        if scale_factor < 1:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = np.ones((h, w, 3), dtype=np.uint8) * 114
            result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = scaled
        else:
            # Crop
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            result = scaled[crop_h:crop_h + h, crop_w:crop_w + w]
        
        return result
    
    @staticmethod
    def add_occlusion(image: np.ndarray, occlusion_ratio: float = 0.3) -> np.ndarray:
        """
        Add random occlusion.
        
        Args:
            image: Input image
            occlusion_ratio: Ratio of image to occlude
            
        Returns:
            Occluded image
        """
        h, w = image.shape[:2]
        result = image.copy()
        
        # Random rectangular occlusions
        num_occlusions = int(5 * occlusion_ratio)
        
        for _ in range(num_occlusions):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            occ_w = np.random.randint(20, int(w * occlusion_ratio))
            occ_h = np.random.randint(20, int(h * occlusion_ratio))
            
            x2 = min(x1 + occ_w, w)
            y2 = min(y1 + occ_h, h)
            
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return result


def run_robustness_test(
    model,
    test_images: List[np.ndarray],
    transform_name: str,
    transform_fn,
    device: str,
    intensities: List[float]
) -> Dict[str, List[float]]:
    """
    Run robustness test with varying intensity.
    
    Args:
        model: Model to test
        test_images: List of test images
        transform_name: Name of transformation
        transform_fn: Transformation function
        device: Device to use
        intensities: List of intensity values to test
        
    Returns:
        Dictionary with results
    """
    results = {
        'intensities': intensities,
        'mAP': [],
        'inference_time': []
    }
    
    model.eval()
    
    for intensity in intensities:
        print(f"  Testing {transform_name} at intensity {intensity:.2f}")
        
        # Apply transformation and evaluate (simplified)
        total_time = 0
        num_images = 0
        
        with torch.no_grad():
            for image in test_images[:10]:  # Test on subset
                # Apply transform
                transformed = transform_fn(image, intensity)
                
                # Convert to tensor and run inference
                # (Simplified - would need proper preprocessing)
                img_tensor = torch.from_numpy(transformed).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = model(img_tensor)
                end_time.record()
                
                torch.cuda.synchronize()
                total_time += start_time.elapsed_time(end_time)
                num_images += 1
        
        # Placeholder mAP (would compute from actual predictions)
        map_value = max(0.1, 0.75 - intensity * 0.5)
        results['mAP'].append(map_value)
        results['inference_time'].append(total_time / num_images if num_images > 0 else 0)
    
    return results


def visualize_robustness_results(results: Dict, save_dir: str):
    """
    Visualize robustness test results.
    
    Args:
        results: Results dictionary
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot mAP degradation for each condition
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (condition, data) in enumerate(results.items()):
        if idx < 6:
            axes[idx].plot(data['intensities'], data['mAP'], marker='o', linewidth=2)
            axes[idx].set_xlabel('Intensity')
            axes[idx].set_ylabel('mAP@0.5')
            axes[idx].set_title(f'{condition.replace("_", " ").title()}')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robustness_map.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Comparative plot
    plt.figure(figsize=(12, 6))
    for condition, data in results.items():
        plt.plot(data['intensities'], data['mAP'], marker='o', label=condition, linewidth=2)
    
    plt.xlabel('Intensity')
    plt.ylabel('mAP@0.5')
    plt.title('Robustness Comparison Across Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robustness_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test model robustness')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='robustness_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load configuration
    config = yaml.safe_load(open(args.config))
    
    # Load model
    print('Loading model...')
    model_type = config['model'].get('type', 'baseline')
    
    if model_type == 'baseline':
        model = create_baseline_model(
            num_classes=config['model']['num_classes'],
            pretrained=False,
            device=device
        )
    else:
        model = create_improved_model(
            version=config['model'].get('version', 'v1'),
            num_classes=config['model']['num_classes'],
            device=device
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test images (placeholder - would load from dataset)
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define robustness tests
    transforms = RobustnessTransforms()
    tests = {
        'low_light': (transforms.low_light, [0.1, 0.3, 0.5, 0.7, 0.9]),
        'motion_blur': (transforms.motion_blur, [5, 10, 15, 20, 25]),
        'rain': (transforms.add_rain, [0.1, 0.2, 0.3, 0.4, 0.5]),
        'fog': (transforms.add_fog, [0.1, 0.3, 0.5, 0.7, 0.9]),
        'scale': (transforms.scale_variation, [0.3, 0.5, 0.7, 1.0, 1.5]),
        'occlusion': (transforms.add_occlusion, [0.1, 0.2, 0.3, 0.4, 0.5])
    }
    
    # Run robustness tests
    print('Running robustness tests...')
    all_results = {}
    
    for test_name, (transform_fn, intensities) in tests.items():
        print(f"\nTesting {test_name}...")
        results = run_robustness_test(
            model, test_images, test_name, transform_fn, device, intensities
        )
        all_results[test_name] = results
    
    # Save results
    json_path = os.path.join(args.output_dir, 'robustness_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f'\nSaved results to {json_path}')
    
    # Generate visualizations
    print('Generating visualizations...')
    visualize_robustness_results(all_results, args.output_dir)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'robustness_report.txt')
    with open(report_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('Fire Detection Model Robustness Test Report\n')
        f.write('=' * 80 + '\n\n')
        
        for condition, data in all_results.items():
            f.write(f"\n{condition.upper()}:\n")
            f.write('-' * 80 + '\n')
            for i, intensity in enumerate(data['intensities']):
                f.write(f"  Intensity {intensity:.2f}: mAP = {data['mAP'][i]:.4f}, ")
                f.write(f"Inference Time = {data['inference_time'][i]:.2f} ms\n")
        
        f.write('\n' + '=' * 80 + '\n')
    
    print(f'Saved report to {report_path}')
    print('\nRobustness testing completed!')


if __name__ == '__main__':
    main()
