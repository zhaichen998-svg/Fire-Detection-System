"""
Model export script for fire detection models.

Supports multiple export formats:
- ONNX export
- Quantization (INT8, FP16)
- TensorRT conversion (if available)
- Edge device optimization
"""

import os
import sys
import argparse
import yaml
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_baseline_model, create_improved_model


def export_to_onnx(
    model,
    save_path: str,
    img_size: int = 640,
    batch_size: int = 1,
    dynamic_axes: bool = True
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        img_size: Input image size
        batch_size: Batch size for export
        dynamic_axes: Whether to use dynamic axes
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    # Define dynamic axes
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None
    
    print(f'Exporting to ONNX: {save_path}')
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_dict
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    print(f'ONNX model exported successfully')
    
    # Get model size
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f'ONNX model size: {model_size_mb:.2f} MB')
    
    return save_path


def test_onnx_inference(
    onnx_path: str,
    img_size: int = 640,
    num_tests: int = 10
):
    """
    Test ONNX model inference.
    
    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
        num_tests: Number of inference tests
        
    Returns:
        Average inference time in milliseconds
    """
    print('Testing ONNX inference...')
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference tests
    total_time = 0
    
    for i in range(num_tests):
        # Create random input
        input_data = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        
        # Run inference
        import time
        start_time = time.time()
        outputs = session.run([output_name], {input_name: input_data})
        end_time = time.time()
        
        total_time += (end_time - start_time) * 1000  # Convert to ms
    
    avg_time = total_time / num_tests
    print(f'Average ONNX inference time: {avg_time:.2f} ms')
    
    return avg_time


def quantize_model_int8(
    model,
    save_path: str,
    img_size: int = 640,
    calibration_data: list = None
):
    """
    Quantize model to INT8.
    
    Args:
        model: PyTorch model
        save_path: Path to save quantized model
        img_size: Input image size
        calibration_data: Calibration data for quantization
    """
    print('Quantizing model to INT8...')
    
    # Prepare model for quantization
    model.eval()
    model.cpu()
    
    # Configure quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with sample data
    if calibration_data is None:
        # Use random data for calibration
        calibration_data = [torch.randn(1, 3, img_size, img_size) for _ in range(10)]
    
    with torch.no_grad():
        for data in calibration_data:
            model(data)
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    # Save quantized model
    torch.save(model.state_dict(), save_path)
    
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f'INT8 quantized model size: {model_size_mb:.2f} MB')
    
    return save_path


def quantize_model_fp16(model, save_path: str):
    """
    Quantize model to FP16.
    
    Args:
        model: PyTorch model
        save_path: Path to save quantized model
    """
    print('Quantizing model to FP16...')
    
    model.eval()
    model.half()
    
    # Save FP16 model
    torch.save(model.state_dict(), save_path)
    
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f'FP16 quantized model size: {model_size_mb:.2f} MB')
    
    return save_path


def export_for_edge_device(
    model,
    output_dir: str,
    img_size: int = 640,
    device_type: str = 'jetson'
):
    """
    Export model optimized for edge devices.
    
    Args:
        model: PyTorch model
        output_dir: Output directory
        img_size: Input image size
        device_type: Target device type ('jetson', 'raspberry_pi', 'mobile')
    """
    print(f'Exporting for edge device: {device_type}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if device_type == 'jetson':
        # Export ONNX for TensorRT conversion
        onnx_path = os.path.join(output_dir, 'model_jetson.onnx')
        export_to_onnx(model, onnx_path, img_size, dynamic_axes=False)
        
        # Also export FP16 for faster inference
        fp16_path = os.path.join(output_dir, 'model_jetson_fp16.pt')
        quantize_model_fp16(model.clone(), fp16_path)
        
    elif device_type == 'raspberry_pi':
        # Export INT8 quantized for CPU
        int8_path = os.path.join(output_dir, 'model_rpi_int8.pt')
        quantize_model_int8(model.clone(), int8_path, img_size)
        
        # Also export ONNX
        onnx_path = os.path.join(output_dir, 'model_rpi.onnx')
        export_to_onnx(model, onnx_path, img_size)
        
    elif device_type == 'mobile':
        # Export to TorchScript for mobile
        mobile_path = os.path.join(output_dir, 'model_mobile.pt')
        model.eval()
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, mobile_path)
        
        print(f'Mobile model size: {os.path.getsize(mobile_path) / (1024 * 1024):.2f} MB')
    
    print(f'Edge device export completed to {output_dir}')


def generate_export_info(export_paths: dict, save_path: str):
    """
    Generate export information file.
    
    Args:
        export_paths: Dictionary of export paths
        save_path: Path to save info file
    """
    import json
    
    info = {
        'exports': {},
        'usage_instructions': {}
    }
    
    for export_type, path in export_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            info['exports'][export_type] = {
                'path': path,
                'size_mb': round(size_mb, 2),
                'format': os.path.splitext(path)[1]
            }
    
    # Add usage instructions
    info['usage_instructions'] = {
        'onnx': 'Use with ONNX Runtime: session = ort.InferenceSession("model.onnx")',
        'tensorrt': 'Convert ONNX to TensorRT engine using trtexec',
        'mobile': 'Load in PyTorch Mobile: torch.jit.load("model.pt")',
        'quantized': 'Load as regular PyTorch model with quantization enabled'
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f'Export info saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Export fire detection model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='exported_models', help='Output directory')
    parser.add_argument('--format', type=str, choices=['onnx', 'int8', 'fp16', 'all'], 
                       default='all', help='Export format')
    parser.add_argument('--edge-device', type=str, choices=['jetson', 'raspberry_pi', 'mobile'],
                       help='Export for specific edge device')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    args = parser.parse_args()
    
    # Load configuration
    config = yaml.safe_load(open(args.config))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print('Loading model...')
    device = 'cpu'  # Export on CPU
    
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
    
    print(f'Model loaded from {args.checkpoint}')
    
    # Export paths
    export_paths = {}
    
    # Export based on format
    if args.format == 'onnx' or args.format == 'all':
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        export_to_onnx(model, onnx_path, args.img_size)
        test_onnx_inference(onnx_path, args.img_size)
        export_paths['onnx'] = onnx_path
    
    if args.format == 'int8' or args.format == 'all':
        int8_path = os.path.join(args.output_dir, 'model_int8.pt')
        quantize_model_int8(model, int8_path, args.img_size)
        export_paths['int8'] = int8_path
    
    if args.format == 'fp16' or args.format == 'all':
        fp16_path = os.path.join(args.output_dir, 'model_fp16.pt')
        quantize_model_fp16(model, fp16_path)
        export_paths['fp16'] = fp16_path
    
    # Export for edge device if specified
    if args.edge_device:
        edge_dir = os.path.join(args.output_dir, args.edge_device)
        export_for_edge_device(model, edge_dir, args.img_size, args.edge_device)
        export_paths[f'edge_{args.edge_device}'] = edge_dir
    
    # Generate export info
    info_path = os.path.join(args.output_dir, 'export_info.json')
    generate_export_info(export_paths, info_path)
    
    print('\n' + '=' * 80)
    print('Model export completed!')
    print('=' * 80)
    print(f'Output directory: {args.output_dir}')
    print('Exported formats:')
    for export_type, path in export_paths.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                print(f'  {export_type}: {path}/ (directory)')
            else:
                size = os.path.getsize(path) / (1024 * 1024)
                print(f'  {export_type}: {path} ({size:.2f} MB)')
    print('=' * 80)


if __name__ == '__main__':
    main()
