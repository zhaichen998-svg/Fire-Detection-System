# Fire Detection System Architecture

## Overview

This document describes the architecture of the fire detection system, including model design, data flow, and key components.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Layer                              │
│  Raw Images (640x640 or 416x416 RGB)                        │
└───────────────┬─────────────────────────────────────────────┘
                │
                ├──► Data Augmentation (optional during training)
                │    • Mosaic, MixUp, CutMix
                │    • Color jittering (HSV)
                │    • Geometric transforms
                │
┌───────────────▼─────────────────────────────────────────────┐
│              Preprocessing                                   │
│  • Resize to target size                                    │
│  • Normalize (ImageNet stats)                               │
│  • Convert to tensor                                        │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│            Model Backbone (YOLOv8)                          │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │ Feature Extraction Network                    │          │
│  │  • CSPDarknet backbone                        │          │
│  │  • Multi-scale feature extraction             │          │
│  │  • Residual connections                       │          │
│  └────┬──────────────┬───────────────┬───────────┘          │
│       │              │               │                       │
│    P3 (80x80)    P4 (40x40)     P5 (20x20)                 │
│       │              │               │                       │
│  ┌────▼──────────────▼───────────────▼───────────┐          │
│  │ Feature Pyramid Network (FPN)                  │          │
│  │  • Top-down pathway                            │          │
│  │  • Lateral connections                         │          │
│  │  • Feature fusion                              │          │
│  └────┬──────────────┬───────────────┬───────────┘          │
│       │              │               │                       │
│  ┌────▼──────────────▼───────────────▼───────────┐          │
│  │ Path Aggregation Network (PAN)                 │          │
│  │  • Bottom-up pathway                           │          │
│  │  • Enhanced feature fusion                     │          │
│  └────┬──────────────┬───────────────┬───────────┘          │
└───────┼──────────────┼───────────────┼───────────────────────┘
        │              │               │
┌───────▼──────────────▼───────────────▼───────────────────────┐
│              Detection Heads                                 │
│                                                              │
│  For each scale (P3, P4, P5):                               │
│  ┌──────────────────────────────────────────────┐          │
│  │ Classification Head                           │          │
│  │  • Conv layers                                │          │
│  │  • Predicts class probabilities               │          │
│  │  • Output: [batch, classes, H, W]             │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │ Regression Head                               │          │
│  │  • Conv layers                                │          │
│  │  • Predicts bounding box coordinates          │          │
│  │  • Output: [batch, 4, H, W]                   │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │ Objectness Head                               │          │
│  │  • Conv layers                                │          │
│  │  • Predicts object confidence                 │          │
│  │  • Output: [batch, 1, H, W]                   │          │
│  └──────────────────────────────────────────────┘          │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│           Post-Processing                                    │
│  • Non-Maximum Suppression (NMS)                            │
│  • Confidence filtering                                     │
│  • Coordinate denormalization                               │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│              Output Layer                                    │
│  Detected objects with:                                     │
│  • Bounding boxes [x1, y1, x2, y2]                         │
│  • Class labels (flame/smoke)                               │
│  • Confidence scores                                        │
└─────────────────────────────────────────────────────────────┘
```

## Model Variants

### Baseline Model (YOLOv8n)

- **Backbone**: CSPDarknet with 5 stages
- **Neck**: PAN-FPN for multi-scale feature fusion
- **Head**: Anchor-free detection head
- **Parameters**: ~3.2M
- **Input Size**: 640x640

### Improved Models

#### V1: Data Augmentation Enhanced
- Base architecture: Same as baseline
- Enhancement: Advanced augmentation pipeline
- Key features:
  - Mosaic augmentation
  - MixUp and CutMix
  - Adaptive augmentation strength

#### V2: Custom Loss Functions
- Base architecture: YOLOv8s (larger than baseline)
- Enhancement: Specialized loss functions
- Key features:
  - Focal Loss for class imbalance
  - Weighted IoU for better localization
  - Dynamic loss weighting

#### V3: Architecture Enhancements
- Base architecture: Modified YOLOv8s
- Enhancement: Attention mechanisms
- Key features:
  - CBAM (Convolutional Block Attention Module)
  - SE (Squeeze-and-Excitation) blocks
  - Enhanced feature pyramid

#### V4: Advanced Training
- Base architecture: Same as baseline
- Enhancement: Training strategy
- Key features:
  - Learning rate warmup
  - Cosine annealing with restarts
  - Label smoothing
  - Mixed precision training (FP16)

#### V5: Lightweight Design
- Base architecture: Optimized YOLOv8n
- Enhancement: Model compression
- Key features:
  - Structured pruning (30% sparsity)
  - Knowledge distillation
  - Quantization (INT8/FP16)

## Data Flow

### Training Pipeline

1. **Data Loading**
   - Read image and YOLO format labels
   - Parse bounding boxes and class IDs

2. **Augmentation** (if enabled)
   - Apply augmentation transforms
   - Adjust bounding boxes accordingly

3. **Batching**
   - Collate multiple samples
   - Pad to uniform size if needed

4. **Forward Pass**
   - Extract features through backbone
   - Fuse features in neck
   - Generate predictions from heads

5. **Loss Computation**
   - Classification loss (BCE or Focal)
   - Localization loss (GIoU or WIoU)
   - Objectness loss (BCE)
   - Combine with weights

6. **Backward Pass**
   - Compute gradients
   - Update model parameters
   - Update learning rate

### Inference Pipeline

1. **Preprocessing**
   - Resize image to input size
   - Normalize pixel values
   - Convert to tensor

2. **Forward Pass**
   - Run model inference
   - Get multi-scale predictions

3. **Post-Processing**
   - Filter by confidence threshold
   - Apply NMS
   - Convert to original image coordinates

4. **Output**
   - Return detected objects
   - Boxes, labels, scores

## Loss Functions

### Classification Loss

**Options:**
- Binary Cross-Entropy (BCE)
- Focal Loss: `FL(p_t) = -α_t(1-p_t)^γ log(p_t)`

### Localization Loss

**Options:**
- IoU Loss: `L = 1 - IoU`
- GIoU Loss: `L = 1 - GIoU`
- WIoU Loss: `L = w(IoU) * (1 - IoU)`

### Combined Loss

```
L_total = λ_cls * L_cls + λ_box * L_box + λ_obj * L_obj
```

Where:
- `λ_cls`: Classification loss weight (default: 1.0)
- `λ_box`: Box regression loss weight (default: 5.0)
- `λ_obj`: Objectness loss weight (default: 1.0)

## Attention Mechanisms

### CBAM (Convolutional Block Attention Module)

1. **Channel Attention**
   ```
   F_c = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
   F' = F ⊗ F_c
   ```

2. **Spatial Attention**
   ```
   F_s = σ(Conv([AvgPool(F'); MaxPool(F')]))
   F'' = F' ⊗ F_s
   ```

### SE (Squeeze-and-Excitation)

```
F_se = σ(W_2 * ReLU(W_1 * GAP(F)))
F' = F ⊗ F_se
```

## Performance Considerations

### Training Optimizations

- **Mixed Precision**: Use FP16 for faster training
- **Gradient Accumulation**: Simulate larger batches
- **EMA**: Exponential Moving Average of weights
- **Gradient Clipping**: Prevent gradient explosion

### Inference Optimizations

- **ONNX Export**: Platform-independent format
- **TensorRT**: NVIDIA GPU acceleration
- **Quantization**: INT8 for 4x speedup
- **Pruning**: Remove redundant parameters

## Hardware Requirements

### Training

- **Minimum**: 
  - GPU: NVIDIA GTX 1660 (6GB VRAM)
  - RAM: 16GB
  - Storage: 50GB

- **Recommended**:
  - GPU: NVIDIA RTX 3080 (10GB+ VRAM)
  - RAM: 32GB
  - Storage: 100GB SSD

### Inference

- **Desktop/Server**:
  - GPU: Any CUDA-capable GPU
  - CPU: Multi-core for batching

- **Edge Devices**:
  - NVIDIA Jetson Nano/Xavier
  - Raspberry Pi 4 (with optimization)
  - Coral Edge TPU

## Design Decisions

### Why YOLOv8?

1. **State-of-the-art Performance**: Best accuracy-speed tradeoff
2. **Anchor-free**: Simpler training, better generalization
3. **Modular Design**: Easy to modify and extend
4. **Active Development**: Regular updates and improvements

### Why Multi-scale Detection?

Fire can appear at various scales in images. Multi-scale detection ensures:
- Small fires (distant) detected at high-resolution features (P3)
- Large fires (close) detected at low-resolution features (P5)

### Why Custom Losses?

Standard YOLO loss may not be optimal for fire detection:
- **Focal Loss**: Addresses class imbalance (more background than fire)
- **Weighted IoU**: Emphasizes accurate localization for safety-critical application

## Future Improvements

- [ ] Temporal consistency for video input
- [ ] 3D convolutions for spatiotemporal features
- [ ] Transformer-based neck for long-range dependencies
- [ ] Few-shot learning for rare fire types
- [ ] Uncertainty estimation for reliability
