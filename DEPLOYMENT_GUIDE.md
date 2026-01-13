# Edge Deployment Guide

Complete guide for deploying fire detection models on edge devices.

## Supported Platforms

- NVIDIA Jetson (Nano, Xavier NX, AGX Xavier)
- Raspberry Pi 4
- Coral Edge TPU
- Mobile devices (Android/iOS)
- Generic x86 edge servers

## Quick Start

### 1. Export Model

```bash
# Export for Jetson
python scripts/export_model.py \
    --checkpoint runs/best_model.pt \
    --config configs/edge_deployment.yaml \
    --edge-device jetson \
    --output-dir exported_models/jetson

# Export for Raspberry Pi
python scripts/export_model.py \
    --checkpoint runs/best_model.pt \
    --config configs/edge_deployment.yaml \
    --edge-device raspberry_pi \
    --output-dir exported_models/rpi
```

### 2. Deploy on Device

Transfer exported model to edge device and run inference (see device-specific sections below).

## NVIDIA Jetson Deployment

### Prerequisites

```bash
# Install JetPack (includes CUDA, cuDNN, TensorRT)
# Version: JetPack 4.6 or higher

# Install Python dependencies
pip3 install torch torchvision
pip3 install onnx onnxruntime-gpu
pip3 install opencv-python
```

### Model Optimization

**Option 1: FP16 (Recommended)**
```python
import torch

model = torch.load('model.pt')
model.half()  # Convert to FP16
model.eval()
```

**Option 2: TensorRT**
```bash
# Convert ONNX to TensorRT engine
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --workspace=4096
```

### Inference Code

```python
import cv2
import torch
import numpy as np

# Load model
model = torch.load('model_fp16.pt')
model.cuda()
model.eval()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    img = cv2.resize(frame, (416, 416))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img.half().cuda()
    
    # Inference
    with torch.no_grad():
        outputs = model(img)
    
    # Post-process and display
    # ... (see full example in notebooks)
    
    cv2.imshow('Fire Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Performance

| Device | Model | FPS | Power | Temp |
|--------|-------|-----|-------|------|
| Nano | V5 INT8 | 25 | 5W | 45°C |
| Nano | V5 FP16 | 20 | 7W | 50°C |
| Xavier NX | Baseline FP16 | 45 | 15W | 55°C |
| Xavier NX | V3 FP16 | 30 | 18W | 60°C |

## Raspberry Pi 4 Deployment

### Prerequisites

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install dependencies
pip3 install opencv-python-headless
pip3 install onnxruntime
pip3 install numpy
```

### Model Optimization

**INT8 Quantization (Required)**
```python
# Model is already quantized during export
# Use INT8 version for best performance
```

### Inference Code

```python
import onnxruntime as ort
import cv2
import numpy as np

# Load ONNX model
session = ort.InferenceSession(
    'model_int8.onnx',
    providers=['CPUExecutionProvider']
)

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    img = cv2.resize(frame, (416, 416))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # Inference
    outputs = session.run(
        [output_name],
        {input_name: img}
    )
    
    # Post-process and display
    # ... (see full example)
    
    cv2.imshow('Fire Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Performance

| Model | FPS | CPU Usage | Temp |
|-------|-----|-----------|------|
| V5 INT8 | 8-12 | 85% | 65°C |
| Baseline INT8 | 5-8 | 95% | 70°C |

### Tips for RPi

1. Use INT8 quantization
2. Reduce input size to 416x416 or 320x320
3. Use lightweight model (V5)
4. Add cooling fan
5. Overclock if needed (careful!)

## Mobile Deployment (Android)

### Prerequisites

- Android Studio
- PyTorch Mobile
- Converted TorchScript model

### Model Conversion

```python
import torch

# Load model
model = torch.load('model.pt')
model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(model)

# Optimize for mobile
optimized_model = optimize_for_mobile(scripted_model)

# Save
optimized_model._save_for_lite_interpreter('model_mobile.ptl')
```

### Android Integration

```java
// MainActivity.java
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class MainActivity extends AppCompatActivity {
    private Module module;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Load model
        module = Module.load(assetFilePath("model_mobile.ptl"));
    }
    
    private Tensor preprocessImage(Bitmap bitmap) {
        // Resize and normalize
        bitmap = Bitmap.createScaledBitmap(
            bitmap, 416, 416, false
        );
        
        return TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            new float[]{0.485f, 0.456f, 0.406f},
            new float[]{0.229f, 0.224f, 0.225f}
        );
    }
    
    private void runInference(Bitmap bitmap) {
        Tensor inputTensor = preprocessImage(bitmap);
        
        // Run inference
        Tensor outputTensor = module.forward(
            IValue.from(inputTensor)
        ).toTensor();
        
        // Process output
        float[] scores = outputTensor.getDataAsFloatArray();
        // ... parse detections
    }
}
```

### Performance

| Device | Model | FPS |
|--------|-------|-----|
| Pixel 5 | V5 | 15-20 |
| Galaxy S21 | V5 | 20-25 |
| iPhone 12 | V5 | 25-30 |

## Optimization Techniques

### 1. Input Size Reduction

```yaml
# configs/edge_deployment.yaml
model:
  img_size: 416  # Instead of 640
```

**Impact:**
- FPS: +40-60%
- Accuracy: -2-3%

### 2. Batch Processing

Process multiple frames in batch:

```python
# Accumulate frames
frames_batch = []
for _ in range(batch_size):
    ret, frame = cap.read()
    frames_batch.append(preprocess(frame))

# Batch inference
batch_tensor = torch.stack(frames_batch)
outputs = model(batch_tensor)
```

### 3. Frame Skipping

Process every Nth frame:

```python
frame_count = 0
skip_frames = 2

while True:
    ret, frame = cap.read()
    
    if frame_count % skip_frames == 0:
        # Run detection
        outputs = model(frame)
    
    # Always display
    cv2.imshow('Detection', frame)
    frame_count += 1
```

### 4. Region of Interest (ROI)

Process only relevant regions:

```python
# Define ROI (e.g., center of frame)
h, w = frame.shape[:2]
roi = frame[h//4:3*h//4, w//4:3*w//4]

# Process ROI only
outputs = model(roi)
```

## Monitoring and Diagnostics

### Temperature Monitoring

```python
# For Jetson
def get_jetson_temp():
    with open('/sys/devices/virtual/thermal/thermal_zone1/temp') as f:
        temp = float(f.read()) / 1000.0
    return temp

# For Raspberry Pi
def get_rpi_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        temp = float(f.read()) / 1000.0
    return temp
```

### Performance Logging

```python
import time
import json

metrics = {
    'fps': [],
    'inference_time': [],
    'temperature': [],
    'timestamp': []
}

while True:
    start = time.time()
    
    # Inference
    outputs = model(frame)
    
    inference_time = (time.time() - start) * 1000
    
    metrics['inference_time'].append(inference_time)
    metrics['fps'].append(1.0 / (time.time() - start))
    metrics['temperature'].append(get_temp())
    metrics['timestamp'].append(time.time())
    
    # Save periodically
    if len(metrics['fps']) % 100 == 0:
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)
```

## Troubleshooting

### Low FPS

**Solutions:**
1. Reduce input size
2. Use lighter model (V5)
3. Apply quantization (INT8)
4. Enable frame skipping
5. Use hardware acceleration

### High Temperature

**Solutions:**
1. Add cooling fan
2. Reduce input size
3. Lower power mode
4. Frame skipping
5. Thermal paste reapplication

### Out of Memory

**Solutions:**
1. Reduce batch size
2. Use smaller model
3. Lower input resolution
4. Enable model quantization
5. Clear cache periodically

## Best Practices

1. **Always test on target device** - Simulations don't match reality
2. **Monitor temperature** - Prevents thermal throttling
3. **Use appropriate model** - Don't over-engineer
4. **Implement fallback** - Handle edge cases
5. **Log metrics** - Track performance over time
6. **Update regularly** - Keep dependencies current

## Example Configurations

### Low Power (Battery)
```yaml
model: V5
input_size: 320
quantization: INT8
frame_skip: 3
fps_target: 10
```

### Balanced
```yaml
model: V5
input_size: 416
quantization: INT8
frame_skip: 1
fps_target: 20
```

### High Performance
```yaml
model: Baseline
input_size: 640
quantization: FP16
frame_skip: 0
fps_target: 30
```

## Support

For deployment issues:
1. Check device compatibility
2. Verify model export
3. Test with sample images
4. Review logs
5. Open GitHub issue

## Resources

- [PyTorch Mobile Documentation](https://pytorch.org/mobile)
- [NVIDIA Jetson Developer Guide](https://developer.nvidia.com/embedded/jetson)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
