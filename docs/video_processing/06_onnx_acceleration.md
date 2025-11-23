# Hardware Acceleration with ONNX

## Overview

ONNX (Open Neural Network Exchange) is a format that enables hardware-accelerated inference across different platforms. This document explains how to use ONNX models for faster pose detection.

## Why ONNX?

### Advantages

1. **Hardware Acceleration**: Supports DirectML (Windows), CoreML (macOS), TensorRT (NVIDIA)
2. **Cross-Platform**: Works on AMD, Intel, and NVIDIA GPUs
3. **Optimized**: Better performance than PyTorch on CPU
4. **Smaller**: More compact model format
5. **No PyTorch Dependency**: Lighter runtime requirements

### Performance Comparison

Typical inference times (640x640 resolution):

| Platform | PyTorch | ONNX | Speedup |
|----------|---------|------|---------|
| CPU (Intel i7) | 150ms | 80ms | 1.9x |
| DirectML (NVIDIA) | 150ms | 25ms | 6.0x |
| DirectML (AMD) | 150ms | 30ms | 5.0x |
| CUDA (NVIDIA) | 15ms | 15ms | 1.0x |

## Converting Models to ONNX

### Method 1: Using Ultralytics Export

```python
from ultralytics import YOLO

# Load PyTorch model
model = YOLO("yolo11m-pose.pt")

# Export to ONNX
model.export(format="onnx", dynamic=False, simplify=True)

# This creates: yolo11m-pose.onnx
```

**Export options:**

```python
model.export(
    format="onnx",           # Export format
    dynamic=False,           # Fixed input size (faster)
    simplify=True,           # Optimize ONNX graph
    half=False,              # Use FP16 (half precision)
    opset=12,                # ONNX opset version
)
```

### Method 2: Using Export Script

Create `export_pose.py`:

```python
from ultralytics import YOLO

# Export pose model
model = YOLO("yolo11m-pose.pt")
model.export(format="onnx")

print("Export complete: yolo11m-pose.onnx")
```

Run the script:

```bash
python export_pose.py
```

### Verification

Verify the exported model:

```python
import onnx

# Load ONNX model
onnx_model = onnx.load("yolo11m-pose.onnx")

# Check model validity
onnx.checker.check_model(onnx_model)
print("Model is valid!")

# Print model info
print(f"Input shape: {onnx_model.graph.input[0].type}")
print(f"Output shape: {onnx_model.graph.output[0].type}")
```

## Using ONNX Models

### Installation

Install ONNX Runtime:

```bash
# CPU only
pip install onnxruntime

# With DirectML (Windows GPU acceleration)
pip install onnxruntime-directml

# With CUDA (NVIDIA GPU acceleration)
pip install onnxruntime-gpu
```

### Basic ONNX Inference

```python
import cv2
import numpy as np
import onnxruntime as ort

# Create inference session
session = ort.InferenceSession(
    "yolo11m-pose.onnx",
    providers=['CPUExecutionProvider']
)

# Get input name
input_name = session.get_inputs()[0].name

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = cv2.resize(frame, (640, 640))
    img_input = img[:, :, ::-1] / 255.0  # BGR→RGB, normalize
    img_input = img_input.transpose(2, 0, 1).astype(np.float32)  # HWC→CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
    
    # Run inference
    outputs = session.run(None, {input_name: img_input})
    
    # Process outputs
    # outputs[0] shape: (1, 56, 8400)
    
    cv2.imshow("ONNX Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## DirectML Acceleration (Windows)

### Setup

DirectML enables GPU acceleration on Windows with AMD, Intel, or NVIDIA GPUs.

```python
import onnxruntime as ort

# Configure providers
providers = [
    ("DmlExecutionProvider", {"device_id": 0}),  # GPU
    "CPUExecutionProvider"                        # Fallback
]

# Create session with DirectML
session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)

# Verify provider
print("Using providers:", session.get_providers())
# Should print: ['DmlExecutionProvider', 'CPUExecutionProvider']
```

### Complete DirectML Example

See `run_pose_onnx_dml.py` in the repository:

```python
import cv2
import numpy as np
import onnxruntime as ort

# DirectML acceleration
providers = [
    ("DmlExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider"
]
session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)
input_name = session.get_inputs()[0].name

print("Using providers:", session.get_providers())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (640, 640))
    img_input = img[:, :, ::-1] / 255.0  # BGR → RGB
    img_input = img_input.transpose(2, 0, 1).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)

    # ONNX inference
    outputs = session.run(None, {input_name: img_input})[0]
    outputs = outputs[0]

    # Parse YOLO11 Pose output
    # Each detection: 56 dimensions
    # 0-3: bbox, 4: score, 5: class, 6-55: keypoints
    scores = outputs[4]
    mask = scores > 0.4

    filtered = outputs[:, mask].T  # (N, 56)

    # Visualize
    for det in filtered:
        x, y, w, h = det[:4].astype(int)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Keypoints
        kpts = det[6:].reshape(25, 2)
        for (kx, ky) in kpts.astype(int):
            cv2.circle(frame, (kx, ky), 3, (0, 0, 255), -1)

    cv2.imshow("ONNX Pose (DirectML)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## CUDA Acceleration (NVIDIA)

### Setup

For NVIDIA GPUs with CUDA:

```bash
pip install onnxruntime-gpu
```

### Usage

```python
import onnxruntime as ort

# Configure CUDA provider
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    }),
    'CPUExecutionProvider'
]

session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)
print("Using providers:", session.get_providers())
```

## Preprocessing Pipeline

### Manual Preprocessing

ONNX requires manual preprocessing:

```python
def preprocess_frame(frame, input_size=(640, 640)):
    """
    Preprocess frame for ONNX model
    
    Args:
        frame: BGR image from OpenCV
        input_size: Target size (width, height)
    
    Returns:
        Preprocessed numpy array ready for inference
    """
    # Resize
    img = cv2.resize(frame, input_size)
    
    # BGR to RGB
    img = img[:, :, ::-1]
    
    # Normalize to [0, 1]
    img = img / 255.0
    
    # HWC to CHW (Height, Width, Channels → Channels, Height, Width)
    img = img.transpose(2, 0, 1)
    
    # Convert to float32
    img = img.astype(np.float32)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
```

### Preprocessing with Letterboxing

For better accuracy, maintain aspect ratio:

```python
def letterbox_image(img, target_size=(640, 640)):
    """
    Resize image with letterboxing (maintain aspect ratio)
    
    Returns:
        Resized image and scaling factors
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new size
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create canvas
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Place resized image on canvas
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return canvas, scale, pad_x, pad_y
```

## Output Parsing

### Understanding ONNX Output

YOLO11-pose ONNX output format:

```
Shape: (1, 56, 8400)
- 1: Batch size
- 56: Features per detection
- 8400: Number of detection boxes (80×80 + 40×40 + 20×20)
```

Feature dimensions (56):
- `[0:4]`: Bounding box (x, y, w, h)
- `[4]`: Objectness score
- `[5]`: Class ID
- `[6:56]`: Keypoints (25 keypoints × 2 coordinates)

### Parsing Function

```python
def parse_pose_output(output, conf_threshold=0.4):
    """
    Parse YOLO11-pose ONNX output
    
    Args:
        output: Raw model output (1, 56, 8400)
        conf_threshold: Confidence threshold
    
    Returns:
        List of detections with bboxes and keypoints
    """
    output = output[0]  # Remove batch dimension: (56, 8400)
    
    # Extract scores
    scores = output[4, :]
    
    # Filter by confidence
    mask = scores > conf_threshold
    filtered = output[:, mask].T  # (N, 56)
    
    detections = []
    for det in filtered:
        bbox = det[:4]  # x, y, w, h
        score = det[4]
        class_id = int(det[5])
        
        # Extract keypoints (25 keypoints, but COCO format uses first 17)
        kpts_raw = det[6:]
        kpts = kpts_raw.reshape(-1, 2)  # Reshape to (25, 2)
        kpts_coco = kpts[:17]  # Take first 17 for COCO format
        
        detections.append({
            'bbox': bbox,
            'score': score,
            'class_id': class_id,
            'keypoints': kpts_coco
        })
    
    return detections
```

## Integration with Action Detection

### Complete ONNX Integration

```python
import cv2
import numpy as np
import onnxruntime as ort

# ... include distance() and detect_custom_actions() ...

# Initialize ONNX session
providers = [
    ("DmlExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider"
]
session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)
action_logs = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    img_input = preprocess_frame(frame)
    
    # Inference
    outputs = session.run(None, {input_name: img_input})
    
    # Parse
    detections = parse_pose_output(outputs[0])
    
    # Action detection
    frame_actions = []
    for det in detections:
        kp = det['keypoints']
        actions = detect_custom_actions(kp)
        frame_actions.extend(actions)
    
    # Visualize
    y = 30
    for act in set(frame_actions):
        cv2.putText(frame, f"ACTION: {act}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30
    
    cv2.imshow("ONNX Interview System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Optimization Tips

### 1. Use Half Precision (FP16)

For compatible hardware:

```python
# Export with half precision
model.export(format="onnx", half=True)

# Inference with FP16
session = ort.InferenceSession(
    "yolo11m-pose.onnx",
    providers=[("DmlExecutionProvider", {"device_id": 0})]
)
```

### 2. Dynamic vs Static Shapes

```python
# Static shape (faster, recommended)
model.export(format="onnx", dynamic=False)

# Dynamic shape (flexible, slower)
model.export(format="onnx", dynamic=True)
```

### 3. Batch Inference

Process multiple frames at once:

```python
# Collect frames
frames = []
for _ in range(4):
    ret, frame = cap.read()
    if ret:
        frames.append(preprocess_frame(frame))

# Batch inference
batch_input = np.concatenate(frames, axis=0)  # (4, 3, 640, 640)
outputs = session.run(None, {input_name: batch_input})
```

### 4. Async Inference

For overlapping I/O and compute:

```python
# Not directly supported in ONNX Runtime Python API
# Use threading for parallelism instead
```

## Performance Profiling

### Measure Inference Time

```python
import time

# Warmup
for _ in range(10):
    outputs = session.run(None, {input_name: img_input})

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    outputs = session.run(None, {input_name: img_input})
    times.append(time.time() - start)

avg_time = np.mean(times) * 1000
print(f"Average inference time: {avg_time:.2f} ms")
print(f"FPS: {1000/avg_time:.1f}")
```

### Memory Usage

```python
import psutil
import os

process = psutil.Process(os.getpid())

# Before inference
mem_before = process.memory_info().rss / 1024 / 1024  # MB

# Run inference
outputs = session.run(None, {input_name: img_input})

# After inference
mem_after = process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory used: {mem_after - mem_before:.2f} MB")
```

## Troubleshooting

### Common Issues

#### 1. DirectML Not Available

```python
# Check available providers
print(ort.get_available_providers())

# If 'DmlExecutionProvider' not listed:
# - Install: pip install onnxruntime-directml
# - Update GPU drivers
# - Check Windows version (requires Windows 10+)
```

#### 2. Model Compatibility

```python
# Check ONNX opset
import onnx
model = onnx.load("yolo11m-pose.onnx")
print(f"ONNX opset: {model.opset_import[0].version}")

# Re-export with compatible opset
model.export(format="onnx", opset=12)
```

#### 3. Wrong Output Shape

```python
# Print actual output shape
outputs = session.run(None, {input_name: img_input})
print(f"Output shape: {outputs[0].shape}")

# Adjust parsing logic accordingly
```

#### 4. Performance Issues

```python
# Enable session optimizations
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "yolo11m-pose.onnx",
    sess_options=so,
    providers=providers
)
```

## Provider Selection

### Auto-Select Best Provider

```python
def get_optimal_providers():
    """Automatically select best available providers"""
    available = ort.get_available_providers()
    
    # Priority order
    priority = [
        'CUDAExecutionProvider',      # NVIDIA GPU (fastest)
        'DmlExecutionProvider',        # Windows GPU
        'CoreMLExecutionProvider',     # Apple Silicon
        'CPUExecutionProvider'         # Fallback
    ]
    
    selected = []
    for provider in priority:
        if provider in available:
            selected.append(provider)
            break
    
    # Always add CPU as fallback
    if 'CPUExecutionProvider' not in selected:
        selected.append('CPUExecutionProvider')
    
    return selected

# Use optimal providers
providers = get_optimal_providers()
session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)
print(f"Using: {session.get_providers()}")
```

## Best Practices

1. **Export models with simplify=True**: Optimizes ONNX graph
2. **Use static shapes**: Faster than dynamic shapes
3. **Choose appropriate provider**: DirectML for Windows, CUDA for NVIDIA
4. **Profile before deploying**: Test on target hardware
5. **Handle preprocessing efficiently**: Minimize Python overhead
6. **Batch when possible**: Better GPU utilization
7. **Warm up the model**: First inference is always slower

## Further Reading

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [DirectML Documentation](https://learn.microsoft.com/en-us/windows/ai/directml/)
- [Ultralytics Export Guide](https://docs.ultralytics.com/modes/export/)
- [ONNX Model Zoo](https://github.com/onnx/models)
