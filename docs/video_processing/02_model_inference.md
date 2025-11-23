# Model Inference Pipeline

## Overview

The Main Thread uses YOLO (You Only Look Once) pose estimation models to detect human poses in real-time. This document covers the model inference pipeline, optimization strategies, and hardware acceleration options.

## Pose Estimation Models

### Supported Models

The system supports multiple YOLO pose models:

| Model | File | Parameters | Speed | Accuracy |
|-------|------|------------|-------|----------|
| YOLOv8n-pose | yolov8n-pose.pt | 3.3M | Fast | Good |
| YOLO11m-pose | yolo11m-pose.pt | 20.1M | Medium | Excellent |
| YOLO11m-pose | yolo11m-pose.onnx | 20.1M | Fast (with hardware) | Excellent |

### Model Selection

```python
# PyTorch model (recommended for development)
model = YOLO("yolo11m-pose.pt")

# ONNX model (recommended for production)
session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)
```

**Choosing a model:**
- **yolov8n-pose**: Use for resource-constrained systems
- **yolo11m-pose.pt**: Use for best accuracy with GPU
- **yolo11m-pose.onnx**: Use for CPU/DirectML acceleration

## Inference Pipeline

### 1. Frame Preprocessing (PyTorch)

```python
# No explicit preprocessing needed with Ultralytics
results = model(frame, device="cpu", verbose=False)
```

The Ultralytics library handles preprocessing internally:
- Resizes frame to model input size (typically 640x640)
- Normalizes pixel values to [0, 1]
- Converts BGR to RGB
- Adds batch dimension

### 2. Frame Preprocessing (ONNX)

```python
# Manual preprocessing for ONNX
img = cv2.resize(frame, (640, 640))
img_input = img[:, :, ::-1] / 255.0  # BGR → RGB, normalize
img_input = img_input.transpose(2, 0, 1).astype(np.float32)  # HWC → CHW
img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
```

### 3. Model Inference

#### PyTorch Inference

```python
results = model(frame, device="cpu", verbose=False)

# Parameters:
# - frame: Input image (BGR format)
# - device: "cpu" or "cuda:0" for GPU
# - verbose: False to suppress output
```

#### ONNX Inference

```python
outputs = session.run(None, {input_name: img_input})[0]
# Shape: (1, 56, 8400)
# 1: batch size
# 56: feature dimensions per detection
# 8400: number of detection boxes
```

### 4. Output Parsing

#### PyTorch Output

```python
for r in results:
    if r.keypoints is None:
        continue
    
    for person in r.keypoints.xy:
        kp = person.cpu().numpy()  # Shape: (17, 2)
        # kp[i] = [x, y] for keypoint i
```

#### ONNX Output

```python
outputs = outputs[0]  # Remove batch dimension: (56, 8400)

# Each detection has 56 dimensions:
# [0-3]: bbox coordinates (x, y, w, h)
# [4]: confidence score
# [5]: class ID
# [6-55]: 25 keypoints × 2 (x, y) values

scores = outputs[4]
mask = scores > 0.4  # Filter low-confidence detections

filtered = outputs[:, mask].T  # (N, 56)

for det in filtered:
    bbox = det[:4]
    score = det[4]
    kpts = det[6:].reshape(25, 2)  # 25 keypoints
```

## COCO Keypoint Format

YOLO pose models output 17 keypoints in COCO format:

```python
KEYPOINT_NAMES = [
    0:  "nose",
    1:  "left_eye",
    2:  "right_eye",
    3:  "left_ear",
    4:  "right_ear",
    5:  "left_shoulder",
    6:  "right_shoulder",
    7:  "left_elbow",
    8:  "right_elbow",
    9:  "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
]
```

### Keypoint Structure

Each keypoint consists of:
- **x**: Horizontal position in pixels
- **y**: Vertical position in pixels
- **confidence**: Detection confidence (ONNX only)

```python
# Accessing keypoints
nose = kp[0]           # [x, y]
left_wrist = kp[9]     # [x, y]
right_wrist = kp[10]   # [x, y]
```

## Device Selection

### CPU Inference

```python
# PyTorch
results = model(frame, device="cpu", verbose=False)
```

**Pros:**
- Works on any system
- No special drivers needed
- Consistent performance

**Cons:**
- Slower inference (~100-200ms per frame)
- Limited by CPU speed

### GPU Inference

```python
# PyTorch with CUDA
results = model(frame, device="cuda:0", verbose=False)
```

**Pros:**
- Fast inference (~10-30ms per frame)
- Can handle higher resolution

**Cons:**
- Requires NVIDIA GPU
- Needs CUDA installation
- Higher power consumption

### DirectML Inference (ONNX)

```python
# ONNX with DirectML (Windows GPU acceleration)
providers = [
    ("DmlExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider"
]
session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)
```

**Pros:**
- Works with AMD, Intel, and NVIDIA GPUs
- Windows-native acceleration
- Good performance on laptops

**Cons:**
- Windows-only
- Requires ONNX model conversion

## Performance Optimization

### 1. Model Quantization

Convert models to smaller data types:

```python
# Export to ONNX with half precision
model.export(format="onnx", half=True)
```

### 2. Batch Processing

Process multiple frames together:

```python
# Not recommended for real-time, but useful for offline analysis
frames = [frame1, frame2, frame3]
results = model(frames, device="cpu")
```

### 3. Resolution Reduction

Reduce input resolution for faster inference:

```python
# Resize before inference
small_frame = cv2.resize(frame, (480, 480))
results = model(small_frame, device="cpu")
```

### 4. Frame Skipping

Skip frames when pose changes are minimal:

```python
frame_count = 0
PROCESS_EVERY_N_FRAMES = 3

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        results = model(frame, device="cpu")
```

## Inference Timing

Typical inference times on different hardware:

| Hardware | Model | Resolution | Time |
|----------|-------|------------|------|
| CPU (i7-10750H) | yolo11m-pose.pt | 640x640 | ~150ms |
| CPU (i7-10750H) | yolov8n-pose.pt | 640x640 | ~50ms |
| GPU (RTX 3060) | yolo11m-pose.pt | 640x640 | ~15ms |
| DirectML (RTX 3060) | yolo11m-pose.onnx | 640x640 | ~25ms |

## Error Handling

### Model Loading Errors

```python
try:
    model = YOLO("yolo11m-pose.pt")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Make sure yolo11m-pose.pt is in the current directory")
    exit(1)
```

### Inference Errors

```python
try:
    results = model(frame, device="cpu", verbose=False)
except Exception as e:
    print(f"Inference failed: {e}")
    continue  # Skip this frame
```

### No Detection Handling

```python
for r in results:
    if r.keypoints is None:
        # No person detected in frame
        continue
    
    if len(r.keypoints.xy) == 0:
        # Keypoints tensor is empty
        continue
```

## Model Confidence and Filtering

### Confidence Thresholds

```python
# Set confidence threshold for detections
results = model(frame, device="cpu", conf=0.5, verbose=False)
# Only detections with confidence > 0.5 will be returned
```

### Multiple Person Handling

```python
for r in results:
    for person_idx, person in enumerate(r.keypoints.xy):
        kp = person.cpu().numpy()
        print(f"Person {person_idx}: {kp.shape}")
        # Process each detected person separately
```

## Exporting Models

### Export to ONNX

```python
from ultralytics import YOLO

# Load PyTorch model
model = YOLO("yolo11m-pose.pt")

# Export to ONNX
model.export(format="onnx", dynamic=False, simplify=True)
# Creates: yolo11m-pose.onnx
```

### Export Script

See `export_pose.py`:

```python
from ultralytics import YOLO

model = YOLO("yolo11m-pose.pt")
model.export(format="onnx")
```

## Best Practices

1. **Load models once**: Initialize models before the main loop
2. **Use appropriate device**: GPU for speed, CPU for compatibility
3. **Handle no-detection cases**: Always check if keypoints exist
4. **Set verbose=False**: Avoid console spam during inference
5. **Monitor performance**: Log inference times to identify bottlenecks
6. **Use ONNX for production**: Better cross-platform support
7. **Validate keypoints**: Check for invalid coordinates (e.g., [0, 0])

## Debugging Inference Issues

### Print Model Information

```python
model = YOLO("yolo11m-pose.pt")
print(f"Model task: {model.task}")
print(f"Model names: {model.names}")
print(f"Input shape: {model.model.args}")
```

### Visualize Raw Output

```python
# Draw keypoints on frame
for r in results:
    if r.keypoints is None:
        continue
    for person in r.keypoints.xy:
        kp = person.cpu().numpy()
        for i, (x, y) in enumerate(kp):
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (int(x), int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
```

## Further Reading

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)
