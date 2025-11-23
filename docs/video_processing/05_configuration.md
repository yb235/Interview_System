# Configuration and Customization

## Overview

This document explains how to configure and customize the Main Thread (Video Processing) module for different use cases, environments, and requirements.

## Configuration Options

### 1. Model Selection

#### Choosing a Pose Model

```python
# In interview_system.py (line 14)
model = YOLO("yolo11m-pose.pt")  # Default: medium accuracy, medium speed
```

**Available options:**

```python
# Fast, lower accuracy (recommended for low-end hardware)
model = YOLO("yolov8n-pose.pt")

# Medium accuracy and speed (default, balanced)
model = YOLO("yolo11m-pose.pt")

# Higher accuracy, slower (if you have a GPU)
model = YOLO("yolo11l-pose.pt")  # Large model
model = YOLO("yolo11x-pose.pt")  # Extra large model
```

**Factors to consider:**
- **CPU-only systems**: Use `yolov8n-pose.pt`
- **GPU available**: Use `yolo11m-pose.pt` or larger
- **Real-time requirements**: Prioritize speed over accuracy
- **Offline analysis**: Can use larger, more accurate models

### 2. Device Selection

#### CPU vs GPU

```python
# CPU inference (default, works everywhere)
results = model(frame, device="cpu", verbose=False)

# GPU inference (requires CUDA-capable GPU)
results = model(frame, device="cuda:0", verbose=False)

# Automatic device selection
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
results = model(frame, device=device, verbose=False)
```

#### Multiple GPU Systems

```python
# Use specific GPU
results = model(frame, device="cuda:0", verbose=False)  # First GPU
results = model(frame, device="cuda:1", verbose=False)  # Second GPU
```

### 3. Camera Configuration

#### Camera Selection

```python
# Default camera (usually built-in webcam)
cap = cv2.VideoCapture(0)

# External USB camera
cap = cv2.VideoCapture(1)

# Specific camera by index
cap = cv2.VideoCapture(2)

# IP camera
cap = cv2.VideoCapture("rtsp://camera_ip:port/stream")

# Video file (for testing)
cap = cv2.VideoCapture("path/to/video.mp4")
```

#### Camera Properties

```python
# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Set frame rate
cap.set(cv2.CAP_PROP_FPS, 30)

# Set exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

# Set brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
```

**Example configuration:**

```python
def initialize_camera(camera_id=0, width=1280, height=720, fps=30):
    """Initialize camera with custom settings"""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera initialized: {actual_width}x{actual_height}")
    
    return cap
```

### 4. Action Detection Thresholds

#### Global Threshold Scaling

```python
# Add to top of file
THRESHOLD_SCALE = 1.0  # Adjust for different camera distances

def detect_custom_actions(kp):
    # ... keypoint extraction ...
    
    # Scale all thresholds
    ARMS_CROSSED_THRESHOLD = 80 * THRESHOLD_SCALE
    HANDS_CLASPED_THRESHOLD = 60 * THRESHOLD_SCALE
    CHIN_REST_THRESHOLD = 70 * THRESHOLD_SCALE
    
    # Use scaled thresholds
    if (distance(l_wrist, r_elbow) < ARMS_CROSSED_THRESHOLD and 
        distance(r_wrist, l_elbow) < ARMS_CROSSED_THRESHOLD):
        actions.append("arms_crossed")
```

#### Individual Action Thresholds

Create a configuration dictionary:

```python
# At top of file
ACTION_CONFIG = {
    "arms_crossed": {
        "enabled": True,
        "threshold": 80,
        "description": "Arms folded across chest"
    },
    "hands_clasped": {
        "enabled": True,
        "threshold": 60,
        "description": "Hands held together"
    },
    "chin_rest": {
        "enabled": True,
        "threshold": 70,
        "description": "Hand supporting chin"
    },
    "touch_face": {
        "enabled": False,  # Disable this action
        "threshold": 70,
        "description": "Hand touching face"
    },
    # ... more actions ...
}

def detect_custom_actions(kp):
    # ... keypoint extraction ...
    
    actions = []
    
    # Arms crossed (only if enabled)
    if ACTION_CONFIG["arms_crossed"]["enabled"]:
        threshold = ACTION_CONFIG["arms_crossed"]["threshold"]
        if (distance(l_wrist, r_elbow) < threshold and 
            distance(r_wrist, l_elbow) < threshold):
            actions.append("arms_crossed")
    
    # Continue for other actions...
```

#### Loading Configuration from File

```python
import json

# Load from JSON file
def load_action_config(config_file="action_config.json"):
    """Load action configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_file} not found, using defaults")
        return DEFAULT_CONFIG

# Use it
ACTION_CONFIG = load_action_config()
```

**Example `action_config.json`:**

```json
{
  "threshold_scale": 1.2,
  "actions": {
    "arms_crossed": {
      "enabled": true,
      "threshold": 80
    },
    "hands_clasped": {
      "enabled": true,
      "threshold": 60
    },
    "fidget_hands": {
      "enabled": true,
      "threshold": 25
    }
  }
}
```

### 5. Speech-to-Text Configuration

#### Audio Settings

```python
# At top of file (interview_system.py)
SAMPLE_RATE = 16000          # Audio sample rate (Hz)
VOICE_CHUNK_SECONDS = 4      # Recording chunk duration (seconds)
```

**Adjustment guidelines:**

```python
# For better accuracy (slower)
VOICE_CHUNK_SECONDS = 6

# For faster response (may miss words)
VOICE_CHUNK_SECONDS = 2

# Higher quality audio
SAMPLE_RATE = 44100
```

#### Whisper Model Selection

```python
def stt_worker():
    # Available models: tiny, base, small, medium, large
    
    # Fastest, least accurate (recommended for real-time)
    model_stt = WhisperModel("tiny", device="cpu")
    
    # Good balance
    model_stt = WhisperModel("base", device="cpu")
    
    # Better accuracy, slower
    model_stt = WhisperModel("small", device="cpu")
    
    # Best accuracy with GPU
    model_stt = WhisperModel("medium", device="cuda")
```

#### Language Configuration

```python
# In stt_worker() transcription call
segments, _ = model_stt.transcribe(
    audio_mono,
    beam_size=1,
    language="en"  # Set language explicitly
)

# Options:
# language="en"    # English
# language="zh"    # Chinese
# language="es"    # Spanish
# language="fr"    # French
# language=None    # Auto-detect (slower)
```

### 6. Visualization Options

#### Display Window Size

```python
# Resize display window
cv2.namedWindow("Interview Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Interview Monitor", 1280, 720)
```

#### Text Overlay Customization

```python
# Customize overlay appearance
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  # Green (BGR)
TEXT_BG_COLOR = (0, 0, 0)  # Black background

# With background for better visibility
def draw_text_with_background(frame, text, position, 
                               font_scale=0.8, thickness=2):
    """Draw text with semi-transparent background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(frame, 
                 (x - 5, y - text_height - 5),
                 (x + text_width + 5, y + baseline + 5),
                 (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, 
                font_scale, TEXT_COLOR, thickness)
```

#### Frame Rate Display

```python
import time

fps_counter = 0
fps_start_time = time.time()
current_fps = 0

while True:
    ret, frame = cap.read()
    # ... processing ...
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    
    # Display FPS
    cv2.putText(frame, f"FPS: {current_fps}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Interview Monitor", frame)
```

### 7. Logging Configuration

#### Output File Names

```python
# At end of main() function
def save_logs(prefix="interview"):
    """Save logs with custom prefix"""
    with open(f"{prefix}_action_log.json", "w", encoding="utf-8") as f:
        json.dump(action_logs, f, ensure_ascii=False, indent=2)
    
    with open(f"{prefix}_transcription_log.json", "w", encoding="utf-8") as f:
        json.dump(speech_logs, f, ensure_ascii=False, indent=2)
    
    with open(f"{prefix}_combined_log.json", "w", encoding="utf-8") as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=2)

# Use with timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_logs(prefix=f"interview_{timestamp}")
```

#### Logging Frequency

```python
# Log every N seconds instead of every frame
LOGGING_INTERVAL = 1.0  # seconds
last_log_time = 0

while True:
    # ... frame processing ...
    
    current_time = time.time() - start_time
    
    # Only log at intervals
    if current_time - last_log_time >= LOGGING_INTERVAL:
        if frame_actions:
            action_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "actions": list(set(frame_actions))
            })
        last_log_time = current_time
```

### 8. Performance Optimization

#### Frame Skipping

```python
PROCESS_EVERY_N_FRAMES = 2  # Process every other frame

frame_counter = 0
last_actions = []

while True:
    ret, frame = cap.read()
    frame_counter += 1
    
    # Only process every Nth frame
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        results = model(frame, device="cpu", verbose=False)
        # ... detection logic ...
        last_actions = frame_actions
    else:
        # Use previous frame's actions
        frame_actions = last_actions
    
    # ... visualization ...
```

#### Resolution Scaling

```python
INFERENCE_SCALE = 0.5  # Process at 50% resolution

while True:
    ret, frame = cap.read()
    
    # Resize for inference
    small_frame = cv2.resize(frame, None, 
                            fx=INFERENCE_SCALE, 
                            fy=INFERENCE_SCALE)
    
    # Run inference on smaller frame
    results = model(small_frame, device="cpu", verbose=False)
    
    # Scale keypoints back to original resolution
    for r in results:
        for person in r.keypoints.xy:
            kp = person.cpu().numpy()
            kp = kp / INFERENCE_SCALE  # Scale back up
            # ... use scaled keypoints ...
    
    # Display original resolution
    cv2.imshow("Monitor", frame)
```

### 9. Multi-Person Handling

#### Single Person Mode (Default)

```python
# Process only first detected person
for r in results:
    if r.keypoints is None:
        continue
    
    # Take first person only
    if len(r.keypoints.xy) > 0:
        person = r.keypoints.xy[0]
        kp = person.cpu().numpy()
        actions = detect_custom_actions(kp)
        break  # Stop after first person
```

#### Multi-Person Mode

```python
# Process all detected persons
for r in results:
    if r.keypoints is None:
        continue
    
    for person_idx, person in enumerate(r.keypoints.xy):
        kp = person.cpu().numpy()
        actions = detect_custom_actions(kp)
        
        # Log with person identifier
        if actions:
            action_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "person_id": person_idx,
                "actions": actions
            })
```

### 10. Environment-Specific Configurations

#### Configuration Profiles

```python
# Define configuration profiles
CONFIGS = {
    "low_end": {
        "model": "yolov8n-pose.pt",
        "device": "cpu",
        "process_every_n_frames": 3,
        "inference_scale": 0.5,
        "camera_resolution": (640, 480),
    },
    "balanced": {
        "model": "yolo11m-pose.pt",
        "device": "cpu",
        "process_every_n_frames": 1,
        "inference_scale": 1.0,
        "camera_resolution": (1280, 720),
    },
    "high_performance": {
        "model": "yolo11m-pose.pt",
        "device": "cuda:0",
        "process_every_n_frames": 1,
        "inference_scale": 1.0,
        "camera_resolution": (1920, 1080),
    }
}

# Select and use profile
PROFILE = "balanced"  # Change based on system
config = CONFIGS[PROFILE]

model = YOLO(config["model"])
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera_resolution"][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_resolution"][1])
```

## Complete Configuration Example

Here's a complete example with a configuration file:

**config.yaml:**

```yaml
# Video Processing Configuration

# Camera settings
camera:
  device_id: 0
  width: 1280
  height: 720
  fps: 30

# Model settings
model:
  name: "yolo11m-pose.pt"
  device: "cpu"  # or "cuda:0"
  confidence: 0.5

# Performance settings
performance:
  process_every_n_frames: 1
  inference_scale: 1.0

# Action detection
actions:
  threshold_scale: 1.0
  enabled_actions:
    - arms_crossed
    - hands_clasped
    - chin_rest
    - lean_forward
    - lean_back
    - head_down
    - touch_face
    - fidget_hands
  
  thresholds:
    arms_crossed: 80
    hands_clasped: 60
    chin_rest: 70
    fidget_hands: 25

# STT settings
stt:
  enabled: true
  model: "tiny"
  language: "en"
  sample_rate: 16000
  chunk_seconds: 4

# Visualization
display:
  window_name: "Interview Monitor"
  font_scale: 0.8
  show_fps: true
  show_keypoints: false

# Logging
logging:
  output_dir: "./logs"
  prefix: "interview"
  add_timestamp: true
  log_interval: 1.0
```

**Load and use configuration:**

```python
import yaml

def load_config(config_file="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()

# Apply configuration
model = YOLO(config['model']['name'])
cap = cv2.VideoCapture(config['camera']['device_id'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])

THRESHOLD_SCALE = config['actions']['threshold_scale']
PROCESS_EVERY_N_FRAMES = config['performance']['process_every_n_frames']
```

## Testing Configurations

### Quick Configuration Test

```python
def test_configuration():
    """Test current configuration"""
    print("Testing configuration...")
    
    # Test camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed to open")
        return False
    print("✓ Camera initialized")
    
    # Test model
    try:
        model = YOLO("yolo11m-pose.pt")
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Model failed to load: {e}")
        return False
    
    # Test inference
    ret, frame = cap.read()
    if ret:
        try:
            results = model(frame, device="cpu", verbose=False)
            print("✓ Inference successful")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return False
    
    cap.release()
    print("✓ Configuration test passed")
    return True

if __name__ == "__main__":
    test_configuration()
```

## Troubleshooting

### Common Configuration Issues

1. **Low FPS**: Reduce model size, increase frame skipping
2. **High memory usage**: Use smaller model, reduce resolution
3. **False detections**: Adjust action thresholds
4. **No detections**: Check camera position, lighting
5. **Audio not working**: Check microphone permissions, sample rate

### Performance Profiling

```python
import time

def profile_performance():
    """Profile performance of different components"""
    timings = {
        "frame_capture": [],
        "inference": [],
        "action_detection": [],
        "visualization": []
    }
    
    for _ in range(100):  # Test 100 frames
        # Frame capture
        t0 = time.time()
        ret, frame = cap.read()
        timings["frame_capture"].append(time.time() - t0)
        
        # Inference
        t0 = time.time()
        results = model(frame, device="cpu", verbose=False)
        timings["inference"].append(time.time() - t0)
        
        # Action detection
        t0 = time.time()
        # ... detection code ...
        timings["action_detection"].append(time.time() - t0)
    
    # Print averages
    for component, times in timings.items():
        avg_ms = np.mean(times) * 1000
        print(f"{component}: {avg_ms:.2f} ms")
```
