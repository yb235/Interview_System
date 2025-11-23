# Main Thread (Video Processing) Architecture

## Overview

The Main Thread is the core video processing component of the Interview System. It operates in real-time to analyze video frames from the webcam, detect poses, recognize actions, and coordinate with other system modules.

## System Components

### 1. Multi-Threading Architecture

The system uses a multi-threaded architecture to handle concurrent operations:

```
┌─────────────────────────────────────────────────────────┐
│                    Main Thread                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Video Capture & Processing Loop                 │  │
│  │  - Frame capture from webcam                     │  │
│  │  - Pose inference                                │  │
│  │  - Action detection                              │  │
│  │  - Emotion detection (optional)                  │  │
│  │  - Frame visualization                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                   STT Thread (Daemon)                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Speech-to-Text Processing                       │  │
│  │  - Audio capture                                 │  │
│  │  - Whisper transcription                         │  │
│  │  - Voice emotion detection                       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Shared State                          │
│  - start_time: Global timing reference                  │
│  - stop_flag: Thread coordination                       │
│  - action_logs: Detected actions with timestamps        │
│  - speech_logs: Transcribed text                        │
│  - emotion_logs: Detected emotions                      │
│  - current_subtitle: Display state                      │
└─────────────────────────────────────────────────────────┘
```

### 2. Main Thread Workflow

The main video processing loop follows this sequence:

```python
1. Initialize camera capture
2. Start STT daemon thread
3. Enter main processing loop:
   a. Read frame from camera
   b. Set start_time on first frame
   c. Run pose inference (YOLO model)
   d. Extract keypoints from results
   e. Detect custom actions
   f. Detect facial emotions (optional)
   g. Draw overlays on frame
   h. Display frame
   i. Check for quit signal
4. Cleanup and save logs
```

### 3. Key Components

#### 3.1 Video Capture

```python
cap = cv2.VideoCapture(0)  # 0 = default webcam
```

The system captures video at the native resolution of the webcam. Frames are processed sequentially in the main thread.

#### 3.2 Timing System

```python
start_time = None  # Set when first frame is captured
ts = time.time() - start_time  # Elapsed time in seconds
```

All logs use consistent timestamps relative to `start_time`, enabling synchronization across different data streams (video, audio, emotions).

#### 3.3 Model Loading

```python
# PyTorch model (versions 1-5)
model = YOLO("yolo11m-pose.pt")

# ONNX model (run_pose_onnx_dml.py)
session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)
```

Models are loaded once at startup to avoid repeated initialization overhead.

#### 3.4 Frame Processing Pipeline

Each frame goes through this pipeline:

```
Raw Frame → Pose Detection → Keypoint Extraction → Action Detection → Visualization
```

### 4. Thread Coordination

#### 4.1 Synchronization Variables

- **start_time**: Shared timing reference, set by main thread
- **stop_flag**: Coordination flag for graceful shutdown
- **current_subtitle**: STT output for display overlay
- **current_emotion**: Latest emotion for display

#### 4.2 Thread Safety

- Global variables are read/written by threads
- STT thread waits for `start_time` to be set before processing
- Main thread sets `stop_flag` to signal STT thread to exit

### 5. Data Flow

```
Camera → Frame → Pose Model → Keypoints → Action Detector → Logs
   ↓                                                           ↓
Display ← Visualization ← Overlay ← Current State ← Shared State
```

### 6. System Versions

The system has evolved through multiple versions:

| Version | Features |
|---------|----------|
| v1 (interview_system.py) | Pose + Actions + STT |
| v2 (interview_system_v2.py) | Added DeepFace emotions |
| v3 (interview_system_v3.py) | Enhanced emotion + STT integration |
| v4 (interview_system_v4.py) | Added voice emotion detection |
| v5 (interview_system_v5.py) | Separate facial emotion logging |

### 7. Performance Considerations

- **CPU vs GPU**: Most versions use `device="cpu"` for compatibility
- **Frame rate**: Limited by pose inference time (~30-50ms per frame)
- **Threading**: STT runs asynchronously to avoid blocking video
- **Emotion detection**: Can be slow; v4/v5 optimize by reducing frequency

### 8. Error Handling

```python
# Camera initialization
if not cap.isOpened():
    print("Error: cannot open camera.")
    stop_flag = True
    return

# Frame capture
ret, frame = cap.read()
if not ret:
    print("Error: cannot read frame.")
    break
```

The system handles common failure cases gracefully:
- Camera unavailable
- Frame read failures
- Model inference errors (try-except in emotion detection)

### 9. Cleanup and Logging

At shutdown, the main thread:
1. Releases camera resources
2. Destroys OpenCV windows
3. Sets stop_flag for STT thread
4. Waits for thread completion
5. Saves all logs to JSON files

### 10. Extension Points

The architecture supports extensions:
- **New actions**: Add to `detect_custom_actions()`
- **New modalities**: Add threads (e.g., gaze tracking)
- **New models**: Swap YOLO model or add additional models
- **Custom visualizations**: Modify frame overlay logic
