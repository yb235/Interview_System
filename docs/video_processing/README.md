# Main Thread (Video Processing) Documentation

This directory contains in-depth documentation for the Main Thread (Video Processing) module of the Interview System.

## Overview

The Main Thread handles real-time video processing, including:
- Pose detection and keypoint extraction
- Custom action/posture recognition
- Model inference management
- Frame-by-frame analysis
- Integration with other system components

## Documentation Structure

- **[01_architecture.md](01_architecture.md)** - System architecture and design
- **[02_model_inference.md](02_model_inference.md)** - Model inference pipeline and optimization
- **[03_motion_recognition.md](03_motion_recognition.md)** - Motion and posture recognition system
- **[04_adding_new_actions.md](04_adding_new_actions.md)** - Guide to adding new motion postures
- **[05_configuration.md](05_configuration.md)** - Configuration and customization options
- **[06_onnx_acceleration.md](06_onnx_acceleration.md)** - Hardware acceleration with ONNX

## Quick Start

For a quick understanding of the video processing module:
1. Start with [Architecture](01_architecture.md) to understand the overall design
2. Review [Model Inference](02_model_inference.md) to understand how pose detection works
3. Read [Motion Recognition](03_motion_recognition.md) to see how actions are detected
4. Follow [Adding New Actions](04_adding_new_actions.md) to extend the system

## System Requirements

- Python 3.8+
- OpenCV (cv2)
- Ultralytics YOLO
- NumPy
- CUDA (optional, for GPU acceleration)
- ONNX Runtime (optional, for hardware acceleration)
