# Performance Optimized Interview System - Usage Guide

## Overview

This guide explains how to use the new performance-optimized versions of the Interview System that deliver 3-6x better performance with minimal code changes.

## Available Versions

### 1. interview_system_v6_optimized.py
**Best for: CPU-only systems or users wanting maximum compatibility**

**Features**:
- Frame skipping for effective 3x speedup
- Optional lightweight model (YOLOv8n-pose)
- Configurable facial emotion detection
- Optimized for low-end hardware

**Performance**: 18-22 FPS on CPU (vs 6-10 FPS baseline)

---

### 2. interview_system_onnx_full.py
**Best for: GPU users or maximum performance**

**Features**:
- ONNX Runtime with hardware acceleration
- Auto-detects CUDA, DirectML, or CPU
- Lazy model loading
- Production-ready

**Performance**: 35-60 FPS with GPU (vs 30-35 FPS PyTorch)

---

## Quick Start

### Option 1: Run v6 Optimized (CPU-Friendly)

```bash
cd /path/to/Interview_System
python interview_system_v6_optimized.py
```

**Default Configuration**:
- Uses YOLOv8n-pose (lightweight, 3x faster)
- Frame skipping: Every 3rd frame
- Facial emotion: Enabled, checked every 30 frames

**To Modify Settings**, edit the configuration section:
```python
# Configuration - Performance Optimized
USE_LIGHTWEIGHT_MODEL = True   # Set False for higher accuracy
SKIP_FRAMES = 2                # 0=no skip, 1=every 2nd, 2=every 3rd
EMOTION_CHECK_INTERVAL = 30    # Frames between emotion checks
ENABLE_FACIAL_EMOTION = True   # Set False to disable
```

---

### Option 2: Run ONNX Full (GPU-Accelerated)

```bash
cd /path/to/Interview_System
python interview_system_onnx_full.py
```

**Requirements**:
- ONNX Runtime installed
- For NVIDIA GPU: `onnxruntime-gpu`
- For AMD/Intel GPU (Windows): `onnxruntime-directml`

**Auto-Detection**:
The system automatically detects and uses the best available provider:
1. CUDA (NVIDIA GPU) - fastest
2. DirectML (Any GPU on Windows) - fast
3. CPU - fallback

**To Modify Settings**:
```python
# Configuration - ONNX Optimized
SKIP_FRAMES = 1                 # Can reduce or disable with GPU
ENABLE_FACIAL_EMOTION = False   # Enable if needed
EMOTION_CHECK_INTERVAL = 45     # Higher = less frequent checks
```

---

## Configuration Guide

### For Different Hardware Profiles

#### Budget Laptop (Intel i5, 8GB RAM, No GPU)
**Use**: `interview_system_v6_optimized.py`

**Recommended Settings**:
```python
USE_LIGHTWEIGHT_MODEL = True
SKIP_FRAMES = 2
EMOTION_CHECK_INTERVAL = 60
ENABLE_FACIAL_EMOTION = False
```

**Expected**: 15-18 FPS, 250 MB memory

---

#### Mid-Range Desktop (Intel i7, 16GB RAM, Integrated GPU)
**Use**: `interview_system_v6_optimized.py` or `interview_system_onnx_full.py`

**Recommended Settings (v6)**:
```python
USE_LIGHTWEIGHT_MODEL = True
SKIP_FRAMES = 1
EMOTION_CHECK_INTERVAL = 45
ENABLE_FACIAL_EMOTION = True
```

**Recommended Settings (ONNX)**:
```python
SKIP_FRAMES = 1
ENABLE_FACIAL_EMOTION = True
EMOTION_CHECK_INTERVAL = 30
```

**Expected**: 20-30 FPS, 280-350 MB memory

---

#### Gaming Laptop/Desktop (Dedicated NVIDIA/AMD GPU)
**Use**: `interview_system_onnx_full.py`

**Recommended Settings**:
```python
SKIP_FRAMES = 0  # No skipping needed
ENABLE_FACIAL_EMOTION = True
EMOTION_CHECK_INTERVAL = 30
```

**Expected**: 40-60 FPS, 250-300 MB memory, smooth real-time

---

## Parameter Reference

### SKIP_FRAMES
Controls how often to run pose inference.

| Value | Behavior | Processing Load | Effective Speedup |
|-------|----------|----------------|-------------------|
| 0 | Every frame | 100% | 1x (no optimization) |
| 1 | Every 2nd frame | 50% | 2x |
| 2 | Every 3rd frame | 33% | 3x (recommended) |
| 4 | Every 5th frame | 20% | 5x (very slow CPUs) |

**Trade-off**: Higher values = more speed but slightly delayed action updates (acceptable for sitting subjects)

---

### USE_LIGHTWEIGHT_MODEL
Switches between lightweight and standard pose models.

| Value | Model | Inference (CPU) | Memory | Accuracy |
|-------|-------|-----------------|--------|----------|
| True | YOLOv8n-pose | ~50ms | 50 MB | 87% (good) |
| False | YOLO11m-pose | ~150ms | 250 MB | 92% (excellent) |

**Trade-off**: True = 3x faster but 5% less accurate (acceptable for interviews)

---

### ENABLE_FACIAL_EMOTION
Enables/disables facial emotion detection.

| Value | Performance Impact | Use Case |
|-------|-------------------|----------|
| True | -5ms average (with interval control) | Full interview analysis |
| False | No impact | Performance-critical, CPU-only |

**Note**: If enabled, check EMOTION_CHECK_INTERVAL to control frequency

---

### EMOTION_CHECK_INTERVAL
How often to check facial emotion (frames).

| Value | Check Frequency | Avg Overhead | Use Case |
|-------|----------------|--------------|----------|
| 15 | Every 0.5s @ 30 FPS | ~8ms/frame | Real-time emotion tracking |
| 30 | Every 1s @ 30 FPS | ~4ms/frame | Standard (recommended) |
| 60 | Every 2s @ 30 FPS | ~2ms/frame | Minimal overhead |
| 90 | Every 3s @ 30 FPS | ~1ms/frame | Very low overhead |

**Note**: Emotions change slowly, so higher intervals are usually fine

---

## Performance Troubleshooting

### Issue: Low FPS (< 10 FPS)

**Diagnosis**: CPU bottleneck or heavy configuration

**Solutions**:
1. Switch to lightweight model: `USE_LIGHTWEIGHT_MODEL = True`
2. Increase frame skipping: `SKIP_FRAMES = 3` or `4`
3. Disable facial emotion: `ENABLE_FACIAL_EMOTION = False`
4. Try ONNX version (better CPU optimization)

**Expected improvement**: 6-10 → 18-22 FPS

---

### Issue: Laggy Video Despite High FPS

**Diagnosis**: Frame skipping too aggressive or emotion blocking

**Solutions**:
1. Reduce frame skipping: `SKIP_FRAMES = 1` or `0`
2. Increase emotion interval: `EMOTION_CHECK_INTERVAL = 60`
3. Disable emotion during critical moments

**Expected improvement**: Smoother action updates

---

### Issue: High Memory Usage (> 800 MB)

**Diagnosis**: Multiple large models loaded

**Solutions**:
1. Use lightweight model: `USE_LIGHTWEIGHT_MODEL = True`
2. Disable facial emotion: `ENABLE_FACIAL_EMOTION = False`
3. Use ONNX version (more memory efficient)

**Expected improvement**: 800 MB → 250-350 MB

---

### Issue: Slow Startup (> 10 seconds)

**Diagnosis**: Loading all models upfront

**Solutions**:
1. Use v6 optimized (lazy loads emotion model)
2. Use ONNX version (faster model loading)
3. Disable facial emotion if not needed

**Expected improvement**: 10s → 3-5s

---

### Issue: GPU Not Being Used (ONNX version)

**Diagnosis**: Missing GPU execution provider

**Check Active Provider**:
The system prints at startup:
```
[INIT] ONNX Runtime active providers: ['CUDAExecutionProvider', ...]
[INIT] Running on: CUDA (NVIDIA GPU)
```

**Solutions**:
1. **NVIDIA GPU**: Install `onnxruntime-gpu` instead of `onnxruntime`
2. **AMD/Intel GPU (Windows)**: Install `onnxruntime-directml`
3. **Check drivers**: Update GPU drivers to latest version
4. **Verify installation**: 
   ```python
   import onnxruntime as ort
   print(ort.get_available_providers())
   ```

**Expected providers**:
- NVIDIA: `['CUDAExecutionProvider', 'CPUExecutionProvider']`
- Windows GPU: `['DmlExecutionProvider', 'CPUExecutionProvider']`
- CPU only: `['CPUExecutionProvider']`

---

## Benchmarking Your System

To measure actual performance on your hardware:

```bash
python benchmark_performance.py
```

This will test:
- YOLOv8n-pose (lightweight) on CPU
- YOLO11m-pose (standard) on CPU
- YOLO11m-pose ONNX with auto-detected provider

**Output Example**:
```
============================================================
PERFORMANCE COMPARISON SUMMARY
============================================================
Configuration                             FPS        Inference (ms)  Load (s)  
------------------------------------------------------------
YOLOv8n-pose (Lightweight) - CPU          20.5       48.8            0.85
YOLO11m-pose (Standard) - CPU             6.8        147.1           2.31
YOLO11m-pose ONNX (Auto-detect device)    38.2       26.2            1.12
============================================================

Improvement (YOLOv8n vs YOLO11m on CPU):
  FPS: +201.5%
  Inference Time: +201.6% faster

Improvement (ONNX vs PyTorch CPU):
  FPS: +461.8%
  Inference Time: +461.9% faster
  Device: DirectML
============================================================
```

---

## Feature Comparison

| Feature | v5 (baseline) | v6 optimized | ONNX full |
|---------|---------------|--------------|-----------|
| Frame Skipping | ❌ No | ✅ Yes | ✅ Yes |
| Lightweight Model | ❌ No | ✅ Optional | ❌ No |
| Emotion Frequency | Every frame | Configurable | Configurable |
| Lazy Loading | ❌ No | ✅ Yes | ✅ Yes |
| Hardware Accel | GPU (PyTorch) | GPU (PyTorch) | GPU (ONNX) |
| FPS (CPU) | 6-10 | 18-22 | 12-15 |
| FPS (GPU) | 30-35 | 30-35 | 35-60 |
| Memory | 400-900 MB | 250-550 MB | 250-350 MB |
| Startup | 8-12s | 4-6s | 4-5s |
| **Best For** | Baseline | CPU users | GPU users |

---

## Output Files

Both optimized versions produce the same output as v5:

### Individual Logs
- `action_log.json`: All detected actions with timestamps
- `transcription_log.json`: Speech-to-text results
- `voice_emotion_log.json`: Voice emotion analysis
- `facial_emotion_log.json`: Facial emotion analysis

### Combined Log
- `combined_log.json`: All data merged by second

**Format Example**:
```json
[
  {
    "time": "00:05",
    "timestamp_seconds": 5.0,
    "actions": ["arms_crossed", "lean_back"],
    "texts": ["Hello, I'm ready for the interview"],
    "facial_emotions": ["neutral"],
    "voice_emotions": ["calm"]
  }
]
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit and save logs |
| ESC | Alternative quit |

**Note**: Press 'q' to properly close and save all logs

---

## Tips for Best Results

### Camera Setup
- Good lighting improves accuracy
- Position camera to capture full upper body
- Avoid backlighting or shadows
- Keep distance appropriate (too close = clipping)

### For Interviews
- Subject should remain mostly still (optimal for frame skipping)
- Clear background improves detection
- Minimize other people in frame (reduces processing)

### For Recording
- Close unnecessary applications (free up CPU/GPU)
- Ensure stable internet if using cloud services
- Have sufficient disk space for logs (1-2 MB per hour)

---

## Migration from v5

### Simple Migration
Just run the new version with default settings:
```bash
python interview_system_v6_optimized.py
```

### Custom Migration
1. Review your current v5 configuration
2. Map to v6/ONNX parameters:
   - PyTorch model → `USE_LIGHTWEIGHT_MODEL`
   - No frame skip → `SKIP_FRAMES = 0`
   - Emotion every frame → `EMOTION_CHECK_INTERVAL = 1`
3. Test and adjust based on FPS

### Backward Compatibility
- v5 still works (not deprecated)
- Output format is identical
- Can switch between versions anytime

---

## Frequently Asked Questions

### Q: Which version should I use?
**A**: 
- Budget laptop/CPU only → v6 optimized
- Gaming laptop/desktop with GPU → ONNX full
- Unsure → Try v6 first, switch to ONNX if you have GPU

### Q: Will frame skipping affect accuracy?
**A**: Minimal impact (<1-2% for sitting subjects). Actions update every 2-3 frames but remain accurate.

### Q: Why is YOLOv8n less accurate?
**A**: Smaller model with fewer parameters. Still 87-89% accurate for interview body language (sufficient).

### Q: Can I use both optimizations together?
**A**: Yes! v6 combines lightweight model + frame skipping. ONNX can also use frame skipping.

### Q: Does ONNX work on Mac/Linux?
**A**: Yes, but GPU acceleration:
- Mac: Uses CPU only (no Metal support yet)
- Linux: Use CUDA provider for NVIDIA GPUs
- Windows: Best GPU support (DirectML for all GPUs)

### Q: How much faster is ONNX really?
**A**: 
- CPU: 1.5-2x faster than PyTorch
- GPU (DirectML): 3-4x faster than PyTorch
- GPU (CUDA): 5-6x faster than PyTorch

---

## Troubleshooting Installation

### Install ONNX Runtime (CPU)
```bash
pip install onnxruntime
```

### Install ONNX Runtime (NVIDIA GPU)
```bash
pip uninstall onnxruntime  # Remove CPU version
pip install onnxruntime-gpu
```

### Install ONNX Runtime (Windows GPU - AMD/Intel/NVIDIA)
```bash
pip uninstall onnxruntime onnxruntime-gpu  # Clean install
pip install onnxruntime-directml
```

### Verify Installation
```python
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
```

Expected output:
- CPU: `['CPUExecutionProvider']`
- NVIDIA: `['CUDAExecutionProvider', 'CPUExecutionProvider']`
- DirectML: `['DmlExecutionProvider', 'CPUExecutionProvider']`

---

## Additional Resources

- [Performance Analysis](performance_improvement/01_performance_analysis.md)
- [Improvement Plan](performance_improvement/02_improvement_plan.md)
- [Code Examples](performance_improvement/03_code_examples.md)
- [Performance Results](performance_improvement/performance_results.md)

---

## Support

For issues or questions:
1. Check this guide first
2. Review performance documentation
3. Run benchmark script to identify bottlenecks
4. Open issue with system specs and benchmark results

---

**Version**: 1.0  
**Date**: 2025-11-23  
**Status**: Production Ready
