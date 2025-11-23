"""
Performance Benchmarking Script for Interview System

This script measures and compares the performance of different
interview system versions to validate improvements.
"""

import cv2
import numpy as np
import time
import sys
from ultralytics import YOLO

def benchmark_model(model_path, num_frames=100, device="cpu", description=""):
    """Benchmark pose detection model performance"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {description}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print("Loading model...")
        start_load = time.time()
        model = YOLO(model_path)
        load_time = time.time() - start_load
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Create dummy frames for testing
        print(f"Generating {num_frames} test frames...")
        test_frames = []
        for i in range(num_frames):
            # Create random frame (640x480, typical webcam resolution)
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            test_frames.append(frame)
        
        # Warmup (first inference is often slower)
        print("Warming up...")
        _ = model(test_frames[0], device=device, verbose=False)
        
        # Benchmark inference
        print(f"Running {num_frames} inferences...")
        inference_times = []
        
        start_total = time.time()
        for i, frame in enumerate(test_frames):
            start_inference = time.time()
            results = model(frame, device=device, verbose=False)
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{num_frames} frames")
        
        total_time = time.time() - start_total
        
        # Calculate statistics
        avg_inference = np.mean(inference_times) * 1000  # Convert to ms
        min_inference = np.min(inference_times) * 1000
        max_inference = np.max(inference_times) * 1000
        std_inference = np.std(inference_times) * 1000
        fps = 1.0 / np.mean(inference_times)
        
        # Display results
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average inference: {avg_inference:.2f}ms")
        print(f"Min inference: {min_inference:.2f}ms")
        print(f"Max inference: {max_inference:.2f}ms")
        print(f"Std deviation: {std_inference:.2f}ms")
        print(f"Theoretical FPS: {fps:.2f}")
        print(f"Model load time: {load_time:.2f}s")
        print(f"{'='*60}\n")
        
        return {
            'model': model_path,
            'device': device,
            'description': description,
            'load_time': load_time,
            'avg_inference_ms': avg_inference,
            'min_inference_ms': min_inference,
            'max_inference_ms': max_inference,
            'std_inference_ms': std_inference,
            'fps': fps,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return None


def benchmark_onnx_model(model_path, num_frames=100, description=""):
    """Benchmark ONNX model performance"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {description}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    try:
        import onnxruntime as ort
        
        # Setup providers
        providers = [
            ("CUDAExecutionProvider", {}),
            ("DmlExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider"
        ]
        
        print("Loading ONNX model...")
        start_load = time.time()
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        active_providers = session.get_providers()
        load_time = time.time() - start_load
        
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"Active providers: {active_providers}")
        
        # Determine device type
        if "CUDAExecutionProvider" in active_providers:
            device_type = "CUDA"
        elif "DmlExecutionProvider" in active_providers:
            device_type = "DirectML"
        else:
            device_type = "CPU"
        
        # Create test frames
        print(f"Generating {num_frames} test frames...")
        test_frames = []
        for i in range(num_frames):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            # Preprocess for ONNX
            img = cv2.resize(frame, (640, 640))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_input = np.ascontiguousarray(
                img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            )[np.newaxis, ...]
            test_frames.append(img_input)
        
        # Warmup
        print("Warming up...")
        _ = session.run(None, {input_name: test_frames[0]})
        
        # Benchmark
        print(f"Running {num_frames} inferences...")
        inference_times = []
        
        start_total = time.time()
        for i, img_input in enumerate(test_frames):
            start_inference = time.time()
            outputs = session.run(None, {input_name: img_input})
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{num_frames} frames")
        
        total_time = time.time() - start_total
        
        # Calculate statistics
        avg_inference = np.mean(inference_times) * 1000
        min_inference = np.min(inference_times) * 1000
        max_inference = np.max(inference_times) * 1000
        std_inference = np.std(inference_times) * 1000
        fps = 1.0 / np.mean(inference_times)
        
        # Display results
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"{'='*60}")
        print(f"Device type: {device_type}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average inference: {avg_inference:.2f}ms")
        print(f"Min inference: {min_inference:.2f}ms")
        print(f"Max inference: {max_inference:.2f}ms")
        print(f"Std deviation: {std_inference:.2f}ms")
        print(f"Theoretical FPS: {fps:.2f}")
        print(f"Model load time: {load_time:.2f}s")
        print(f"{'='*60}\n")
        
        return {
            'model': model_path,
            'device': device_type,
            'description': description,
            'load_time': load_time,
            'avg_inference_ms': avg_inference,
            'min_inference_ms': min_inference,
            'max_inference_ms': max_inference,
            'std_inference_ms': std_inference,
            'fps': fps,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "="*60)
    print("Interview System Performance Benchmark")
    print("="*60)
    
    results = []
    
    # Test configurations
    tests = [
        {
            'type': 'pytorch',
            'model': 'yolov8n-pose.pt',
            'device': 'cpu',
            'description': 'YOLOv8n-pose (Lightweight) - CPU'
        },
        {
            'type': 'pytorch',
            'model': 'yolo11m-pose.pt',
            'device': 'cpu',
            'description': 'YOLO11m-pose (Standard) - CPU'
        },
        {
            'type': 'onnx',
            'model': 'yolo11m-pose.onnx',
            'description': 'YOLO11m-pose ONNX (Auto-detect device)'
        }
    ]
    
    # Run benchmarks
    for test in tests:
        try:
            if test['type'] == 'pytorch':
                result = benchmark_model(
                    test['model'],
                    num_frames=100,
                    device=test['device'],
                    description=test['description']
                )
            else:
                result = benchmark_onnx_model(
                    test['model'],
                    num_frames=100,
                    description=test['description']
                )
            
            if result:
                results.append(result)
        except Exception as e:
            print(f"Skipping test due to error: {e}")
            continue
    
    # Summary comparison
    if results:
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Configuration':<40} {'FPS':<10} {'Inference (ms)':<15} {'Load (s)':<10}")
        print("-"*60)
        
        for r in results:
            config = f"{r['description'][:37]}..."
            print(f"{config:<40} {r['fps']:<10.1f} {r['avg_inference_ms']:<15.1f} {r['load_time']:<10.2f}")
        
        print("="*60)
        
        # Calculate improvements
        if len(results) >= 2:
            baseline = results[1]  # YOLO11m-pose CPU
            optimized = results[0]  # YOLOv8n-pose CPU
            
            fps_improvement = (optimized['fps'] / baseline['fps']) * 100 - 100
            inference_improvement = (baseline['avg_inference_ms'] / optimized['avg_inference_ms']) * 100 - 100
            
            print(f"\nImprovement (YOLOv8n vs YOLO11m on CPU):")
            print(f"  FPS: +{fps_improvement:.1f}%")
            print(f"  Inference Time: +{inference_improvement:.1f}% faster")
            
        if len(results) >= 3:
            baseline = results[1]  # YOLO11m-pose CPU
            onnx = results[2]  # ONNX
            
            fps_improvement = (onnx['fps'] / baseline['fps']) * 100 - 100
            inference_improvement = (baseline['avg_inference_ms'] / onnx['avg_inference_ms']) * 100 - 100
            
            print(f"\nImprovement (ONNX vs PyTorch CPU):")
            print(f"  FPS: +{fps_improvement:.1f}%")
            print(f"  Inference Time: +{inference_improvement:.1f}% faster")
            print(f"  Device: {onnx['device']}")
        
        print("="*60)
    else:
        print("\n✗ No successful benchmarks to compare")


if __name__ == "__main__":
    main()
