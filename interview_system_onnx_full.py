import cv2
import numpy as np
import time
import json
import threading
import sounddevice as sd
from faster_whisper import WhisperModel
import onnxruntime as ort

# ==============================
# Configuration - ONNX Optimized
# ==============================

# ONNX Model and Providers
ONNX_MODEL = "yolo11m-pose.onnx"

# Auto-detect best execution provider
# Priority: CUDA > DirectML > CPU
PROVIDERS = [
    ("CUDAExecutionProvider", {}),      # NVIDIA GPU
    ("DmlExecutionProvider", {"device_id": 0}),  # AMD/Intel/NVIDIA on Windows
    "CPUExecutionProvider"              # Fallback
]

# Frame Skipping: Less needed with ONNX, but still beneficial
SKIP_FRAMES = 1  # 0=no skip, 1=every 2nd frame

# Facial Emotion Detection (optional, can be disabled for max performance)
ENABLE_FACIAL_EMOTION = False  # Set to True if needed
EMOTION_CHECK_INTERVAL = 45

# Audio Settings
SAMPLE_RATE = 16000
VOICE_CHUNK_SECONDS = 4

# Shared globals
start_time = None
stop_flag = False

prev_left_wrist = None
prev_right_wrist = None

action_logs = []
speech_logs = []
voice_emotion_logs = []
facial_emotion_logs = []

current_subtitle = ""
current_voice_emotion = ""
current_facial_emotion = ""


# ================================================================
# Utility
# ================================================================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ================================================================
# ONNX Session Setup
# ================================================================
print("[INIT] Initializing ONNX Runtime session...")
try:
    session = ort.InferenceSession(ONNX_MODEL, providers=PROVIDERS)
    input_name = session.get_inputs()[0].name
    active_providers = session.get_providers()
    print(f"[INIT] ONNX Runtime active providers: {active_providers}")
    
    # Determine device type for display
    if "CUDAExecutionProvider" in active_providers:
        device_type = "CUDA (NVIDIA GPU)"
    elif "DmlExecutionProvider" in active_providers:
        device_type = "DirectML (GPU)"
    else:
        device_type = "CPU"
    print(f"[INIT] Running on: {device_type}")
except Exception as e:
    print(f"[ERROR] Failed to initialize ONNX session: {e}")
    exit(1)


# ================================================================
# Pose / Action Detection with ONNX
# ================================================================
def detect_custom_actions(kp):
    global prev_left_wrist, prev_right_wrist

    # YOLO pose keypoint indices (17 keypoints standard format)
    # 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows
    # 9-10: wrists, 11-12: hips
    if len(kp) < 13:
        return []

    nose = kp[0]
    left_eye, right_eye = kp[1], kp[2]
    left_ear, right_ear = kp[3], kp[4]
    l_shoulder, r_shoulder = kp[5], kp[6]
    l_elbow, r_elbow = kp[7], kp[8]
    l_wrist, r_wrist = kp[9], kp[10]
    l_hip, r_hip = kp[11], kp[12]

    actions = []

    shoulder_center = (
        (l_shoulder[0] + r_shoulder[0]) / 2,
        (l_shoulder[1] + r_shoulder[1]) / 2
    )
    hip_center = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    )

    # 1. Arms Crossed
    if distance(l_wrist, r_elbow) < 80 and distance(r_wrist, l_elbow) < 80:
        actions.append("arms_crossed")

    # 2. Hands Clasped
    if distance(l_wrist, r_wrist) < 60:
        actions.append("hands_clasped")

    # 3. Chin Rest
    if distance(l_wrist, nose) < 70 or distance(r_wrist, nose) < 70:
        actions.append("chin_rest")

    # 4. Lean forward or backward
    torso_height = abs(shoulder_center[1] - hip_center[1])
    if torso_height < 120:
        actions.append("lean_forward")
    if torso_height > 200:
        actions.append("lean_back")

    # 5. Head Down
    if nose[1] > shoulder_center[1] + 40:
        actions.append("head_down")

    # 6. Touch Face
    face_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )
    if distance(l_wrist, face_center) < 70 or distance(r_wrist, face_center) < 70:
        actions.append("touch_face")

    # 7. Touch Nose
    if distance(l_wrist, nose) < 40 or distance(r_wrist, nose) < 40:
        actions.append("touch_nose")

    # 8. Fix Hair
    if (
        distance(l_wrist, left_ear) < 60 or
        distance(r_wrist, right_ear) < 60 or
        distance(l_wrist, right_ear) < 60 or
        distance(r_wrist, left_ear) < 60
    ):
        actions.append("fix_hair")

    # 9. Fidget Hands
    fidget_detected = False
    if prev_left_wrist is not None:
        if distance(prev_left_wrist, l_wrist) > 25:
            fidget_detected = True
    if prev_right_wrist is not None:
        if distance(prev_right_wrist, r_wrist) > 25:
            fidget_detected = True
    if fidget_detected:
        actions.append("fidget_hands")

    prev_left_wrist = l_wrist
    prev_right_wrist = r_wrist

    return list(set(actions))


def preprocess_frame(frame):
    """Optimized frame preprocessing for ONNX"""
    img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.ascontiguousarray(
        img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    )[np.newaxis, ...]
    return img_input


def parse_onnx_output(outputs, original_shape, conf_threshold=0.4):
    """Parse YOLO pose ONNX output"""
    # YOLO11 Pose output: (1, 56, 8400)
    # 0-3: bbox (x, y, w, h)
    # 4: objectness score
    # 5: class confidence (usually 0 for person)
    # 6-55: 17 keypoints × 3 (x, y, confidence) = 51 values (total 56)
    
    outputs = outputs[0]  # Remove batch dimension: (56, 8400)
    scores = outputs[4]
    mask = scores > conf_threshold
    
    filtered = outputs[:, mask].T  # (N, 56) where N is number of detections
    
    detections = []
    h, w = original_shape[:2]
    scale_x = w / 640
    scale_y = h / 640
    
    for det in filtered:
        # Extract keypoints (17 keypoints, indices 6-55, but in format x,y,conf)
        kpts_raw = det[6:].reshape(17, 3)  # (17, 3) - x, y, confidence
        
        # Scale keypoints back to original frame size
        keypoints = []
        for kpt in kpts_raw:
            x = kpt[0] * scale_x
            y = kpt[1] * scale_y
            conf = kpt[2]
            keypoints.append([x, y, conf])
        
        detections.append({
            'keypoints': np.array(keypoints),
            'score': det[4]
        })
    
    return detections


# ================================================================
# Facial Emotion Detection (optional)
# ================================================================
facial_emotion_model = None

def detect_facial_emotion(frame):
    global current_facial_emotion, facial_emotion_model
    
    if not ENABLE_FACIAL_EMOTION:
        return None
    
    # Lazy load DeepFace only if needed
    if facial_emotion_model is None:
        from deepface import DeepFace
        facial_emotion_model = DeepFace
        print("[INIT] DeepFace loaded (lazy initialization)")
    
    try:
        res = facial_emotion_model.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        emo = res[0]["dominant_emotion"]
        current_facial_emotion = emo
        return emo
    except Exception as e:
        return None


# ================================================================
# Voice Emotion Detection
# ================================================================
def detect_voice_emotion_simple(audio_chunk):
    global current_voice_emotion
    energy = float(np.mean(audio_chunk ** 2))
    
    if energy > 0.03:
        emotion = "agitated"
    elif energy > 0.005:
        emotion = "neutral"
    else:
        emotion = "calm"
    
    current_voice_emotion = emotion
    return emotion


# ================================================================
# Speech-to-Text (Whisper)
# ================================================================
def stt_worker():
    global stop_flag, start_time, speech_logs, current_subtitle
    
    print("[STT] Loading Whisper tiny on CPU...")
    model_stt = WhisperModel("tiny", device="cpu")
    
    while start_time is None and not stop_flag:
        time.sleep(0.1)
    
    print("[STT] STT thread active.")
    
    while not stop_flag:
        duration = VOICE_CHUNK_SECONDS
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        
        if stop_flag:
            break
        
        audio_mono = audio.flatten()
        
        # STT
        try:
            segments, _ = model_stt.transcribe(
                audio_mono,
                beam_size=1,
                language="en"
            )
        except Exception as e:
            print("[STT] Transcribe error:", e)
            continue
        
        text_full = ""
        for seg in segments:
            text_full += seg.text.strip() + " "
        
        if text_full.strip():
            ts = time.time() - start_time
            mm = int(ts // 60)
            ss = int(ts % 60)
            
            speech_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "text": text_full.strip()
            })
            
            current_subtitle = text_full.strip()
            print(f"[STT {mm:02d}:{ss:02d}] {text_full.strip()}")
        
        # Voice emotion
        emotion = detect_voice_emotion_simple(audio_mono)
        ts = time.time() - start_time
        mm = int(ts // 60)
        ss = int(ts % 60)
        
        voice_emotion_logs.append({
            "time": f"{mm:02d}:{ss:02d}",
            "timestamp_seconds": round(ts, 2),
            "voice_emotion": emotion
        })
    
    print("[STT] Thread stopped.")


# ================================================================
# Main - ONNX Full System
# ================================================================
def main():
    global start_time, stop_flag
    
    print("=" * 60)
    print("Interview System - ONNX Full (High Performance)")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Model: {ONNX_MODEL}")
    print(f"  - Device: {device_type}")
    print(f"  - Frame Skip: {SKIP_FRAMES}")
    print(f"  - Facial Emotion: {'Enabled' if ENABLE_FACIAL_EMOTION else 'Disabled'}")
    print("=" * 60)
    
    # Start STT thread
    threading.Thread(target=stt_worker, daemon=True).start()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera.")
        stop_flag = True
        return
    
    frame_count = 0
    last_actions = []
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    print("[SYSTEM] Starting video capture...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if start_time is None:
            start_time = time.time()
            print("[SYSTEM] Timer started from first frame.")
        
        frame_count += 1
        fps_frame_count += 1
        
        ts = time.time() - start_time
        mm = int(ts // 60)
        ss = int(ts % 60)
        
        # Frame skipping
        if frame_count % (SKIP_FRAMES + 1) == 0:
            # Preprocess
            img_input = preprocess_frame(frame)
            
            # ONNX inference
            outputs = session.run(None, {input_name: img_input})[0]
            
            # Parse detections
            detections = parse_onnx_output(outputs, frame.shape)
            
            # Detect actions for all persons
            frame_actions = []
            for det in detections:
                kp = det['keypoints'][:, :2]  # Extract only x, y (ignore confidence)
                actions = detect_custom_actions(kp)
                frame_actions.extend(actions)
            
            last_actions = frame_actions
            
            if frame_actions:
                action_logs.append({
                    "time": f"{mm:02d}:{ss:02d}",
                    "timestamp_seconds": round(ts, 2),
                    "actions": list(set(frame_actions))
                })
        else:
            frame_actions = last_actions
        
        # Facial emotion (optional)
        if ENABLE_FACIAL_EMOTION and frame_count % EMOTION_CHECK_INTERVAL == 0:
            facial_emo = detect_facial_emotion(frame)
            if facial_emo:
                facial_emotion_logs.append({
                    "time": f"{mm:02d}:{ss:02d}",
                    "timestamp_seconds": round(ts, 2),
                    "facial_emotion": facial_emo
                })
        else:
            facial_emo = current_facial_emotion
        
        # Calculate FPS
        if fps_frame_count >= 30:
            elapsed = time.time() - fps_start_time
            current_fps = fps_frame_count / elapsed
            fps_start_time = time.time()
            fps_frame_count = 0
        
        # Display
        y = 30
        
        # FPS
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Device info
        cv2.putText(frame, f"Device: {device_type}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Actions
        for act in set(frame_actions):
            cv2.putText(frame, f"ACTION: {act}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30
        
        # Facial Emotion
        if facial_emo and ENABLE_FACIAL_EMOTION:
            cv2.putText(frame, f"FACE: {facial_emo}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y += 30
        
        # Voice Emotion
        if current_voice_emotion:
            cv2.putText(frame, f"VOICE: {current_voice_emotion}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
            y += 30
        
        # Subtitles
        if current_subtitle:
            h, w, _ = frame.shape
            cv2.putText(frame, current_subtitle, (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Interview System ONNX Full (High Performance)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    stop_flag = True
    time.sleep(0.3)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save logs
    with open("action_log.json", "w", encoding="utf-8") as f:
        json.dump(action_logs, f, ensure_ascii=False, indent=2)
    
    with open("transcription_log.json", "w", encoding="utf-8") as f:
        json.dump(speech_logs, f, ensure_ascii=False, indent=2)
    
    with open("voice_emotion_log.json", "w", encoding="utf-8") as f:
        json.dump(voice_emotion_logs, f, ensure_ascii=False, indent=2)
    
    with open("facial_emotion_log.json", "w", encoding="utf-8") as f:
        json.dump(facial_emotion_logs, f, ensure_ascii=False, indent=2)
    
    # Build combined log
    combined = {}
    for entry in action_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "facial_emotions": [],
                "voice_emotions": []
            }
        combined[sec]["actions"].extend(entry["actions"])
    
    for entry in speech_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "facial_emotions": [],
                "voice_emotions": []
            }
        combined[sec]["texts"].append(entry["text"])
    
    for entry in facial_emotion_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "facial_emotions": [],
                "voice_emotions": []
            }
        combined[sec]["facial_emotions"].append(entry["facial_emotion"])
    
    for entry in voice_emotion_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "facial_emotions": [],
                "voice_emotions": []
            }
        combined[sec]["voice_emotions"].append(entry["voice_emotion"])
    
    combined_list = []
    for sec in sorted(combined.keys()):
        item = combined[sec]
        item["actions"] = sorted(list(set(item["actions"])))
        item["facial_emotions"] = sorted(list(set(item["facial_emotions"])))
        item["voice_emotions"] = sorted(list(set(item["voice_emotions"])))
        combined_list.append(item)
    
    with open("combined_log.json", "w", encoding="utf-8") as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("Saved logs:")
    print("  - action_log.json")
    print("  - transcription_log.json")
    print("  - voice_emotion_log.json")
    print("  - facial_emotion_log.json")
    print("  - combined_log.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
