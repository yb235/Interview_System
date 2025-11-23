# Motion and Posture Recognition System

## Overview

The motion recognition system analyzes pose keypoints to detect interview-relevant actions and postures. This document explains how the system recognizes different motions and the algorithms used for detection.

## Detection Framework

### Core Function

All motion detection is handled by the `detect_custom_actions()` function:

```python
def detect_custom_actions(kp):
    """
    Detect interview-related body actions from pose keypoints.
    
    Args:
        kp: numpy array with shape (17, 2) -> (x, y) for each keypoint
        
    Returns:
        list of action strings (e.g., ["arms_crossed", "head_down"])
    """
```

### Keypoint Mapping

The function uses COCO keypoint indices:

```python
nose = kp[0]
left_eye, right_eye = kp[1], kp[2]
left_ear, right_ear = kp[3], kp[4]

l_shoulder, r_shoulder = kp[5], kp[6]
l_elbow, r_elbow = kp[7], kp[8]
l_wrist, r_wrist = kp[9], kp[10]

l_hip, r_hip = kp[11], kp[12]
```

## Distance-Based Detection

### Utility Function

```python
def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))
```

Most actions are detected by measuring distances between keypoints and comparing them to thresholds.

## Recognized Actions

### 1. Arms Crossed

**Description**: Arms folded across the chest, indicating defensiveness or discomfort.

**Detection Logic**:
```python
if (distance(l_wrist, r_elbow) < 80 and 
    distance(r_wrist, l_elbow) < 80):
    actions.append("arms_crossed")
```

**How it works**:
- Left wrist near right elbow AND
- Right wrist near left elbow
- Threshold: 80 pixels

**Visual**:
```
    L_elbow -------- R_wrist
        \              /
         \            /
          \          /
           L_wrist--R_elbow
```

### 2. Hands Clasped

**Description**: Hands held together, often indicating nervousness or formality.

**Detection Logic**:
```python
if distance(l_wrist, r_wrist) < 60:
    actions.append("hands_clasped")
```

**How it works**:
- Left wrist close to right wrist
- Threshold: 60 pixels
- Detects when hands are touching or very close

### 3. Chin Rest

**Description**: Hand supporting chin, indicating boredom or deep thought.

**Detection Logic**:
```python
if distance(l_wrist, nose) < 70 or distance(r_wrist, nose) < 70:
    actions.append("chin_rest")
```

**How it works**:
- Either wrist near the nose
- Threshold: 70 pixels
- May also trigger for other face-touching gestures near nose

### 4. Lean Forward

**Description**: Body leaning forward, indicating engagement or aggression.

**Detection Logic**:
```python
shoulder_center = (
    (l_shoulder[0] + r_shoulder[0]) / 2,
    (l_shoulder[1] + r_shoulder[1]) / 2
)
hip_center = (
    (l_hip[0] + r_hip[0]) / 2,
    (l_hip[1] + r_hip[1]) / 2
)

torso_height = abs(shoulder_center[1] - hip_center[1])
if torso_height < 120:
    actions.append("lean_forward")
```

**How it works**:
- Calculates vertical distance between shoulder center and hip center
- Shorter distance = more compressed torso = leaning forward
- Threshold: < 120 pixels

**Note**: This assumes the person is sitting upright in the default position.

### 5. Lean Back

**Description**: Body leaning backward, indicating relaxation or disengagement.

**Detection Logic**:
```python
torso_height = abs(shoulder_center[1] - hip_center[1])
if torso_height > 200:
    actions.append("lean_back")
```

**How it works**:
- Larger vertical distance between shoulders and hips
- Threshold: > 200 pixels
- Indicates person is stretching or leaning back

### 6. Head Down

**Description**: Head lowered, indicating lack of confidence or sadness.

**Detection Logic**:
```python
if nose[1] > shoulder_center[1] + 40:
    actions.append("head_down")
```

**How it works**:
- Nose y-coordinate below shoulder center
- Threshold: 40 pixels below shoulders
- Uses y-axis where larger values are lower on screen

### 7. Touch Face

**Description**: Hand touching face area, often a nervous gesture.

**Detection Logic**:
```python
face_center = (
    (left_eye[0] + right_eye[0]) / 2,
    (left_eye[1] + right_eye[1]) / 2
)
if distance(l_wrist, face_center) < 70 or distance(r_wrist, face_center) < 70:
    actions.append("touch_face")
```

**How it works**:
- Calculates face center from eye positions
- Either wrist within 70 pixels of face center
- General face-touching detection

### 8. Touch Nose

**Description**: Hand touching nose specifically, potentially indicating deception or stress.

**Detection Logic**:
```python
if distance(l_wrist, nose) < 40 or distance(r_wrist, nose) < 40:
    actions.append("touch_nose")
```

**How it works**:
- More specific than "touch_face"
- Tighter threshold: 40 pixels
- Specifically targets nose keypoint

**Note**: Can co-occur with "chin_rest" or "touch_face"

### 9. Fix Hair

**Description**: Hand near ears/hair, indicating grooming or nervousness.

**Detection Logic**:
```python
if (distance(l_wrist, left_ear) < 60 or 
    distance(r_wrist, right_ear) < 60 or
    distance(l_wrist, right_ear) < 60 or 
    distance(r_wrist, left_ear) < 60):
    actions.append("fix_hair")
```

**How it works**:
- Checks all combinations of wrists and ears
- Threshold: 60 pixels
- Detects grooming behaviors

### 10. Fidget Hands

**Description**: Rapid hand movements, indicating nervousness or restlessness.

**Detection Logic**:
```python
global prev_left_wrist, prev_right_wrist

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
```

**How it works**:
- Tracks wrist positions across frames
- Compares current position to previous frame
- Threshold: > 25 pixels of movement
- Requires frame-to-frame memory

**Important**: This is the only temporal detection (requires state across frames)

## Detection Parameters Summary

| Action | Keypoints Used | Threshold | Type |
|--------|---------------|-----------|------|
| Arms Crossed | Wrists, Elbows | 80px | Distance |
| Hands Clasped | Both Wrists | 60px | Distance |
| Chin Rest | Wrist, Nose | 70px | Distance |
| Lean Forward | Shoulders, Hips | < 120px | Height |
| Lean Back | Shoulders, Hips | > 200px | Height |
| Head Down | Nose, Shoulders | 40px offset | Position |
| Touch Face | Wrist, Eyes | 70px | Distance |
| Touch Nose | Wrist, Nose | 40px | Distance |
| Fix Hair | Wrists, Ears | 60px | Distance |
| Fidget Hands | Wrists (temporal) | 25px/frame | Motion |

## Threshold Tuning

### Why These Values?

Thresholds are calibrated for typical webcam setups:
- **Resolution**: 640x480 to 1920x1080
- **Distance**: Person sitting 50-100cm from camera
- **View**: Upper body visible (head to waist minimum)

### Adjusting Thresholds

To tune for different setups:

```python
# For closer camera or higher resolution
THRESHOLD_SCALE = 1.5
if distance(l_wrist, r_wrist) < 60 * THRESHOLD_SCALE:
    actions.append("hands_clasped")

# For farther camera or lower resolution
THRESHOLD_SCALE = 0.7
if distance(l_wrist, r_wrist) < 60 * THRESHOLD_SCALE:
    actions.append("hands_clasped")
```

## Multi-Action Detection

### Duplicate Removal

```python
actions = list(set(actions))  # Remove duplicates
```

Multiple actions can be detected simultaneously:
- "touch_nose" and "chin_rest" can both trigger
- "touch_face" and "fix_hair" can co-occur
- "arms_crossed" and "lean_back" can combine

### Example Output

```python
# Single frame might detect:
["arms_crossed", "lean_back", "fidget_hands"]
```

## State Management

### Global State Variables

```python
prev_left_wrist = None   # Previous left wrist position
prev_right_wrist = None  # Previous right wrist position
```

These are needed for temporal detection (fidget detection).

### State Initialization

```python
# On first frame or reset
prev_left_wrist = None
prev_right_wrist = None
```

### State Update

```python
# At end of detect_custom_actions()
prev_left_wrist = l_wrist
prev_right_wrist = r_wrist
```

## Performance Considerations

### Computational Complexity

Each action detection is O(1):
- Simple distance calculations
- No loops or recursive operations
- Very fast (~0.1ms for all 10 actions)

### Frame Rate Impact

Action detection has negligible impact:
- Main bottleneck is pose inference (~50-150ms)
- Action detection adds < 1ms per frame

## Coordinate System

### OpenCV Coordinate System

```
(0,0) -----------------> X (width)
  |
  |
  |
  v
  Y (height)
```

- X increases to the right
- Y increases downward
- Origin is top-left corner

### Important Implications

1. **Vertical comparisons**: 
   - `y1 > y2` means point1 is BELOW point2
   - This is why "head_down" uses `nose[1] > shoulder_center[1]`

2. **Distance is symmetric**:
   - `distance(p1, p2) == distance(p2, p1)`
   - Direction doesn't matter for distance-based detection

## Debugging Actions

### Visualizing Detection

```python
def detect_custom_actions(kp):
    # ... detection logic ...
    
    # Debug: print detected actions
    if actions:
        print(f"Detected: {actions}")
    
    return list(set(actions))
```

### Drawing Keypoints

```python
# In main loop
for r in results:
    for person in r.keypoints.xy:
        kp = person.cpu().numpy()
        
        # Draw keypoint connections
        nose, l_wrist, r_wrist = kp[0], kp[9], kp[10]
        
        # Draw distance line (for debugging)
        cv2.line(frame, 
                tuple(l_wrist.astype(int)), 
                tuple(r_wrist.astype(int)), 
                (255, 0, 0), 2)
        
        # Show distance value
        dist = distance(l_wrist, r_wrist)
        mid = ((l_wrist + r_wrist) / 2).astype(int)
        cv2.putText(frame, f"{dist:.0f}", tuple(mid),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
```

### Logging Thresholds

```python
def detect_custom_actions(kp):
    # ... extract keypoints ...
    
    # Log distances for calibration
    wrist_dist = distance(l_wrist, r_wrist)
    torso_height = abs(shoulder_center[1] - hip_center[1])
    
    print(f"Wrist distance: {wrist_dist:.1f}")
    print(f"Torso height: {torso_height:.1f}")
    
    # ... continue with detection ...
```

## Limitations

### Current System Limitations

1. **Single Person**: Optimized for one person in frame
2. **Fixed Thresholds**: Not adaptive to person size or distance
3. **Occlusion**: May fail if keypoints are occluded
4. **Side Views**: Designed for frontal/slightly angled views
5. **Sitting Position**: Some actions assume seated posture

### False Positives

Common false positive scenarios:
- **Arms Crossed**: May trigger when gesturing
- **Fidget Hands**: May trigger during normal speech gestures
- **Touch Face**: May trigger when drinking or eating
- **Lean Forward/Back**: Sensitive to person's height and sitting position

### False Negatives

Common false negative scenarios:
- **Arms Crossed**: May miss if arms are loosely crossed
- **Fidget Hands**: May miss subtle movements < 25px
- **Head Down**: May miss if person has naturally low head position

## Interview Relevance

### Behavioral Insights

Each action has psychological implications:

| Action | Interpretation | Interview Impact |
|--------|---------------|------------------|
| Arms Crossed | Defensive, closed-off | Negative |
| Hands Clasped | Nervous, formal | Neutral |
| Chin Rest | Bored, thinking | Mixed |
| Lean Forward | Engaged, interested | Positive |
| Lean Back | Relaxed, disinterested | Mixed |
| Head Down | Unconfident, sad | Negative |
| Touch Face | Nervous, self-soothing | Negative |
| Touch Nose | Stressed, possibly deceptive | Negative |
| Fix Hair | Grooming, nervous | Negative |
| Fidget Hands | Restless, anxious | Negative |

### Usage in Analysis

The action logs can be analyzed to:
- Identify stress patterns during specific questions
- Track engagement levels over time
- Correlate body language with speech content
- Provide feedback on non-verbal communication
