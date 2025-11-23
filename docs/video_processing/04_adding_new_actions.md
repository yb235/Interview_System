# Adding New Motion Posture Recognition

## Overview

This guide walks you through the process of adding new actions to the motion recognition system. Follow these steps to extend the system with custom posture detection.

## Step-by-Step Guide

### Step 1: Understand the Keypoint Structure

First, familiarize yourself with the available keypoints:

```python
# COCO 17-keypoint format
kp[0]  = nose
kp[1]  = left_eye
kp[2]  = right_eye
kp[3]  = left_ear
kp[4]  = right_ear
kp[5]  = left_shoulder
kp[6]  = right_shoulder
kp[7]  = left_elbow
kp[8]  = right_elbow
kp[9]  = left_wrist
kp[10] = right_wrist
kp[11] = left_hip
kp[12] = right_hip
kp[13] = left_knee
kp[14] = right_knee
kp[15] = left_ankle
kp[16] = right_ankle
```

### Step 2: Define Your Action

Clearly describe what you want to detect:
- **Action name**: e.g., "hand_wave"
- **Description**: What does this action look like?
- **Keypoints involved**: Which body parts are relevant?
- **Detection criteria**: What makes this action unique?

### Step 3: Implement Detection Logic

Add your detection to the `detect_custom_actions()` function in the interview system file.

#### Example 1: Hand Wave

**Goal**: Detect when a person waves their hand (hand raised above shoulder)

```python
def detect_custom_actions(kp):
    # ... existing code ...
    
    # Extract keypoints
    l_shoulder, r_shoulder = kp[5], kp[6]
    l_wrist, r_wrist = kp[9], kp[10]
    
    actions = []
    
    # ... existing actions ...
    
    # 11. Hand Wave (new action)
    # Detect if either wrist is above the shoulder line
    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    
    if l_wrist[1] < shoulder_y - 50 or r_wrist[1] < shoulder_y - 50:
        actions.append("hand_wave")
    
    return list(set(actions))
```

**Explanation**:
- Calculate average shoulder height
- Check if wrist is significantly above (y < shoulder_y - 50)
- The 50-pixel buffer prevents false positives from normal arm positions

#### Example 2: Scratching Head

**Goal**: Detect when hand is on top of head

```python
def detect_custom_actions(kp):
    # ... existing code ...
    
    nose = kp[0]
    l_wrist, r_wrist = kp[9], kp[10]
    
    actions = []
    
    # ... existing actions ...
    
    # 11. Scratching Head (new action)
    # Hand near top of head (above nose)
    if (l_wrist[1] < nose[1] - 20 and distance(l_wrist, nose) < 80) or \
       (r_wrist[1] < nose[1] - 20 and distance(r_wrist, nose) < 80):
        actions.append("scratching_head")
    
    return list(set(actions))
```

**Explanation**:
- Wrist must be above nose (y < nose[1] - 20)
- Wrist must be close to head (distance < 80)
- Checks both left and right hands

#### Example 3: Slouching

**Goal**: Detect poor posture (shoulders significantly below normal)

```python
def detect_custom_actions(kp):
    # ... existing code ...
    
    l_shoulder, r_shoulder = kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]
    nose = kp[0]
    
    actions = []
    
    # ... existing actions ...
    
    # 11. Slouching (new action)
    # Calculate shoulder center and hip center
    shoulder_center = (
        (l_shoulder[0] + r_shoulder[0]) / 2,
        (l_shoulder[1] + r_shoulder[1]) / 2
    )
    hip_center = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    )
    
    # Slouching: shoulders too close to hips AND head forward
    torso_height = abs(shoulder_center[1] - hip_center[1])
    if torso_height < 100 and nose[1] > shoulder_center[1]:
        actions.append("slouching")
    
    return list(set(actions))
```

**Explanation**:
- Compressed torso (torso_height < 100)
- Head forward relative to shoulders
- Combines multiple conditions for accuracy

#### Example 4: Pointing

**Goal**: Detect when person is pointing (arm extended)

```python
def detect_custom_actions(kp):
    # ... existing code ...
    
    l_shoulder, r_shoulder = kp[5], kp[6]
    l_elbow, r_elbow = kp[7], kp[8]
    l_wrist, r_wrist = kp[9], kp[10]
    
    actions = []
    
    # ... existing actions ...
    
    # 11. Pointing (new action)
    # Arm extended: shoulder-elbow-wrist roughly aligned and stretched
    
    # Check left arm
    left_arm_length = distance(l_shoulder, l_elbow) + distance(l_elbow, l_wrist)
    left_direct_dist = distance(l_shoulder, l_wrist)
    left_straightness = left_direct_dist / left_arm_length if left_arm_length > 0 else 0
    
    # Check right arm
    right_arm_length = distance(r_shoulder, r_elbow) + distance(r_elbow, r_wrist)
    right_direct_dist = distance(r_shoulder, r_wrist)
    right_straightness = right_direct_dist / right_arm_length if right_arm_length > 0 else 0
    
    # If arm is mostly straight (ratio > 0.85), consider it pointing
    if left_straightness > 0.85 or right_straightness > 0.85:
        actions.append("pointing")
    
    return list(set(actions))
```

**Explanation**:
- Calculates arm straightness ratio
- Ratio close to 1.0 means arm is fully extended
- Works for any direction of pointing

### Step 4: Test Your Action

#### Manual Testing

1. Run the interview system
2. Perform the action in front of camera
3. Check if it's detected in real-time

```python
# In main loop, temporary debug code:
if "your_action_name" in frame_actions:
    print(f"[DEBUG] Action detected at {ts:.2f}s")
```

#### Calibration Testing

Find the right thresholds:

```python
def detect_custom_actions(kp):
    # ... extract keypoints ...
    
    # Temporary debug output
    test_distance = distance(l_wrist, nose)
    print(f"[CALIBRATE] Wrist-nose distance: {test_distance:.1f}")
    
    # Try different thresholds
    if test_distance < 70:  # Adjust this value
        actions.append("your_action")
```

### Step 5: Optimize Thresholds

Use this script to find optimal thresholds:

```python
# Save as calibrate_action.py
import cv2
import numpy as np
from ultralytics import YOLO

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

model = YOLO("yolo11m-pose.pt")
cap = cv2.VideoCapture(0)

print("Perform your action. Press 'c' to capture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, device="cpu", verbose=False)
    
    for r in results:
        if r.keypoints is None:
            continue
        for person in r.keypoints.xy:
            kp = person.cpu().numpy()
            
            # Calculate your distance/metric
            l_wrist = kp[9]
            nose = kp[0]
            dist = distance(l_wrist, nose)
            
            # Display on frame
            cv2.putText(frame, f"Distance: {dist:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Calibration", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        print(f"Captured: {dist:.1f}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Step 6: Document Your Action

Add documentation to the action detection section:

```python
# 11. Your Action Name
# Description: Brief description of what this action represents
# Psychological interpretation: What does this mean in an interview context?
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good
actions.append("looking_away")
actions.append("hands_on_table")

# Bad
actions.append("action1")
actions.append("a11")
```

### 2. Avoid Overlapping Detections

Make your action distinct from existing ones:

```python
# Problem: Too similar to existing "touch_face"
if distance(l_wrist, nose) < 100:  # Too broad
    actions.append("nose_scratch")

# Solution: Make it more specific
if 40 < distance(l_wrist, nose) < 60 and prev_distance < 40:
    actions.append("nose_scratch")  # Detected movement toward nose
```

### 3. Consider Temporal Aspects

For actions involving movement:

```python
global prev_keypoints, frame_counter

# Initialize on first call
if prev_keypoints is None:
    prev_keypoints = kp.copy()
    frame_counter = 0
    return []

frame_counter += 1

# Detect movement every N frames
if frame_counter % 5 == 0:
    movement = distance(kp[9], prev_keypoints[9])
    if movement > 30:
        actions.append("rapid_hand_movement")
    prev_keypoints = kp.copy()
```

### 4. Handle Edge Cases

```python
# Check for valid keypoints
if np.any(np.isnan(kp)) or np.any(kp == 0):
    return []  # Invalid keypoints, skip detection

# Handle occlusion
if distance(l_wrist, [0, 0]) < 5:  # Likely occluded
    return []  # Don't try to detect with occluded keypoints
```

### 5. Use Relative Measurements

Prefer relative measurements over absolute ones:

```python
# Good: Relative to body size
shoulder_width = distance(l_shoulder, r_shoulder)
threshold = shoulder_width * 0.8  # 80% of shoulder width

# Less good: Fixed threshold
threshold = 100  # May not work for different distances/resolutions
```

## Common Patterns

### Pattern 1: Proximity Detection

When two body parts are close:

```python
if distance(point_a, point_b) < threshold:
    actions.append("action_name")
```

### Pattern 2: Relative Position

When one part is above/below/beside another:

```python
# Above
if point_a[1] < point_b[1] - offset:
    actions.append("action_name")

# Below
if point_a[1] > point_b[1] + offset:
    actions.append("action_name")

# Left of
if point_a[0] < point_b[0] - offset:
    actions.append("action_name")
```

### Pattern 3: Region Detection

When a point is within a region:

```python
# Define region (rectangle)
region_x1, region_y1 = 100, 100
region_x2, region_y2 = 300, 300

if (region_x1 < point[0] < region_x2 and 
    region_y1 < point[1] < region_y2):
    actions.append("in_region")
```

### Pattern 4: Angle Detection

When detecting specific angles:

```python
def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

# Detect arm bent at 90 degrees
elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
if 80 < elbow_angle < 100:
    actions.append("arm_bent_90")
```

### Pattern 5: Multiple Conditions

Combining multiple criteria:

```python
# Both conditions must be true
if (distance(l_wrist, nose) < 50 and 
    l_wrist[1] < nose[1] and
    l_elbow[1] < l_shoulder[1]):
    actions.append("specific_gesture")
```

## Complete Example: Adding "Phone Gesture"

Here's a complete example adding detection for holding a phone to the ear:

```python
def detect_custom_actions(kp):
    """
    Detect interview-related body actions from pose keypoints.
    
    Args:
        kp: numpy array with shape (17, 2) -> (x, y) for each keypoint
        
    Returns:
        list of action strings
    """
    global prev_left_wrist, prev_right_wrist
    
    # Extract keypoints
    nose = kp[0]
    left_eye, right_eye = kp[1], kp[2]
    left_ear, right_ear = kp[3], kp[4]
    
    l_shoulder, r_shoulder = kp[5], kp[6]
    l_elbow, r_elbow = kp[7], kp[8]
    l_wrist, r_wrist = kp[9], kp[10]
    
    l_hip, r_hip = kp[11], kp[12]
    
    actions = []
    
    # Calculate common reference points
    shoulder_center = (
        (l_shoulder[0] + r_shoulder[0]) / 2,
        (l_shoulder[1] + r_shoulder[1]) / 2
    )
    hip_center = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    )
    
    # ... existing actions (1-10) ...
    
    # 11. Phone Gesture (NEW)
    # Hand near ear, elbow bent, indicating phone call gesture
    
    # Check left hand
    left_ear_dist = distance(l_wrist, left_ear)
    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    
    if left_ear_dist < 80 and 70 < left_elbow_angle < 110:
        actions.append("phone_gesture")
    
    # Check right hand
    right_ear_dist = distance(r_wrist, right_ear)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    
    if right_ear_dist < 80 and 70 < right_elbow_angle < 110:
        actions.append("phone_gesture")
    
    # Update state for fidget detection
    prev_left_wrist = l_wrist
    prev_right_wrist = r_wrist
    
    return list(set(actions))


def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3 in degrees"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)
```

## Troubleshooting

### Action Not Detecting

1. **Verify keypoints are visible**
   - Print keypoint values
   - Check if they're not [0, 0] (occluded)

2. **Adjust thresholds**
   - Start with loose thresholds
   - Gradually tighten based on testing

3. **Check coordinate system**
   - Remember Y increases downward
   - Verify your comparison operators

### Too Many False Positives

1. **Add more conditions**
   - Require multiple criteria to be true
   - Use temporal consistency

2. **Tighten thresholds**
   - Reduce distance tolerances
   - Increase angle requirements

3. **Add exclusion logic**
   - Prevent conflicting actions
   - Check for impossible combinations

### Inconsistent Detection

1. **Add frame buffering**
   - Detect action over multiple frames
   - Use moving average

2. **Handle occlusion**
   - Skip detection if keypoints unreliable
   - Use previous frame data

3. **Normalize for distance**
   - Calculate thresholds relative to body size
   - Use ratios instead of absolute distances

## Testing Checklist

Before finalizing your new action:

- [ ] Action detects when performed
- [ ] No false positives during normal movement
- [ ] Works at different distances from camera
- [ ] Works at different angles (within reason)
- [ ] Doesn't conflict with existing actions
- [ ] Thresholds are documented
- [ ] Code includes comments
- [ ] Action name is descriptive
- [ ] Psychological interpretation documented

## Next Steps

After adding your action:
1. Test thoroughly in various scenarios
2. Update the documentation in `03_motion_recognition.md`
3. Add the action to the interview analysis logic
4. Consider adding visualization for debugging
5. Share your findings and thresholds with the team
