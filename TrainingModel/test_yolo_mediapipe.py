import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import math

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def classify_posture(landmarks, image_shape):
    """Classify posture based on multiple body angles and measurements"""
    if not landmarks:
        return "Unknown", 0.0
    
    # Convert landmarks to pixel coordinates
    h, w = image_shape[:2]
    points = {}
    for idx, landmark in enumerate(landmarks.landmark):
        points[idx] = (int(landmark.x * w), int(landmark.y * h))
    
    try:
        # Calculate multiple angles for better accuracy
        
        # 1. Neck angle (ear through shoulder to hip)
        neck_angle = calculate_angle(
            points[mp_pose.PoseLandmark.RIGHT_EAR.value],
            points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        
        # 2. Upper back angle (shoulder through hip to knee)
        upper_back_angle = calculate_angle(
            points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[mp_pose.PoseLandmark.RIGHT_HIP.value],
            points[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        
        # 3. Head tilt (ear to ear through nose)
        head_tilt = calculate_angle(
            points[mp_pose.PoseLandmark.LEFT_EAR.value],
            points[mp_pose.PoseLandmark.NOSE.value],
            points[mp_pose.PoseLandmark.RIGHT_EAR.value]
        )
        
        # 4. Shoulder alignment (left to right shoulder)
        shoulder_angle = calculate_angle(
            points[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            (points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] + 100, 
             points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])  # Point to the right
        )

        # Calculate confidence scores for each metric
        # Neck angle (should be close to 170 degrees for good posture)
        neck_score = 1.0 - min(abs(170 - neck_angle) / 40.0, 1.0)
        
        # Upper back angle (should be close to 175 degrees for good posture)
        back_score = 1.0 - min(abs(175 - upper_back_angle) / 40.0, 1.0)
        
        # Head tilt (should be close to 180 degrees for good posture)
        head_score = 1.0 - min(abs(180 - head_tilt) / 30.0, 1.0)
        
        # Shoulder alignment (should be close to 180 degrees for level shoulders)
        shoulder_score = 1.0 - min(abs(180 - shoulder_angle) / 30.0, 1.0)
        
        # Weight the scores (emphasize neck and back angles more)
        weights = {
            'neck': 0.35,      # 35% weight
            'back': 0.35,      # 35% weight
            'head': 0.15,      # 15% weight
            'shoulder': 0.15    # 15% weight
        }
        
        # Calculate weighted confidence score
        confidence = (
            (neck_score * weights['neck']) +
            (back_score * weights['back']) +
            (head_score * weights['head']) +
            (shoulder_score * weights['shoulder'])
        )
        
        # Clamp between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        # Simplified classification (just good/bad)
        if confidence > 0.7:
            posture = "Good Posture"
        else:
            posture = "Bad Posture"
        
        return posture, confidence
            
    except KeyError:
        return "Unknown", 0.0

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Use the most accurate model
    min_detection_confidence=0.7,  # Increase confidence threshold
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Get COCO class names
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model(frame, verbose=False)  # Disable verbose output
    persons = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Only process if it's a person (class 0)
            if int(box.cls) == 0 and box.conf[0] > 0.7:  # Add confidence threshold
                persons.append(box.xyxy[0].cpu().numpy())

    # Process each detected person
    for person_box in persons:
        x1, y1, x2, y2 = map(int, person_box)
        
        # Ensure the box coordinates are within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Skip if box is too small
        if x2 - x1 < 50 or y2 - y1 < 50:
            continue
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            continue
            
        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        landmark_results = pose.process(person_img_rgb)
        
        if landmark_results.pose_landmarks:
            # Draw landmarks on the person image
            mp_drawing.draw_landmarks(
                person_img,
                landmark_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Classify posture
            posture, confidence = classify_posture(landmark_results.pose_landmarks, person_img.shape)
            
            # Display posture classification and confidence as percentage
            label = f"{posture} ({confidence*100:.1f}%)"
            
            # Simple color coding (green for good, red for bad)
            color = (0, 255, 0) if "Good" in posture else (0, 0, 255)
                
            # Add background rectangle for better text visibility
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1-text_height-10), (x1 + text_width, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Add instructions
    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('PosturEase - Human Posture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 