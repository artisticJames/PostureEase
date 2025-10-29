import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize YOLO model
model = YOLO('runs/classify/train3/weights/best.pt')

def draw_posture_status(image, prediction, conf):
    """Draw posture status on image."""
    text = f"Posture: {prediction} ({conf:.2f})"
    color = (0, 255, 0) if "Good" in prediction else (0, 0, 255)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add instructions
    cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read from webcam")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect poses
        results = pose.process(image_rgb)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Make prediction on the frame
            results = model(image_rgb)
            class_name = results[0].names[results[0].probs.top1]
            confidence = results[0].probs.top1conf
            
            # Draw the prediction on the image
            draw_posture_status(image, class_name, confidence)
        
        # Show the image
        cv2.imshow('PostureEase - Real-time Posture Detection', image)
        
        # Exit on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    pose.close()
    cap.release()
    cv2.destroyAllWindows()