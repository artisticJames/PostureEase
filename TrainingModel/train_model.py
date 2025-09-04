import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import cv2
from ultralytics import YOLO
import mediapipe as mp

# Load YOLO model for person detection
yolo_model = YOLO('yolov8n.pt')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_persons_and_landmarks(image):
    # Detect persons using YOLO
    results = yolo_model(image)
    persons = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # Class 0 is person in COCO dataset
                persons.append(box.xyxy[0].cpu().numpy())
    
    landmarks_list = []
    for person_box in persons:
        x1, y1, x2, y2 = map(int, person_box)
        person_img = image[y1:y2, x1:x2]
        if person_img.size == 0:
            continue
        # Convert to RGB for MediaPipe
        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        # Get landmarks using MediaPipe
        landmark_results = pose.process(person_img_rgb)
        if landmark_results.pose_landmarks:
            landmarks = []
            for landmark in landmark_results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks_list.append(landmarks)
    return landmarks_list

def prepare_data():
    # Read the CSV file
    df = pd.read_csv('pose_landmarks.csv')
    
    # Convert landmarks string to numpy array
    df['landmarks'] = df['landmarks'].apply(eval)
    
    # Create features (X) and target (y)
    X = np.array(df['landmarks'].tolist())
    
    # Create binary labels: 1 for good posture, 0 for bad posture
    y = (df['category'] == 'good').astype(int)
    
    return X, y

def train_model():
    # Prepare data
    X, y = prepare_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
    
    print("Training the model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print model performance
    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'posture_model.pkl')
    print("\nModel saved as 'posture_model.pkl'")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'landmark_{i}' for i in range(X.shape[1])],
        'importance': model.feature_importances_
    })
    print("\nTop 10 most important landmarks:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))

if __name__ == "__main__":
    train_model() 