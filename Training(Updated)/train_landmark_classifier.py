#!/usr/bin/env python3
"""
Landmark-based Posture Classification Training Pipeline
Uses MediaPipe pose landmarks to extract engineered features and train classifiers.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import math
import warnings
warnings.filterwarnings('ignore')

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PostureFeatureExtractor:
    """Extract engineered features from MediaPipe pose landmarks."""
    
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def calculate_distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def extract_features(self, landmarks) -> Dict[str, float]:
        """Extract engineered features from pose landmarks."""
        if not landmarks:
            return self._get_default_features()
        
        # Convert landmarks to pixel coordinates
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y)
        
        try:
            # 1. Neck angle (ear through shoulder to hip)
            neck_angle = self.calculate_angle(
                points[mp_pose.PoseLandmark.RIGHT_EAR.value],
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[mp_pose.PoseLandmark.RIGHT_HIP.value]
            )
            
            # 2. Upper back angle (shoulder through hip to knee)
            upper_back_angle = self.calculate_angle(
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[mp_pose.PoseLandmark.RIGHT_HIP.value],
                points[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            )
            
            # 3. Head tilt (ear to ear through nose)
            head_tilt = self.calculate_angle(
                points[mp_pose.PoseLandmark.LEFT_EAR.value],
                points[mp_pose.PoseLandmark.NOSE.value],
                points[mp_pose.PoseLandmark.RIGHT_EAR.value]
            )
            
            # 4. Shoulder alignment (left to right shoulder)
            shoulder_alignment = self.calculate_angle(
                points[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                (points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] + 0.1, 
                 points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])
            )
            
            # 5. Front angle (nose relative to shoulder midpoint)
            ls = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            nose = points[mp_pose.PoseLandmark.NOSE.value]
            
            shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            vx = nose[0] - shoulder_mid[0]
            vy = nose[1] - shoulder_mid[1]
            front_angle = abs(np.arctan2(vx, -vy) * 180.0 / np.pi)
            
            # 6. Ratios for asymmetry and proportions
            ear_nose_asym = abs(
                self.calculate_distance(points[mp_pose.PoseLandmark.LEFT_EAR.value], nose) -
                self.calculate_distance(points[mp_pose.PoseLandmark.RIGHT_EAR.value], nose)
            )
            
            shoulder_span = self.calculate_distance(ls, rs)
            
            ear_shoulder_r = self.calculate_distance(
                points[mp_pose.PoseLandmark.RIGHT_EAR.value],
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            )
            
            ear_shoulder_l = self.calculate_distance(
                points[mp_pose.PoseLandmark.LEFT_EAR.value],
                points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            )
            
            shoulder_hip_r = self.calculate_distance(
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[mp_pose.PoseLandmark.RIGHT_HIP.value]
            )
            
            shoulder_hip_l = self.calculate_distance(
                points[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                points[mp_pose.PoseLandmark.LEFT_HIP.value]
            )
            
            nose_shoulder_mid = self.calculate_distance(nose, shoulder_mid)
            
            return {
                'angle_neck': neck_angle,
                'angle_upper_back': upper_back_angle,
                'angle_head_tilt': head_tilt,
                'angle_shoulder_alignment': shoulder_alignment,
                'angle_front': front_angle,
                'ratio_ear_nose_asym': ear_nose_asym,
                'ratio_shoulder_span': shoulder_span,
                'ratio_ear_shoulder_r': ear_shoulder_r,
                'ratio_ear_shoulder_l': ear_shoulder_l,
                'ratio_shoulder_hip_r': shoulder_hip_r,
                'ratio_shoulder_hip_l': shoulder_hip_l,
                'ratio_nose_shoulder_mid': nose_shoulder_mid
            }
            
        except (KeyError, IndexError) as e:
            print(f"Error extracting features: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when landmarks are not available."""
        return {
            'angle_neck': 0.0,
            'angle_upper_back': 0.0,
            'angle_head_tilt': 0.0,
            'angle_shoulder_alignment': 0.0,
            'angle_front': 0.0,
            'ratio_ear_nose_asym': 0.0,
            'ratio_shoulder_span': 0.0,
            'ratio_ear_shoulder_r': 0.0,
            'ratio_ear_shoulder_l': 0.0,
            'ratio_shoulder_hip_r': 0.0,
            'ratio_shoulder_hip_l': 0.0,
            'ratio_nose_shoulder_mid': 0.0
        }

def load_dataset(data_path: Path) -> Tuple[List[Dict], List[str]]:
    """Load dataset and extract features from all images."""
    feature_extractor = PostureFeatureExtractor()
    features_list = []
    labels_list = []
    
    # Get all class directories
    train_path = data_path / "processed_frames" / "train"
    test_path = data_path / "processed_frames" / "test"
    
    total_images = 0
    processed_images = 0
    
    for split_path in [train_path, test_path]:
        if not split_path.exists():
            continue
            
        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            print(f"Processing {class_name}...")
            
            for img_path in class_dir.glob("*.jpg"):
                total_images += 1
                
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract landmarks
                results = feature_extractor.pose.process(img_rgb)
                
                if results.pose_landmarks:
                    # Extract features
                    features = feature_extractor.extract_features(results.pose_landmarks)
                    features_list.append(features)
                    labels_list.append(class_name)
                    processed_images += 1
                    
                    if processed_images % 100 == 0:
                        print(f"  Processed {processed_images} images...")
    
    print(f"\nDataset Summary:")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_images}")
    if total_images > 0:
        print(f"Success rate: {processed_images/total_images*100:.1f}%")
    else:
        print("Success rate: N/A (no images found)")
    
    return features_list, labels_list

def train_classifiers(X: np.ndarray, y: List[str]) -> Dict:
    """Train multiple classifiers and return the best one."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'logistic_regression': LogisticRegression(
            max_iter=2000, 
            random_state=42,
            multi_class='ovr'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    results = {}
    
    print("\nTraining and evaluating classifiers...")
    print("=" * 50)
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train model
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = clf.score(X_train_scaled, y_train)
        test_score = clf.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed classification report
        y_pred = clf.predict(X_test_scaled)
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results[name] = {
            'model': clf,
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        # Track best model
        if test_score > best_score:
            best_score = test_score
            best_model = clf
            best_name = name
    
    print(f"\nBest model: {best_name} with test accuracy: {best_score:.4f}")
    
    return {
        'best_model': best_model,
        'best_name': best_name,
        'scaler': scaler,
        'results': results,
        'X_test': X_test,
        'y_test': y_test
    }

def create_task_specific_models(X: np.ndarray, y: List[str]) -> Dict:
    """Create separate models for different tasks."""
    
    # Parse labels to extract task components
    posture_quality = []  # good/bad
    orientation = []     # front/side/back
    activity = []        # sitting/standing
    
    for label in y:
        parts = label.split('_')
        if len(parts) >= 3:
            activity.append(parts[0])  # sitting/standing
            posture_quality.append(parts[1])  # good/bad
            orientation.append(parts[2])  # front/side/back
        else:
            # Fallback for unexpected format
            activity.append('unknown')
            posture_quality.append('unknown')
            orientation.append('unknown')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Extract task labels for training
    train_activity = [activity[i] for i in range(len(y_train))]
    train_posture = [posture_quality[i] for i in range(len(y_train))]
    train_orientation = [orientation[i] for i in range(len(y_train))]
    
    # Train task-specific models
    task_models = {}
    
    # Posture Quality Model (good/bad)
    print("\nTraining Posture Quality Model (good/bad)...")
    posture_model = RandomForestClassifier(n_estimators=100, random_state=42)
    posture_model.fit(X_train_scaled, train_posture)
    task_models['posture_quality'] = posture_model
    
    # Orientation Model (front/side/back)
    print("Training Orientation Model (front/side/back)...")
    orientation_model = RandomForestClassifier(n_estimators=100, random_state=42)
    orientation_model.fit(X_train_scaled, train_orientation)
    task_models['orientation'] = orientation_model
    
    # Activity Model (sitting/standing)
    print("Training Activity Model (sitting/standing)...")
    activity_model = RandomForestClassifier(n_estimators=100, random_state=42)
    activity_model.fit(X_train_scaled, train_activity)
    task_models['activity'] = activity_model
    
    return {
        'models': task_models,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test
    }

def main():
    """Main training pipeline."""
    print("Posture Classification Training Pipeline")
    print("=" * 50)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir  # Current directory is Training(Updated)
    
    print(f"Data path: {data_path}")
    print(f"Looking for dataset in: {data_path / 'processed_frames'}")
    
    # Load dataset
    print("\nLoading dataset and extracting features...")
    features_list, labels_list = load_dataset(data_path)
    
    if not features_list:
        print("No features extracted! Check dataset path and image quality.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    X = df.values
    y = labels_list
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of classes: {len(set(y))}")
    print(f"Classes: {sorted(set(y))}")
    
    # Save feature names
    feature_names = list(df.columns)
    with open(script_dir / "feature_names.json", "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)
    
    print(f"\nFeature names saved to: {script_dir / 'feature_names.json'}")
    
    # Train multi-class classifier
    print("\n" + "="*50)
    print("TRAINING MULTI-CLASS CLASSIFIER")
    print("="*50)
    
    multi_class_results = train_classifiers(X, y)
    
    # Save best multi-class model
    best_model = multi_class_results['best_model']
    scaler = multi_class_results['scaler']
    
    model_path = script_dir / "posture_landmark_classifier.pkl"
    scaler_path = script_dir / "posture_scaler.pkl"
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nBest model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Train task-specific models
    print("\n" + "="*50)
    print("TRAINING TASK-SPECIFIC MODELS")
    print("="*50)
    
    task_results = create_task_specific_models(X, y)
    
    # Save task-specific models
    task_models_path = script_dir / "task_specific_models.pkl"
    joblib.dump(task_results, task_models_path)
    
    print(f"\nTask-specific models saved to: {task_models_path}")
    
    print("\nTraining complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
