#!/usr/bin/env python3
"""
Model Evaluation Script
Tests trained models and shows performance metrics.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mediapipe as mp

# MediaPipe setup
mp_pose = mp.solutions.pose

class ModelEvaluator:
    """Evaluate trained posture classification models."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.load_models()
        
    def load_models(self):
        """Load trained models and scaler."""
        try:
            # Load multi-class model
            self.multi_class_model = joblib.load(self.model_dir / "posture_landmark_classifier.pkl")
            self.scaler = joblib.load(self.model_dir / "posture_scaler.pkl")
            
            # Load task-specific models
            task_models = joblib.load(self.model_dir / "task_specific_models.pkl")
            self.task_models = task_models['models']
            self.task_scaler = task_models['scaler']
            
            # Load feature names
            with open(self.model_dir / "feature_names.json", "r") as f:
                self.feature_names = json.load(f)["feature_names"]
                
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def extract_features_from_image(self, img_path: Path) -> np.ndarray:
        """Extract features from a single image."""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return None
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        results = pose.process(img_rgb)
        
        if not results.pose_landmarks:
            return None
            
        # Extract features (same as training)
        features = self._extract_landmark_features(results.pose_landmarks)
        return np.array([features[f] for f in self.feature_names])
    
    def _extract_landmark_features(self, landmarks):
        """Extract features from landmarks (same as training)."""
        # Convert landmarks to coordinates
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y)
        
        try:
            # Calculate angles and ratios (same as training script)
            neck_angle = self._calculate_angle(
                points[mp_pose.PoseLandmark.RIGHT_EAR.value],
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[mp_pose.PoseLandmark.RIGHT_HIP.value]
            )
            
            upper_back_angle = self._calculate_angle(
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[mp_pose.PoseLandmark.RIGHT_HIP.value],
                points[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            )
            
            head_tilt = self._calculate_angle(
                points[mp_pose.PoseLandmark.LEFT_EAR.value],
                points[mp_pose.PoseLandmark.NOSE.value],
                points[mp_pose.PoseLandmark.RIGHT_EAR.value]
            )
            
            shoulder_alignment = self._calculate_angle(
                points[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                (points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] + 0.1, 
                 points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])
            )
            
            # Front angle
            ls = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            nose = points[mp_pose.PoseLandmark.NOSE.value]
            
            shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            vx = nose[0] - shoulder_mid[0]
            vy = nose[1] - shoulder_mid[1]
            front_angle = abs(np.arctan2(vx, -vy) * 180.0 / np.pi)
            
            # Ratios
            ear_nose_asym = abs(
                self._calculate_distance(points[mp_pose.PoseLandmark.LEFT_EAR.value], nose) -
                self._calculate_distance(points[mp_pose.PoseLandmark.RIGHT_EAR.value], nose)
            )
            
            shoulder_span = self._calculate_distance(ls, rs)
            ear_shoulder_r = self._calculate_distance(
                points[mp_pose.PoseLandmark.RIGHT_EAR.value],
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            )
            ear_shoulder_l = self._calculate_distance(
                points[mp_pose.PoseLandmark.LEFT_EAR.value],
                points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            )
            shoulder_hip_r = self._calculate_distance(
                points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[mp_pose.PoseLandmark.RIGHT_HIP.value]
            )
            shoulder_hip_l = self._calculate_distance(
                points[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                points[mp_pose.PoseLandmark.LEFT_HIP.value]
            )
            nose_shoulder_mid = self._calculate_distance(nose, shoulder_mid)
            
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
            
        except (KeyError, IndexError):
            return {name: 0.0 for name in self.feature_names}
    
    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def _calculate_distance(self, a, b):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def predict_single_image(self, img_path: Path):
        """Predict posture for a single image."""
        features = self.extract_features_from_image(img_path)
        if features is None:
            return None, None, None
            
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Multi-class prediction
        multi_class_pred = self.multi_class_model.predict(features_scaled)[0]
        multi_class_prob = self.multi_class_model.predict_proba(features_scaled)[0]
        
        # Task-specific predictions
        task_features_scaled = self.task_scaler.transform(features.reshape(1, -1))
        
        posture_pred = self.task_models['posture_quality'].predict(task_features_scaled)[0]
        orientation_pred = self.task_models['orientation'].predict(task_features_scaled)[0]
        activity_pred = self.task_models['activity'].predict(task_features_scaled)[0]
        
        return {
            'multi_class': multi_class_pred,
            'multi_class_prob': multi_class_prob,
            'posture_quality': posture_pred,
            'orientation': orientation_pred,
            'activity': activity_pred
        }
    
    def evaluate_on_test_set(self, test_path: Path):
        """Evaluate models on test dataset."""
        print("Evaluating models on test set...")
        
        predictions = []
        true_labels = []
        
        # Process test images
        for class_dir in sorted(test_path.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            print(f"Processing {class_name}...")
            
            for img_path in class_dir.glob("*.jpg"):
                pred = self.predict_single_image(img_path)
                if pred is not None:
                    predictions.append(pred)
                    true_labels.append(class_name)
        
        if not predictions:
            print("No valid predictions made!")
            return
        
        # Extract multi-class predictions
        multi_class_preds = [p['multi_class'] for p in predictions]
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, multi_class_preds)
        print(f"\nMulti-class Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, multi_class_preds))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, multi_class_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Multi-class Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.model_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }

def main():
    """Main evaluation function."""
    print("Model Evaluation")
    print("=" * 30)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent
    
    # Initialize evaluator
    evaluator = ModelEvaluator(script_dir)
    
    # Evaluate on test set
    test_path = data_path / "Training(Updated)" / "processed_frames" / "test"
    if test_path.exists():
        results = evaluator.evaluate_on_test_set(test_path)
    else:
        print(f"Test path not found: {test_path}")
    
    # Test single image prediction
    print("\nTesting single image prediction...")
    sample_img = test_path / "sitting_good_front" / "sitting_good_front_1.jpg"
    if sample_img.exists():
        pred = evaluator.predict_single_image(sample_img)
        if pred:
            print(f"Sample prediction: {pred}")
    else:
        print("No sample image found for testing")

if __name__ == "__main__":
    main()
