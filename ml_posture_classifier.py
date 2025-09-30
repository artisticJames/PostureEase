#!/usr/bin/env python3
"""
ML-based Posture Classifier
Replaces rule-based classification with trained machine learning models.
"""

import os
import cv2
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import mediapipe as mp
from collections import deque
import logging

logger = logging.getLogger(__name__)

class MLPostureClassifier:
    """Machine Learning-based posture classifier using trained models."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the ML posture classifier.
        
        Args:
            model_path: Path to the directory containing trained models
        """
        self.model_path = Path(model_path) if model_path else Path(__file__).parent / "Training(Updated)"
        self.models_loaded = False
        self.temporal_buffer = deque(maxlen=5)  # Buffer for temporal smoothing
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.3,  # Further lowered for better side/back detection
            min_tracking_confidence=0.3   # Further lowered for better side/back detection
        )
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scalers."""
        try:
            # Load multi-class model
            self.multi_class_model = joblib.load(self.model_path / "posture_landmark_classifier.pkl")
            self.scaler = joblib.load(self.model_path / "posture_scaler.pkl")
            
            # Load task-specific models
            task_models = joblib.load(self.model_path / "task_specific_models.pkl")
            self.task_models = task_models['models']
            self.task_scaler = task_models['scaler']
            
            # Load feature names
            with open(self.model_path / "feature_names.json", "r") as f:
                self.feature_names = json.load(f)["feature_names"]
            
            self.models_loaded = True
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.models_loaded = False
    
    def _calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def _calculate_distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _extract_landmark_features(self, landmarks) -> Dict[str, float]:
        """Extract engineered features from pose landmarks."""
        if not landmarks:
            return self._get_default_features()
        
        # Convert landmarks to coordinates
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y)
        
        try:
            # 1. Neck angle (ear through shoulder to hip)
            neck_angle = self._calculate_angle(
                points[self.mp_pose.PoseLandmark.RIGHT_EAR.value],
                points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            )
            
            # 2. Upper back angle (shoulder through hip to knee)
            upper_back_angle = self._calculate_angle(
                points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            )
            
            # 3. Head tilt (ear to ear through nose)
            head_tilt = self._calculate_angle(
                points[self.mp_pose.PoseLandmark.LEFT_EAR.value],
                points[self.mp_pose.PoseLandmark.NOSE.value],
                points[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
            )
            
            # 4. Shoulder alignment (left to right shoulder)
            shoulder_alignment = self._calculate_angle(
                points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                (points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] + 0.1, 
                 points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])
            )
            
            # 5. Front angle (nose relative to shoulder midpoint)
            ls = points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            nose = points[self.mp_pose.PoseLandmark.NOSE.value]
            
            shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            vx = nose[0] - shoulder_mid[0]
            vy = nose[1] - shoulder_mid[1]
            front_angle = abs(np.arctan2(vx, -vy) * 180.0 / np.pi)
            
            # 6. Ratios for asymmetry and proportions
            ear_nose_asym = abs(
                self._calculate_distance(points[self.mp_pose.PoseLandmark.LEFT_EAR.value], nose) -
                self._calculate_distance(points[self.mp_pose.PoseLandmark.RIGHT_EAR.value], nose)
            )
            
            shoulder_span = self._calculate_distance(ls, rs)
            
            ear_shoulder_r = self._calculate_distance(
                points[self.mp_pose.PoseLandmark.RIGHT_EAR.value],
                points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            )
            
            ear_shoulder_l = self._calculate_distance(
                points[self.mp_pose.PoseLandmark.LEFT_EAR.value],
                points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            )
            
            shoulder_hip_r = self._calculate_distance(
                points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            )
            
            shoulder_hip_l = self._calculate_distance(
                points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
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
            
        except (KeyError, IndexError) as e:
            logger.warning(f"Error extracting features: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when landmarks are not available."""
        return {name: 0.0 for name in self.feature_names}
    
    def _calculate_feature_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on feature quality and posture indicators."""
        try:
            # Define ideal ranges for good posture features
            ideal_ranges = {
                'angle_neck': (120, 160),        # Good neck angle
                'angle_upper_back': (150, 180),  # Straight back
                'angle_head_tilt': (170, 190),   # Head straight
                'angle_shoulder_alignment': (170, 190),  # Shoulders level
                'angle_front': (0, 15),          # Facing forward
                'ratio_ear_nose_asym': (0, 0.05), # Symmetrical
                'ratio_shoulder_span': (0.15, 0.25),  # Reasonable shoulder width
                'ratio_ear_shoulder_r': (0.08, 0.15),  # Proportional
                'ratio_ear_shoulder_l': (0.08, 0.15),  # Proportional
                'ratio_shoulder_hip_r': (0.20, 0.35),  # Proportional
                'ratio_shoulder_hip_l': (0.20, 0.35),  # Proportional
                'ratio_nose_shoulder_mid': (0.10, 0.20)  # Proportional
            }
            
            confidence_scores = []
            
            for feature_name, (min_val, max_val) in ideal_ranges.items():
                if feature_name in features:
                    value = features[feature_name]
                    
                    # Calculate how close the value is to the ideal range
                    if min_val <= value <= max_val:
                        # Perfect score for values in ideal range
                        score = 1.0
                    else:
                        # Calculate distance from ideal range
                        if value < min_val:
                            distance = min_val - value
                            range_size = max_val - min_val
                        else:  # value > max_val
                            distance = value - max_val
                            range_size = max_val - min_val
                        
                        # Score decreases with distance from ideal range
                        score = max(0.0, 1.0 - (distance / (range_size * 2)))
                    
                    confidence_scores.append(score)
            
            # Return average confidence, but ensure it's reasonable
            if confidence_scores:
                avg_confidence = np.mean(confidence_scores)
                # Scale to reasonable range (0.3 to 0.9)
                scaled_confidence = 0.3 + (avg_confidence * 0.6)
                return min(0.9, max(0.3, scaled_confidence))
            else:
                return 0.5  # Default moderate confidence
                
        except Exception as e:
            logger.warning(f"Error calculating feature confidence: {e}")
            return 0.5  # Default moderate confidence
    
    def _determine_posture_from_features(self, features: Dict[str, float]) -> str:
        """Determine posture quality based on feature analysis."""
        try:
            # Define thresholds for good posture
            good_posture_thresholds = {
                'angle_neck': (120, 160),        # Good neck angle
                'angle_upper_back': (150, 180),  # Straight back
                'angle_head_tilt': (170, 190),   # Head straight
                'angle_shoulder_alignment': (170, 190),  # Shoulders level
                'angle_front': (0, 15),          # Facing forward
                'ratio_ear_nose_asym': (0, 0.05), # Symmetrical
            }
            
            good_scores = 0
            total_checks = 0
            
            for feature_name, (min_val, max_val) in good_posture_thresholds.items():
                if feature_name in features:
                    value = features[feature_name]
                    total_checks += 1
                    
                    # Check if value is in good posture range
                    if min_val <= value <= max_val:
                        good_scores += 1
            
            # Determine posture based on how many features are in good ranges
            if total_checks > 0:
                good_ratio = good_scores / total_checks
                # If 70% or more features are in good ranges, consider it good posture
                return 'good' if good_ratio >= 0.7 else 'bad'
            else:
                return 'bad'  # Default to bad if no features available
                
        except Exception as e:
            logger.warning(f"Error determining posture from features: {e}")
            return 'bad'  # Default to bad on error
    
    def _apply_temporal_smoothing(self, prediction: str, confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to reduce prediction flicker."""
        # Add current prediction to buffer
        self.temporal_buffer.append((prediction, confidence))
        
        if len(self.temporal_buffer) < 3:
            return prediction, confidence
        
        # Count predictions in buffer
        prediction_counts = {}
        confidence_sum = {}
        
        for pred, conf in self.temporal_buffer:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                confidence_sum[pred] = 0
            prediction_counts[pred] += 1
            confidence_sum[pred] += conf
        
        # Find most frequent prediction
        most_frequent = max(prediction_counts, key=prediction_counts.get)
        avg_confidence = confidence_sum[most_frequent] / prediction_counts[most_frequent]
        
        return most_frequent, avg_confidence
    
    def classify_posture(self, landmarks, image_shape: Tuple[int, int, int]) -> Tuple[str, float]:
        """
        Classify posture using trained ML models.
        
        Args:
            landmarks: MediaPipe pose landmarks
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            Tuple of (posture_class, confidence)
        """
        if not self.models_loaded:
            logger.warning("ML models not loaded, falling back to default")
            return "Unknown", 0.0
        
        if not landmarks:
            return "Unknown", 0.0
        
        try:
            # Extract features
            features = self._extract_landmark_features(landmarks)
            feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(feature_vector)
            
            # Multi-class prediction (kept for debugging/telemetry)
            multi_class_pred = self.multi_class_model.predict(features_scaled)[0]
            multi_class_prob = self.multi_class_model.predict_proba(features_scaled)[0]
            
            # Task-specific predictions
            task_features_scaled = self.task_scaler.transform(feature_vector)
            
            posture_quality_model = self.task_models['posture_quality']
            posture_quality = posture_quality_model.predict(task_features_scaled)[0]
            # Use binary posture quality probability as final confidence
            if hasattr(posture_quality_model, 'predict_proba'):
                pq_proba = posture_quality_model.predict_proba(task_features_scaled)[0]
                # Map probability to the predicted class index
                pq_classes = list(getattr(posture_quality_model, 'classes_', []))
                try:
                    pred_idx = pq_classes.index(posture_quality)
                    raw_confidence = float(pq_proba[pred_idx])
                except Exception:
                    raw_confidence = float(np.max(pq_proba))
                
                # Simple 70% threshold system as per manager requirements
                # Below 70% = Bad Posture, Above 70% = Good Posture
                from config import Config
                threshold = Config.POSTURE_GOOD_CONFIDENCE_THRESHOLD  # 0.7 (70%)
                
                # Use raw model confidence for simple threshold comparison
                raw_confidence = float(np.max(pq_proba))
                
                # Apply simple threshold logic
                if raw_confidence >= threshold:
                    posture_quality = 'good'
                    pq_confidence = raw_confidence  # Use actual confidence
                else:
                    posture_quality = 'bad'
                    pq_confidence = raw_confidence  # Use actual confidence
            else:
                # Fallback if model lacks predict_proba
                pq_confidence = 0.85 if posture_quality == 'good' else 0.15
            orientation = self.task_models['orientation'].predict(task_features_scaled)[0]
            activity = self.task_models['activity'].predict(task_features_scaled)[0]
            
            # Simple confidence system - no complex adjustments
            # Use the raw confidence from the model as-is
            final_confidence = pq_confidence
            
            # Apply temporal smoothing on posture quality prediction
            smoothed_pred, smoothed_conf = self._apply_temporal_smoothing(
                'good' if posture_quality == 'good' else 'bad', final_confidence
            )
            
            # Convert to simple good/bad classification for compatibility
            if 'good' in smoothed_pred:
                simple_class = "Good Posture"
            elif 'bad' in smoothed_pred:
                simple_class = "Bad Posture"
            else:
                simple_class = "Unknown"
            
            # Store detailed predictions for debugging
            self.last_detailed_prediction = {
                'multi_class': multi_class_pred,
                'multi_class_confidence': float(np.max(multi_class_prob)),
                'posture_quality': posture_quality,
                'posture_quality_confidence': pq_confidence,
                'orientation': orientation,
                'activity': activity,
                'confidence': smoothed_conf,
                'raw_confidence': raw_confidence,
                'threshold_used': threshold,
                'classification_logic': f"{'Good' if posture_quality == 'good' else 'Bad'} (confidence: {pq_confidence:.2f}, threshold: {threshold:.2f})"
            }
            
            return simple_class, smoothed_conf
            
        except Exception as e:
            logger.error(f"Error in ML classification: {e}")
            return "Unknown", 0.0
    
    def get_detailed_prediction(self) -> Optional[Dict]:
        """Get the last detailed prediction breakdown."""
        return getattr(self, 'last_detailed_prediction', None)
    
    def reset_temporal_buffer(self):
        """Reset the temporal smoothing buffer."""
        self.temporal_buffer.clear()

# Global instance for the app
ml_classifier = None

def initialize_ml_classifier(model_path: str = None):
    """Initialize the global ML classifier instance."""
    global ml_classifier
    try:
        ml_classifier = MLPostureClassifier(model_path)
        return ml_classifier.models_loaded
    except Exception as e:
        logger.error(f"Failed to initialize ML classifier: {e}")
        return False

def get_ml_classifier():
    """Get the global ML classifier instance."""
    return ml_classifier
