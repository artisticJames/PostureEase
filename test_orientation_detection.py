#!/usr/bin/env python3
"""
Test Orientation Detection
Quick test to verify side and back orientation detection is working.
"""

import cv2
import numpy as np
from pathlib import Path
from ml_posture_classifier import initialize_ml_classifier, get_ml_classifier
import mediapipe as mp

def test_orientation_detection():
    """Test if the ML model can detect side and back orientations."""
    print("🧪 Testing Orientation Detection")
    print("=" * 40)
    
    # Initialize ML classifier
    if not initialize_ml_classifier():
        print("❌ Failed to initialize ML classifier")
        return False
    
    ml_classifier = get_ml_classifier()
    if not ml_classifier:
        print("❌ ML classifier not available")
        return False
    
    print("✅ ML classifier loaded successfully")
    
    # Test with sample images from different orientations
    test_path = Path("Training(Updated)/processed_frames/test")
    
    orientation_tests = {
        'front': [],
        'side': [],
        'back': []
    }
    
    # Collect test images by orientation
    for class_dir in test_path.iterdir():
        if class_dir.is_dir():
            orientation = class_dir.name.split('_')[-1]  # Get last part (front/side/back)
            if orientation in orientation_tests:
                for img_path in list(class_dir.glob("*.jpg"))[:3]:  # Test first 3 images
                    orientation_tests[orientation].append((img_path, class_dir.name))
    
    print(f"\n📊 Found test images:")
    for orient, images in orientation_tests.items():
        print(f"  {orient}: {len(images)} images")
    
    # Test each orientation
    results = {}
    
    for orientation, test_images in orientation_tests.items():
        if not test_images:
            continue
            
        print(f"\n🎯 Testing {orientation.upper()} orientation:")
        print("-" * 30)
        
        correct_detections = 0
        total_tests = 0
        
        for img_path, true_class in test_images:
            try:
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract landmarks
                mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
                
                results_pose = pose.process(img_rgb)
                
                if results_pose.pose_landmarks:
                    # Get ML prediction
                    posture, confidence = ml_classifier.classify_posture(results_pose.pose_landmarks, img.shape)
                    detailed = ml_classifier.get_detailed_prediction()
                    
                    if detailed:
                        predicted_orientation = detailed['orientation']
                        predicted_activity = detailed['activity']
                        predicted_quality = detailed['posture_quality']
                        
                        # Check if orientation is correctly detected
                        is_correct_orientation = predicted_orientation == orientation
                        if is_correct_orientation:
                            correct_detections += 1
                        
                        total_tests += 1
                        
                        print(f"  {img_path.name}:")
                        print(f"    True: {orientation} | Predicted: {predicted_orientation} {'✅' if is_correct_orientation else '❌'}")
                        print(f"    Activity: {predicted_activity} | Quality: {predicted_quality}")
                        print(f"    Confidence: {confidence:.3f}")
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                continue
        
        if total_tests > 0:
            accuracy = correct_detections / total_tests
            results[orientation] = accuracy
            print(f"\n  📈 {orientation.upper()} Detection Accuracy: {accuracy:.1%} ({correct_detections}/{total_tests})")
        else:
            results[orientation] = 0.0
            print(f"\n  ❌ No valid tests for {orientation}")
    
    # Summary
    print(f"\n🎯 ORIENTATION DETECTION SUMMARY")
    print("=" * 40)
    
    overall_accuracy = sum(results.values()) / len(results) if results else 0
    
    for orientation, accuracy in results.items():
        status = "✅ EXCELLENT" if accuracy >= 0.8 else "⚠️ MODERATE" if accuracy >= 0.5 else "❌ POOR"
        print(f"  {orientation.upper()}: {accuracy:.1%} {status}")
    
    print(f"\n📊 Overall Orientation Accuracy: {overall_accuracy:.1%}")
    
    if overall_accuracy >= 0.7:
        print("🎉 Orientation detection is working well!")
    elif overall_accuracy >= 0.5:
        print("⚠️ Orientation detection needs improvement")
    else:
        print("❌ Orientation detection needs significant improvement")
    
    return overall_accuracy >= 0.5

if __name__ == "__main__":
    test_orientation_detection()
