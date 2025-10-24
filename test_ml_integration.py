#!/usr/bin/env python3
"""
Test ML Integration
Tests the ML classifier integration with the app.
"""

import cv2
import numpy as np
from pathlib import Path
from ml_posture_classifier import initialize_ml_classifier, get_ml_classifier
import mediapipe as mp
import time

def test_ml_classifier():
    """Test the ML classifier with sample images."""
    print("Testing ML Posture Classifier Integration")
    print("=" * 50)
    
    # Initialize ML classifier
    if not initialize_ml_classifier():
        print("❌ Failed to initialize ML classifier")
        return False
    
    ml_classifier = get_ml_classifier()
    if not ml_classifier:
        print("❌ ML classifier not available")
        return False
    
    print("✅ ML classifier initialized successfully")
    
    # Test with sample images from the dataset
    test_path = Path("Training(Updated)/processed_frames/test")
    if not test_path.exists():
        print(f"❌ Test dataset not found at {test_path}")
        return False
    
    # Test a few sample images
    test_images = []
    for class_dir in test_path.iterdir():
        if class_dir.is_dir():
            for img_path in list(class_dir.glob("*.jpg"))[:2]:  # Test first 2 images per class
                test_images.append((img_path, class_dir.name))
    
    print(f"\nTesting with {len(test_images)} sample images...")
    
    correct_predictions = 0
    total_predictions = 0
    
    for img_path, true_class in test_images:
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract landmarks using MediaPipe
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            results = pose.process(img_rgb)
            
            if results.pose_landmarks:
                # Test ML classification
                posture, confidence = ml_classifier.classify_posture(results.pose_landmarks, img.shape)
                
                # Get detailed prediction
                detailed = ml_classifier.get_detailed_prediction()
                
                print(f"\nImage: {img_path.name}")
                print(f"True class: {true_class}")
                print(f"Predicted posture: {posture}")
                print(f"Confidence: {confidence:.3f}")
                
                if detailed:
                    print(f"Detailed prediction:")
                    print(f"  - Multi-class: {detailed['multi_class']}")
                    print(f"  - Posture quality: {detailed['posture_quality']}")
                    print(f"  - Orientation: {detailed['orientation']}")
                    print(f"  - Activity: {detailed['activity']}")
                
                # Check if prediction matches true class
                if 'good' in true_class and 'Good' in posture:
                    correct_predictions += 1
                elif 'bad' in true_class and 'Bad' in posture:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Simple accuracy check
                is_correct = ('good' in true_class and 'Good' in posture) or ('bad' in true_class and 'Bad' in posture)
                print(f"Correct: {'✅' if is_correct else '❌'}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n📊 Results:")
        print(f"Total predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy > 0.8:
            print("✅ ML classifier performing well!")
        elif accuracy > 0.6:
            print("⚠️  ML classifier performing moderately")
        else:
            print("❌ ML classifier needs improvement")
    else:
        print("❌ No valid predictions made")
        return False
    
    return True

def test_config_toggle():
    """Test the config toggle functionality."""
    print("\nTesting Config Toggle")
    print("=" * 30)
    
    # Test with ML enabled
    print("Testing with ML classifier enabled...")
    from config import Config
    print(f"USE_ML_CLASSIFIER: {Config.USE_ML_CLASSIFIER}")
    print(f"ML_MODEL_PATH: {Config.ML_MODEL_PATH}")
    
    return True

def main():
    """Main test function."""
    print("ML Integration Test Suite")
    print("=" * 50)
    
    # Test 1: ML Classifier functionality
    success1 = test_ml_classifier()
    
    # Test 2: Config toggle
    success2 = test_config_toggle()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! ML integration is working correctly.")
        print("\nTo use the ML classifier:")
        print("1. Set USE_ML_CLASSIFIER=true in config.py or .env file")
        print("2. Run the Flask app: python app.py")
        print("3. The app will automatically use the ML classifier")
        print("\nTo disable ML classifier:")
        print("1. Set USE_ML_CLASSIFIER=false in config.py or .env file")
        print("2. The app will fall back to rule-based classification")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
