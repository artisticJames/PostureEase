"""
YOLOv8 Training Script for PosturerEaseModel

This script trains a YOLOv8 classification model using the Ultralytics library.
It expects the dataset to be organized in the following structure:

Training Again/yolo_dataset/
    train/
        class1/
            img1.jpg
            ...
        class2/
            ...
    val/
        class1/
            ...
        class2/
            ...
    posture.yaml  # class names

Usage:
    python train_yolo.py

Requirements:
    - ultralytics
    - torch
    - (see requirements.txt)
"""

from ultralytics import YOLO
import os

# Paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'yolo_dataset')
YAML_PATH = os.path.join(DATASET_DIR, 'posture.yaml')
MODEL = 'yolov8n-cls.pt'  # Change to your preferred model

# Training parameters
EPOCHS = 100
BATCH = 32
IMG_SIZE = 224
NAME = 'posture_yolov8_cls'

if __name__ == '__main__':
    # Check dataset and config
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"YAML file not found: {YAML_PATH}")
    if not os.path.exists(MODEL):
        raise FileNotFoundError(f"Model weights not found: {MODEL}")

    # Load YOLO model
    model = YOLO(MODEL)

    # Train
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name=NAME,
        project=os.path.join(os.path.dirname(__file__), 'runs', 'classify'),
        exist_ok=True
    )

    print("Training complete. Results saved in ./runs/classify/")
