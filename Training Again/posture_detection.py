"""Real-time posture detection using MediaPipe + Ultralytics YOLOv8.

This script reads from webcam, extracts pose keypoints using MediaPipe and runs a
YOLO classification model (trained with keypoints or image-level classifier) to
predict posture class. It falls back gracefully if the model file is missing.
"""

import argparse
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import sys

try:
	from ultralytics import YOLO
except Exception:
	YOLO = None


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def draw_posture_status(image, prediction: str, conf: float):
	text = f"Posture: {prediction} ({conf:.2f})"
	color = (0, 255, 0) if "Good" in prediction else (0, 0, 255)
	cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
	cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)


def main(model_path: Path):
	pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

	if YOLO is None:
		print("Ultralytics YOLO not installed. Install 'ultralytics' to run this script.")
		print("Falling back to drawing only MediaPipe landmarks.")
		model = None
	else:
		if model_path and model_path.exists():
			print(f"Loading YOLO model from {model_path}")
			model = YOLO(str(model_path))
		else:
			print(f"Model not found at {model_path}. Continuing without classifier.")
			model = None

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Couldn't open webcam. Exiting.")
		sys.exit(1)

	try:
		while cap.isOpened():
			ok, frame = cap.read()
			if not ok:
				continue

			frame = cv2.flip(frame, 1)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = pose.process(rgb)

			if results.pose_landmarks:
				mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

				if model is not None:
					# Pass the RGB image to the model; works if model is an image classifier
					yolo_results = model(rgb)
					# Try to get top prediction in a few safe ways
					try:
						# ultralytics v8 Results object: .probs.top1 / .probs.top1conf
						class_name = yolo_results[0].names[yolo_results[0].probs.top1]
						conf = float(yolo_results[0].probs.top1conf)
					except Exception:
						try:
							# fallback: .boxes or .pred
							pred = yolo_results[0]
							# If classifier, .names and .probs may be present
							if hasattr(pred, 'probs'):
								idx = int(pred.probs.argmax())
								class_name = pred.names[idx]
								conf = float(pred.probs.max())
							else:
								# as a last resort present raw string
								class_name = str(pred)
								conf = 0.0
						except Exception:
							class_name, conf = "unknown", 0.0

					draw_posture_status(frame, class_name, conf)

			cv2.imshow('PostureEase', frame)
			if cv2.waitKey(5) & 0xFF == ord('q'):
				break

	finally:
		pose.close()
		cap.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('--model', type=Path, default=Path('runs/classify/train3/weights/best.pt'), help='Path to YOLO weights/classifier')
	args = p.parse_args()
	main(args.model)
