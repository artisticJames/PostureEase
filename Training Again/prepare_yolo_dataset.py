"""Prepare dataset files for YOLO training.

Creates a `yolo_dataset` folder (if missing) with `train.txt` and `val.txt` listing
absolute paths to images. This script is idempotent and safe to run multiple times.

Behavior:
- If `yolo_dataset/train` and `yolo_dataset/val` don't exist but `dataset/train` exists,
  it will call `organize_dataset.organize_dataset` to create the splits.
- Writes `yolo_dataset/train.txt` and `yolo_dataset/val.txt` with one image path per line.
"""

from pathlib import Path
import argparse
import sys
import random


def build_image_list(split_dir: Path):
	"""Return sorted list of image file paths (jpg, jpeg, png) under split_dir."""
	exts = ("*.jpg", "*.jpeg", "*.png")
	images = []
	for e in exts:
		images.extend(split_dir.rglob(e))
	images = [p.resolve() for p in images if p.is_file()]
	images.sort()
	return images


def write_list_file(path: Path, items):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		for p in items:
			f.write(str(p) + "\n")


def main(root: Path):
	project_root = root.resolve()
	yolo_dir = project_root / "yolo_dataset"
	alt_split = project_root / "yolo_dataset_split"

	# If yolo_dataset/train and val exist, use them. Otherwise try to build from dataset/train
	train_dir = yolo_dir / "train"
	val_dir = yolo_dir / "val"

	# If a separate standardized split folder exists, prefer it
	if not (train_dir.exists() and val_dir.exists()):
		if alt_split.exists():
			# support alt_split/train/images and alt_split/val/images
			alt_train_images = alt_split / 'train' / 'images'
			alt_val_images = alt_split / 'val' / 'images'
			alt_train_labels = alt_split / 'train' / 'labels'
			alt_val_labels = alt_split / 'val' / 'labels'
			if alt_train_images.exists() and alt_val_images.exists():
				print(f"Detected alternate split at {alt_split}. Using that to build lists.")
				train_images = build_image_list(alt_train_images)
				val_images = build_image_list(alt_val_images)

				# Basic validation: counts of images vs labels (if labels exist)
				def count_basenames(folder):
					return {p.stem for p in folder.iterdir() if p.is_file()} if folder.exists() else set()

				train_img_basenames = {p.stem for p in train_images}
				train_lbl_basenames = count_basenames(alt_train_labels)
				val_img_basenames = {p.stem for p in val_images}
				val_lbl_basenames = count_basenames(alt_val_labels)

				if train_lbl_basenames and len(train_img_basenames - train_lbl_basenames) > 0:
					missing = list(sorted(train_img_basenames - train_lbl_basenames))[:5]
					print(f"Warning: {len(train_img_basenames - train_lbl_basenames)} train images without labels. Examples: {missing}")

				if val_lbl_basenames and len(val_img_basenames - val_lbl_basenames) > 0:
					missing = list(sorted(val_img_basenames - val_lbl_basenames))[:5]
					print(f"Warning: {len(val_img_basenames - val_lbl_basenames)} val images without labels. Examples: {missing}")

				# Write lists and finish
				write_list_file(yolo_dir / "train.txt", train_images)
				write_list_file(yolo_dir / "val.txt", val_images)
				print(f"Wrote {len(train_images)} train and {len(val_images)} val image paths to {yolo_dir}")
				return
			else:
				print(f"Found {alt_split} but missing images/labels subfolders. Falling back.")

		dataset_train = project_root / "dataset" / "train"
		if dataset_train.exists():
			# Call organize_dataset (import local module) to create yolo_dataset splits if available
			try:
				from organize_dataset import organize_dataset

				print("Creating yolo_dataset train/val splits from dataset/train using organize_dataset...")
				organize_dataset(dataset_train, yolo_dir)
			except Exception as e:
				print("Failed to automatically organize dataset:", e)
				print("Please run `organize_dataset.py` manually or create yolo_dataset/train and yolo_dataset/val directories.")
				sys.exit(1)
		else:
			print("Neither yolo_dataset splits nor dataset/train nor yolo_dataset_split found. Nothing to prepare.")
			sys.exit(1)

	# Build file lists
	train_images = build_image_list(train_dir)
	val_images = build_image_list(val_dir)

	if len(train_images) == 0 and len(val_images) == 0:
		print("No images found under yolo_dataset/train or yolo_dataset/val. Exiting.")
		sys.exit(1)

	# Shuffle for randomness but keep reproducible
	random.seed(42)
	random.shuffle(train_images)
	random.shuffle(val_images)

	write_list_file(yolo_dir / "train.txt", train_images)
	write_list_file(yolo_dir / "val.txt", val_images)

	print(f"Wrote {len(train_images)} train and {len(val_images)} val image paths to {yolo_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Prepare YOLO dataset file lists")
	parser.add_argument("--root", default=Path(__file__).parent, type=Path, help="Project root directory")
	args = parser.parse_args()
	main(args.root)
