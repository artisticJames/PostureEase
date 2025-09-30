#!/usr/bin/env python3
"""
Simple Dataset Split Script
Moves 20% of images from train/<category>/ to test/<category>/ for front, side, back categories.
"""

import os
import shutil
import random
from pathlib import Path

def get_image_files(folder_path: str):
    """Get all image files from a folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if not os.path.exists(folder_path):
        return image_files
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    return image_files

def split_category(train_folder: str, test_folder: str, move_percentage: float = 0.2):
    """
    Move 20% of images from train folder to test folder for a specific category.
    
    Args:
        train_folder: Path to train/<category> folder
        test_folder: Path to test/<category> folder  
        move_percentage: Percentage of images to move (0.0 to 1.0)
    """
    if not os.path.exists(train_folder):
        print(f"Train folder does not exist: {train_folder}")
        return 0, 0
    
    # Create test folder if it doesn't exist
    os.makedirs(test_folder, exist_ok=True)
    
    # Get all image files in train folder
    image_files = get_image_files(train_folder)
    
    if not image_files:
        print(f"No images found in {train_folder}")
        return 0, 0
    
    # Calculate number of images to move (20%)
    total_images = len(image_files)
    images_to_move = int(total_images * move_percentage)
    
    if images_to_move == 0:
        print(f"Not enough images to move from {train_folder} (need at least {1/move_percentage:.0f} images)")
        return 0, 0
    
    # Randomly select images to move
    random.shuffle(image_files)
    selected_images = image_files[:images_to_move]
    
    moved_count = 0
    skipped_count = 0
    
    print(f"Moving {images_to_move} out of {total_images} images from {os.path.basename(train_folder)}")
    
    for image_file in selected_images:
        source_path = os.path.join(train_folder, image_file)
        dest_path = os.path.join(test_folder, image_file)
        
        # Skip if file already exists in test folder
        if os.path.exists(dest_path):
            print(f"  Skipping {image_file} (already exists in test)")
            skipped_count += 1
            continue
        
        try:
            shutil.move(source_path, dest_path)
            print(f"  Moved {image_file}")
            moved_count += 1
        except Exception as e:
            print(f"  Error moving {image_file}: {e}")
            skipped_count += 1
    
    return moved_count, skipped_count

def main():
    """Main function to split dataset for front, side, back categories."""
    # Set random seed for reproducible results
    random.seed(42)
    
    # Configuration
    BASE_PATH = "processed_frames"  # Change this to your dataset path
    CATEGORIES = ["front", "side", "back"]
    MOVE_PERCENTAGE = 0.2  # Move 20% of images
    
    print("Simple Dataset Split Script")
    print("=" * 50)
    print(f"Base path: {BASE_PATH}")
    print(f"Categories: {', '.join(CATEGORIES)}")
    print(f"Move percentage: {MOVE_PERCENTAGE*100:.0f}%")
    print()
    
    total_moved = 0
    total_skipped = 0
    
    for category in CATEGORIES:
        train_folder = os.path.join(BASE_PATH, "train", category)
        test_folder = os.path.join(BASE_PATH, "test", category)
        
        print(f"\nProcessing category: {category}")
        print("-" * 30)
        
        moved, skipped = split_category(train_folder, test_folder, MOVE_PERCENTAGE)
        
        total_moved += moved
        total_skipped += skipped
        
        print(f"Result: {moved} moved, {skipped} skipped")
    
    print("\n" + "=" * 50)
    print(f"Dataset split complete!")
    print(f"Total images moved: {total_moved}")
    print(f"Total images skipped: {total_skipped}")

if __name__ == "__main__":
    main()
