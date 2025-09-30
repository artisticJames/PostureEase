#!/usr/bin/env python3
"""
Dataset Split Script
Moves 20% of images from train folders to test folders for each category.
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple

def get_image_files(folder_path: str) -> List[str]:
    """Get all image files from a folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if not os.path.exists(folder_path):
        return image_files
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    return image_files

def move_images_to_test(train_folder: str, test_folder: str, move_percentage: float = 0.2) -> Tuple[int, int]:
    """
    Move a percentage of images from train to test folder.
    
    Args:
        train_folder: Path to train folder
        test_folder: Path to test folder  
        move_percentage: Percentage of images to move (0.0 to 1.0)
    
    Returns:
        Tuple of (moved_count, skipped_count)
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
    
    # Calculate number of images to move
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

def split_dataset(base_path: str = "processed_frames", move_percentage: float = 0.2):
    """
    Split dataset by moving images from train to test folders.
    
    Args:
        base_path: Base path containing train and test folders
        move_percentage: Percentage of images to move (0.0 to 1.0)
    """
    base_path = Path(base_path)
    train_path = base_path / "train"
    test_path = base_path / "test"
    
    if not train_path.exists():
        print(f"Train folder not found: {train_path}")
        return
    
    if not test_path.exists():
        print(f"Test folder not found: {test_path}")
        return
    
    print(f"Splitting dataset: moving {move_percentage*100:.0f}% of images from train to test")
    print("=" * 60)
    
    total_moved = 0
    total_skipped = 0
    categories_processed = 0
    
    # Get all train categories
    train_categories = [d for d in os.listdir(train_path) 
                       if os.path.isdir(train_path / d)]
    
    for category in sorted(train_categories):
        train_category_path = train_path / category
        test_category_path = test_path / category
        
        print(f"\nProcessing category: {category}")
        print("-" * 40)
        
        moved, skipped = move_images_to_test(
            str(train_category_path), 
            str(test_category_path), 
            move_percentage
        )
        
        total_moved += moved
        total_skipped += skipped
        categories_processed += 1
        
        print(f"  Result: {moved} moved, {skipped} skipped")
    
    print("\n" + "=" * 60)
    print(f"Dataset split complete!")
    print(f"Categories processed: {categories_processed}")
    print(f"Total images moved: {total_moved}")
    print(f"Total images skipped: {total_skipped}")

def main():
    """Main function to run the dataset split."""
    # Set random seed for reproducible results
    random.seed(42)
    
    # You can modify these parameters
    BASE_PATH = "processed_frames"  # Change this to your dataset path
    MOVE_PERCENTAGE = 0.2  # Move 20% of images
    
    print("Dataset Split Script")
    print("=" * 60)
    print(f"Base path: {BASE_PATH}")
    print(f"Move percentage: {MOVE_PERCENTAGE*100:.0f}%")
    print()
    
    # Confirm before proceeding
    response = input("Do you want to proceed with the split? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    split_dataset(BASE_PATH, MOVE_PERCENTAGE)

if __name__ == "__main__":
    main()