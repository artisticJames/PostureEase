import os
import shutil
from pathlib import Path
import random

def organize_dataset(source_dir: Path, output_dir: Path, train_ratio: float = 0.8):
    """Organize dataset into train and val splits for YOLO training."""
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in source_dir.glob("*") if d.is_dir()]
    
    for class_dir in class_dirs:
        # Create corresponding directories in train and val
        (train_dir / class_dir.name).mkdir(exist_ok=True)
        (val_dir / class_dir.name).mkdir(exist_ok=True)
        
        # Get all images in the class
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        
        # Split into train and val
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(img, train_dir / class_dir.name / img.name)
        
        for img in val_images:
            shutil.copy2(img, val_dir / class_dir.name / img.name)
    
    # Create data.yaml
    with open(output_dir / "data.yaml", "w") as f:
        f.write(f"train: {train_dir}\n")
        f.write(f"val: {val_dir}\n")
        f.write(f"nc: {len(class_dirs)}\n")
        f.write("names: [" + ", ".join(f"'{d.name}'" for d in class_dirs) + "]")

if __name__ == "__main__":
    source_dir = Path("dataset/train")
    output_dir = Path("yolo_dataset")
    random.seed(42)
    organize_dataset(source_dir, output_dir)