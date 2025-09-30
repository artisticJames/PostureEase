from pathlib import Path
from typing import List
import random
import cv2
import matplotlib.pyplot as plt


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def show_random_images(folder: Path, num_images: int = 10) -> None:
    images = list_images(folder)
    if not images:
        print(f"No images found in {folder}")
        return

    sample = images if len(images) <= num_images else random.sample(images, num_images)

    cols = len(sample)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]

    for ax, img_path in zip(axes, sample):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            ax.set_title("Failed to load")
            ax.axis("off")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(img_path.name)
        ax.axis("off")

    fig.suptitle(f"Random samples from: {folder}")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Default path example: processed_frames/train/good_posture
    script_dir = Path(__file__).resolve().parent
    default_folder = script_dir / "processed_frames" / "train" / "good_posture"

    folder = default_folder
    try:
        import sys
        if len(sys.argv) > 1:
            folder = Path(sys.argv[1])
    except Exception:
        pass

     show_random_images(folder, num_images=10)


if __name__ == "__main__":
    main()


