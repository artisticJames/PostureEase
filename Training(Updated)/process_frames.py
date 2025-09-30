from pathlib import Path
from typing import List
import cv2


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    frames_root = (script_dir / "frames") if (script_dir / "frames").exists() else (script_dir / "Frames")
    output_root = script_dir / "processed_frames"
    output_root.mkdir(parents=True, exist_ok=True)

    if not frames_root.exists():
        print(f"Input folder not found: {frames_root}")
        return

    images = list_images(frames_root)
    if not images:
        print("No images found in frames directory.")
        return

    target_size = (224, 224)
    count = 0

    for img_path in images:
        rel = img_path.relative_to(frames_root)
        out_path = output_root / rel
        ensure_parent(out_path)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), resized)
        count += 1

    print(f"Processed {count} images to {output_root}")


if __name__ == "__main__":
    main()


