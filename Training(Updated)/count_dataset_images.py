from pathlib import Path
from typing import Dict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images_in_dir(directory: Path) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def collect_class_counts(split_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not split_root.exists():
        return counts
    for class_dir in sorted([p for p in split_root.iterdir() if p.is_dir()]):
        counts[class_dir.name] = count_images_in_dir(class_dir)
    return counts


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir / "processed_frames"

    # Support both layouts:
    # 1) processed_frames/train and processed_frames/test
    # 2) train and test at the project root
    if (dataset_root / "train").exists() or (dataset_root / "test").exists():
        train_root = dataset_root / "train"
        test_root = dataset_root / "test"
    else:
        train_root = script_dir / "train"
        test_root = script_dir / "test"

    train_counts = collect_class_counts(train_root)
    test_counts = collect_class_counts(test_root)

    if not train_counts and not test_counts:
        print("No train/test folders found. Expected 'processed_frames/train' and 'processed_frames/test' or top-level 'train' and 'test'.")
        return

    def print_counts(title: str, counts: Dict[str, int]) -> None:
        print(f"\n{title} ({sum(counts.values())} images total):")
        for cls, cnt in counts.items():
            print(f"  {cls}: {cnt}")

    print_counts("Train", train_counts)
    print_counts("Test", test_counts)


if __name__ == "__main__":
    main()


