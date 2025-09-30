import os
from pathlib import Path
from typing import Optional, List, Dict, Set
import cv2


def ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def find_existing_path(root: Path, relative_path: str) -> Optional[Path]:
    candidate = root / relative_path
    if candidate.exists():
        return candidate
    # Try case-insensitive resolution by walking components
    current = root
    for part in Path(relative_path).parts:
        if not current.exists() or not current.is_dir():
            return None
        matches = [p for p in current.iterdir() if p.name.lower() == part.lower()]
        if not matches:
            return None
        current = matches[0]
    return current


def list_videos_in_directory(directory: Path) -> List[Path]:
    if not directory or not directory.exists():
        return []
    exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts])


def compute_frame_interval(fps: float) -> int:
    # Aim for ~1 frame per second; handle edge cases robustly
    if fps is None or fps != fps or fps <= 0:
        return 30  # conservative default
    # Round to nearest int, but ensure at least 1
    interval = int(round(fps))
    return max(interval, 1)


def extract_frames_from_video(video_path: Path, output_dir: Path, filename_prefix: str, start_index: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = compute_frame_interval(fps)

    saved_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_interval == 1:
            # Save every second based on time if fps ~1; use timestamp seek for better spacing
            # But fall back to modulo when timestamp seek is unreliable
            pass  # modulo path below handles this uniformly

        if frame_index % frame_interval == 0:
            out_name = f"{filename_prefix}_{start_index + saved_count}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    return saved_count


def main():
    # Default dataset root is the directory containing this script
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir

    # Decide frames root: prefer existing 'Frames', else use 'frames'
    frames_root = dataset_root / "Frames"
    if not frames_root.exists():
        frames_root = dataset_root / "frames"

    # Define categories and map to source directories (support both cases)
    categories = [
        ("Sitting/Good", "sitting_good"),
        ("Sitting/Bad", "sitting_bad"),
        ("Standing/Good", "standing_good"),
        ("Standing/Bad", "standing_bad"),
        ("sitting/good", "sitting_good"),
        ("sitting/bad", "sitting_bad"),
        ("standing/good", "standing_good"),
        ("standing/bad", "standing_bad"),
    ]

    # Initialize counters per output category
    counters: Dict[str, int] = {
        "sitting_good": 0,
        "sitting_bad": 0,
        "standing_good": 0,
        "standing_bad": 0,
    }

    # Ensure output directories exist
    for out_name in counters.keys():
        ensure_dir(frames_root / out_name)

    # Process each category (skip duplicates by tracking resolved paths)
    seen_src_dirs: Set[Path] = set()

    for src_rel, out_name in categories:
        src_dir = find_existing_path(dataset_root, src_rel)
        if src_dir is None or src_dir in seen_src_dirs:
            continue
        seen_src_dirs.add(src_dir)

        videos = list_videos_in_directory(src_dir)
        if not videos:
            continue

        output_dir = frames_root / out_name
        ensure_dir(output_dir)

        for video in videos:
            start_idx = counters[out_name]
            saved = extract_frames_from_video(video, output_dir, out_name, start_idx)
            counters[out_name] += saved

    # Optional: print summary to console
    for out_name, count in counters.items():
        print(f"Saved {count} frames to {frames_root / out_name}")


if __name__ == "__main__":
    main()


