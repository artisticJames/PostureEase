from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_with_labels(root: Path) -> Tuple[List[Path], List[str]]:
    images: List[Path] = []
    labels: List[str] = []
    if not root.exists():
        return images, labels
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cls = class_dir.name
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                images.append(img_path)
                labels.append(cls)
    return images, labels


def extract_hog_features(paths: List[Path], img_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    features: List[np.ndarray] = []
    for p in paths:
        img = imread(p)
        if img is None:
            features.append(np.zeros(1, dtype=np.float32))
            continue
        if img.ndim == 3:
            img = rgb2gray(img)
        img_resized = resize(img, img_size, anti_aliasing=True)
        feat = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )
        features.append(feat.astype(np.float32))
    return np.vstack(features)


def show_samples_with_predictions(paths: List[Path], y_true: List[str], y_pred: List[str], y_proba: np.ndarray, class_names: List[str], num_samples: int = 10) -> None:
    if not paths:
        return
    idxs = list(range(len(paths)))
    random.shuffle(idxs)
    idxs = idxs[: min(num_samples, len(idxs))]

    cols = len(idxs)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    for ax, i in zip(axes, idxs):
        img = imread(paths[i])
        ax.imshow(img if img.ndim == 3 else img, cmap="gray")
        pred = y_pred[i]
        if y_proba is not None:
            pred_idx = class_names.index(pred)
            conf = float(y_proba[i, pred_idx])
            title = f"pred: {pred} ({conf:.2f})\ntrue: {y_true[i]}"
        else:
            title = f"pred: {pred}\ntrue: {y_true[i]}"
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    train_root = script_dir / "train"
    test_root = script_dir / "test"

    if not train_root.exists() or not test_root.exists():
        print("Expected 'train' and 'test' folders at the same level as this script.")
        return

    print("Loading train set...")
    X_train_paths, y_train = list_images_with_labels(train_root)
    print(f"Train images: {len(X_train_paths)}")
    print("Extracting HOG features for train...")
    X_train = extract_hog_features(X_train_paths)

    print("Loading test set...")
    X_test_paths, y_test = list_images_with_labels(test_root)
    print(f"Test images: {len(X_test_paths)}")
    print("Extracting HOG features for test...")
    X_test = extract_hog_features(X_test_paths)

    if len(set(y_train)) < 2:
        print("Need at least two classes to train a classifier.")
        return

    print("Training Logistic Regression (with probabilities)...")
    clf = LogisticRegression(max_iter=2000, n_jobs=None, solver="lbfgs")
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    class_names = sorted(list(set(y_train)))
    print("\nShowing sample predictions with confidence...")
    show_samples_with_predictions(X_test_paths, y_test, y_pred, y_proba, class_names, num_samples=10)


if __name__ == "__main__":
    main()


