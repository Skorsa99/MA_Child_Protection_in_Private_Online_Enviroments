"""
Evaluate class-wise accuracy for a saved Keras model on a folder of images.

Configure the settings block below, then run:
    python3 test_model_class_biases.py

The data directory should contain subfolders named exactly like the model's
classes (e.g. "safe", "unsafe", ...). Images can be nested in subfolders.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Match the training setup so legacy-keras models load cleanly
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
DATA_DIR: Path = Path("data/reddit_pics")   # Root folder that contains one subfolder per class
MODEL_DIR: Path = Path("models/V_4_2")      # Folder containing the saved model + labels.json
MODEL_FILE: Path | None = None              # Optional explicit path to a .keras or .h5 file
LABELS_PATH: Path | None = None             # Optional explicit labels.json path
IMAGE_SIZE: int = 256                       # Resize images to IMAGE_SIZE x IMAGE_SIZE
BATCH_SIZE: int = 32                        # Number of images per prediction batch
MAX_IMAGES_PER_CLASS: int = None            # Limit images evaluated per class (None = all)
# --------------------------------------------------------------------------------------

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".jfif",
    ".heic",
}


def load_labels(labels_path: Path) -> List[str]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Could not find labels file at {labels_path}")

    with labels_path.open("r", encoding="utf-8") as fh:
        labels = json.load(fh)

    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"labels.json must be a list of strings, got: {labels}")
    return labels


def find_model_file(model_dir: Path, explicit_model: Path | None) -> Path:
    if explicit_model:
        return explicit_model

    keras_candidates = sorted(model_dir.glob("*.keras"))
    if keras_candidates:
        return keras_candidates[0]

    h5_candidates = sorted(model_dir.glob("*.h5"))
    if h5_candidates:
        return h5_candidates[0]

    raise FileNotFoundError(
        f"No .keras or .h5 model file found in {model_dir}. "
        "Use --model-file to point at a specific file."
    )


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def load_batch(paths: Sequence[Path], image_size: int) -> np.ndarray:
    batch = []
    for img_path in paths:
        try:
            img = tf.keras.utils.load_img(
                img_path, color_mode="rgb", target_size=(image_size, image_size)
            )
            arr = tf.keras.utils.img_to_array(img, dtype=np.float32) / 255.0
            batch.append(arr)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Skipping {img_path} ({exc})")
    if not batch:
        return np.empty((0, image_size, image_size, 3), dtype=np.float32)
    return np.stack(batch)


def evaluate_class(
    model: tf.keras.Model,
    class_name: str,
    class_idx: int,
    folder: Path,
    image_size: int,
    batch_size: int,
    max_images: int | None,
) -> Tuple[int, int]:
    paths = list(iter_images(folder))
    paths.sort()

    if max_images is not None:
        paths = paths[:max_images]

    total = len(paths)
    if total == 0:
        print(f"- {class_name}: no images found under {folder}")
        return 0, 0

    correct = 0
    for start in range(0, total, batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = load_batch(batch_paths, image_size)
        if batch.size == 0:
            continue

        preds = model.predict(batch, verbose=0)
        pred_idx = np.argmax(preds, axis=1)

        correct += int((pred_idx == class_idx).sum())

    return correct, total


def main() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    labels_path = LABELS_PATH or (MODEL_DIR / "model_tfjs" / "labels.json")
    labels = load_labels(labels_path)
    label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}

    model_path = find_model_file(MODEL_DIR, MODEL_FILE)
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded classes ({len(labels)}): {', '.join(labels)}")

    results: Dict[str, Tuple[int, int]] = {}
    for class_name in labels:
        class_folder = DATA_DIR / class_name
        if not class_folder.exists():
            print(f"- {class_name}: folder not found at {class_folder}, skipping")
            continue

        correct, total = evaluate_class(
            model=model,
            class_name=class_name,
            class_idx=label_to_idx[class_name],
            folder=class_folder,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            max_images=MAX_IMAGES_PER_CLASS,
        )
        results[class_name] = (correct, total)

    print("\nPer-class accuracy:")
    for class_name in labels:
        correct, total = results.get(class_name, (0, 0))
        if total == 0:
            print(f"- {class_name}: no samples evaluated")
            continue
        acc = correct / total
        print(f"- {class_name}: {correct}/{total} correct ({acc:.2%})")


if __name__ == "__main__":
    main()
