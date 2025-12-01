"""
Evaluate class-wise accuracy for a saved Keras model on a folder of images.

Configure the settings block below, then run:
    python3 test_model_class_biases.py

The data directory should contain subfolders named exactly like the model's
classes (e.g. "safe", "unsafe", ...). Images can be nested in subfolders.
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Match the training setup so legacy-keras models load cleanly
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from custom_logging import log_extensive_testing

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


def _get_ourdir(logs_dir="logs/model_testing_logs") -> str:
    os.makedirs(logs_dir, exist_ok=True)
    pattern = re.compile(rf"test_(\d+)$")
    subversions = []

    for name in os.listdir(logs_dir):
        match = pattern.match(name)
        if match:
            subversions.append(int(match.group(1)))

    next_sub = max(subversions) + 1 if subversions else 0
    return f"{logs_dir}/test_{next_sub}"

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
) -> Tuple[int, int, List[int], List[int]]:
    paths = list(iter_images(folder))
    paths.sort()

    if max_images is not None:
        paths = paths[:max_images]

    total_images = len(paths)
    if total_images == 0:
        print(f"- {class_name}: no images found under {folder}")
        return 0, 0, [], []

    correct = 0
    actual_total = 0
    true_labels: List[int] = []
    pred_labels: List[int] = []
    for start in range(0, total_images, batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = load_batch(batch_paths, image_size)
        if batch.size == 0:
            continue

        preds = model.predict(batch, verbose=0)
        pred_idx = np.argmax(preds, axis=1)

        batch_size_actual = len(pred_idx)
        actual_total += batch_size_actual

        correct += int((pred_idx == class_idx).sum())
        true_labels.extend([class_idx] * batch_size_actual)
        pred_labels.extend(pred_idx.tolist())

    return correct, actual_total, true_labels, pred_labels


def test_label_accuracy():
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
    all_true: List[int] = []
    all_pred: List[int] = []
    for class_name in labels:
        class_folder = DATA_DIR / class_name
        if not class_folder.exists():
            print(f"- {class_name}: folder not found at {class_folder}, skipping")
            continue

        correct, total, class_true, class_pred = evaluate_class(
            model=model,
            class_name=class_name,
            class_idx=label_to_idx[class_name],
            folder=class_folder,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            max_images=MAX_IMAGES_PER_CLASS,
        )
        results[class_name] = (correct, total)
        all_true.extend(class_true)
        all_pred.extend(class_pred)

    label_results = []

    print("\nPer-class accuracy:")
    for class_name in labels:
        correct, total = results.get(class_name, (0, 0))
        if total == 0:
            print(f"- {class_name}: no samples evaluated")
            continue
        acc = correct / total
        print(f"- {class_name}: {correct}/{total} correct ({acc:.2%})")
        label_results.append(f"{class_name}: {correct}/{total} correct ({acc:.2%})")

    return label_results, labels, all_true, all_pred


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    out_dir: Path,
) -> Path:
    """
    Build and save a confusion matrix plot for the given predictions.
    """
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t, p] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(6, num_classes * 1.2), max(4, num_classes * 0.9))
    )
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
            )

    fig.tight_layout()
    out_path = out_dir / "confusion_matrix.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out_dir = Path(_get_ourdir())

    log_extensive_testing(out_dir, f"--- Test Started -----------------------------------")
    log_extensive_testing(out_dir, f"Model Tested  : {MODEL_DIR}")
    log_extensive_testing(out_dir, f"Dataset       : {DATA_DIR}")
    log_extensive_testing(out_dir, f"Batch Size    : {BATCH_SIZE}")
    log_extensive_testing(out_dir, f"Test-Set-Size : {MAX_IMAGES_PER_CLASS}")
    log_extensive_testing(out_dir, f"----------------------------------------------------")

    label_results, labels, y_true, y_pred = test_label_accuracy()
    log_extensive_testing(out_dir, f"Accuracy per Label:")
    for label in label_results:
        log_extensive_testing(out_dir, f" - {label}")

    if y_true and y_pred:
        cm_path = save_confusion_matrix(y_true, y_pred, labels, out_dir)
        log_extensive_testing(out_dir, f"Confusion matrix saved to: {cm_path}")
    else:
        log_extensive_testing(out_dir, "Confusion matrix not created (no predictions).")
