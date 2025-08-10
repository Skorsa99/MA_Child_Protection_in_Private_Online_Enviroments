

#!/usr/bin/env python3
"""
Train a lightweight 3-class CNN for on-device (browser) inference.

Targets:
- Classes: ["unsafe", "safe", "empty"]
- Extremely lightweight → MobileNetV3Small with width multiplier (alpha)
- Export to TensorFlow.js for running in the browser
- Uses JSONL manifest produced by create_manifest.py

Expected JSONL fields per line:
  {"path": "unsafe/source/img.jpg", "label": "unsafe", "label_id": 0, "source": "source", "split": "train"}

Usage:
  Just set the CONFIG section below and run:
    python train_model.py

Outputs:
  - ./exports/keras/best_model.keras          (Keras saved model)
  - ./exports/tfjs/                           (TensorFlow.js model.json + weights)
  - ./exports/history.json                    (training curves)
  - ./exports/metrics.json                    (eval metrics)

Notes:
  - If tensorflowjs is not installed, TF.js export is skipped with a hint.
  - You can tweak alpha/img_size/augments to trade accuracy vs. size/speed.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np, tensorflow as tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from custom_logging import log_data_training


# =====================
# CONFIG
# =====================

# Dataset root the manifest paths are relative to
DATASET_ROOT = Path("data/reddit_pics")

# Path to the manifest.jsonl to use (versioned folder supported). Adjust as needed.
MANIFEST_PATH = Path("data/manifests/V001_80_10_10_per_source/manifest.jsonl")

# Class names in label_id order
CLASS_NAMES = ["unsafe", "safe", "empty"]
NUM_CLASSES = len(CLASS_NAMES)

# Model / training
IMG_SIZE = 160              # 128–192 is a good range; 160 is a balanced default
WIDTH_MULT = 0.75           # MobileNetV3 width multiplier (0.75/1.0)
BATCH_SIZE = 64
EPOCHS_HEAD = 4             # train classifier head with base frozen
EPOCHS_FINE = 8             # fine-tune top layers
BASE_LR = 1e-3
FINE_TUNE_LR = 2e-4
DROPOUT = 0.2
SEED = 42
IMAGE_NORMALIZER = "resize" # "resize" | "crop" | "bars"

def _get_version(manual_number, base_dir):
    # Find subfolders matching pattern V_{manual}_{auto}
    subfolders = [
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith(f"V_{manual_number}_")
    ]

    # Extract the part after the second underscore and turn into int
    nums = []
    for folder in subfolders:
        parts = folder.name.split("_")
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            nums.append(int(parts[2]))

    next_count = (max(nums) + 1) if nums else 1

    return f"V_{manual_number}_{next_count:03d}"

# Export dirs
EXPORT_DIR = Path("models")
MANUAL_VERSION = 0
VERSION = _get_version(MANUAL_VERSION, EXPORT_DIR)
KERAS_DIR = EXPORT_DIR / VERSION
TFJS_DIR = EXPORT_DIR / VERSION

# Optional metadata flags from your manifest generator
INCLUDE_SHA256 = False
INCLUDE_SIZE = False
INCLUDE_IMAGE_DIMS = False

AUTOTUNE = tf.data.AUTOTUNE


# =====================
# Helpers
# =====================

def _read_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _filter_split(items: List[dict], split: str) -> List[dict]:
    return [x for x in items if x.get("split") == split]


def _abs_paths_and_labels(records: List[dict], dataset_root: Path) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    for r in records:
        p = dataset_root / r["path"]
        paths.append(str(p))
        labels.append(int(r["label_id"]))
    return paths, labels


def _make_class_weights(train_labels: List[int], num_classes: int) -> Dict[int, float]:
    """Balanced class weights: total / (num_classes * count_c)."""
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = {}
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        weights[c] = (total / (num_classes * cnt)) if cnt > 0 else 0.0
    return weights


def _build_datasets(manifest_path: Path, dataset_root: Path, img_size: int, batch_size: int, seed: int):
    items = _read_jsonl(manifest_path)
    train_items = _filter_split(items, "train")
    val_items = _filter_split(items, "val")
    test_items = _filter_split(items, "test")

    train_paths, train_labels = _abs_paths_and_labels(train_items, dataset_root)
    val_paths, val_labels = _abs_paths_and_labels(val_items, dataset_root)
    test_paths, test_labels = _abs_paths_and_labels(test_items, dataset_root)

    def decode(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)

        if (IMAGE_NORMALIZER == "resize"):
            img = tf.image.resize(img, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR)
        elif (IMAGE_NORMALIZER == "crop"):
            # Resize so the shorter side == img_size, preserve aspect ratio
            h = tf.shape(img)[0]; w = tf.shape(img)[1]
            scale = tf.cast(img_size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
            new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
            new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
            img = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)
            img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)  # center crop to square
        elif (IMAGE_NORMALIZER == "bars"):
            img = tf.image.resize_with_pad(img, img_size, img_size, method=tf.image.ResizeMethod.BILINEAR)
        else:
            raise ValueError(f"Unknown IMAGE_NORMALIZER: {IMAGE_NORMALIZER}")

        img = tf.cast(img, tf.float32)
        return img, tf.cast(label, tf.int32)

    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.1),
    ], name="augment")

    def preprocess(img, label):
        # MobileNetV3 expects inputs scaled to [-1, 1]
        img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
        return img, label

    def make_ds(paths, labels, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(buffer_size=min(8192, len(paths)), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(train_paths, train_labels, training=True)
    val_ds   = make_ds(val_paths,   val_labels,   training=False)
    test_ds  = make_ds(test_paths,  test_labels,  training=False)

    class_weights = _make_class_weights(train_labels, NUM_CLASSES)

    info = {
        "num_train": len(train_paths),
        "num_val": len(val_paths),
        "num_test": len(test_paths),
        "class_weights": class_weights,
    }
    return train_ds, val_ds, test_ds, info


# =====================
# Model
# =====================

def build_model(img_size: int, num_classes: int, width_mult: float, dropout: float) -> keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 3))

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=width_mult,
        pooling=None,
    )
    base.trainable = False

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

def compile_model(model: keras.Model, lr: float):
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )


# Helper: Compute confusion matrix and macro precision/recall for multi-class
def eval_confmat_and_pr(ds, model, num_classes: int):
    """Return (cm, per_class_precision, per_class_recall, macro_precision, macro_recall)."""
    y_true, y_pred = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.append(y.numpy())
        y_pred.append(np.argmax(p, axis=1))
    y_true = np.concatenate(y_true) if len(y_true) else np.array([], dtype=np.int32)
    y_pred = np.concatenate(y_pred) if len(y_pred) else np.array([], dtype=np.int32)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy().astype(int)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    macro_p = float(precision.mean())
    macro_r = float(recall.mean())
    return cm, precision.tolist(), recall.tolist(), macro_p, macro_r


# =====================
# Train & Evaluate
# =====================

class LogBestValAcc(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best = -float("inf")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get("val_acc") or logs.get("val_accuracy")
        if val is not None and val > self.best:
            self.best = val
            log_data_training(f"New best val_acc: {val:.4f} at epoch {epoch+1}")

def train_and_eval():
    # Pull in global variables
    global DATASET_ROOT
    global MANIFEST_PATH
    global CLASS_NAMES
    global NUM_CLASSES
    global IMG_SIZE
    global WIDTH_MULT
    global BATCH_SIZE
    global EPOCHS_HEAD
    global EPOCHS_FINE
    global BASE_LR
    global FINE_TUNE_LR
    global DROPOUT
    global SEED
    global IMAGE_NORMALIZER

    # Data
    train_ds, val_ds, test_ds, info = _build_datasets(MANIFEST_PATH, DATASET_ROOT, IMG_SIZE, BATCH_SIZE, SEED)
    log_data_training(f"Data loaded | train: {info['num_train']} val: {info['num_val']} test: {info['num_test']}")

    # Model
    model = build_model(IMG_SIZE, NUM_CLASSES, WIDTH_MULT, DROPOUT)
    compile_model(model, BASE_LR)
    log_data_training(f"Model built | MobileNetV3Small alpha={WIDTH_MULT}, img={IMG_SIZE}, dropout={DROPOUT}")

    # Callbacks
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    KERAS_DIR.mkdir(parents=True, exist_ok=True)
    TFJS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = KERAS_DIR / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]
    callbacks.append(LogBestValAcc())
    log_data_training(f"Checkpoint → {ckpt_path}")

    # Class weights for imbalance
    class_weights = info["class_weights"]

    # Phase 1: train head
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        verbose=1,
        callbacks=callbacks,
    )
    try:
        best1 = max(history1.history.get('val_acc', []))
        log_data_training(f"Phase 1 done | best val_acc: {best1:.4f}")
    except Exception:
        pass

    # Phase 2: fine-tune top layers
    # Unfreeze last ~30% of layers
    base = model.layers[1]  # the MobileNetV3 layer
    if isinstance(base, keras.Model):
        n = len(base.layers)
        cutoff = int(n * 0.7)
        log_data_training(f"Fine-tuning: unfreezing top {n - cutoff} / {n} layers")
        for l in base.layers[cutoff:]:
            if not isinstance(l, layers.BatchNormalization):
                l.trainable = True
    compile_model(model, FINE_TUNE_LR)
    log_data_training("Fine-tuning started")

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weights,
        verbose=1,
        callbacks=callbacks,
    )
    try:
        best2 = max(history2.history.get('val_acc', []))
        log_data_training(f"Phase 2 done | best val_acc: {best2:.4f}")
    except Exception:
        pass

    # Evaluate
    eval_val = model.evaluate(val_ds, verbose=0)
    eval_test = model.evaluate(test_ds, verbose=0)

    metrics = {
        "val": {m.name if hasattr(m, "name") else m: float(v) for m, v in zip(model.metrics, eval_val)},
        "test": {m.name if hasattr(m, "name") else m: float(v) for m, v in zip(model.metrics, eval_test)},
        "info": info,
    }
    va = metrics.get('val', {})
    te = metrics.get('test', {})
    def _fmt(d, k):
        v = d.get(k)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

    # Compute confusion matrices and macro P/R for multi-class
    cm_val, prec_val, rec_val, macro_p_val, macro_r_val = eval_confmat_and_pr(val_ds, model, NUM_CLASSES)
    cm_test, prec_test, rec_test, macro_p_test, macro_r_test = eval_confmat_and_pr(test_ds, model, NUM_CLASSES)

    # Attach to metrics for saving
    metrics["val_cm"] = cm_val.tolist()
    metrics["test_cm"] = cm_test.tolist()
    metrics["val_pr"] = {
        "per_class_precision": prec_val,
        "per_class_recall": rec_val,
        "macro_precision": macro_p_val,
        "macro_recall": macro_r_val,
    }
    metrics["test_pr"] = {
        "per_class_precision": prec_test,
        "per_class_recall": rec_test,
        "macro_precision": macro_p_test,
        "macro_recall": macro_r_test,
    }

    # Log the settings and final results (using module-level constants)
    log_data_training(f"- Settings -------------------------------------------------------------------------------------------")
    log_data_training(f"Image Size:              {IMG_SIZE}")
    log_data_training(f"Multiplier Width:        {WIDTH_MULT}")
    log_data_training(f"Batchsize:               {BATCH_SIZE}")
    log_data_training(f"Epochs Head:             {EPOCHS_HEAD}")
    log_data_training(f"Epochs Fine:             {EPOCHS_FINE}")
    log_data_training(f"Base Learning Rate:      {BASE_LR}")
    log_data_training(f"Fine_Tune Learning Rate: {FINE_TUNE_LR}")
    log_data_training(f"Dropout:                 {DROPOUT}")
    log_data_training(f"Seed:                    {SEED}")
    log_data_training(f"- Results --------------------------------------------------------------------------------------------")
    log_data_training(f"Validation | acc: {_fmt(va,'acc')}  macro P: {macro_p_val:.4f}  macro R: {macro_r_val:.4f}")
    log_data_training(f"Test       | acc: {_fmt(te,'acc')}  macro P: {macro_p_test:.4f}  macro R: {macro_r_test:.4f}")
    log_data_training(f"------------------------------------------------------------------------------------------------------")

    # Save training curves & metrics
    hist = {"phase1": history1.history, "phase2": history2.history}
    with (EXPORT_DIR / "history.json").open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    with (EXPORT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save Keras & TF.js
    model.save(str(ckpt_path))
    log_data_training(f"Saved Keras model → {ckpt_path}")
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, str(TFJS_DIR))
        log_data_training(f"Saved TF.js model → {TFJS_DIR}")
        print(f"[OK] TF.js model saved to: {TFJS_DIR}")
    except Exception as e:
        log_data_training("TF.js export skipped (install tensorflowjs)")
        print("[WARN] Skipping TF.js export. Install 'tensorflowjs' to enable (pip install tensorflowjs).", e)

    log_data_training(f"Training completed for {VERSION}\n\n")
    print("[OK] Training done.")
    print("Val:", metrics["val"])  
    print("Test:", metrics["test"])


if __name__ == "__main__":
    log_data_training(f"Started process for: {VERSION}") # Log-Entry

    # Reproducibility
    tf.keras.utils.set_random_seed(SEED)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    train_and_eval()