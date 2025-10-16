import os, json, math, re, csv, sys

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import random
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed

# Backbone: ResNet152V2 with [-1,1] preprocessing (TFJS-friendly)
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import tensorflowjs as tfjs

from custom_logging import log_data_training


# -----------------------------------------------------------------------------------------------------------------------------------
# - Config --------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
MAIN_VERSION = 2

MANIFEST_PATH = "data/manifests/V006_70_20_10_per_source/manifest.jsonl" # Holds the Path to the manifest
BASE_PATH = "data/reddit_pics" # Holds the Path to the basefolder that contains the image data
ACCEPTED_EXTS = (".jpeg", ".png", ".bmp")  # ← no .webp

IMG_SIZE         = 224                  # Size of each image (squared)
IMAGE_NORMALIZER = "resize"             # "resize", "crop", "bars"
EPOCHS           = 20
FINETUNE_EPOCHS  = 0                    # Keep this at 0, right now fine tuning seems to cause issues
BATCH_SIZE       = 265
BASE_LR          = 5e-4
FINE_TUNE_LR     = 1e-5
DROPOUT          = 0.2
UNFREEZE_FROM    = -20                  # last N layers of the base model
PATIENCE         = 5                    # early stopping
LABEL_SMOOTHING  = 0.0                  # e.g. 0.05 if you want it
SEED             = 42                   # Seed for reproducability
tf.keras.utils.set_random_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs


# -----------------------------------------------------------------------------------------------------------------------------------
# - Helperfunctions -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
def _get_next_model_version(main_version: int, models_dir="models") -> str:
    os.makedirs(models_dir, exist_ok=True)
    pattern = re.compile(rf"^V_{main_version}_(\d+)$")
    subversions = []

    for name in os.listdir(models_dir):
        match = pattern.match(name)
        if match:
            subversions.append(int(match.group(1)))

    next_sub = max(subversions) + 1 if subversions else 0
    return f"V_{main_version}_{next_sub}"

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f]
    return manifest

def _get_pairs(manifest, split):
    return [
        (os.path.join(BASE_PATH, item["path"]), item["label_id"])
        for item in manifest if item["split"] == split
    ]

def _is_image_valid(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def _filter_invalid_pairs(pairs, max_workers=8):
    if not pairs:
        return []
    valid = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_is_image_valid, p): (p, y) for (p, y) in pairs}
        for fut in as_completed(futs):
            p, y = futs[fut]
            ok = False
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                valid.append((p, y))
    removed = len(pairs) - len(valid)
    if removed:
        print(f"Filtered out {removed} unreadable images (kept {len(valid)})")
    return valid

# --- TF preprocess -----------------------------------------------------------
def _preprocess_tf(path, label_id, training=False, img_size=IMG_SIZE, normalizer=IMAGE_NORMALIZER):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)

    if normalizer == "resize":
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
    elif normalizer == "crop":
        h = tf.shape(img)[0]; w = tf.shape(img)[1]
        scale = tf.cast(img_size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
        new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        img = tf.image.resize(img, (new_h, new_w), method="bilinear")
        img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)
    elif normalizer == "bars":
        img = tf.image.resize_with_pad(img, img_size, img_size, method="bilinear")
    else:
        raise ValueError(f"Unknown IMAGE_NORMALIZER: {normalizer}")

    img = tf.cast(img, tf.float32)
    if training:
        img = tf.image.random_flip_left_right(img)

    # ResNetV2 expects [-1, 1]
    img = preprocess_input(img)

    img.set_shape([img_size, img_size, 3])
    return img, tf.cast(label_id, tf.int32)

def _make_dataset(pairs, training=False):
    if not pairs:
        return tf.data.Dataset.from_tensor_slices(([], [])).batch(BATCH_SIZE)

    paths, labels = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        ds = ds.shuffle(buffer_size=min(len(pairs), 4096))

    num_calls = min(8, (os.cpu_count() or 8))
    ds = ds.map(lambda p, y: _preprocess_tf(p, y, training), num_parallel_calls=num_calls)

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    ds = ds.with_options(opts)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def _num_classes_from_pairs(pairs):
    return int(max(l for _, l in pairs)) + 1 if pairs else 0

def _class_weights_from_pairs(pairs):
    cnt = Counter(l for _, l in pairs)
    if not cnt: return None
    total = sum(cnt.values())
    k = len(cnt)
    return {c: total/(k*cnt[c]) for c in cnt}

def _labels_from_manifest(manifest):
    """Returns a list index->name (falls back to 'class_{id}' if no name)."""
    id_to_name = {}
    for item in manifest:
        lid = int(item["label_id"])
        name = item.get("label", None)
        if lid not in id_to_name:
            id_to_name[lid] = name if isinstance(name, str) and len(name) else f"class_{lid}"
    if not id_to_name:
        return []
    max_id = max(id_to_name.keys())
    return [id_to_name.get(i, f"class_{i}") for i in range(max_id + 1)]

def _save_history_artifacts(history, out_dir: str):
    _ensure_dir(out_dir)
    h = history.history  # dict: metric -> list per epoch

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(h, f, indent=2)

    keys = sorted(h.keys())
    rows = zip(*[h[k] for k in keys]) if keys else []
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i, row in enumerate(rows):
            w.writerow([i+1] + list(row))

    base_metrics = [k for k in keys if not k.startswith("val_")]
    for m in base_metrics:
        plt.figure()
        plt.plot(h[m], label=m)
        val_key = f"val_{m}"
        if val_key in h:
            plt.plot(h[val_key], label=val_key)
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{m}.png"), dpi=160)
        plt.close()

# -----------------------------------------------------------------------------------------------------------------------------------
# - Mainfunction --------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
def main():
    print(tf.config.list_physical_devices('GPU'))
    print("TF version:", tf.__version__)

    version_name = _get_next_model_version(MAIN_VERSION, models_dir="models")
    version_dir = os.path.join("models", version_name)
    os.makedirs(version_dir, exist_ok=True)
    print("Saving to:", version_dir)

    log_data_training(f"Started prozess for training Version: {version_name}")

    manifest = _load_manifest()

    train_pairs = _get_pairs(manifest, "train")
    val_pairs   = _get_pairs(manifest, "val")
    test_pairs  = _get_pairs(manifest, "test")

    def _counts(pairs): return dict(sorted(Counter(y for _,y in pairs).items()))
    print("train_prefiltered:", _counts(train_pairs))
    print("val_prefiltered:  ", _counts(val_pairs))
    print("test_prefiltered: ", _counts(test_pairs))

    # Drop undecodable images fast (parallel PIL verify)
    train_pairs = _filter_invalid_pairs(train_pairs, max_workers=8)
    val_pairs   = _filter_invalid_pairs(val_pairs,   max_workers=8)
    test_pairs  = _filter_invalid_pairs(test_pairs,  max_workers=8)

    print("train_filtered:", _counts(train_pairs))
    print("val_filtered:  ", _counts(val_pairs))
    print("test_filtered: ", _counts(test_pairs))

    labels = _labels_from_manifest(manifest)

    train_ds = _make_dataset(train_pairs, training=True)
    val_ds   = _make_dataset(val_pairs,   training=False)
    test_ds  = _make_dataset(test_pairs,  training=False)

    steps_per_epoch = math.ceil(len(train_pairs) / BATCH_SIZE) if train_pairs else 0
    val_steps       = math.ceil(len(val_pairs) / BATCH_SIZE)   if val_pairs else 0

    num_classes = _num_classes_from_pairs(train_pairs)
    class_weights = _class_weights_from_pairs(train_pairs)

    log_data_training("Settings:")
    log_data_training(f"Image Size:             {IMG_SIZE}")
    log_data_training(f"Image Normalization:    {IMAGE_NORMALIZER}")
    log_data_training(f"Epochs:                 {EPOCHS}")
    log_data_training(f"Finetune Epochs:        {FINETUNE_EPOCHS}")
    log_data_training(f"Batch Size:             {BATCH_SIZE}")
    log_data_training(f"Base Learning Rate:     {BASE_LR}")
    log_data_training(f"Finetune Learning Rate: {FINE_TUNE_LR}")
    log_data_training(f"Dropout:                {DROPOUT}")
    log_data_training(f"Unfreeze From:          {UNFREEZE_FROM}")
    log_data_training(f"Patience:               {PATIENCE}")
    log_data_training(f"Label Smoothing:        {LABEL_SMOOTHING}")
    log_data_training(f"Seed:                   {SEED}")
    log_data_training(f"Classes:                {num_classes}")

    # ----------------------------------------------------------
    # Define model (ResNet152V2 backbone)
    # ----------------------------------------------------------
    base_model = ResNet152V2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_1")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=BASE_LR),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.summary()

    early_stop = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)

    log_data_training("Starting initial training phase")

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[early_stop],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
    )

    # Save curves and CSV
    _save_history_artifacts(history, out_dir=version_dir)

    # ----------------------------------------------------------
    # Optional Fine-tuning
    # ----------------------------------------------------------
    if FINETUNE_EPOCHS > 0:
        log_data_training("Starting fine-tuning phase")
        base_model.trainable = True
        for layer in base_model.layers[:UNFREEZE_FROM]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        fine_tune_history = model.fit(
            train_ds,
            epochs=EPOCHS + FINETUNE_EPOCHS,
            initial_epoch=history.epoch[-1] + 1,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=[early_stop],
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
        )
        _save_history_artifacts(fine_tune_history, out_dir=version_dir)

    # ----------------------------------------------------------
    # Evaluate Model
    # ----------------------------------------------------------
    if test_pairs:
        print("Evaluating on test set…")
        test_ds  = _make_dataset(test_pairs,  training=False)
        test_steps = math.ceil(len(test_pairs) / BATCH_SIZE)
        test_loss, test_acc = model.evaluate(test_ds, steps=test_steps, verbose=1)
        log_data_training(f"TEST — loss: {test_loss:.4f}, acc: {test_acc:.4f}")
    else:
        print("No test data found — skipping test evaluation.")

    # ----------------------------------------------------------
    # Save final model
    # ----------------------------------------------------------
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    model.save(os.path.join(version_dir, "model.keras"))

    def _strip_regularizers_for_export(m):
        for layer in m.layers:
            for attr in ("kernel_regularizer", "bias_regularizer", "activity_regularizer"):
                if hasattr(layer, attr):
                    setattr(layer, attr, None)
        return m

    export_model = tf.keras.models.clone_model(model)
    export_model.set_weights(model.get_weights())
    export_model = _strip_regularizers_for_export(export_model)
    tfjs.converters.save_keras_model(export_model, os.path.join(version_dir, "model_tfjs"))

    # Create labels.json for the frontend
    with open(os.path.join(version_dir, "model_tfjs", "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    log_data_training(f"Saved labels.json with {len(labels)} entries.")

    log_data_training("Training finished")


if __name__ == "__main__":
    main()
