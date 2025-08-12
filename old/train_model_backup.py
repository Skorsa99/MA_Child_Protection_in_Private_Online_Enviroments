import matplotlib
import tensorflow as tf
import os, json, math, re, csv
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from tensorflow.keras.applications.efficientnet import preprocess_input

from custom_logging import log_data_training

matplotlib.use("Agg")  # headless


# -----------------------------------------------------------------------------------------------------------------------------------
# - Config --------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
MAIN_VERSION = 0

MANIFEST_PATH = "data/manifests/V001_80_10_10_per_source/manifest.jsonl" # Holds the Path to the manifest
BASE_PATH = "data/reddit_pics" # Holds the Path to the basefolder that contains the image data
ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"} # Image types accepted by Tensorflow

IMG_SIZE         = 224                  # Size of each image (squared)
IMAGE_NORMALIZER = "resize"             # Different options for image normalization: "resize", "crop", "bars"
EPOCHS           = 10
FINETUNE_EPOCHS  = 5
BATCH_SIZE       = 64                   # Number of Samples per Batches
BASE_LR          = 1e-4
FINE_TUNE_LR     = 1e-5
DROPOUT          = 0.2
UNFREEZE_FROM    = -20                  # last N layers of the base model
PATIENCE         = 3                    # early stopping
LABEL_SMOOTHING  = 0.0                  # e.g. 0.05 if you want it
SEED             = 42                   # Seed for reproducability
tf.keras.utils.set_random_seed(SEED)    # Seed for reproducability

AUTOTUNE = tf.data.AUTOTUNE # Add Comment


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

def _is_decodable_np(path_bytes) -> bool:
    try:
        path = path_bytes.decode("utf-8")
        # try read + decode using TF (same decoder as training)
        img_bytes = tf.io.read_file(path).numpy()
        tf.io.decode_image(img_bytes, channels=3, expand_animations=False).numpy()
        return True
    except Exception:
        return False

def _is_decodable_tf(path, label):
    ok = tf.numpy_function(_is_decodable_np, [path], tf.bool)
    # ensure shapes preserved through filter
    ok.set_shape(())
    return ok

def _safe_preprocess(path, label_id, training=False, img_size=IMG_SIZE, normalizer=IMAGE_NORMALIZER):
    def _load_and_process(p) -> tf.Tensor:
        try:
            # --- Path as Python string
            path_str = p.numpy().decode("utf-8")  # <- this is safe inside py_function
            lower_path = path_str.lower()

            # --- Try fast TF decode first (JPEG/PNG/GIF/BMP)
            try:
                img_bytes = tf.io.read_file(path_str)
                img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
            except Exception:
                # Fallback: PIL handles a lot more formats (incl. WEBP)
                with Image.open(path_str) as im:
                    im = im.convert("RGB")
                    img = tf.convert_to_tensor(np.array(im), dtype=tf.uint8)

            # --- (Optional) normalize exotic extensions to JPEG in-memory
            if not lower_path.endswith(tuple(ACCEPTED_EXTS)):
                img = tf.io.decode_jpeg(tf.io.encode_jpeg(tf.cast(img, tf.uint8)), channels=3)

            # --- Spatial normalization (unchanged)
            if normalizer == "resize":
                img = tf.image.resize(img, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR)
            elif normalizer == "crop":
                h = tf.shape(img)[0]; w = tf.shape(img)[1]
                scale = tf.cast(img_size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
                new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
                new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
                img = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)
                img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)
            elif normalizer == "bars":
                img = tf.image.resize_with_pad(img, img_size, img_size, method=tf.image.ResizeMethod.BILINEAR)
            else:
                raise ValueError(f"Unknown IMAGE_NORMALIZER: {normalizer}")

            img = tf.cast(img, tf.float32)
            if training:
                img = tf.image.random_flip_left_right(img)

        except Exception as e:
            # Throttle logging: comment out for speed or keep a single line
            # tf.print("⚠ bad image, replaced:", path_str, "Error:", str(e))
            img = tf.zeros((img_size, img_size, 3), tf.float32)

        return tf.cast(img, tf.float32)

    # Use py_function to handle Python-side try/except
    img = tf.py_function(_load_and_process, [path], tf.float32)
    img.set_shape([img_size, img_size, 3])  # ensure static shape for batching
    return img, tf.cast(label_id, tf.int32)

def _preprocess_tf(path, label_id, training=False, img_size=IMG_SIZE, normalizer=IMAGE_NORMALIZER):
    img_bytes = tf.io.read_file(path)                            # TF op
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)  # TF op

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

    img = tf.cast(img, tf.float32)  # no /255 if you use preprocess_input later
    if training:
        img = tf.image.random_flip_left_right(img)

    img.set_shape([img_size, img_size, 3])  # static shape for batching
    return img, tf.cast(label_id, tf.int32)

def _make_dataset(pairs, training=False):
    if not pairs:
        return tf.data.Dataset.from_tensor_slices(([], [])).batch(BATCH_SIZE)

    paths, labels = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        ds = ds.shuffle(buffer_size=min(len(pairs), 4096))

    # DROP bad files here
    ds = ds.filter(_is_decodable_tf)

    # map with pure-TF preprocessing (no py_function)
    num_calls = min(8, (os.cpu_count() or 8))
    ds = ds.map(lambda p, y: _preprocess_tf(p, y, training),
                num_parallel_calls=num_calls)

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
    # inverse frequency
    return {c: total/(k*cnt[c]) for c in cnt}

def save_history_artifacts(history, out_dir: str):
    """
    Saves: history.json, history.csv, and one PNG per metric (train vs val).
    Works with keys like: 'loss', 'val_loss', 'accuracy', 'val_accuracy', etc.
    """
    _ensure_dir(out_dir)
    h = history.history  # dict: metric -> list per epoch

    # 1) save raw history
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(h, f, indent=2)

    # 2) CSV (epochs as rows)
    # collect union of keys
    keys = sorted(h.keys())
    rows = zip(*[h[k] for k in keys]) if keys else []
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i, row in enumerate(rows):
            w.writerow([i+1] + list(row))

    # 3) plots: for each base metric (without 'val_'), plot train + val
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
def build_model(num_classes: int):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_input(inputs)                 # safe regardless of Keras version
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)     # <— use DROPOUT
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),           # <— BASE_LR
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING), # This should reduce overfitting but throws an error right now so it is taken out, might try to put it back in later
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def train(train_ds, val_ds, train_pairs, epochs=EPOCHS, use_class_weights=True, out_dir=None):
    num_classes = _num_classes_from_pairs(train_pairs)
    if num_classes <= 1:
        raise ValueError("num_classes <= 1 — check your labels.")
    class_weight = _class_weights_from_pairs(train_pairs) if use_class_weights else None

    model = build_model(num_classes)
    callbacks = []
    if out_dir:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(out_dir, "best.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=PATIENCE,
                restore_best_weights=True
            )
        ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks
    )
    return model, history

def fine_tune(model, train_ds, val_ds, lr=FINE_TUNE_LR, unfreeze_from=UNFREEZE_FROM,
              epochs=FINETUNE_EPOCHS, out_dir=None):
    # unfreeze last N layers of the base model
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[unfreeze_from:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),               # <— FINE_TUNE_LR
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )
    callbacks = []
    if out_dir:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(out_dir, "best_finetune.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=PATIENCE,
                restore_best_weights=True
            )
        ]
    hist_ft = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return hist_ft


# -----------------------------------------------------------------------------------------------------------------------------------
# - Controlfunction -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
def main():
    print(tf.config.list_physical_devices('GPU'))
    print(tf.__version__)
    print(tf.keras.losses.SparseCategoricalCrossentropy.__init__)

    version_name = _get_next_model_version(MAIN_VERSION, models_dir="models")
    version_dir = os.path.join("models", version_name)
    os.makedirs(version_dir, exist_ok=True)
    print("Saving to:", version_dir)

    log_data_training(f"Started prozess for training Version: {version_name}") # Create Log

    manifest = _load_manifest()

    print("loaded manifest-------------------------------------------------------------------------")

    train_pairs = _get_pairs(manifest, "train")
    val_pairs   = _get_pairs(manifest, "val")
    test_pairs  = _get_pairs(manifest, "test")

    print("created pairs---------------------------------------------------------------------------")

    train_ds = _make_dataset(train_pairs, training=True)
    val_ds   = _make_dataset(val_pairs, training=False)
    test_ds  = _make_dataset(test_pairs, training=False)

    print("created datasets------------------------------------------------------------------------")

    log_data_training(f"Settings:") # Create Log
    log_data_training(f"Image Size:             {IMG_SIZE}") # Create Log
    log_data_training(f"Image Normalization:    {IMAGE_NORMALIZER}") # Create Log
    log_data_training(f"Epochs:                 {EPOCHS}") # Create Log
    log_data_training(f"Finetune Epochs:        {FINETUNE_EPOCHS}") # Create Log
    log_data_training(f"Batch Size:             {BATCH_SIZE}") # Create Log
    log_data_training(f"Base Learning Rate:     {BASE_LR}") # Create Log
    log_data_training(f"Finetune Learning Rate: {FINE_TUNE_LR}") # Create Log
    log_data_training(f"Dropout:                {DROPOUT}") # Create Log
    log_data_training(f"Unfreeze From:          {UNFREEZE_FROM}") # Create Log
    log_data_training(f"Patience:               {PATIENCE}") # Create Log
    log_data_training(f"Label Smoothing:        {LABEL_SMOOTHING}") # Create Log
    log_data_training(f"Seed:                   {SEED}") # Create Log

    # Stage 1
    model, history = train(train_ds, val_ds, train_pairs, out_dir=version_dir)
    save_history_artifacts(history, out_dir=os.path.join(version_dir, "stage1"))

    log_data_training(f"Training complete") # Create Log
    
    print("completed training----------------------------------------------------------------------")

    # Stage 2
    hist_ft = fine_tune(model, train_ds, val_ds, out_dir=version_dir)
    save_history_artifacts(hist_ft, out_dir=os.path.join(version_dir, "finetune"))

    print("completed finetuning--------------------------------------------------------------------")

    log_data_training(f"Fine-Tuning complete") # Create Log

    # Save final model folder + evaluate
    model.save(version_dir)
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test acc: {test_acc:.4f}")

    log_data_training(f"Test Loss: {test_loss} | Test Accuracy: {test_acc}\n\n") # Create Log

if __name__ == "__main__":
    main()