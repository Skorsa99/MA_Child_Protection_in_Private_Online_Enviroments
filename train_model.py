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

# ✅ Option A (recommended with Keras 3 / tf-keras shim)
# from keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import tensorflowjs as tfjs

from custom_logging import log_data_training


# -----------------------------------------------------------------------------------------------------------------------------------
# - Config --------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
MAIN_VERSION = 1

MANIFEST_PATH = "data/manifests/V005_70_20_10_per_source/manifest.jsonl" # Holds the Path to the manifest
BASE_PATH = "data/reddit_pics" # Holds the Path to the basefolder that contains the image data
ACCEPTED_EXTS = (".jpeg", ".png", ".bmp")  # ← no .webp

IMG_SIZE         = 224                  # Size of each image (squared)
IMAGE_NORMALIZER = "resize"             # "resize", "crop", "bars"
EPOCHS           = 10
FINETUNE_EPOCHS  = 0                    # Keep this at 0, right now fine tuning seems to cause issues
BATCH_SIZE       = 265 # 64
BASE_LR          = 5e-4 # 1e-4
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

def _is_decodable_np(path_bytes) -> bool:
    try:
        path = path_bytes.decode("utf-8")
        img_bytes = tf.io.read_file(path).numpy()
        tf.io.decode_image(img_bytes, channels=3, expand_animations=False).numpy()
        return True
    except Exception:
        return False

def _is_decodable_tf(path, label):
    ok = tf.numpy_function(_is_decodable_np, [path], tf.bool)
    ok.set_shape(())
    return ok

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

# --- 2) Pure-TF preprocess (NO py_function, NO PIL in the map) ---------------
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

    # MobileNetV3 expects [-1, 1]
    img = preprocess_input(img)

    img.set_shape([img_size, img_size, 3])
    return img, tf.cast(label_id, tf.int32)

def _preprocess_tf_more_regulazation(path, label_id, training=False, img_size=IMG_SIZE, normalizer=IMAGE_NORMALIZER):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)

    # size normalization (as you have it)
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

    img = tf.cast(img, tf.float32)

    if training:
        # jitter in [0..1], then scale back
        x = tf.image.convert_image_dtype(img, tf.float32)            # [0..1]
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.08)            # ±8% light
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1)        # ±10% contrast
        x = tf.image.random_saturation(x, lower=0.9, upper=1.1)      # ±10% saturation
        # optional tiny zoom/crop:
        # pad 4px and random-crop back to 224
        x = tf.image.resize_with_pad(x, img_size+4, img_size+4)
        x = tf.image.random_crop(x, size=[img_size, img_size, 3])
        x = tf.clip_by_value(x, 0.0, 1.0)
        img = x * 255.0                                              # back to [0..255]

    # now match MobileNetV2
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)   # [-1, 1]
    img.set_shape([img_size, img_size, 3])
    return img, tf.cast(label_id, tf.int32)

# --- 3) Dataset builder -------------------------------------------------------
def _make_dataset(pairs, training=False):
    if not pairs:
        return tf.data.Dataset.from_tensor_slices(([], [])).batch(BATCH_SIZE)

    paths, labels = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        ds = ds.shuffle(buffer_size=min(len(pairs), 4096))

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

def save_history_artifacts(history, out_dir: str):
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

def _show_examples(ds, class_names, n_per_class=10):
    """
    Display n_per_class images for each class in columns.
    - ds: tf.data.Dataset yielding (images, labels), already preprocessed.
    - class_names: list like ['unsafe','neutral','safe'] (index -> name).
    - n_per_class: how many rows per class (default: 10).
    """
    num_classes = len(class_names)
    buckets = {i: [] for i in range(num_classes)}

    # Collect up to n_per_class per class by scanning the dataset
    for batch_imgs, batch_labels in ds:
        imgs = batch_imgs.numpy()
        labs = batch_labels.numpy()
        for img, y in zip(imgs, labs):
            y = int(y)
            if y in buckets and len(buckets[y]) < n_per_class:
                buckets[y].append(img)
        if all(len(buckets[c]) >= n_per_class for c in range(num_classes)):
            break

    # Prepare the figure
    fig, axes = plt.subplots(n_per_class, num_classes, figsize=(3.2*num_classes, 2.2*n_per_class))
    if n_per_class == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plot each column = one class
    for c in range(num_classes):
        samples = buckets[c]
        # If fewer than requested exist, pad with blanks
        while len(samples) < n_per_class:
            samples.append(None)

        for r in range(n_per_class):
            ax = axes[r, c]
            ax.axis("off")
            if r == 0:
                ax.set_title(class_names[c], fontsize=12)
            img = samples[r]
            if img is None:
                continue
            # MobileNet preprocess_input put values in [-1,1]; convert back for display
            disp = (img + 1.0) / 2.0
            disp = np.clip(disp, 0, 1)
            ax.imshow(disp)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------
# - Mainfunction --------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------
# - Controlfunction -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
def main():
    print(tf.config.list_physical_devices('GPU'))
    print("TF version:", tf.__version__)
    print("Loss init:", tf.keras.losses.SparseCategoricalCrossentropy.__init__)

    version_name = _get_next_model_version(MAIN_VERSION, models_dir="models")
    version_dir = os.path.join("models", version_name)
    os.makedirs(version_dir, exist_ok=True)
    print("Saving to:", version_dir)

    log_data_training(f"Started prozess for training Version: {version_name}")  # Create Log

    manifest = _load_manifest()
    print("loaded manifest-------------------------------------------------------------------------")

    train_pairs = _get_pairs(manifest, "train")
    val_pairs   = _get_pairs(manifest, "val")
    test_pairs  = _get_pairs(manifest, "test")

    def _counts(pairs): return dict(sorted(Counter(y for _,y in pairs).items())) # -----------------------------
    print("train_prefiltered:", _counts(train_pairs)) # --------------------------------------------------------
    print("val_prefiltered:  ", _counts(val_pairs)) # ----------------------------------------------------------
    print("test_prefiltered: ", _counts(test_pairs)) # ---------------------------------------------------------

    # Pre-scan (drops bad files fast, parallel)
    train_pairs = _filter_invalid_pairs(train_pairs, max_workers=8)
    val_pairs   = _filter_invalid_pairs(val_pairs,   max_workers=8)
    test_pairs  = _filter_invalid_pairs(test_pairs,  max_workers=8)

    print("train_filtered:", _counts(train_pairs)) # -----------------------------------------------------------
    print("val_filtered:  ", _counts(val_pairs)) # -------------------------------------------------------------
    print("test_filtered: ", _counts(test_pairs)) # ------------------------------------------------------------

    labels_train = sorted({y for _,y in train_pairs}) # --------------------------------------------------------
    labels_val   = sorted({y for _,y in val_pairs}) # ----------------------------------------------------------
    labels_test  = sorted({y for _,y in test_pairs}) # ---------------------------------------------------------
    print("labels_train:", labels_train) # ---------------------------------------------------------------------
    print("labels_val:  ", labels_val) # -----------------------------------------------------------------------
    print("labels_test: ", labels_test) # ----------------------------------------------------------------------

    print("created pairs---------------------------------------------------------------------------")
    
    labels = _labels_from_manifest(manifest)

    train_ds = _make_dataset(train_pairs, training=True)
    val_ds   = _make_dataset(val_pairs,   training=False)
    test_ds  = _make_dataset(test_pairs,  training=False)

    # _show_examples(train_ds, labels, n_per_class=10)

    steps_per_epoch = math.ceil(len(train_pairs) / BATCH_SIZE) if train_pairs else 0
    val_steps       = math.ceil(len(val_pairs) / BATCH_SIZE)   if val_pairs else 0

    print("created datasets------------------------------------------------------------------------")

    num_classes = _num_classes_from_pairs(train_pairs)
    class_weights = _class_weights_from_pairs(train_pairs)

    print()
    print("class_weights:", class_weights)   # should be {0:...,1:...,2:...}

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
    # Define model
    # ----------------------------------------------------------
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze initially

    """ Previous model without the smaller head and aditional Dropout"""
    # inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_1")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    """
    # inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_1")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(256, activation="relu")(x)       # larger head (more trainable parameters)
    # x = layers.BatchNormalization()(x)                # might cause issues
    # x = layers.Dropout(0.3)(x)                        # a bit more regularization
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    """

    # model = tf.keras.Model(inputs, outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=BASE_LR),
        # loss=losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        loss=losses.SparseCategoricalCrossentropy(),  # no label_smoothing
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

    last = model.layers[-1] # -----------------------------------------------------------------------
    W, b = last.get_weights() # ---------------------------------------------------------------------
    print("last-layer bias:", np.round(b, 4))  # z.B. [ -0.12  1.95  -0.88 ] # ----------------------

    y_true, y_pred, maxp = [], [], [] # -------------------------------------------------------------
    for x, y in val_ds: # ---------------------------------------------------------------------------
        p = model.predict(x, verbose=0) # -----------------------------------------------------------
        y_true.extend(y.numpy()) # ------------------------------------------------------------------
        y_pred.extend(p.argmax(axis=1)) # -----------------------------------------------------------
        maxp.extend(p.max(axis=1)) # ----------------------------------------------------------------
    print("mean max prob:", float(np.mean(maxp))) # -------------------------------------------------
    print("share predicted class==1:", float((np.array(y_pred)==1).mean())) # -----------------------

    save_history_artifacts(history, out_dir=version_dir)

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
            # loss=losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
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

        last = model.layers[-1] # -----------------------------------------------------------------------
        W, b = last.get_weights() # ---------------------------------------------------------------------
        print("last-layer bias:", np.round(b, 4))  # z.B. [ -0.12  1.95  -0.88 ] # ----------------------

        y_true, y_pred, maxp = [], [], [] # -------------------------------------------------------------
        for x, y in val_ds: # ---------------------------------------------------------------------------
            p = model.predict(x, verbose=0) # -----------------------------------------------------------
            y_true.extend(y.numpy()) # ------------------------------------------------------------------
            y_pred.extend(p.argmax(axis=1)) # -----------------------------------------------------------
            maxp.extend(p.max(axis=1)) # ----------------------------------------------------------------
        print("mean max prob:", float(np.mean(maxp))) # -------------------------------------------------
        print("share predicted class==1:", float((np.array(y_pred)==1).mean())) # -----------------------

        save_history_artifacts(fine_tune_history, out_dir=version_dir)

    # ----------------------------------------------------------
    # Evaluate Model
    # ----------------------------------------------------------
    if test_pairs:
        print("Evaluating on test set…")
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
    tfjs.converters.save_keras_model(model, os.path.join(version_dir, "model_tfjs"))

    # Create labels.json
    with open(os.path.join(version_dir, "model_tfjs", "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    log_data_training(f"Saved labels.json with {len(labels)} entries.")

    log_data_training("Training finished")


if __name__ == "__main__":
    main()