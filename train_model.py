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

# Backbone: ResNet50V2 with [-1,1] preprocessing (TFJS-friendly)
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import tensorflowjs as tfjs

from custom_logging import log_data_training


# -----------------------------------------------------------------------------------------------------------------------------------
# - Config --------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
MAIN_VERSION = 3

MANIFEST_PATH = "data/manifests/V006_70_20_10_per_source/manifest.jsonl" # Holds the Path to the manifest
BASE_PATH = "data/reddit_pics" # Holds the Path to the basefolder that contains the image data
ACCEPTED_EXTS = (".jpeg", ".png", ".bmp")  # ← no .webp

IMG_SIZE         = 224                  # Size of each image (squared)
IMAGE_NORMALIZER = "resize"             # "resize", "crop", "bars"
EPOCHS           = 20
FINETUNE_EPOCHS  = 0                    # Keep this at 0, right now fine tuning seems to cause issues
BATCH_SIZE       = 64 # 265
BASE_LR          = 5e-4
FINE_TUNE_LR     = 1e-5
DROPOUT          = 0.2
UNFREEZE_FROM    = -20                  # last N layers of the base model
PATIENCE         = 5                    # early stopping
LABEL_SMOOTHING  = 0.05                  # e.g. 0.05 if you want it
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

# -----------------------------------------------------------------------------------------------------------------------------------
# Augmentation configuration (old) --------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
"""
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
"""
# -----------------------------------------------------------------------------------------------------------------------------------
# Augmentation configuration (new) --------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
AUG_RESIZE_DELTA = 32          # extra pixels before random crop
AUG_SCALE_RANGE = (1.0, 1.35)  # random zoom-in factors
COLOR_JITTER = dict(brightness=0.18, contrast=0.18, saturation=0.25, hue=0.05)
MIXUP_ALPHA = 0.25
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5
USE_SOFT_LABELS = False          # False keeps sparse labels for SparseCategoricalCrossentropy

def _preprocess_tf(path, label_id, training=False, img_size=IMG_SIZE, normalizer=IMAGE_NORMALIZER):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0, 1]

    target = img_size + AUG_RESIZE_DELTA
    if normalizer == "bars":
        img = tf.image.resize_with_pad(img, target, target, method="bilinear")
    elif normalizer == "crop":
        h = tf.shape(img)[0]; w = tf.shape(img)[1]
        scale = tf.cast(target, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
        new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        img = tf.image.resize(img, (new_h, new_w), method="bilinear")
    elif normalizer == "resize":
        img = tf.image.resize(img, (target, target), method="bilinear")
    else:
        raise ValueError(f"Unknown IMAGE_NORMALIZER: {normalizer}")

    if training:
        crop_size = tf.cast(
            tf.round(tf.random.uniform([], AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1]) * tf.cast(img_size, tf.float32)),
            tf.int32,
        )
        crop_size = tf.clip_by_value(crop_size, img_size, target)
        img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=COLOR_JITTER["brightness"])
        img = tf.image.random_contrast(
            img,
            lower=1.0 - COLOR_JITTER["contrast"],
            upper=1.0 + COLOR_JITTER["contrast"],
        )
        img = tf.image.random_saturation(
            img,
            lower=1.0 - COLOR_JITTER["saturation"],
            upper=1.0 + COLOR_JITTER["saturation"],
        )
        img = tf.image.random_hue(img, max_delta=COLOR_JITTER["hue"])
        img = tf.clip_by_value(img, 0.0, 1.0)
    else:
        img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)

    img = preprocess_input(img * 255.0)  # ResNetV2 expects [-1, 1]
    img.set_shape([img_size, img_size, 3])
    return img, tf.cast(label_id, tf.int32)


def _sample_beta_distribution(size, concentration):
    gamma_1 = tf.random.gamma(shape=[size], alpha=concentration, dtype=tf.float32)
    gamma_2 = tf.random.gamma(shape=[size], alpha=concentration, dtype=tf.float32)
    return gamma_1 / (gamma_1 + gamma_2)


def _mixup_batch(images, labels, alpha):
    batch = tf.shape(images)[0]

    def _apply():
        weights = _sample_beta_distribution(batch, alpha)
        x_w = tf.reshape(weights, [batch, 1, 1, 1])
        y_w = tf.reshape(weights, [batch, 1])
        shuffle = tf.random.shuffle(tf.range(batch))
        mixed_images = images * x_w + tf.gather(images, shuffle) * (1.0 - x_w)
        mixed_labels = labels * y_w + tf.gather(labels, shuffle) * (1.0 - y_w)
        return mixed_images, mixed_labels

    return tf.cond(batch > 1, _apply, lambda: (images, labels))


def _cutmix_batch(images, labels, alpha):
    batch = tf.shape(images)[0]

    def _apply():
        weights = _sample_beta_distribution(batch, alpha)
        shuffle = tf.random.shuffle(tf.range(batch))
        shuffled_images = tf.gather(images, shuffle)
        shuffled_labels = tf.gather(labels, shuffle)

        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        cut_ratios = tf.sqrt(1.0 - weights)
        cut_heights = tf.maximum(tf.cast(cut_ratios * tf.cast(h, tf.float32), tf.int32), 1)
        cut_widths = tf.maximum(tf.cast(cut_ratios * tf.cast(w, tf.float32), tf.int32), 1)
        ys = tf.random.uniform([batch], 0, h, dtype=tf.int32)
        xs = tf.random.uniform([batch], 0, w, dtype=tf.int32)

        def _cutmix_single_index(idx):
            idx = tf.cast(idx, tf.int32)
            image = images[idx]
            label = labels[idx]
            s_image = shuffled_images[idx]
            s_label = shuffled_labels[idx]
            ch = cut_heights[idx]
            cw = cut_widths[idx]
            cy = ys[idx]
            cx = xs[idx]

            image_h = tf.shape(image)[0]
            image_w = tf.shape(image)[1]

            y1 = tf.clip_by_value(cy - ch // 2, 0, image_h)
            y2 = tf.clip_by_value(y1 + ch, 0, image_h)
            x1 = tf.clip_by_value(cx - cw // 2, 0, image_w)
            x2 = tf.clip_by_value(x1 + cw, 0, image_w)

            mask = tf.pad(
                tf.zeros((y2 - y1, x2 - x1, 3), dtype=tf.float32),
                [[y1, image_h - y2], [x1, image_w - x2], [0, 0]],
                constant_values=1.0,
            )
            mixed_image = image * mask + s_image * (1.0 - mask)

            mix_ratio = tf.cast((y2 - y1) * (x2 - x1), tf.float32) / tf.cast(image_h * image_w, tf.float32)
            mixed_label = (1.0 - mix_ratio) * label + mix_ratio * s_label
            mixed_image.set_shape([IMG_SIZE, IMG_SIZE, 3])
            if label.shape.rank == 1 and label.shape[0] is not None:
                mixed_label.set_shape(label.shape)
            return mixed_image, mixed_label

        label_depth = labels.shape[-1]
        label_spec_dim = int(label_depth) if label_depth is not None else None

        mixed_images, mixed_labels = tf.map_fn(
            _cutmix_single_index,
            elems=tf.range(batch),
            fn_output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(label_spec_dim,), dtype=tf.float32),
            ),
        )
        return mixed_images, mixed_labels

    return tf.cond(batch > 1, _apply, lambda: (images, labels))


def _mixup_cutmix_batch(images, labels):
    return tf.cond(
        tf.less(tf.random.uniform([], 0, 1.0), CUTMIX_PROB),
        lambda: _cutmix_batch(images, labels, CUTMIX_ALPHA),
        lambda: _mixup_batch(images, labels, MIXUP_ALPHA),
    )


def _make_dataset(pairs, training=False, num_classes=None):
    if not pairs:
        empty_images = tf.zeros([0, IMG_SIZE, IMG_SIZE, 3], tf.float32)
        if USE_SOFT_LABELS:
            if num_classes is None or num_classes == 0:
                return tf.data.Dataset.from_tensor_slices(([], [])).batch(BATCH_SIZE)
            empty_labels = tf.zeros([0, num_classes], tf.float32)
        else:
            empty_labels = tf.zeros([0], tf.int32)
        return tf.data.Dataset.from_tensor_slices((empty_images, empty_labels)).batch(BATCH_SIZE)

    if USE_SOFT_LABELS and num_classes is None:
        num_classes = _num_classes_from_pairs(pairs)

    paths, labels = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        ds = ds.shuffle(buffer_size=min(len(pairs), 4096))

    num_calls = min(8, (os.cpu_count() or 8))

    def _map_fn(path, label):
        image, label = _preprocess_tf(path, label, training)
        if USE_SOFT_LABELS:
            label = tf.one_hot(label, num_classes)
        else:
            label = tf.cast(label, tf.int32)
        return image, label

    ds = ds.map(_map_fn, num_parallel_calls=num_calls)

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    ds = ds.with_options(opts)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    if training and USE_SOFT_LABELS:
        ds = ds.map(_mixup_cutmix_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


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
    # h = history.history  # dict: metric -> list per epoch
    h = {
        k: [float(x) for x in v]
        for k, v in history.history.items()
    }


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
    # Define model (ResNet50V2 backbone)
    # ----------------------------------------------------------
    base_model = ResNet50V2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_1")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=BASE_LR),
        # loss=losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        loss = losses.SparseCategoricalCrossentropy(),
        metrics=[
            # "categorical_accuracy"
            "accuracy"
        ]
    )

    model.summary()

    early_stop = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=PATIENCE,
        verbose=1,
        min_lr=1e-6, # Ask if min means that this is the largest vlaue, basically the minimum step, or the smallest basically the floor
    )

    log_data_training("Starting initial training phase")

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
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
            # loss=losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            loss = losses.SparseCategoricalCrossentropy(),
            metrics=[
                # "categorical_accuracy"
                "accuracy"
            ]
        )

        fine_tune_reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE,
            verbose=1,
            min_lr=1e-6, # Ask if min means that this is the largest vlaue, basically the minimum step, or the smallest basically the floor
        )

        fine_tune_history = model.fit(
            train_ds,
            epochs=EPOCHS + FINETUNE_EPOCHS,
            initial_epoch=history.epoch[-1] + 1,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=[early_stop, fine_tune_reduce_lr],
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
