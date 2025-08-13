import os, json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Config ----------------
classes = ('unsafe', 'safe', 'empty')
manifest_path = "data/manifests/V002_80_10_10_per_source/manifest.jsonl"
IMG_SIZE = 32
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
BASE_DIR = "data/reddit_pics"

# -------------- Manifest loading --------------

def load_manifest(path: str):
    path = Path(path)
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # skip bad lines
                continue
    return items


def split_records(items):
    # Ensure label_id exists. If not, map from "label"
    # Use classes tuple for stable mapping if needed
    label_to_id = {lab: i for i, lab in enumerate(classes)}
    for it in items:
        if 'label_id' not in it:
            it['label_id'] = label_to_id.get(it.get('label', ''), 0)

    train = [it for it in items if str(it.get('split')).lower() == 'train']
    val_keys = {'val', 'valid', 'validation'}
    val = [it for it in items if str(it.get('split')).lower() in val_keys]
    test = [it for it in items if str(it.get('split')).lower() == 'test']
    return train, val, test


# -------------- tf.data helpers --------------

def _decode_and_resize(img_bytes):
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def make_dataset(records, training: bool):
    paths = [str(Path(BASE_DIR) / r['path']) for r in records]
    labels = [int(r['label_id']) for r in records]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes)
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
        return img, label

    if training:
        ds = ds.shuffle(buffer_size=max(len(paths), 1000), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# -------------- Build simple CNN (kept from your draft) --------------

model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(len(classes), activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------- Data wiring --------------

items = load_manifest(manifest_path)
train_recs, val_recs, test_recs = split_records(items)

if not train_recs:
    raise SystemExit("No training records found in manifest.")

train_ds = make_dataset(train_recs, training=True)

# If no explicit val split, fall back to test for validation during training
val_ds = make_dataset(val_recs, training=False) if val_recs else (make_dataset(test_recs, training=False) if test_recs else None)

epochs = 10
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
)

# -------------- Quick viz + test eval --------------

def view_classification(image, probabilities):
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 4), ncols=2)
    ax1.imshow(np.clip(image, 0, 1))
    ax1.axis('off')
    ax2.barh(np.arange(len(classes)), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

# Evaluate on test split if present
if test_recs:
    test_ds = make_dataset(test_recs, training=False)
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f'Accuracy on test set: {test_accuracy * 100:.2f}%')

    # Take one image from test for visualization
    for img_batch, lbl_batch in test_ds.take(1):
        img0 = img_batch[0].numpy()
        probs = model.predict(img_batch[:1], verbose=0)[0]
        view_classification(img0, probs)
        break
else:
    print('No test split found in manifest; skipping final evaluation.')

# -------------- Export for browser (TensorFlow.js) --------------
out_dir = Path("models_V2")
out_dir.mkdir(parents=True, exist_ok=True)
try:
    import tensorflowjs as tfjs
    # Save model in TF.js Layers format: creates model.json + shards
    tfjs.converters.save_keras_model(model, str(out_dir))
    # Save metadata for the frontend
    meta = {
        "classes": list(classes),
        "img_size": IMG_SIZE,
        "framework": "tfjs-layers",
        "created_by": "train_model_V2.py"
    }
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved TFJS model to {out_dir}/model.json")
except Exception as e:
    print(f"[WARN] Failed to export TFJS model: {e}\nInstall the converter with: pip install tensorflowjs")