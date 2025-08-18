import os
import re
import PIL
import pathlib
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from custom_logging import log_data_training


# -----------------------------------------------------------------------------------------------------------------------------
# - Config --------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
DATA_PATH = "data/reddit_pics"
DATA_DIR = pathlib.Path(DATA_PATH)

MAIN_VERSION = 0

EPOCHS = 10
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

AUTOTUNE = tf.data.AUTOTUNE


# -----------------------------------------------------------------------------------------------------------------------------
# - Helper --------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------------------------------------------------------
# - Control -------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
def main():
    image_count = len(list(DATA_DIR.with_suffix('').glob('*/*/*.jpeg'))) + len(list(DATA_DIR.with_suffix('').glob('*/*/*.png')))
    print(image_count)

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    """
    safe = [p for p in DATA_DIR.glob("safe/*/*")
        if p.is_file() and p.suffix.lower() in allowed]
    Image.open(str(safe[0])).show()
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(class_names)

    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
    """

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    model = Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()

    print("1---------------------------")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    print("2---------------------------")

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print("3---------------------------")

    epochs_range = range(EPOCHS)

    print("4---------------------------")

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')


    print("5---------------------------")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    print("6---------------------------")

    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    print("7---------------------------")

    with open('models/test/model.tflite', 'wb') as f:
        f.write(tflite_model)
    """

    tfjs.converters.save_keras_model(model, "models/tests/V1")

    print("8---------------------------")


if __name__ == "__main__":
    main()