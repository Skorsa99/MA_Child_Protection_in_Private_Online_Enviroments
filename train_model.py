# import cv2
import imghdr
import pathlib
import numpy as np
import os, json, re

os.environ["TF_USE_LEGACY_KERAS"] = "1" # setup for tensorflow import

import tensorflowjs as tfjs
import tensorflow as tf, inspect
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers, layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, SparseCategoricalAccuracy

from custom_logging import log_data_training


print("TF module file:", getattr(tf, "__file__", "NO __file__"))
print("Has keras?", hasattr(tf, "keras"))

# --------------------------------------------------------------------------------------------------
# - Setup ------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


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

MAIN_VERSION = 4

# override the earlier `version` with an automatic next version string
version = _get_next_model_version(MAIN_VERSION, models_dir="models")
print(f"Auto-selected version: {version}")


# data_dir = "../Master/ARiES/Project/SZ - NSFW Detections/Project 2 - Practical Part/MA_Child_Protection_in_Private_Online_Enviroments/data/reddit_pics"
data_dir = "data/working_data"
out_dir = f"models/{version}"
out_dir = pathlib.Path(out_dir)
out_dir.mkdir(exist_ok=True)


log_data_training("--------------------------------------------------------------------------")
log_data_training(f"Started prozess for training Version: {version}")


# - Config ------------
EPOCHS        = 20
BATCH_SIZE    = 64 # 32
LR            = 1e-5 # little small but lets give it a go
MIN_LR        = 5e-8
PATIENCE      = 2
REDUCE_FACTOR = 0.3
L2            = regularizers.l2(1e-4) # weight-decay strength
IMG_SIZE      = 256
# ---------------------


log_data_training("Settings:")
log_data_training(f"Epochs:                 {EPOCHS}")
log_data_training(f"Batch Size:             {BATCH_SIZE}")
log_data_training(f"Lerningrate:            {LR}")
log_data_training(f"Minimum Lerningrat:     {MIN_LR}")
log_data_training(f"Patience:               {PATIENCE}")
log_data_training(f"REduce Factor:          {REDUCE_FACTOR}")
log_data_training(f"L2:                     {L2}")
log_data_training(f"Image Size:             {IMG_SIZE}")


"""
gpus = tf.config.experimental.list_physical_devices('GPU')              # Wont do anything on mac, but good for nvidia-gpus
for gpu in gpus:                                                        # Wont do anything on mac, but good for nvidia-gpus
    tf. config.experimental.set_memory_growth(gpu, True)                # Wont do anything on mac, but good for nvidia-gpus
"""
    

# --------------------------------------------------------------------------------------------------
# - Remove dodgy images ----------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
img_exts = ['jpeg', 'jpg', 'bmp', 'png']
"""
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2. imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in img_exts:
                print('Image not inext list {}' .format(image_path))
                # os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'. format(image_path))
            #os.remove(image_path)
"""


# --------------------------------------------------------------------------------------------------
# - Load Data --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
seed = 1337

log_data_training(f"Loading Data with Seed: {seed}")

# data = tf.keras.utils.image_dataset_from_directory('data') # Original, but i need it recursively, if that doesnt work ill make it work
raw_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='int',     # or 'categorical', or 'binary' — depending on your setup
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    # shuffle=True,
    shuffle=True,
    seed=seed
)

class_names   = raw_ds.class_names
total_batches = len(raw_ds)

# - Regularization ----
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),       # mirroring
    tf.keras.layers.RandomRotation(0.08),           # ± ≈15°
    tf.keras.layers.RandomZoom(0.1),                # in/out 10 %
    tf.keras.layers.RandomContrast(0.1)             # ± 10 % contrast
], name="augment")

# data = raw_ds.ignore_errors().map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: (x/255., y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
# ---------------------
data = raw_ds.ignore_errors()

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

print("")
print("----------------------------------")
# class_names = data.class_names          # e.g. ['benign', 'mild', 'severe']
print("Class index → name mapping:")
for idx, name in enumerate(class_names):
    print(f"  {idx}: {name}")
print("")
print(batch[1])
print(batch[0].shape)
print("----------------------------------")

log_data_training(f"Found {len(class_names)} classes")


# --------------------------------------------------------------------------------------------------
# - Preprocess Data --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# data = data.map(lambda x,y: (x/255, y)) # Removed because of regularization above
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

print(f"Number of Batches: {total_batches}")
print("")

log_data_training(f"Total Batches: {total_batches} with a batchsize of: {BATCH_SIZE}")


# --------------------------------------------------------------------------------------------------
# - Split Data -------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
train_size = int(total_batches * .7)
val_size   = int(total_batches * .2)
# test_size  = int(total_batches * .1)
test_size  = total_batches - train_size - val_size  # keeps all samples

if train_size + val_size + test_size > total_batches:
    print("")
    print("   -------------------------------")
    print("    Batch missmatch")
    print("   -------------------------------")
    print("")

# Maybe allocate remaining batches to train

# train = data.take(train_size).map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: (x/255., y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
train = data.take(train_size).map(lambda x,y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).map(lambda x,y: (tf.cast(x, tf.float32)/255., y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

rest = data.skip(train_size)

# val   = data.skip(train_size).take(val_size)
val   = rest.take(val_size).map(lambda x, y: (x/255., y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
# test  = data.skip(train_size+val_size).take(test_size)
test = rest.skip(val_size).take(test_size).map(lambda x, y: (x/255., y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

tmp_test = train.take(1)
tmp_batch = next(tmp_test.as_numpy_iterator())
print(f"Scaling check min, should be 0: {tmp_batch[0].min()}")
print(f"Scaling check max, should be 1: {tmp_batch[0].max()}")
print("----------------------------------")

print(f"Train_Batches:      {train_size}")
print(f"Validation_Batches: {val_size}")
print(f"Test_Batches:       {test_size}")
print("----------------------------------")

log_data_training(f"Train Batches: {train_size}")
log_data_training(f"Val Batches:   {val_size}")
log_data_training(f"Test Batches:  {test_size}")


# --------------------------------------------------------------------------------------------------
# - Create Model -----------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
def build_model_V1(n_classes=3):
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(IMG_SIZE, activation='relu'))
    # model.add(Dense(1, activation='sigmoid')) # For binary output, if i decide to remove the empty class
    model.add(Dense(n_classes, activation='softmax'))

    return model

def build_model_V2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                n_classes=3):
    """
    Drop-in upgrade for build_model_V1.
    • 3 conv blocks (64→128→IMG_SIZE filters, two convs each)
    • BatchNorm after every conv
    • L2 weight-decay + Dropout 0.5
    • ~2.4 M parameters – fits on a 4 GB GPU with batch_size 32
    """
    reg = L2

    model = Sequential([
        # Block 1 ------------------------------------------------------------
        Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=reg),
        layers.BatchNormalization(),
        MaxPooling2D(),

        # Block 2 ------------------------------------------------------------
        Conv2D(128, (3, 3), padding='same', activation='relu',
            kernel_regularizer=reg),
        layers.BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu',
            kernel_regularizer=reg),
        layers.BatchNormalization(),
        MaxPooling2D(),

        # Block 3 ------------------------------------------------------------
        Conv2D(IMG_SIZE, (3, 3), padding='same', activation='relu',
            kernel_regularizer=reg),
        layers.BatchNormalization(),
        Conv2D(IMG_SIZE, (3, 3), padding='same', activation='relu',
            kernel_regularizer=reg),
        layers.BatchNormalization(),
        MaxPooling2D(),

        # Classification head -----------------------------------------------
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu', kernel_regularizer=reg),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    return model

# --- Slightly larger CNN ----------------------------------------------------
def build_model_V3(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=3):
    """
    Moderate bump over V1:
        • 3 Conv blocks     (filters 32-64-128)
        • BatchNorm each layer
        • GlobalAveragePooling2D head
        • Dropout 0.3 + L2 1e-4  (uses your global `l2`)
    Params ≈ 0.9 M – small enough for batch_size 64 on a 4 GB GPU.
    """
    reg = L2  # you already defined this above (regularizers.l2(1e-4))

    model = Sequential([
        # Block 1 ------------------------------------------------------------
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        MaxPooling2D(),

        # Block 2 ------------------------------------------------------------
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        layers.BatchNormalization(),
        MaxPooling2D(),

        # Block 3 ------------------------------------------------------------
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        layers.BatchNormalization(),
        MaxPooling2D(),

        # Head ---------------------------------------------------------------
        layers.GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(IMG_SIZE, activation='relu', kernel_regularizer=reg),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    return model

def build_model_V4_transfer(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=3):
    """Using MobileNetV2 as backbone"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base initially
    base_model.trainable = True
    
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(IMG_SIZE, activation='relu', kernel_regularizer=L2),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    return model

model = build_model_V4_transfer(n_classes=len(class_names))

log_data_training(f"Using Modelfunction: build_model_V4_transfer(n_classes=len(class_names))")

# model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy']) # For binary output, if i decide to remove the empty class
model.compile(
    # optimizer='adam',
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


print(model.summary())
print("----------------------------------")


# --------------------------------------------------------------------------------------------------
# - Train Model ------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
log_dir_variable = f"logs/tensorboard_callback_logs/{version}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_variable)

# history = model.fit(train, epochs=EPOCHS, validation_data=val, callbacks=[tensorboard_callback])

# - LR-Scheduler ------
plateau = ReduceLROnPlateau(
    monitor='val_loss',            # watch this metric
    factor=REDUCE_FACTOR,          # new_lr = old_lr * 0.3
    patience=PATIENCE,             # wait 3 epochs with no improv.
    min_lr=MIN_LR,                 # don’t go below this
    verbose=1)

history = model.fit(
    train,
    epochs=EPOCHS,
    validation_data=val,
    callbacks=[tensorboard_callback, plateau])
# ---------------------


# --------------------------------------------------------------------------------------------------
# - Plot Traininghistory ---------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
fig.savefig(out_dir / "loss_curve.png", dpi=300, bbox_inches='tight')

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
fig.savefig(out_dir / "accuracy_curve.png", dpi=300, bbox_inches='tight')


# --------------------------------------------------------------------------------------------------
# - Evaluate Model ---------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
n_classes = len(class_names) # 3
pre  = Precision()
rec  = Recall()
sacc = SparseCategoricalAccuracy() # BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch

    y_pred_vec = model.predict(x, verbose=0)          # (batch, 3)
    y_pred_lbl = tf.argmax(y_pred_vec, axis=1)        # (batch,)

    # one-hot encode
    y_true_oh = tf.one_hot(y,         depth=n_classes)
    y_pred_oh = tf.one_hot(y_pred_lbl, depth=n_classes)

    pre.update_state(y_true_oh, y_pred_oh)
    rec.update_state(y_true_oh, y_pred_oh)
    sacc.update_state(y, y_pred_vec)

result_string = f"Precision: {pre.result().numpy()}; Recall: {rec.result().numpy()}; Accuracy: {sacc.result().numpy()}"
print(result_string)

log_data_training(f"Results:")
log_data_training(f"   {result_string}")


# --------------------------------------------------------------------------------------------------
# - Save Model -------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
model.save(f"{out_dir}/model_{version}.h5")
model.save(os.path.join(out_dir, f"model_{version}.keras"))

def _strip_regularizers_for_export(m):
    for layer in m.layers:
        for attr in ("kernel_regularizer", "bias_regularizer", "activity_regularizer"):
            if hasattr(layer, attr):
                setattr(layer, attr, None)
    return m


def _patch_tfjs_input_layers(model_json_path: pathlib.Path) -> None:
    """Ensure TFJS InputLayers expose batch_input_shape for compatibility."""
    try:
        data = json.loads(model_json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return

    changed = False

    def _fix(node):
        nonlocal changed
        if isinstance(node, dict):
            if node.get("class_name") == "InputLayer":
                cfg = node.get("config", {})
                if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                    cfg["batch_input_shape"] = cfg["batch_shape"]
                    changed = True
            for value in node.values():
                _fix(value)
        elif isinstance(node, list):
            for item in node:
                _fix(item)

    _fix(data)
    if changed:
        model_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

export_model = tf.keras.models.clone_model(model)
export_model.set_weights(model.get_weights())
export_model = _strip_regularizers_for_export(export_model)
tfjs.converters.save_keras_model(export_model, os.path.join(out_dir, "model_tfjs"))
_patch_tfjs_input_layers(out_dir / "model_tfjs" / "model.json")

# Create labels.json for the frontend
with open(os.path.join(out_dir, "model_tfjs", "labels.json"), "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2, ensure_ascii=False)

log_data_training(f"Model saved")