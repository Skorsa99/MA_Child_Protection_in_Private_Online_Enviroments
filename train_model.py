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
MAIN_VERSION = 2

MANIFEST_PATH = "data/manifests/V005_70_20_10_per_source/manifest.jsonl" # Holds the Path to the manifest
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


# -----------------------------------------------------------------------------------------------------------------------------------
# - Mainfunctions -------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------
# - Controllsection -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------