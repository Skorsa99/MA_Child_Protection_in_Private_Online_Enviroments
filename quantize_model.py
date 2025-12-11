#!/usr/bin/env python3
"""
Quantize an existing TensorFlow.js layers model (model.json + weight shards)
and save the quantized copy as a subfolder inside the same model version
directory.

Edit the config block below or pass CLI flags to point at the model you want
to quantize. Example:

    python quantize_model.py --model-dir models/V_4_9 --output-subdir model_tfjs_q8
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# =====================
# Config defaults
# =====================

MODELS_ROOT = Path("models")                # Root folder containing versioned model folders
MODEL_VERSION = "V_4_6_2"                   # Version folder name (overridden by --model-dir)
TFJS_SUBDIR_NAME = "model_tfjs"             # Folder with model.json + weight shards
OUTPUT_SUBDIR_NAME = "model_tfjs_q8"        # Where the quantized model will be written
QUANTIZATION_BYTES = 1                      # 1=8-bit, 2=16-bit, 4=32-bit (no compression)
WEIGHT_SHARD_SIZE_BYTES = 4 * 1024 * 1024   # Shard size used by the converter
OVERWRITE_EXISTING = False                  # If True, delete an existing output folder first
COPY_LABELS = True                          # Copy labels.json into the quantized folder


def _resolve_converter() -> list[str]:
    """
    Return the tensorflowjs_converter command as a list suitable for subprocess.
    Prefers the binary on PATH; raises if not found.
    """
    binary = shutil.which("tensorflowjs_converter")
    if binary:
        return [binary]
    raise FileNotFoundError(
        "tensorflowjs_converter not found. Install tensorflowjs (`pip install tensorflowjs`)."
    )


def quantize_tfjs_layers_model(
    model_dir: Path,
    tfjs_subdir: str,
    output_subdir: str,
    quantization_bytes: int,
    weight_shard_size_bytes: int,
    overwrite: bool = False,
    copy_labels: bool = True,
) -> Path:
    """
    Quantize a TFJS layers model using tensorflowjs_converter.

    Args:
        model_dir: Folder containing the TFJS subfolder (e.g., models/V_4_9).
        tfjs_subdir: Name of the folder inside model_dir with model.json.
        output_subdir: Name of the folder to create for the quantized model.
        quantization_bytes: 1, 2, or 4 bytes per weight (1 = 8-bit).
        weight_shard_size_bytes: Desired shard size for the output weights.
        overwrite: Whether to delete an existing output folder first.
        copy_labels: Copy labels.json into the quantized folder if present.

    Returns:
        Path to the quantized model directory.
    """
    input_dir = model_dir / tfjs_subdir
    model_json = input_dir / "model.json"
    labels_file = input_dir / "labels.json"
    output_dir = model_dir / output_subdir

    if not model_json.is_file():
        raise FileNotFoundError(f"Missing model.json at {model_json}")

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output folder {output_dir} already exists; use --overwrite to replace it."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        _resolve_converter()
        + [
            "--input_format",
            "tfjs_layers_model",
            "--output_format",
            "tfjs_layers_model",
            "--quantization_bytes",
            str(quantization_bytes),
            "--weight_shard_size_bytes",
            str(weight_shard_size_bytes),
            str(model_json),
            str(output_dir),
        ]
    )

    print("Running converter:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Quantization failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if copy_labels and labels_file.is_file():
        shutil.copy2(labels_file, output_dir / "labels.json")

    print(f"Quantized model written to: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a TensorFlow.js layers model and save it to a subfolder."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_ROOT / MODEL_VERSION,
        help="Path to the model version folder containing the TFJS subfolder.",
    )
    parser.add_argument(
        "--tfjs-subdir",
        default=TFJS_SUBDIR_NAME,
        help="Subfolder under model-dir that contains model.json.",
    )
    parser.add_argument(
        "--output-subdir",
        default=OUTPUT_SUBDIR_NAME,
        help="Name of the subfolder to write the quantized model into.",
    )
    parser.add_argument(
        "--quantization-bytes",
        type=int,
        choices=[1, 2, 4],
        default=QUANTIZATION_BYTES,
        help="Number of bytes per weight (1=8-bit, 2=16-bit, 4=32-bit/no quantization).",
    )
    parser.add_argument(
        "--weight-shard-size",
        type=int,
        default=WEIGHT_SHARD_SIZE_BYTES,
        help="Weight shard size in bytes for the output model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_EXISTING,
        help="Delete the output folder if it already exists.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Do not copy labels.json to the quantized output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantize_tfjs_layers_model(
        model_dir=args.model_dir,
        tfjs_subdir=args.tfjs_subdir,
        output_subdir=args.output_subdir,
        quantization_bytes=args.quantization_bytes,
        weight_shard_size_bytes=args.weight_shard_size,
        overwrite=args.overwrite,
        copy_labels=not args.skip_labels,
    )


if __name__ == "__main__":
    main()
