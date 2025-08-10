#!/usr/bin/env python3
"""
make_manifest.py — Build JSON manifests for a 3-class image classification dataset.

Assumptions
-----------
Directory layout (files can be nested inside each source folder):
    DATASET_ROOT/
        unsafe/
            <source_A>/
                img1.jpg
                subdir/img2.png
            <source_B>/...
        safe/
            <source_C>/...
        empty/
            <source_D>/...

What it produces (in OUTPUT_DIR):
    - dataset_meta.json        : Info about classes, ratios, and split counts.
    - all.jsonl                : One JSON per line; each record includes its split.
    - train.jsonl / val.jsonl / test.jsonl

Each JSON record looks like:
{
  "path": "unsafe/source_A/img1.jpg",   # always relative to DATASET_ROOT
  "label": "unsafe",
  "label_id": 0,
  "source": "source_A",
  "split": "train"
  // optional fields if enabled: "sha256", "size_bytes", "width", "height"
}

You can edit the config block below or override via CLI flags.
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import random
import hashlib
import argparse
import dataclasses
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from custom_logging import log_data_manifest

# =====================
# Config (defaults)
# =====================

DATASET_ROOT = Path("data/reddit_pics") # Root directory containing class folders (unsafe/safe/empty)
CLASS_NAMES = ["unsafe", "safe", "empty"] # Class names (order defines label IDs). Must match subfolder names under DATASET_ROOT.
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10} # Split ratios must sum to 1.0 (within small tolerance)
# Split modes: "per_source" | "flat" | "keep_groups"
SPLIT_MODE = "per_source"  # default spreads each source across train/val/test
RANDOM_SEED = 42 # Random seed for reproducibility
OUTPUT_DIR = Path("data/manifests") # Where to write the output manifests
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"} # File extensions to include

# Whether to compute extra metadata (hash, size, width/height). Width/height requires Pillow.
INCLUDE_SHA256 = False
INCLUDE_SIZE = False
INCLUDE_IMAGE_DIMS = False  # requires Pillow; auto-disabled if Pillow isn't available


# =====================
# Implementation
# =====================

try:
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

@dataclasses.dataclass
class Record:
    path: str          # relative to DATASET_ROOT
    label: str
    label_id: int
    source: str        # first-level subfolder inside the class folder
    split: str         # train/val/test
    sha256: str | None = None
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None

    def to_json(self) -> str:
        d = {
            "path": self.path,
            "label": self.label,
            "label_id": self.label_id,
            "source": self.source,
            "split": self.split,
        }
        if self.sha256 is not None:
            d["sha256"] = self.sha256
        if self.size_bytes is not None:
            d["size_bytes"] = self.size_bytes
        if self.width is not None and self.height is not None:
            d["width"] = self.width
            d["height"] = self.height
        return json.dumps(d, ensure_ascii=False)

def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in ALLOWED_EXTS:
                yield p

def _first_level_subfolder(class_dir: Path, file_path: Path) -> str:
    """
    Return the first-level subfolder name under class_dir that contains file_path.
    Example: class_dir=/.../unsafe, file_path=/.../unsafe/sourceA/x.jpg -> "sourceA"
    If file is directly under class_dir, returns "" (empty string).
    """
    try:
        rel = file_path.relative_to(class_dir)
    except Exception:
        return ""
    parts = rel.parts
    return parts[0] if len(parts) > 1 else ""

def _groupped_split(counts: Dict[str, int], ratios: Dict[str, float]) -> Dict[str, int]:
    """
    Given counts per split target (desired total items) and a list of group sizes,
    we allocate greedily to get as close as possible to the targets.
    Not used directly; see split_groups().
    """
    return counts  # unused; kept for clarity

def split_groups(groups: List[Tuple[str, int]], ratios: Dict[str, float], seed: int) -> Dict[str, List[str]]:
    """
    Split groups (identified by name) into train/val/test by approximately matching
    total item ratios. `groups` is a list of (group_name, group_size).
    Returns a dict: {split_name: [group_name, ...]}.
    Greedy allocator with random shuffle for reproducibility.
    """
    rng = random.Random(seed)
    rng.shuffle(groups)  # in-place

    total = sum(size for _, size in groups)
    targets = {k: ratios[k] * total for k in ratios}
    assigned: Dict[str, List[str]] = {k: [] for k in ratios}
    loads = {k: 0 for k in ratios}

    # Greedy: for each group, assign to the split currently furthest below its target
    for gname, gsize in groups:
        deficits = {k: targets[k] - loads[k] for k in ratios}
        # pick the split with maximum deficit; tie-break random but deterministic
        best_split = max(deficits.items(), key=lambda kv: (kv[1], rng.random()))[0]
        assigned[best_split].append(gname)
        loads[best_split] += gsize

    return assigned

def split_items(n_items: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """
    Given n_items and desired ratios, compute integer counts per split that sum to n_items.
    Uses largest remainder method for stable rounding.
    """
    targets = {k: ratios[k] * n_items for k in ratios}
    floor_counts = {k: int(math.floor(targets[k])) for k in ratios}
    remainder = n_items - sum(floor_counts.values())
    # distribute remainders by largest fractional parts
    fracs = sorted(((k, targets[k] - floor_counts[k]) for k in ratios),
                   key=lambda kv: kv[1], reverse=True)
    for i in range(remainder):
        floor_counts[fracs[i % len(fracs)][0]] += 1
    return floor_counts

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def discover_dataset(dataset_root: Path, class_names: List[str]) -> Dict[str, Dict[str, List[Path]]]:
    """
    Returns a nested dict: {class_name: {source_name: [file_paths...]}}.
    source_name is first-level subfolder under the class folder; can be "" if none.
    """
    result: Dict[str, Dict[str, List[Path]]] = {}
    for cls in class_names:
        class_dir = dataset_root / cls
        if not class_dir.is_dir():
            print(f"[WARN] Missing class dir: {class_dir}", file=sys.stderr)
            result[cls] = {}
            continue
        buckets: Dict[str, List[Path]] = {}
        for fp in _iter_files(class_dir):
            src = _first_level_subfolder(class_dir, fp)
            buckets.setdefault(src, []).append(fp)
        result[cls] = buckets
    return result

def build_records(dataset_root: Path,
                  structure: Dict[str, Dict[str, List[Path]]],
                  class_to_id: Dict[str, int],
                  ratios: Dict[str, float],
                  split_mode: str,
                  seed: int,
                  extra_sha256: bool,
                  extra_size: bool,
                  extra_dims: bool) -> Tuple[List[Record], Dict[str, int]]:
    """
    Create Record list and return (records, split_counts).
    """
    rng = random.Random(seed)
    records: List[Record] = []
    split_counts = {"train": 0, "val": 0, "test": 0}

    for cls, sources in structure.items():
        label_id = class_to_id[cls]
        if split_mode == "keep_groups":
            # prepare groups
            groups = [(src, len(files)) for src, files in sources.items() if len(files) > 0]
            # edge case: if there are no sources but direct files (src == ""), it's still one group
            if not groups and "" in sources:
                groups = [("", len(sources[""]))]
            assigned_groups = split_groups(groups, ratios, seed + label_id)

            for split, src_list in assigned_groups.items():
                for src in src_list:
                    files = sources.get(src, [])
                    for fp in files:
                        rec = _file_to_record(dataset_root, fp, cls, label_id, src,
                                              split, extra_sha256, extra_size, extra_dims, rng)
                        records.append(rec)
                        split_counts[split] += 1

        elif split_mode == "flat":
            # flat list of files, split by count per class
            all_files: List[Path] = [fp for files in sources.values() for fp in files]
            rng.shuffle(all_files)
            counts = split_items(len(all_files), ratios)
            idx = 0
            for split in ("train", "val", "test"):
                for _ in range(counts[split]):
                    fp = all_files[idx]; idx += 1
                    src = _first_level_subfolder(dataset_root / cls, fp)
                    rec = _file_to_record(dataset_root, fp, cls, label_id, src,
                                          split, extra_sha256, extra_size, extra_dims, rng)
                    records.append(rec)
                    split_counts[split] += 1

        elif split_mode == "per_source":
            # split inside each source independently so each source contributes to all splits
            for src, files in sources.items():
                if not files:
                    continue
                files = files[:]  # copy
                rng.shuffle(files)
                counts = split_items(len(files), ratios)
                idx = 0
                for split in ("train", "val", "test"):
                    for _ in range(counts[split]):
                        fp = files[idx]; idx += 1
                        rec = _file_to_record(dataset_root, fp, cls, label_id, src,
                                              split, extra_sha256, extra_size, extra_dims, rng)
                        records.append(rec)
                        split_counts[split] += 1
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

    return records, split_counts

def _file_to_record(dataset_root: Path, fp: Path, cls: str, label_id: int, src: str,
                    split: str, extra_sha256: bool, extra_size: bool, extra_dims: bool,
                    rng: random.Random) -> Record:
    rel_path = str(fp.relative_to(dataset_root).as_posix())

    sha = None
    if extra_sha256:
        try:
            sha = compute_sha256(fp)
        except Exception as e:
            print(f"[WARN] sha256 failed for {fp}: {e}", file=sys.stderr)

    size_bytes = None
    if extra_size:
        try:
            size_bytes = fp.stat().st_size
        except Exception as e:
            print(f"[WARN] size stat failed for {fp}: {e}", file=sys.stderr)

    width = height = None
    if extra_dims and _PIL_AVAILABLE:
        try:
            with Image.open(fp) as im:
                width, height = im.size
        except Exception as e:
            print(f"[WARN] reading dims failed for {fp}: {e}", file=sys.stderr)
    elif extra_dims and not _PIL_AVAILABLE:
        print("[WARN] INCLUDE_IMAGE_DIMS=True but Pillow is not available. Skipping dims.", file=sys.stderr)

    return Record(
        path=rel_path,
        label=cls,
        label_id=label_id,
        source=src,
        split=split,
        sha256=sha,
        size_bytes=size_bytes,
        width=width,
        height=height,
    )

def validate_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    keys = set(ratios.keys())
    required = {"train", "val", "test"}
    if keys != required:
        missing = required - keys
        extra = keys - required
        if missing:
            raise ValueError(f"SPLIT_RATIOS missing keys: {missing}")
        if extra:
            raise ValueError(f"SPLIT_RATIOS has extra keys: {extra}")

    total = sum(ratios.values())
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"SPLIT_RATIOS must sum to 1.0, got {total}")
    return ratios

def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_json())
            f.write("\n")


def _ratios_tag(r: dict) -> str:
    # 0.8,0.1,0.1 -> "80-10-10"
    def pct(x: float) -> str:
        return f"{int(round(x * 100)):02d}"
    return f"{pct(r['train'])}_{pct(r['val'])}_{pct(r['test'])}"

def _next_version_dir(base: Path, ratios: dict, split_mode: str) -> Path:
    """
    Find the next version folder name: V###_<ratios>_<split_mode>
    Increments the numeric prefix regardless of the tag.
    """
    max_n = -1
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^V(\d+)_", p.name)
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except ValueError:
                pass
    n = max_n + 1
    tag = f"{_ratios_tag(ratios)}_{split_mode}"
    return base / f"V{n:03d}_{tag}"

def main() -> int:
    dataset_root: Path = DATASET_ROOT
    output_dir: Path = OUTPUT_DIR
    class_names: List[str] = list(CLASS_NAMES)
    ratios = dict(SPLIT_RATIOS)

    # Validate ratios
    try:
        ratios = validate_ratios(ratios)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Build class map
    class_to_id = {cls: i for i, cls in enumerate(class_names)}

    # Basic checks
    if not dataset_root.exists():
        print(f"ERROR: dataset_root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    # Discover files
    structure = discover_dataset(dataset_root, class_names)

    # Create records
    records, split_counts = build_records(
        dataset_root=dataset_root,
        structure=structure,
        class_to_id=class_to_id,
        ratios=ratios,
        split_mode=SPLIT_MODE,
        seed=RANDOM_SEED,
        extra_sha256=INCLUDE_SHA256,
        extra_size=INCLUDE_SIZE,
        extra_dims=INCLUDE_IMAGE_DIMS,
    )

    # Versioned output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    version_dir = _next_version_dir(output_dir, ratios, SPLIT_MODE)
    version_dir.mkdir(parents=True, exist_ok=False)

    # Write outputs
    write_jsonl(version_dir / "manifest.jsonl", records)

    # If you want per-split files later, uncomment:
    # for split in ("train", "val", "test"):
    #     split_recs = (r for r in records if r.split == split)
    #     write_jsonl(version_dir / f"{split}.jsonl", split_recs)

    meta = {
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "class_to_id": class_to_id,
        "split_ratios": ratios,
        "split_mode": SPLIT_MODE,
        "random_seed": RANDOM_SEED,
        "allowed_exts": sorted(ALLOWED_EXTS),
        "counts": {
            "total": len(records),
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
        },
        "version_dir": str(version_dir),
    }
    with (version_dir / "manifest_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    message = f"[OK] Wrote manifests to: {version_dir}"
    print(message)
    log_data_manifest(message)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
