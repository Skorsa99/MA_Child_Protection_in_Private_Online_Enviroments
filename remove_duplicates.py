import os
import sys
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Optional, Union

# PIL for pixel-based hashing (ignores metadata)
try:
    from PIL import Image, ImageOps, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # be tolerant with slightly corrupted files
except ImportError as e:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e

from custom_logging import log_duplicate_remove, image_tally

# Default root folder (can be overridden by function arg or CLI arg)
CATEGORY = "duplicate_test"
outer_dir = f"data/reddit_pics/{CATEGORY}"
deleted_path = "data/deleted_pics/duplicates"


def _pixel_hash(path: Path) -> str:
    """Return SHA-256 of normalized pixel data (size+pixels), ignoring metadata/EXIF.

    Steps:
    - Open with PIL
    - Apply EXIF orientation (so rotated-via-metadata images normalize)
    - Convert to RGB; flatten alpha to white if present
    - Hash image size and raw pixel bytes
    """
    with Image.open(path) as img:
        # Normalize orientation from EXIF
        img = ImageOps.exif_transpose(img)

        # Normalize mode: flatten alpha onto white background
        if img.mode in ("RGBA", "LA"):
            if img.mode == "LA":
                img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # use alpha channel
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Compute hash from size + pixel bytes (metadata excluded)
        h = hashlib.sha256()
        h.update(str(img.size).encode("utf-8"))  # include dimensions to avoid collisions across reshapes
        h.update(img.tobytes())
        return h.hexdigest()


def remove_duplicates(root_dir: Optional[Union[str, Path]] = None) -> int:
    """
    Recursively walk `root_dir` and move *pixel-identical* images (duplicates) to a separate folder,
    regardless of filename or metadata. Keeps the first encountered file in place and moves later duplicates.

    Returns the number of removed files.
    """
    if root_dir is None:
        root_dir = outer_dir

    root = Path(root_dir)
    if not root.exists():
        print(f"Path does not exist: {root}")
        return 0
    
    # Determine destination root for moved duplicates
    dest_root = Path(deleted_path) if deleted_path else (root / "_duplicates")
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create destination folder '{dest_root}': {e}")
        return 0

    # Image extensions to consider (lowercase)
    img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.webp', '.jfif', '.heic'}

    seen_hashes: Dict[str, Path] = {}
    removed = 0

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if not p.is_file():
                    continue
                # Only consider likely image files by extension
                if p.suffix.lower() not in img_exts:
                    continue

                digest = _pixel_hash(p)
                if digest in seen_hashes:
                    # Duplicate detected -> move this one
                    try:
                        # preserve subfolder structure under dest_root
                        rel_dir = Path(dirpath).relative_to(root)
                        dest_dir = dest_root / rel_dir
                        dest_dir.mkdir(parents=True, exist_ok=True)

                        dest_file = dest_dir / p.name
                        if dest_file.exists():
                            # avoid collisions in destination by appending a short hash + counter
                            dest_file = dest_dir / f"{p.stem}__dup_{removed+1}_{digest[:8]}{p.suffix}"

                        shutil.move(str(p), str(dest_file))
                        removed += 1
                        image_tally(-1, CATEGORY)
                        # print(f"Moved duplicate: {p} -> {dest_file} (same pixels as {seen_hashes[digest]})")
                    except Exception as e:
                        print(f"Failed to move {p}: {e}")
                else:
                    seen_hashes[digest] = p
            except Exception as e:
                # Skip unreadable/problematic files but continue processing
                print(f"Skipping {p}: {e}")

    end_message = f"Removed {removed} pixel-identical duplicate image(s) in '{root.resolve()}'."
    print(end_message)
    log_duplicate_remove(end_message)
    return removed


if __name__ == "__main__":
    arg_dir = "data/reddit_pics"
    remove_duplicates(arg_dir)