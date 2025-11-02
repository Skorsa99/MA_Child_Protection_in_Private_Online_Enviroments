from pathlib import Path
from typing import Iterable, List, Set


def collect_file_names(directory: Path) -> Set[str]:
    """Return the set of file names (with extensions) found under directory."""
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    return {path.name for path in directory.rglob("*") if path.is_file()}

def iter_image_files(directory: Path) -> Iterable[Path]:
    """Yield file paths under directory; keeps logic isolated for future filtering."""
    for path in directory.rglob("*"):
        if path.is_file():
            yield path

def find_matching_folders(tmp_dir: Path, category_dir: Path) -> List[str]:
    """Return names of subfolders under category_dir containing files matching tmp_dir."""
    tmp_names = collect_file_names(tmp_dir)
    matches: List[str] = []

    for subfolder in sorted([p for p in category_dir.iterdir() if p.is_dir()]):
        for file_path in iter_image_files(subfolder):
            if file_path.name in tmp_names:
                matches.append(subfolder.name)
                break

    return matches


def main() -> None:
    # Update these paths to point at the tmp directory and target category.
    tmp_dir      = Path("data/working_data/tmp_unsafe").resolve()
    category_dir = Path("data/reddit_pics/unsafe").resolve()

    if not tmp_dir.is_dir():
        raise SystemExit(f"tmp directory does not exist: {tmp_dir}")
    if not category_dir.is_dir():
        raise SystemExit(f"category directory does not exist: {category_dir}")

    matching_folders = find_matching_folders(tmp_dir, category_dir)

    if not matching_folders:
        print("No matching folders found.")
        return

    for folder_name in matching_folders:
        print(folder_name)


if __name__ == "__main__":
    main()
