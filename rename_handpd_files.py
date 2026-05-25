"""
Standardize the HandPD dataset into participant-aware folders and filenames.

Expected output format:
- Healthy:   HPHP001/HPHP001_hand_spiral_01.png
- Parkinson: HPPD001/HPPD001_hand_spiral_01.png

The script copies files into a new standardized tree so the raw dataset remains
untouched.
"""

from collections import defaultdict
from pathlib import Path
import re
import shutil


RAW_ROOT = Path(r"D:/Final Semester/Thesis Work/Codes/Dataset/Spiral_Handwriting/HandPD/Spiral_HandPD")
STANDARDIZED_ROOT = RAW_ROOT.parent / "Spiral_HandPD_Standardized"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

CLASS_ALIASES = {
    "healthy": "healthy",
    "control": "healthy",
    "hc": "healthy",
    "spiralcontrol": "healthy",
    "parkinson": "parkinson",
    "pd": "parkinson",
    "patient": "parkinson",
    "spiralpatients": "parkinson",
}

CLASS_PREFIX = {
    "healthy": "HPHP",
    "parkinson": "HPPD",
}


def canonical_class_name(folder_name: str) -> str:
    return CLASS_ALIASES.get(folder_name.strip().lower(), folder_name.strip().lower())


def find_class_folders(root: Path):
    class_folders = {}

    for child in root.iterdir():
        if child.is_dir():
            canonical = canonical_class_name(child.name)
            if canonical in CLASS_PREFIX:
                class_folders[canonical] = child

    if not class_folders:
        for child in root.iterdir():
            if child.is_dir():
                lowered = child.name.strip().lower()
                if lowered == "spiralcontrol":
                    class_folders["healthy"] = child
                elif lowered == "spiralpatients":
                    class_folders["parkinson"] = child

    return class_folders


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def extract_subject_token(path: Path, class_folder: Path) -> str:
    parent_name = path.parent.name

    if parent_name != class_folder.name:
        standardized_parent = re.match(r"^(HPHP|HPPD)(\d{3})$", parent_name, flags=re.IGNORECASE)
        if standardized_parent:
            return standardized_parent.group(0).upper()

        if re.search(r"\d", parent_name):
            return parent_name

    stem = path.stem
    if "_" in stem:
        return stem.split("_")[0]
    if "-" in stem:
        return stem.split("-")[0]

    return stem


def extract_sample_index(stem: str, fallback_index: int) -> int:
    match = re.search(r"(?:hand_spiral[_-]?|[_-])(\d+)$", stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    return fallback_index


def normalize_subject_serial(subject_token: str) -> str:
    digits = re.findall(r"\d+", str(subject_token))
    if not digits:
        raise ValueError(f"Cannot derive numeric subject id from '{subject_token}'")

    return f"{int(''.join(digits)):03d}"


def build_subject_id(class_name: str, subject_token: str) -> str:
    prefix = CLASS_PREFIX[class_name]
    return f"{prefix}{normalize_subject_serial(subject_token)}"


def main():
    expected_tail = ("HandPD", "Spiral_HandPD")
    if RAW_ROOT.parts[-2:] != expected_tail:
        raise RuntimeError(
            f"Refusing to run outside HandPD root. Expected path ending in {expected_tail}, got {RAW_ROOT}"
        )

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"HandPD dataset folder not found: {RAW_ROOT}")

    class_folders = find_class_folders(RAW_ROOT)
    if not class_folders:
        raise FileNotFoundError(
            f"Could not find healthy/parkinson class folders under {RAW_ROOT}"
        )

    STANDARDIZED_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for class_name, class_folder in sorted(class_folders.items()):
        subject_groups = defaultdict(list)

        for image_path in iter_images(class_folder):
            subject_token = extract_subject_token(image_path, class_folder)
            sample_index = extract_sample_index(image_path.stem, 0)
            subject_groups[subject_token].append((sample_index, image_path))

        for subject_token in sorted(subject_groups, key=lambda value: normalize_subject_serial(value)):
            subject_id = build_subject_id(class_name, subject_token)
            target_subject_folder = STANDARDIZED_ROOT / class_name / subject_id
            target_subject_folder.mkdir(parents=True, exist_ok=True)

            ordered_samples = sorted(
                subject_groups[subject_token],
                key=lambda item: (item[0] if item[0] > 0 else 10**9, item[1].name.lower())
            )

            for fallback_index, (sample_index, source_path) in enumerate(ordered_samples, start=1):
                final_sample_index = sample_index if sample_index > 0 else fallback_index
                standardized_filename = f"{subject_id}_hand_spiral_{final_sample_index:02d}{source_path.suffix.lower()}"
                target_path = target_subject_folder / standardized_filename

                shutil.copy2(source_path, target_path)

                manifest_rows.append({
                    "dataset_name": "HandPD",
                    "class_label": class_name,
                    "raw_subject_id": subject_token,
                    "subject_id": subject_id,
                    "sample_index": final_sample_index,
                    "original_filename": source_path.name,
                    "standardized_filename": standardized_filename,
                    "source_path": str(source_path),
                    "standardized_path": str(target_path),
                })

                print(f"{source_path.name} -> {target_path}")

    manifest_path = STANDARDIZED_ROOT / "handpd_standardization_manifest.csv"

    import pandas as pd

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print("HandPD standardization complete")
    print(f"Standardized root: {STANDARDIZED_ROOT}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()