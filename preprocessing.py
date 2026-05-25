"""Preprocess handwriting spiral images for downstream model training and analysis."""

import os
import cv2
import random
import warnings
import re
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent

# Default input/output (kept for backward compatibility)
DEFAULT_INPUT_BASE = Path(r"D:/Final Semester/Thesis Work/Codes/Dataset/Spiral_Handwriting")

OUTPUT_BASE = PROJECT_ROOT / "preprocessed_images"

GRAY_DIR = OUTPUT_BASE / "grayscale"

BINARY_DIR = OUTPUT_BASE / "binary"

QC_DIR = OUTPUT_BASE / "quality_check"

STEP_VIS_DIR = OUTPUT_BASE / "step_visualizations"

STEP_VISUALIZATION_INDICES = {

    "healthy": {1, 2},

    "parkinson": {6, 7}
}

CLASSES = ["healthy", "parkinson"]

CLASS_ALIASES = {
    "healthy": "healthy",
    "health": "healthy",
    "control": "healthy",
    "hc": "healthy",
    "parkinson": "parkinson",
    "pd": "parkinson",
    "patient": "parkinson",
}

HANDPD_SUBJECT_PREFIX = {
    "healthy": "HPHP",
    "parkinson": "HPPD",
}

HANDPD_BLUE_LOWER = np.array([95, 60, 60])
HANDPD_BLUE_UPPER = np.array([135, 255, 255])

IMG_SIZE = 224

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

MIN_COMPONENT_AREA = 20

for base_dir in [GRAY_DIR, BINARY_DIR]:

    for cls in CLASSES:

        os.makedirs(base_dir / cls, exist_ok=True)

os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(STEP_VIS_DIR, exist_ok=True)

metadata_records = []

def validate_image(img):

    if img is None:
        return False

    if len(img.shape) < 2:
        return False

    h, w = img.shape[:2]

    if h < 50 or w < 50:
        return False

    return True

def extract_patient_id(filename):
    stem = Path(filename).stem

    if "_" in stem:
        return stem.split("_")[0]

    if "-" in stem:
        return stem.split("-")[0]

    return stem

def canonical_dataset_name(dataset_name: Optional[str]) -> str:
    if not dataset_name:
        return "default"

    return str(dataset_name).strip()

def canonical_class_name(class_name: str) -> str:
    normalized = str(class_name).strip().lower()
    return CLASS_ALIASES.get(normalized, normalized)

def is_handpd_dataset(dataset_name: str) -> bool:
    return dataset_name.lower() == "handpd"

def infer_class_folder(input_base: Path, class_name: str) -> Path:
    canonical = canonical_class_name(class_name)

    direct_match = input_base / canonical
    if direct_match.exists():
        return direct_match

    for child in input_base.iterdir():
        if child.is_dir() and canonical_class_name(child.name) == canonical:
            return child

    raise FileNotFoundError(
        f"Missing folder for class '{class_name}' under:\n{input_base}"
    )

def iter_image_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path

def extract_sample_index(stem: str, fallback_index: int) -> int:
    matches = re.findall(r"(?:hand_spiral[_-]?|[_-])(\d+)$", stem, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])

    trailing_match = re.search(r"[-_](\d+)$", stem)
    if trailing_match:
        return int(trailing_match.group(1))

    return fallback_index

def normalize_subject_serial(subject_token: str) -> str:
    digits = re.findall(r"\d+", str(subject_token))
    if not digits:
        return str(subject_token).strip()

    serial_value = int("".join(digits))
    return f"{serial_value:03d}"

def build_subject_id(class_name: str, subject_token: str, dataset_name: str) -> str:
    canonical_class = canonical_class_name(class_name)

    if is_handpd_dataset(dataset_name):
        prefix = HANDPD_SUBJECT_PREFIX.get(canonical_class)
        if prefix is None:
            raise ValueError(f"Unsupported HandPD class: {class_name}")

        return f"{prefix}{normalize_subject_serial(subject_token)}"

    return str(subject_token).strip()

def parse_subject_and_sample(path: Path, class_name: str, dataset_name: str, fallback_index: int):
    stem = path.stem
    parent_name = path.parent.name
    canonical_class = canonical_class_name(class_name)

    if is_handpd_dataset(dataset_name):
        standardized_subject_match = re.match(r"^(HPHP|HPPD)(\d{3})$", stem, flags=re.IGNORECASE)
        if standardized_subject_match:
            subject_token = standardized_subject_match.group(0).upper()
            sample_index = extract_sample_index(stem, fallback_index)
            return subject_token, sample_index

        parent_standardized_match = re.match(r"^(HPHP|HPPD)(\d{3})$", parent_name, flags=re.IGNORECASE)
        if parent_standardized_match:
            subject_token = parent_standardized_match.group(0).upper()
        else:
            candidate_token = parent_name if parent_name != canonical_class else extract_patient_id(stem)
            subject_token = build_subject_id(canonical_class, candidate_token, dataset_name)

        sample_index = extract_sample_index(stem, fallback_index)
        return subject_token, sample_index

    subject_token = extract_patient_id(stem)
    sample_index = extract_sample_index(stem, fallback_index)

    return subject_token, sample_index

def build_output_filename(subject_id: str, sample_index: int, dataset_name: str, suffix: str) -> str:
    if is_handpd_dataset(dataset_name):
        return f"{subject_id}_hand_spiral_{sample_index:02d}{suffix}"

    return f"{subject_id}{suffix}"

def normalize_foreground(binary_img):
    white_ratio = np.mean(binary_img > 127)

    if white_ratio > 0.5:

        binary_img = cv2.bitwise_not(binary_img)

    return binary_img

def remove_small_components(binary_img, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8
    )

    cleaned = np.zeros_like(binary_img)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:

            cleaned[labels == i] = 255

    return cleaned

def crop_to_content(img, dataset_name="default"):
    coords = cv2.findNonZero(img)

    if coords is None:

        return img

    x, y, w, h = cv2.boundingRect(coords)

    padding = 40 if dataset_name.lower() == "bd_dataset" else 20

    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, img.shape[1] - x)
    h = min(h + 2 * padding, img.shape[0] - y)
    
    return img[y:y+h, x:x+w]

def resize_with_padding(img, size=224):
    h, w = img.shape[:2]

    scale = size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas

def build_handpd_blue_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, HANDPD_BLUE_LOWER, HANDPD_BLUE_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return normalize_foreground(blue_mask)

def preprocess_image(img, dataset_name="default"):
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=10
    )

    if dataset_name.lower() == "bd_dataset":

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

    else:

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    enhanced = clahe.apply(denoised)
    enhanced = cv2.medianBlur(enhanced, 3)

    if is_handpd_dataset(dataset_name):
        binary = build_handpd_blue_mask(img)
    else:
        binary = cv2.adaptiveThreshold(

            enhanced,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY_INV,

            31,

            11
        )

    raw_threshold = binary.copy()

    if dataset_name.lower() == "bd_dataset":
        kernel = np.ones((3, 3), np.uint8)
    else:
        kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    morphology_output = binary.copy()

    if dataset_name.lower() == "bd_dataset":
        binary = remove_small_components(binary, min_area=20)
    else:
        binary = remove_small_components(binary, min_area=40)

    binary = normalize_foreground(binary)

    masked_gray = cv2.bitwise_and(
        enhanced,
        enhanced,
        mask=binary
    )

    cropped_gray = crop_to_content(enhanced, dataset_name=dataset_name)

    cropped_binary = crop_to_content(binary, dataset_name=dataset_name)

    final_gray = resize_with_padding(
        cropped_gray,
        IMG_SIZE
    )

    final_binary = resize_with_padding(
        cropped_binary,
        IMG_SIZE
    )

    steps = {
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced,
        "threshold": raw_threshold if not is_handpd_dataset(dataset_name) else binary.copy(),
        "morphology": morphology_output,
        "cropped_gray": cropped_gray,
        "final_gray": final_gray,
        "final_binary": final_binary
    }

    return final_gray, final_binary, steps

def save_quality_check(

    original,
    gray,
    binary,
    filename
):

    original = cv2.resize(
        original,
        (224, 224)
    )

    if len(original.shape) == 2:

        original = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2BGR
        )

    gray_bgr = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    binary_bgr = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR
    )

    combined = np.hstack([

        original,
        gray_bgr,
        binary_bgr
    ])

    save_path = QC_DIR / filename

    cv2.imwrite(
        str(save_path),
        combined
    )

def save_step_visualization(steps, original, filename):

    step_names = [
        "Original",
        "Grayscale",
        "Denoised",
        "CLAHE",
        "Threshold",
        "Morphology",
        "Cropped",
        "Final Gray",
        "Final Binary"
    ]

    images = [
        original,
        steps["gray"],
        steps["denoised"],
        steps["enhanced"],
        steps["threshold"],
        steps["morphology"],
        steps["cropped_gray"],
        steps["final_gray"],
        steps["final_binary"]
    ]

    processed_images = []

    for img in images:

        if len(img.shape) == 2:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR
            )

        img = cv2.resize(img, (224, 224))

        processed_images.append(img)
    rows = []

    for i in range(0, 9, 3):
        row = np.hstack(processed_images[i:i+3])
        rows.append(row)

    combined = np.vstack(rows)

    for idx, name in enumerate(step_names):
        x = (idx % 3) * 224 + 10
        y = (idx // 3) * 224 + 30

        cv2.putText(
            combined,
            name,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3
        )

    save_path = STEP_VIS_DIR / filename
    cv2.imwrite(str(save_path), combined)

def process_dataset(input_base: Optional[Path] = None, output_base: Optional[Path] = None, dataset_name: Optional[str] = None):

    print("\n================================================")
    print("STARTING HANDWRITING PREPROCESSING")
    print("================================================\n")

    # Resolve input/output
    if input_base is None:
        input_base = DEFAULT_INPUT_BASE

    if output_base is None:
        output_base = OUTPUT_BASE

    dataset_tag = canonical_dataset_name(dataset_name or output_base.name or input_base.name)

    # Reset metadata for each run so per-dataset outputs do not mix.
    metadata_records.clear()

    print(f"Dataset: {dataset_tag}")
    print(f"Input Root: {input_base}")
    print(f"Output Root: {output_base}")

    # Per-dataset output directories (keeps different datasets separate)
    global GRAY_DIR, BINARY_DIR, QC_DIR, STEP_VIS_DIR
    GRAY_DIR = output_base / "grayscale"
    BINARY_DIR = output_base / "binary"
    QC_DIR = output_base / "quality_check"
    STEP_VIS_DIR = output_base / "step_visualizations"

    for base_dir in [GRAY_DIR, BINARY_DIR]:
        for cls in CLASSES:
            os.makedirs(base_dir / cls, exist_ok=True)

    os.makedirs(QC_DIR, exist_ok=True)
    os.makedirs(STEP_VIS_DIR, exist_ok=True)

    total_processed = 0
    total_skipped = 0

    for cls in CLASSES:

        input_folder = infer_class_folder(Path(input_base), cls)

        print(f"\nProcessing class: {cls}")

        image_files = list(iter_image_files(input_folder))

        for image_index, img_name in enumerate(tqdm(image_files), start=1):

            img_path = img_name

            try:

                # --------------------------------------------------
                # Load image
                # --------------------------------------------------

                img = cv2.imread(str(img_path))

                # --------------------------------------------------
                # Validate
                # --------------------------------------------------

                if not validate_image(img):

                    total_skipped += 1
                    continue

                if img is None:
                    total_skipped += 1
                    continue

                img_height, img_width = img.shape[:2]

                # --------------------------------------------------
                # Preprocess
                # --------------------------------------------------

                gray_img, binary_img, steps = preprocess_image(img, dataset_name=dataset_tag)

                subject_id, sample_index = parse_subject_and_sample(
                    img_path,
                    cls,
                    dataset_tag,
                    image_index
                )

                output_filename = build_output_filename(
                    subject_id,
                    sample_index,
                    dataset_tag,
                    img_path.suffix.lower()
                )

                # --------------------------------------------------
                # Save outputs
                # --------------------------------------------------

                gray_save_path = GRAY_DIR / cls / output_filename

                binary_save_path = BINARY_DIR / cls / output_filename

                cv2.imwrite(
                    str(gray_save_path),
                    gray_img
                )

                cv2.imwrite(
                    str(binary_save_path),
                    binary_img
                )

                # --------------------------------------------------
                # Quality check
                # --------------------------------------------------

                save_quality_check(
                    img,
                    gray_img,
                    binary_img,
                    output_filename
                )

                if image_index in STEP_VISUALIZATION_INDICES.get(cls, set()):

                    save_step_visualization(
                        steps,
                        img,
                        f"{cls}_{output_filename}"
                    )

                
                # --------------------------------------------------
                # Metadata
                # --------------------------------------------------

                patient_id = subject_id

                metadata_records.append({

                    "dataset_name": dataset_tag,

                    "subject_id": subject_id,

                    "filename": output_filename,

                    "original_filename": img_path.name,

                    "patient_id": patient_id,

                    "master_patient_id": subject_id,

                    "class": cls,

                    "class_label": cls,

                    "sample_index": sample_index,

                    "image_path": str(img_path),

                    "processed_path": str(gray_save_path),

                    "gray_path": str(gray_save_path),

                    "binary_path": str(binary_save_path),

                    "height": img.shape[0],

                    "width": img.shape[1]
                    if img is not None else 0
                })

                total_processed += 1

            except Exception as e:

                print(f"\nError processing: {img_name}")

                print(e)

                total_skipped += 1

    # ==========================================================
    # SAVE METADATA
    # ==========================================================

    metadata_df = pd.DataFrame(metadata_records)

    # Save metadata inside the per-dataset output folder
    metadata_path = output_base / "metadata.csv"

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================

    print("\n================================================")
    print("PREPROCESSING COMPLETE")
    print("================================================")

    print(f"Dataset : {dataset_tag}")

    print(f"\nProcessed : {total_processed}")

    print(f"Skipped   : {total_skipped}")

    print(f"\nMetadata CSV:")

    print(metadata_path)

    print("\nSaved Outputs:")

    print(f"Grayscale -> {GRAY_DIR}")

    print(f"Binary    -> {BINARY_DIR}")

    print("\nQuality Check Folder:")

    print(QC_DIR)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Preprocess handwriting images for a dataset')
    parser.add_argument('--input-base', type=str, default=None,
                        help='Path to dataset root containing Healthy/ and Parkinson/ folders')
    parser.add_argument('--output-base', type=str, default=None,
                        help='Path to write preprocessed images and metadata (overrides default)')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Dataset tag used for dataset-specific preprocessing and metadata')

    args = parser.parse_args()

    input_base = Path(args.input_base) if args.input_base else None
    output_base = Path(args.output_base) if args.output_base else None

    process_dataset(input_base=input_base, output_base=output_base, dataset_name=args.dataset_name)

