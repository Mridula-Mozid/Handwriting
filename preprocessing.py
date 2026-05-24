"""Preprocess handwriting spiral images for downstream model training and analysis."""

import os
import cv2
import random
import warnings
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

IMG_SIZE = 224

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

MIN_COMPONENT_AREA = 40

PADDING = 20

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

    parts = stem.split("_")

    return parts[0]

def normalize_foreground(binary_img):
    white_ratio = np.mean(binary_img > 127)

    if white_ratio > 0.5:

        binary_img = cv2.bitwise_not(binary_img)

    return binary_img

def remove_small_components(binary_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8
    )

    cleaned = np.zeros_like(binary_img)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:

            cleaned[labels == i] = 255

    return cleaned

def crop_to_content(img):
    coords = cv2.findNonZero(img)

    if coords is None:

        return img

    x, y, w, h = cv2.boundingRect(coords)

    x = max(x - PADDING, 0)
    y = max(y - PADDING, 0)

    w = min(w + 2 * PADDING, img.shape[1] - x)
    h = min(h + 2 * PADDING, img.shape[0] - y)

    return img[y:y+h, x:x+w]

def resize_with_padding(img, size=224):
    h, w = img.shape[:2]

    scale = size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas

def preprocess_image(img):
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=10
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    binary = cv2.adaptiveThreshold(

        enhanced,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV,

        31,

        11
    )

    raw_threshold = binary.copy()

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

    binary = remove_small_components(binary)

    binary = normalize_foreground(binary)

    masked_gray = cv2.bitwise_and(
        enhanced,
        enhanced,
        mask=binary
    )

    cropped_gray = crop_to_content(masked_gray)

    cropped_binary = crop_to_content(binary)

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
    "threshold": raw_threshold,
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

def process_dataset(input_base: Path = None, output_base: Path = None):

    print("\n================================================")
    print("STARTING HANDWRITING PREPROCESSING")
    print("================================================\n")

    # Resolve input/output
    if input_base is None:
        input_base = DEFAULT_INPUT_BASE

    if output_base is None:
        output_base = OUTPUT_BASE

    dataset_tag = output_base.name

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

        input_folder = Path(input_base) / cls

        if not input_folder.exists():

            raise FileNotFoundError(
                f"Missing folder:\n{input_folder}"
            )

        print(f"\nProcessing class: {cls}")

        image_files = [
            img_name for img_name in sorted(os.listdir(input_folder))
            if Path(img_name).suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        for image_index, img_name in enumerate(tqdm(image_files), start=1):

            img_path = input_folder / img_name

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

                # --------------------------------------------------
                # Preprocess
                # --------------------------------------------------

                gray_img, binary_img, steps = preprocess_image(img)

                # --------------------------------------------------
                # Save outputs
                # --------------------------------------------------

                gray_save_path = (
                    GRAY_DIR / cls / img_name
                )

                binary_save_path = (
                    BINARY_DIR / cls / img_name
                )

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
                    img_name
                )

                if image_index in STEP_VISUALIZATION_INDICES.get(cls, set()):

                    save_step_visualization(
                        steps,
                        img,
                        f"{cls}_{img_name}"
                    )

                
                # --------------------------------------------------
                # Metadata
                # --------------------------------------------------

                patient_id = extract_patient_id(
                    img_name
                )

                metadata_records.append({

                    "filename": img_name,

                    "patient_id": patient_id,

                    "class": cls,

                    "gray_path": str(gray_save_path),

                    "binary_path": str(binary_save_path),

                    "height": img.shape[0],

                    "width": img.shape[1]
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

    args = parser.parse_args()

    input_base = Path(args.input_base) if args.input_base else None
    output_base = Path(args.output_base) if args.output_base else None

    process_dataset(input_base=input_base, output_base=output_base)