"""
====================================================================
PARKINSON'S HANDWRITING PREPROCESSING PIPELINE
FINAL RESEARCH-GRADE VERSION
====================================================================

MAIN IMPROVEMENTS
-----------------

1. Better background removal
2. Adaptive thresholding
3. Morphological cleanup
4. Stroke-focused preprocessing
5. Noise reduction
6. Better cropping
7. Better Grad-CAM interpretability
8. Reduced shortcut learning
9. Multimodal-ready metadata
10. Publication-oriented preprocessing

====================================================================
"""

import os
import cv2
import random
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================
# REPRODUCIBILITY
# ==============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ==============================================================
# PATHS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

THESIS_ROOT = PROJECT_ROOT.parents[1]

INPUT_BASE = (
    THESIS_ROOT
    / "Handwriting Dataset"
    / "workingData"
    / "drawings"
    / "spiral"
    / "all"
)

OUTPUT_BASE = PROJECT_ROOT / "preprocessed_handwriting"

GRAY_DIR = OUTPUT_BASE / "grayscale"
BINARY_DIR = OUTPUT_BASE / "binary"
QC_DIR = OUTPUT_BASE / "quality_check"

CLASSES = ["healthy", "parkinson"]

IMG_SIZE = 224

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

for base_dir in [GRAY_DIR, BINARY_DIR]:

    for cls in CLASSES:

        os.makedirs(base_dir / cls, exist_ok=True)

os.makedirs(QC_DIR, exist_ok=True)

metadata_records = []

# ==============================================================
# IMAGE VALIDATION
# ==============================================================

def validate_image(img):

    if img is None:
        return False

    if len(img.shape) < 2:
        return False

    h, w = img.shape[:2]

    if h < 50 or w < 50:
        return False

    return True

# ==============================================================
# FOREGROUND NORMALIZATION
# ==============================================================

def normalize_foreground(binary_img):

    white_ratio = np.mean(binary_img > 127)

    if white_ratio > 0.5:
        binary_img = cv2.bitwise_not(binary_img)

    return binary_img

# ==============================================================
# REMOVE SMALL ARTIFACTS
# ==============================================================

def remove_small_components(binary_img):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8
    )

    cleaned = np.zeros_like(binary_img)

    min_area = 40

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:

            cleaned[labels == i] = 255

    return cleaned

# ==============================================================
# CROPPING
# ==============================================================

def crop_to_content(img):

    coords = cv2.findNonZero(img)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    padding = 20

    x = max(x - padding, 0)
    y = max(y - padding, 0)

    w = min(w + 2 * padding, img.shape[1] - x)
    h = min(h + 2 * padding, img.shape[0] - y)

    return img[y:y+h, x:x+w]

# ==============================================================
# RESIZE WITH PADDING
# ==============================================================

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

# ==============================================================
# PATIENT ID
# ==============================================================

def extract_patient_id(filename):

    stem = Path(filename).stem

    parts = stem.split("_")

    if len(parts) >= 2:
        return parts[1]

    return stem

# ==============================================================
# MAIN PREPROCESSING
# ==============================================================

def preprocess_image(img):

    # ----------------------------------------------------------
    # 1. Grayscale
    # ----------------------------------------------------------

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----------------------------------------------------------
    # 2. Denoising
    # ----------------------------------------------------------

    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=10
    )

    # ----------------------------------------------------------
    # 3. CLAHE
    # ----------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    # ----------------------------------------------------------
    # 4. Adaptive Thresholding
    # ----------------------------------------------------------

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11
    )

    # ----------------------------------------------------------
    # 5. Morphological Cleanup
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # 6. Remove Tiny Components
    # ----------------------------------------------------------

    binary = remove_small_components(binary)

    # ----------------------------------------------------------
    # 7. Normalize Foreground
    # ----------------------------------------------------------

    binary = normalize_foreground(binary)

    # ----------------------------------------------------------
    # 8. Stroke-Focused Grayscale
    # ----------------------------------------------------------

    masked_gray = cv2.bitwise_and(
        enhanced,
        enhanced,
        mask=binary
    )

    # ----------------------------------------------------------
    # 9. Crop
    # ----------------------------------------------------------

    cropped_gray = crop_to_content(masked_gray)

    cropped_binary = crop_to_content(binary)

    # ----------------------------------------------------------
    # 10. Resize
    # ----------------------------------------------------------

    final_gray = resize_with_padding(
        cropped_gray,
        IMG_SIZE
    )

    final_binary = resize_with_padding(
        cropped_binary,
        IMG_SIZE
    )

    return final_gray, final_binary

# ==============================================================
# QUALITY CHECK
# ==============================================================

def save_quality_check(original, gray, binary, filename):

    original = cv2.resize(original, (224, 224))

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

    cv2.imwrite(str(save_path), combined)

# ==============================================================
# PROCESS DATASET
# ==============================================================

def process_dataset():

    print("\n================================================")
    print("STARTING PREPROCESSING")
    print("================================================\n")

    total_processed = 0
    total_skipped = 0

    for cls in CLASSES:

        input_folder = INPUT_BASE / cls

        print(f"\nProcessing class: {cls}")

        image_files = sorted(os.listdir(input_folder))

        for img_name in tqdm(image_files):

            ext = Path(img_name).suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            img_path = input_folder / img_name

            try:

                img = cv2.imread(str(img_path))

                if not validate_image(img):

                    total_skipped += 1
                    continue

                gray_img, binary_img = preprocess_image(img)

                gray_save_path = GRAY_DIR / cls / img_name
                binary_save_path = BINARY_DIR / cls / img_name

                cv2.imwrite(
                    str(gray_save_path),
                    gray_img
                )

                cv2.imwrite(
                    str(binary_save_path),
                    binary_img
                )

                save_quality_check(
                    img,
                    gray_img,
                    binary_img,
                    img_name
                )

                patient_id = extract_patient_id(img_name)

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

    metadata_df = pd.DataFrame(metadata_records)

    metadata_path = OUTPUT_BASE / "metadata.csv"

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    print("\n================================================")
    print("PREPROCESSING COMPLETE")
    print("================================================")

    print(f"\nProcessed: {total_processed}")
    print(f"Skipped: {total_skipped}")

    print(f"\nMetadata saved:")
    print(metadata_path)

# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    process_dataset()