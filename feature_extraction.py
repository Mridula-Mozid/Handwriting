"""
========================================================================
RESEARCH-GRADE HANDWRITING FEATURE EXTRACTION
FOR CLASSICAL ML BASELINES
========================================================================

Extracted Features:
-------------------
1. HOG
2. LBP
3. Stroke density
4. Contour morphology
5. Fractal dimension
6. Stroke width statistics
7. Skeleton statistics
8. Skeleton branching statistics
9. Hu moments
10. Image entropy

Outputs:
--------
- handcrafted_features.npy
- class_labels.npy
- patient_identifiers.npy
- handcrafted_features_table.csv

========================================================================
"""

import os
import cv2
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from scipy.stats import entropy

from skimage.feature import (
    hog,
    local_binary_pattern
)

from skimage.morphology import skeletonize

warnings.filterwarnings("ignore")

# =========================================================================
# PATHS
# =========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

import argparse

parser = argparse.ArgumentParser(description='Extract handcrafted features for a dataset')
parser.add_argument('--dataset', type=str, default=None,
                    help='Dataset tag (e.g., Public_Dataset or BD_Dataset). If provided, will read from preprocessed_images/<dataset>')
args = parser.parse_args()

DATA_ROOT = PROJECT_ROOT / "preprocessed_images"

if args.dataset:
    DATA_ROOT = DATA_ROOT / args.dataset

METADATA_PATH = DATA_ROOT / "metadata.csv"

FEATURE_SAVE_DIR = PROJECT_ROOT / "handcrafted_features_classical_ml" / (args.dataset if args.dataset else "default")

DATASET_TAG = args.dataset if args.dataset else "default"

os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)

# =========================================================================
# LOAD METADATA
# =========================================================================

metadata_df = pd.read_csv(METADATA_PATH)

metadata_df["label"] = metadata_df["class"].map({
    "healthy": 0,
    "parkinson": 1
})

# =========================================================================
# FEATURE EXTRACTION
# =========================================================================

def fractal_dimension(binary_img):

    binary = binary_img > 0

    def boxcount(img, k):

        S = np.add.reduceat(
            np.add.reduceat(
                img,
                np.arange(0, img.shape[0], k),
                axis=0
            ),
            np.arange(0, img.shape[1], k),
            axis=1
        )

        return len(np.where(S > 0)[0])

    sizes = 2 ** np.arange(1, 8)

    counts = []

    for size in sizes:
        counts.append(boxcount(binary, size))

    counts = np.maximum(np.array(counts, dtype=float), 1.0)

    coeffs = np.polyfit(
        np.log(sizes),
        np.log(counts),
        1
    )

    return -coeffs[0]

def extract_features(gray_img, binary_img):
    features = []

    gray_img = gray_img.astype(np.float32) / 255.0

    hog_features = hog(
        gray_img,
        orientations=6,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    features.extend(hog_features)

    radius = 1

    n_points = 8

    lbp = local_binary_pattern(
        gray_img,
        n_points,
        radius,
        method='uniform'
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    lbp_hist = lbp_hist.astype("float")

    lbp_hist /= (lbp_hist.sum() + 1e-6)

    features.extend(lbp_hist)

    white_pixels = np.sum(binary_img > 127)
    total_pixels = binary_img.shape[0] * binary_img.shape[1]
    stroke_density = white_pixels / total_pixels
    features.append(stroke_density)

    # Contour-based features for stroke morphology
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_count = len(contours)

    contour_areas = []

    contour_perimeters = []

    contour_circularities = []

    contour_compactness = []

    for c in contours:

        area = cv2.contourArea(c)

        perimeter = cv2.arcLength(c, True)

        contour_areas.append(area)

        contour_perimeters.append(perimeter)

        compactness = area / (perimeter + 1e-8)

        contour_compactness.append(compactness)

        if perimeter > 0:

            circularity = (
                4 * np.pi * area
            ) / (perimeter ** 2)

            contour_circularities.append(circularity)

    features.append(contour_count)

    features.append(
        np.mean(contour_areas)
        if contour_areas else 0
    )

    features.append(
        np.std(contour_areas)
        if contour_areas else 0
    )

    features.append(
        np.mean(contour_perimeters)
        if contour_perimeters else 0
    )

    features.append(
        np.mean(contour_circularities)
        if contour_circularities else 0
    )

    features.append(
        np.mean(contour_compactness)
        if contour_compactness else 0
    )

    fd = fractal_dimension(binary_img)
    features.append(fd)

    distance = cv2.distanceTransform(
        binary_img,
        cv2.DIST_L2,
        5
    )

    foreground_pixels = distance[binary_img > 0]

    if len(foreground_pixels) > 0:

        stroke_width_mean = np.mean(foreground_pixels)

        stroke_width_std = np.std(foreground_pixels)

    else:

        stroke_width_mean = 0

        stroke_width_std = 0

    features.append(stroke_width_mean)

    features.append(stroke_width_std)

    binary_bool = binary_img > 127
    skeleton = skeletonize(binary_bool)
    skeleton_pixels = np.sum(skeleton)
    skeleton_density = skeleton_pixels / total_pixels
    features.append(skeleton_pixels)
    features.append(skeleton_density)

    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ])

    neighbor_count = cv2.filter2D(
        skeleton.astype(np.uint8),
        -1,
        kernel
    )

    branch_points = np.sum(neighbor_count >= 13)

    end_points = np.sum(neighbor_count == 11)

    features.append(branch_points)

    features.append(end_points)

    moments = cv2.moments(binary_img)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = np.log(np.abs(hu_moments) + 1e-8).flatten()
    features.extend(hu_moments)

    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.ravel()
    hist = hist / (hist.sum() + 1e-8)
    image_entropy = entropy(hist)
    features.append(image_entropy)

    return np.array(features)

# =========================================================================
# MAIN EXTRACTION LOOP
# =========================================================================

all_features = []

all_labels = []

all_patient_ids = []

feature_rows = []

print("\n================================================")
print("EXTRACTING HANDCRAFTED FEATURES")
print("================================================\n")
print(f"Dataset: {DATASET_TAG}")
print(f"Metadata: {METADATA_PATH}\n")

for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):

    try:

        gray_path = row["gray_path"]

        binary_path = row["binary_path"]

        skeleton_binary_path = row.get("skeleton_binary_path", binary_path)

        if pd.isna(skeleton_binary_path) or not str(skeleton_binary_path).strip():
            skeleton_binary_path = binary_path

        label = row["label"]

        patient_id = row["patient_id"]

        # ---------------------------------------------------------
        # LOAD IMAGES
        # ---------------------------------------------------------

        gray_img = cv2.imread(

            gray_path,

            cv2.IMREAD_GRAYSCALE
        )

        binary_img = cv2.imread(

            str(skeleton_binary_path),

            cv2.IMREAD_GRAYSCALE
        )

        if gray_img is None or binary_img is None:

            if skeleton_binary_path != binary_path:

                binary_img = cv2.imread(

                    binary_path,

                    cv2.IMREAD_GRAYSCALE
                )

        if gray_img is None or binary_img is None:

            print(f"Skipped: {gray_path}")

            continue

        # ---------------------------------------------------------
        # FEATURE EXTRACTION
        # ---------------------------------------------------------

        features = extract_features(

            gray_img,

            binary_img
        )

        all_features.append(features)

        all_labels.append(label)

        all_patient_ids.append(patient_id)

        row_dict = {

            "patient_id": patient_id,

            "label": label,

            "binary_source_path": str(skeleton_binary_path)
        }

        for i, value in enumerate(features):

            row_dict[f"feature_{i}"] = value

        feature_rows.append(row_dict)

    except Exception as e:

        print(f"\nError processing row {idx}")

        print(e)

# =========================================================================
# CONVERT TO NUMPY
# =========================================================================

all_features = np.array(all_features)

all_labels = np.array(all_labels)

all_patient_ids = np.array(all_patient_ids)

# =========================================================================
# SAVE FILES
# =========================================================================

np.save(FEATURE_SAVE_DIR / "handcrafted_features.npy", all_features)
np.save(FEATURE_SAVE_DIR / "class_labels.npy", all_labels)
np.save(FEATURE_SAVE_DIR / "patient_identifiers.npy", all_patient_ids)

# =========================================================================
# SAVE CSV
# =========================================================================

features_df = pd.DataFrame(feature_rows)
csv_path = FEATURE_SAVE_DIR / "handcrafted_features_table.csv"
features_df.to_csv(csv_path, index=False)
# =========================================================================
# FINAL SUMMARY
# =========================================================================

print("\n================================================")
print("FEATURE EXTRACTION COMPLETE")
print("================================================")

print(f"Dataset: {DATASET_TAG}")

print(f"\nTotal Samples: {len(all_features)}")

print(f"\nFeature Shape: {all_features.shape}")

print("\nSaved Files:")

print(FEATURE_SAVE_DIR / "handcrafted_features.npy")
print(FEATURE_SAVE_DIR / "class_labels.npy")
print(FEATURE_SAVE_DIR / "patient_identifiers.npy")
print(csv_path)