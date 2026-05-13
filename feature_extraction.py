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
5. Skeleton statistics
6. Hu moments
7. Image entropy

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

DATA_ROOT = PROJECT_ROOT / "preprocessed_images"

METADATA_PATH = DATA_ROOT / "metadata.csv"

FEATURE_SAVE_DIR = PROJECT_ROOT / "handcrafted_features_classical_ml"

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

def extract_features(gray_img, binary_img):
    features = []
    hog_features = hog(gray_img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    features.extend(hog_features)
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
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

    for c in contours:

        area = cv2.contourArea(c)

        perimeter = cv2.arcLength(c, True)

        contour_areas.append(area)

        contour_perimeters.append(perimeter)

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

    binary_bool = binary_img > 127
    skeleton = skeletonize(binary_bool)
    skeleton_pixels = np.sum(skeleton)
    skeleton_density = skeleton_pixels / total_pixels
    features.append(skeleton_pixels)
    features.append(skeleton_density)

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

for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):

    try:

        gray_path = row["gray_path"]

        binary_path = row["binary_path"]

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

            "label": label
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

print(f"\nTotal Samples: {len(all_features)}")

print(f"\nFeature Shape: {all_features.shape}")

print("\nSaved Files:")

print(FEATURE_SAVE_DIR / "handcrafted_features.npy")
print(FEATURE_SAVE_DIR / "class_labels.npy")
print(FEATURE_SAVE_DIR / "patient_identifiers.npy")
print(csv_path)