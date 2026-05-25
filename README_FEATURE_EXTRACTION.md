# Handcrafted Feature Extraction for Classical Machine Learning

## Overview

This module extracts seven categories of hand-engineered features from preprocessed handwriting images. It is dataset-aware and writes separate outputs for `Public_Dataset`, `BD_Dataset`, and `HandPD`.

## Feature Categories

- HOG
- LBP
- Stroke density
- Contour morphology
- Skeleton statistics
- Hu moments
- Image entropy

## Output Artifacts

```
handcrafted_features_classical_ml/
├── Public_Dataset/
│   ├── handcrafted_features.npy
│   ├── class_labels.npy
│   ├── patient_identifiers.npy
│   └── handcrafted_features_table.csv
└── BD_Dataset/
    ├── handcrafted_features.npy
    ├── class_labels.npy
    ├── patient_identifiers.npy
    └── handcrafted_features_table.csv
└── HandPD/
    ├── handcrafted_features.npy
    ├── class_labels.npy
    ├── patient_identifiers.npy
    └── handcrafted_features_table.csv
```

## Runtime Labeling

The script now prints dataset context at runtime (dataset tag and metadata path), so you can confirm which dataset is currently being processed.

## Running the Pipeline

```bash
# Public
python feature_extraction.py --dataset Public_Dataset

# BD
python feature_extraction.py --dataset BD_Dataset

# HandPD
python feature_extraction.py --dataset HandPD
```

Prerequisite per dataset:

- `preprocessed_images/<dataset>/metadata.csv`
- corresponding grayscale and binary images inside `preprocessed_images/<dataset>/`

## Notes

The exact feature dimensionality can vary depending on preprocessing/image properties. The generated `.npy` files are the canonical inputs for the classical ML stage.
