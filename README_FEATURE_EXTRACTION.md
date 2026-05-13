# Handcrafted Feature Extraction for Classical Machine Learning

## Overview

This module extracts seven categories of hand-engineered features from preprocessed handwriting images. These features are specifically designed to capture clinical and morphological patterns relevant to Parkinson's disease detection, providing interpretable inputs for classical machine learning models.

## Rationale

While deep learning models automatically discover features during training, classical machine learning requires explicit feature engineering. Our approach combines established computer vision and signal processing techniques to capture different aspects of handwriting degradation associated with Parkinson's disease.

## Feature Categories

**HOG (Histogram of Oriented Gradients)**: Captures directional stroke patterns and edge orientations within 16×16 pixel neighborhoods. The 9-bin orientation histogram reveals how handwriting tends to follow consistent directions versus becoming erratic. This is particularly sensitive to tremor-induced directional noise.

**LBP (Local Binary Patterns)**: Encodes local texture by comparing each pixel with its 8 neighbors, creating a local binary pattern. The uniform LBP method reduces dimensionality while focusing on microstructure. This discriminates fine stroke details and pressure variation patterns between groups.

**Stroke Density**: The proportion of foreground pixels in the binary mask. Parkinsonian handwriting often exhibits reduced density due to smaller, lighter strokes. This simple but effective global feature normalizes for writing size and pressure.

**Contour Morphology**: Extracts 5 geometric features from detected stroke contours: count (fragmentation level), mean area, area standard deviation, mean perimeter, and circularity. These statistics reflect whether strokes are smooth and continuous (low fragmentation) or broken and scattered (high fragmentation).

**Skeleton Features**: The medial skeleton (centerline) of strokes is computed via morphological thinning. Skeleton pixel count and density describe overall stroke structure and continuity independent of stroke thickness. Fragmented skeletons indicate temporal control loss.

**Hu Moments**: Seven moment-based invariants computed from the binary image. These mathematical shape descriptors remain constant under rotation, scaling, and translation. They capture the global shape characteristics of the entire handwriting pattern.

**Image Entropy**: Shannon entropy of the grayscale intensity histogram. High entropy indicates mixed intensity levels suggesting pressure variation and tremor, while low entropy suggests uniform, controlled strokes. This is a proxy for motor control consistency.

## Feature Matrix Dimensions

Each image generates approximately 168 numerical features (HOG: 144 + LBP: 16 + Density: 1 + Contours: 5 + Skeleton: 2 + Hu: 7 + Entropy: 1 + additional statistics).

## Output Artifacts

```
handcrafted_features_classical_ml/
├── handcrafted_features.npy              (N×168 matrix)
├── class_labels.npy                      (N×1, 0=healthy, 1=Parkinson)
├── patient_identifiers.npy               (N×1, unique patient IDs)
└── handcrafted_features_table.csv        (human-readable feature table)
```

Where N is the total number of images (typically 1000+).

## Statistical Properties

Before use in classical ML pipelines, features exhibit different scales and distributions. The downstream classical_ml pipeline applies standardization (zero mean, unit variance) to normalize these differences. PCA further reduces dimensionality to prevent overfitting given typical patient cohort sizes.

## Running the Pipeline

```bash
python feature_extraction.py
```

Prerequisites: Requires `preprocessed_images/metadata.csv` and preprocessed image pairs from the preprocessing stage.

Expected runtime: 5-10 minutes depending on image count.

## Research Transparency

All feature definitions follow established computer vision literature and are thoroughly documented in the code. This approach prioritizes interpretability and reproducibility—researchers can directly inspect which features drive classification decisions.
