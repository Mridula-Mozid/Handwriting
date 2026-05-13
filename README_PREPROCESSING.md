# Image Preprocessing Pipeline

## Overview

The preprocessing module standardizes raw handwriting spiral images into research-grade datasets suitable for both deep learning and classical machine learning approaches. This stage is critical for ensuring data consistency and improving model robustness across all downstream analyses.

## Purpose

Raw handwriting images exhibit significant variability in lighting conditions, scale, orientation, and background quality. This preprocessing pipeline normalizes these variations while preserving clinically meaningful stroke characteristics that distinguish Parkinson's disease patients from healthy controls.

The raw folder is expected to use the research naming convention where healthy samples begin with `HP` and Parkinson samples begin with `PP`, followed by a zero-padded patient index and the spiral sample index, such as `HP001_hand_spiral_01.png` or `PP014_hand_spiral_03.png`.

## Processing Stages

**Grayscale Conversion**: Raw images are converted to single-channel grayscale, removing color channel redundancy while maintaining stroke intensity information.

**Denoising**: The Non-Local Means Denoising algorithm removes sensor noise and acquisition artifacts without destroying fine stroke details. The h=10 parameter balances noise removal with preservation of handwriting micro-features.

**Contrast Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE) improves local contrast without over-amplifying noise. This is particularly effective for low-quality scans where pressure variation may be subtle.

**Adaptive Thresholding**: Rather than global thresholding which fails with variable lighting, adaptive thresholding computes local thresholds for each image region. This produces binary masks that accurately separate handwriting strokes from background.

**Morphological Cleanup**: Opening followed by closing operations remove small noise artifacts while preserving stroke connectivity. This prevents spurious features from micro-noise affecting downstream analysis.

**Artifact Removal**: Connected components with areas smaller than 40 pixels are discarded, eliminating residual noise while retaining meaningful stroke structure.

**Foreground Normalization**: Ensures consistent foreground/background representation by detecting and correcting inverted images where background was originally marked as white.

**Stroke Masking**: The binary mask is applied to the enhanced grayscale image, creating a version where only actual handwriting strokes are visible. This hybrid representation balances intensity information with clear stroke definition.

**Content-Based Cropping**: The bounding box of all non-zero pixels is computed and padded by 20 pixels to provide context while removing excessive whitespace. This tightens the field of view around actual handwriting.

**Standardized Resizing**: All images are resized to 224×224 pixels with aspect ratio preservation using zero-padding. This standardization enables consistent batch processing and fair model comparisons.

## Output Structure

```
preprocessed_images/
├── grayscale/
│   ├── healthy/          (preprocessed grayscale images)
│   └── parkinson/
├── binary/
│   ├── healthy/          (binary masks for classical ML)
│   └── parkinson/
├── quality_check/        (side-by-side preprocessing verification)
└── metadata.csv          (image paths, patient IDs, dimensions)
```

## Quality Control

The pipeline generates side-by-side triplet visualizations (original → grayscale → binary) for manual verification. These quality check images are saved for spot-checking and publication validation.

## Running the Pipeline

```bash
python preprocessing.py
```

Expected output: 1000+ preprocessed image pairs with metadata tracking for reproducibility and patient-level analysis.
