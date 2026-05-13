# Image Preprocessing Pipeline

## Overview

The preprocessing module standardizes raw handwriting spiral images into research-grade datasets suitable for both deep learning and classical machine learning. It is designed to run per dataset so `Public_Dataset` and `BD_Dataset` remain fully separated.

## Purpose

Raw handwriting images vary in lighting, scale, orientation, and background quality. This pipeline normalizes those factors while preserving clinically meaningful stroke patterns.

Naming conventions used in this project:

- Public dataset: healthy `HP...`, Parkinson `PP...`
- BD dataset: healthy `BHP...`, Parkinson `BPP...`

## Processing Stages

- Grayscale conversion
- Non-local means denoising
- CLAHE contrast enhancement
- Adaptive thresholding
- Morphological cleanup
- Small-component filtering
- Foreground normalization
- Content cropping with padding
- Resize to 224x224 with aspect-ratio-preserving padding

## Output Structure

```
preprocessed_images/
├── Public_Dataset/
│   ├── grayscale/
│   ├── binary/
│   ├── quality_check/
│   └── metadata.csv
└── BD_Dataset/
    ├── grayscale/
    ├── binary/
    ├── quality_check/
    └── metadata.csv
```

## Runtime Labeling

When preprocessing starts, the script prints dataset context clearly, including dataset tag, input root, and output root.

## Running the Pipeline

```bash
# Public
python preprocessing.py --input-base "D:\Final Semester\Thesis Work\Codes\Dataset\Spiral_Handwriting\Public_Dataset" --output-base "D:\Final Semester\Thesis Work\Codes\Handwriting\preprocessed_images\Public_Dataset"

# BD
python preprocessing.py --input-base "D:\Final Semester\Thesis Work\Codes\Dataset\Spiral_Handwriting\BD_Dataset" --output-base "D:\Final Semester\Thesis Work\Codes\Handwriting\preprocessed_images\BD_Dataset"
```

Each run produces dataset-specific metadata and images without cross-dataset mixing.
