# Parkinson's Disease Detection from Handwriting Analysis

## Research Overview

This research project develops and validates machine learning approaches for detecting Parkinson's disease from handwritten spiral patterns. The study implements both deep learning (CNNs) and classical machine learning (SVM, Random Forest, XGBoost) pipelines, enabling comprehensive performance comparison and clinical interpretability analysis.

The raw handwriting dataset now uses a consistent naming scheme: healthy samples are named with the `HP` prefix and Parkinson samples with the `PP` prefix, followed by zero-padded patient and spiral image indices. This keeps preprocessing, metadata generation, and downstream model tracking stable and publication-friendly.

## Scientific Motivation

Handwriting is a complex motor task that deteriorates in Parkinson's disease due to progressive motor control loss. Patients exhibit characteristic changes including tremor (uncontrolled waviness), micrographia (progressively smaller writing), and reduced pressure. Automated detection from spiral patterns offers a non-invasive, rapid screening tool complementary to clinical examination.

## Dataset Structure

The research pipeline operates on spiral handwriting images organized by patient and health status:

```
├── preprocessed_images/
│   ├── grayscale/           (intensity-preserved preprocessing)
│   ├── binary/              (binary masks for feature extraction)
│   └── quality_check/       (verification visualizations)
├── handcrafted_features_classical_ml/
│   ├── handcrafted_features.npy
│   ├── class_labels.npy
│   └── patient_identifiers.npy
├── trained_models_checkpoints/
│   └── resnet18_fold_*.pth
├── deep_learning_results/
│   └── fold_*_predictions.csv
├── classical_ml_results/
│   └── classical_ml_results.csv
└── model_interpretability_visualizations/
    └── gradcam_*.png
```

For a plain-English explanation of the generated folders and CSV files, see [README_DATA_FILES.md](README_DATA_FILES.md). For the full journal-style write-up, see [REPORT_JOURNAL_STYLE.md](REPORT_JOURNAL_STYLE.md).

## Execution Pipeline

**Stage 1 - Preprocessing** (`preprocessing.py`):
Standardizes raw spiral images through grayscale conversion, denoising, contrast enhancement, adaptive thresholding, morphological cleanup, and resizing. Generates two image representations: grayscale for deep learning and binary masks for feature extraction. Produces quality control visualizations and metadata CSV.

**Stage 2A - Feature Extraction** (`feature_extraction.py`):
Extracts seven categories of hand-engineered features (HOG, LBP, stroke density, contour morphology, skeleton statistics, Hu moments, entropy) from preprocessed images. Output comprises numpy arrays and CSV for classical ML pipelines.

**Stage 2B - Deep Learning Training** (`train.py`):
Trains a ResNet-18 model using 5-fold stratified group cross-validation. Implements transfer learning from ImageNet, selective layer freezing, data augmentation, and early stopping. Produces per-fold checkpoints and prediction CSVs.

**Stage 3 - Classical ML Baseline** (`H_ML_pipeline.py`):
Trains SVM, Random Forest, and XGBoost on extracted features using identical cross-validation splits. Applies feature standardization and PCA-based dimensionality reduction. Generates confusion matrices and aggregated metrics for baseline comparison.

**Stage 4 - Model Interpretability** (`H_grad_cam.py`):
Generates Grad-CAM visualizations from trained deep learning models, identifying which image regions most influence predictions. Validates that models learned clinically relevant handwriting features rather than spurious artifacts.

## Execution Instructions

### Run Full Pipeline

```bash
# Activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Execute stages sequentially
python preprocessing.py
python feature_extraction.py
python train.py
python H_ML_pipeline.py
python H_grad_cam.py
```

### Run Individual Stages

Each stage is independently executable and requires only its specific input data:

```bash
# Preprocessing only (requires raw images from Thesis dataset)
python preprocessing.py

# Feature extraction only (requires preprocessed images)
python feature_extraction.py

# Deep learning only (requires preprocessed images and metadata)
python train.py

# Classical ML only (requires extracted features)
python H_ML_pipeline.py

# Interpretability only (requires trained models and preprocessed images)
python H_grad_cam.py
```

## Output Artifacts and Naming Convention

All output directories use research-grade descriptive names reflecting their purpose:

- `preprocessed_images/` - Standardized image dataset
- `handcrafted_features_classical_ml/` - Extracted statistical features
- `trained_models_checkpoints/` - Saved neural network weights
- `deep_learning_results/` - DL model performance metrics
- `classical_ml_results/` - Classical ML performance metrics
- `model_interpretability_visualizations/` - Grad-CAM heatmaps

All files within these directories use meaningful, publication-ready names enabling immediate recognition of content and stage origin.

## Methodological Features

**Patient-Level Cross-Validation**: All folds assign complete patient datasets to either training or validation, preventing information leakage where models memorize individual patients' writing styles rather than learning disease patterns.

**Data Augmentation**: Training incorporates rotation, translation, and scaling augmentation to improve model robustness under natural handwriting variation.

**Stratified Sampling**: Class proportions are maintained across all folds, ensuring each fold contains representative samples of both healthy and Parkinson's cases.

**Transfer Learning**: Deep learning models leverage ImageNet pre-training, dramatically reducing the data and computation required to train effective models.

**Multi-Model Comparison**: Both deep learning and classical ML approaches are evaluated on identical data splits, enabling fair performance comparison and methodological transparency.

**Interpretability First**: Grad-CAM analysis validates that learned patterns correspond to clinically meaningful handwriting characteristics.

## Reproducibility

All randomization sources are seeded with SEED=42 for complete reproducibility:

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

CuDNN is configured for deterministic behavior. Running any stage twice produces identical results (bit-for-bit).

## Documentation

Each stage includes a dedicated README:

- [README_PREPROCESSING.md](README_PREPROCESSING.md) - Preprocessing algorithm details and design rationale
- [README_FEATURE_EXTRACTION.md](README_FEATURE_EXTRACTION.md) - Feature definitions and statistical motivation
- [README_DEEP_LEARNING.md](README_DEEP_LEARNING.md) - Training strategy, architecture, and evaluation
- [README_CLASSICAL_ML.md](README_CLASSICAL_ML.md) - Baseline model specifications and comparison framework
- [README_INTERPRETABILITY.md](README_INTERPRETABILITY.md) - Grad-CAM methodology and clinical validation

## Performance Reporting

Results are reported with standard cross-validation metrics:

- **Accuracy**: Overall correctness
- **Sensitivity**: Disease detection rate (clinical priority)
- **Specificity**: Healthy classification rate
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision/recall
- **ROC-AUC**: Threshold-independent performance

Metrics are computed per-fold and reported as mean ± standard deviation to convey both central tendency and variability.

## Research Quality Standards

This pipeline adheres to modern machine learning and clinical research standards:

1. **No Test Set Contamination**: Cross-validation uses strictly separated training/validation folds
2. **Proper Baselines**: Classical ML baseline enables assessment of deep learning necessity
3. **Interpretability**: Grad-CAM validation confirms clinically meaningful predictions
4. **Reproducibility**: Seeds and deterministic settings ensure repeatability
5. **Documentation**: Comprehensive comments and READMEs enable audit and extension
6. **Statistical Rigor**: Multiple metrics capture different aspects of performance
7. **Transparency**: All methods, hyperparameters, and results are fully documented

## Code Quality

Comments throughout the codebase have been refined to remove redundant documentation while preserving explanation of non-obvious algorithmic choices. All variable names, function definitions, and output filenames are research-grade and publication-ready.

## Citation and Usage

This codebase is designed for research publication and regulatory submission. All methodology is documented and reproducible. Results should be cited and archived with code versions for complete traceability.

## Contact and Support

For questions regarding methodology, implementation details, or results interpretation, consult the stage-specific READMEs and inline code documentation.
