# Parkinson's Disease Detection from Handwriting Analysis

## Research Overview

This research project develops and validates machine learning approaches for detecting Parkinson's disease from handwritten spiral patterns. The study implements both deep learning (CNNs) and classical machine learning (SVM, Random Forest, XGBoost) pipelines, enabling comprehensive performance comparison and clinical interpretability analysis.

The raw handwriting datasets now use consistent naming schemes. Public dataset samples use `HP` (healthy) and `PP` (Parkinson), while BD dataset samples use `BHP` (healthy) and `BPP` (Parkinson), followed by zero-padded patient IDs. This keeps preprocessing, metadata generation, and downstream tracking stable and publication-friendly.

## Scientific Motivation

Handwriting is a complex motor task that deteriorates in Parkinson's disease due to progressive motor control loss. Patients exhibit characteristic changes including tremor (uncontrolled waviness), micrographia (progressively smaller writing), and reduced pressure. Automated detection from spiral patterns offers a non-invasive, rapid screening tool complementary to clinical examination.

## Dataset Structure

The research pipeline operates on spiral handwriting images organized by patient and health status:

```
├── preprocessed_images/
│   ├── Public_Dataset/
│   │   ├── grayscale/
│   │   ├── binary/
│   │   ├── quality_check/
│   │   └── metadata.csv
│   └── BD_Dataset/
│       ├── grayscale/
│       ├── binary/
│       ├── quality_check/
│       └── metadata.csv
├── handcrafted_features_classical_ml/
│   ├── Public_Dataset/
│   └── BD_Dataset/
├── trained_models_checkpoints/
│   ├── Public_Dataset/
│   └── BD_Dataset/
├── deep_learning_results/
│   ├── Public_Dataset/
│   └── BD_Dataset/
├── classical_ml_results/
│   ├── Public_Dataset/
│   └── BD_Dataset/
└── model_interpretability_visualizations/
    ├── Public_Dataset/
    └── BD_Dataset/
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

### Run Full Pipeline (Both Datasets)

```bash
# Activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: normalize BD image names to BHP/BPP convention
python rename_bd_files.py

# 1) Preprocess each dataset separately
python preprocessing.py --input-base "D:\Final Semester\Thesis Work\Codes\Dataset\Spiral_Handwriting\Public_Dataset" --output-base "D:\Final Semester\Thesis Work\Codes\Handwriting\preprocessed_images\Public_Dataset"
python preprocessing.py --input-base "D:\Final Semester\Thesis Work\Codes\Dataset\Spiral_Handwriting\BD_Dataset" --output-base "D:\Final Semester\Thesis Work\Codes\Handwriting\preprocessed_images\BD_Dataset"

# 2) Extract handcrafted features per dataset
python feature_extraction.py --dataset Public_Dataset
python feature_extraction.py --dataset BD_Dataset

# 3) Run classical ML per dataset
python H_ML_pipeline.py --dataset Public_Dataset
python H_ML_pipeline.py --dataset BD_Dataset

# 4) Train deep learning per dataset
python train.py --dataset Public_Dataset
python train.py --dataset BD_Dataset

# 5) Generate Grad-CAM per dataset (example uses fold 1)
python H_grad_cam.py --dataset Public_Dataset --model-fold 1
python H_grad_cam.py --dataset BD_Dataset --model-fold 1

# 6) Create cross-dataset comparison outputs
python compare_datasets_results.py
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

- `preprocessed_images/<dataset>/` - Standardized images and per-dataset metadata
- `handcrafted_features_classical_ml/<dataset>/` - Extracted statistical features
- `trained_models_checkpoints/<dataset>/` - Saved neural network weights
- `deep_learning_results/<dataset>/` - DL fold predictions and final summary CSV
- `classical_ml_results/<dataset>/` - Classical ML metrics and confusion matrices
- `model_interpretability_visualizations/<dataset>/` - Grad-CAM heatmaps
- `results/dataset_comparison_summary.csv` - Consolidated Public vs BD comparison table

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
