# Classical Machine Learning Baseline Pipeline

## Overview

This stage trains SVM, Random Forest, and XGBoost on handcrafted features with patient-aware grouped cross-validation. The pipeline is dataset-aware and should be run separately for each dataset.

## Models

- SVM (RBF, class-balanced)
- RandomForest
- XGBoost

## Cross-Validation and PCA

- Split strategy: `StratifiedGroupKFold` (patient-level leakage prevention)
- Folds: 5
- PCA components: computed safely per run based on the minimum train-fold size to avoid failures on smaller datasets

## Runtime Labeling

Logs now include explicit dataset/model/fold prefixes, for example:

- `[BD_Dataset][SVM][Fold 1] ...`
- `[Public_Dataset][RandomForest][Fold 3] ...`

## Output Structure

```
classical_ml_results/
├── Public_Dataset/
│   ├── classical_ml_results.csv
│   ├── SVM_fold_1_cm.png
│   ├── RandomForest_fold_1_cm.png
│   └── XGBoost_fold_1_cm.png
└── BD_Dataset/
    ├── classical_ml_results.csv
    ├── SVM_fold_1_cm.png
    ├── RandomForest_fold_1_cm.png
    └── XGBoost_fold_1_cm.png
```

## Running the Pipeline

```bash
# Public
python H_ML_pipeline.py --dataset Public_Dataset

# BD
python H_ML_pipeline.py --dataset BD_Dataset
```

Prerequisite per dataset:

- `handcrafted_features_classical_ml/<dataset>/handcrafted_features.npy`
- `handcrafted_features_classical_ml/<dataset>/class_labels.npy`
- `handcrafted_features_classical_ml/<dataset>/patient_identifiers.npy`
