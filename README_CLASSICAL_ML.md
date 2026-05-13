# Classical Machine Learning Baseline Pipeline

## Overview

This pipeline trains three classical machine learning models (Support Vector Machine, Random Forest, and XGBoost) on the hand-engineered features extracted from handwriting images. Classical ML provides an essential baseline for validating whether deep learning is necessary and offers superior interpretability for clinical decision support.

## Purpose and Value

Classical machine learning models are fundamentally interpretable: each feature's contribution to predictions can be quantified and visualized. For clinical applications, this transparency is crucial for regulatory compliance and clinical acceptance. Additionally, classical ML typically requires fewer samples to train reliably compared to deep learning, making it attractive for smaller research cohorts.

## Models

**Support Vector Machine (SVM)**: Uses a radial basis function (RBF) kernel to discover non-linear decision boundaries in feature space. SVM is particularly effective for high-dimensional data with moderate sample sizes. The balanced class weight setting ensures equal importance is given to healthy and Parkinson's samples during training.

**Random Forest**: Ensemble of 300 decision trees with maximum depth 10, preventing individual trees from becoming too complex and overfitting. Random Forest naturally handles non-linear relationships and variable interactions. Out-of-bag error estimates provide internal validation without separate holdout sets.

**XGBoost**: Gradient boosting approach that iteratively builds weak learners and combines them. With 200 rounds of boosting and learning rate 0.03, this conservative setup prevents early convergence and excessive adaptation to training noise. The subsample and colsample parameters provide additional regularization.

## Preprocessing Pipeline

**Standardization**: Features are transformed to mean=0 and std=1. This normalization is essential for SVM which is sensitive to feature scaling, and beneficial for gradient boosting algorithms.

**Dimensionality Reduction with PCA**: The handcrafted feature matrix contains ~168 dimensions. Principal Component Analysis projects these into a lower-dimensional space while preserving maximum variance. The number of components is determined adaptively as min(64, N_samples-1, N_features) to prevent numerical instability and overfitting. Typically retains 90%+ of variance in 50-64 dimensions.

**Pipelining**: Standardization and PCA are applied only to the training fold, then the learned transformations are applied identically to the validation fold. This prevents information leakage where the validation set influences preprocessing.

## Cross-Validation Framework

**Stratified Group K-Fold**: Identical to the deep learning pipeline—5 folds with stratified class distribution and patient-level grouping. This ensures fair comparison between classical and deep learning approaches using identical data splits.

**Per-Fold Training**: Each model is trained on 80% of unique patients (training fold) and evaluated on 20% (validation fold). Results are averaged across folds to produce robust performance estimates.

## Evaluation and Output

For each model and fold, the following metrics are computed:

- Accuracy, Precision, Sensitivity, Specificity, F1-Score, and ROC-AUC (identical to deep learning evaluation)
- Per-fold confusion matrices visualized as heatmaps
- Aggregated results across all folds

Output structure:

```
classical_ml_results/
├── SVM_fold_1_confusion_matrix.png
├── SVM_fold_2_confusion_matrix.png
├── ...
├── RandomForest_fold_1_confusion_matrix.png
├── RandomForest_fold_2_confusion_matrix.png
├── ...
├── XGBoost_fold_1_confusion_matrix.png
├── XGBoost_fold_2_confusion_matrix.png
├── ...
└── classical_ml_results.csv          (aggregated metrics across models and folds)
```

## Running the Pipeline

```bash
python H_ML_pipeline.py
```

Prerequisites: Requires handcrafted features from the feature extraction stage.

Expected runtime: 5-15 minutes depending on machine specifications.

## Comparison and Interpretation

The classical ML results serve multiple purposes: (1) baseline performance to assess whether deep learning adds significant value, (2) interpretability analysis where feature importance can be extracted from tree-based models and SVM coefficients, and (3) computational efficiency demonstration—classical ML is typically orders of magnitude faster to train and deploy.

Comparing classical ML and deep learning results reveals dataset characteristics: if classical ML performs nearly as well as deep learning, the problem may be sufficiently linearly separable in feature space; if deep learning substantially outperforms, this suggests complex non-linear patterns that justify the added model complexity and computational cost.
