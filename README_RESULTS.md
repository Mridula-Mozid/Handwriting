# Final Results and Reproducibility Notes

## Purpose

This file is the consolidated results guide for the dual-dataset handwriting pipeline.

- Datasets: `Public_Dataset`, `BD_Dataset`
- Runtime logging format: `[DATASET][Fold X]` and `[DATASET][MODEL][Fold X]`
- All outputs are generated per dataset in separate folders

## Important Current State

Generated CSV files and generated visualization images were intentionally removed so the project can be run cleanly from scratch.

What remains in the workspace:

- Python source files
- README and report documents
- Folder structure for output regeneration

## Verified Run Snapshot

The latest completed and verified 5-fold deep-learning run currently available in the workspace is for `Public_Dataset`.

- Accuracy: 0.9705 ± 0.0270
- Precision: 0.9636 ± 0.0498
- Sensitivity: 0.9818 ± 0.0407
- Specificity: 0.9600 ± 0.0548
- F1-score: 0.9714 ± 0.0261
- ROC-AUC: 0.9922 ± 0.0083

The `HandPD` full 5-fold run was executed and partially verified fold-by-fold, but the final aggregate summary was not available in the captured terminal output at the time of this update.

## Historical Metrics Snapshot (Most Recent Completed Full Run)

These numbers are copied from the latest journal-style report and represent the most recent full run before cleanup.

### Deep Learning (ResNet-18, 5-fold grouped CV)

`Public_Dataset` mean metrics:

- Accuracy: 0.7690
- Precision: 0.7793
- Sensitivity: 0.8133
- Specificity: 0.6889
- F1-score: 0.7670
- ROC-AUC: 0.9563

`BD_Dataset` mean metrics:

- Accuracy: 0.5786
- Precision: 0.5762
- Sensitivity: 0.5500
- Specificity: 0.6000
- F1-score: 0.5067
- ROC-AUC: 0.9067

### Classical ML (Best model by ROC-AUC: SVM)

- `Public_Dataset` SVM: Accuracy 0.7373, ROC-AUC 0.8899
- `BD_Dataset` SVM: Accuracy 0.8071, ROC-AUC 0.8667

## Regenerated Result Files (After You Run Again)

### Deep learning outputs

- `deep_learning_results/Public_Dataset/final_cross_validation_results.csv`
- `deep_learning_results/Public_Dataset/fold_1_predictions.csv` to `fold_5_predictions.csv`
- `deep_learning_results/BD_Dataset/final_cross_validation_results.csv`
- `deep_learning_results/BD_Dataset/fold_1_predictions.csv` to `fold_5_predictions.csv`

### Classical ML outputs

- `classical_ml_results/Public_Dataset/classical_ml_results.csv`
- `classical_ml_results/BD_Dataset/classical_ml_results.csv`
- confusion matrix images per model and fold under each dataset folder

### Interpretability outputs

- `model_interpretability_visualizations/Public_Dataset/*.png`
- `model_interpretability_visualizations/BD_Dataset/*.png`

### Comparison output

- `results/dataset_comparison_summary.csv`
- `..\Visualization\09_Public_vs_BD_Performance_Comparison.png`
- `..\Visualization\09_Public_vs_BD_Performance_Comparison.pdf`

## Canonical Run Order (From Scratch)

```bash
# 1) Preprocessing
python preprocessing.py --input-base "D:\Final Semester\Thesis Work\Codes\Dataset\Spiral_Handwriting\Public_Dataset" --output-base "D:\Final Semester\Thesis Work\Codes\Handwriting\preprocessed_images\Public_Dataset"
python preprocessing.py --input-base "D:\Final Semester\Thesis Work\Codes\Dataset\Spiral_Handwriting\BD_Dataset" --output-base "D:\Final Semester\Thesis Work\Codes\Handwriting\preprocessed_images\BD_Dataset"

# 2) Handcrafted features
python feature_extraction.py --dataset Public_Dataset
python feature_extraction.py --dataset BD_Dataset

# 3) Classical ML
python H_ML_pipeline.py --dataset Public_Dataset
python H_ML_pipeline.py --dataset BD_Dataset

# 4) Deep learning
python train.py --dataset Public_Dataset
python train.py --dataset BD_Dataset

# 5) Grad-CAM (example: fold 1)
python H_grad_cam.py --dataset Public_Dataset --model-fold 1
python H_grad_cam.py --dataset BD_Dataset --model-fold 1

# 6) Cross-dataset comparison
python compare_datasets_results.py
```

## Where to Read More

- `README.md`: project-level end-to-end guide
- `README_PREPROCESSING.md`: preprocessing details
- `README_FEATURE_EXTRACTION.md`: handcrafted features
- `README_CLASSICAL_ML.md`: classical baseline details
- `README_DEEP_LEARNING.md`: deep learning details
- `README_INTERPRETABILITY.md`: Grad-CAM details
- `REPORT_JOURNAL_STYLE.md`: narrative report and discussion
