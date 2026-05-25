# Deep Learning Training Pipeline for Parkinson's Handwriting Detection

## Overview

This module implements a robust 5-fold cross-validation pipeline training a ResNet-18 convolutional neural network to classify handwriting spirals as healthy or Parkinsonian. The architecture prioritizes generalization and clinical interpretability through stratified group cross-validation that prevents patient-level data leakage.

## Clinical Motivation

Deep learning approaches automatically discover discriminative patterns in pixel space without requiring hand-engineered features. However, for clinical applications, it is essential to ensure that models have not simply memorized patient-specific characteristics or learned spurious imaging artifacts. Our cross-validation strategy ensures genuine disease classification capability that would generalize to new patients.

## Training Strategy

**Transfer Learning**: ResNet-18 uses ImageNet pre-trained weights as initialization. The early convolutional layers (learned on natural images) capture generic edge detection and texture patterns, while later layers are fine-tuned specifically for handwriting analysis. This approach dramatically reduces training data requirements compared to training from random initialization.

**Architecture Adaptation**: The first convolutional layer is modified to accept single-channel grayscale input (instead of RGB), and the final classification layer outputs 2 classes (Healthy vs Parkinson). All other layers retain their pre-trained structure.

**Selective Layer Freezing**: Early layers remain frozen during training to preserve general feature extractors. Only the final residual block (layer4) and classification head are trainable, reducing the number of learnable parameters and preventing overfitting on the relatively modest dataset size.

**Class Weighting**: The training loss function applies higher weight (1.2×) to Parkinson's class samples, compensating for any minor class imbalance and ensuring the model remains sensitive to disease cases.

**Data Augmentation**: Training data undergoes:
  - Random rotation (±5 degrees) to handle natural pen angle variation
  - Random translation (±3%) to simulate writing position shifts
  - Random scaling (0.97–1.03×) to account for paper distance variation
  
These transformations improve model robustness without sacrificing the clinical validity of the handwriting characteristics.

## Cross-Validation Design

**Stratified Group K-Fold**: The stratified group k-fold splitter ensures each fold maintains the same healthy/Parkinson's class proportions while assigning all images from a single patient to either training or validation. This prevents information leakage where the model could recognize a specific patient's writing style rather than learning disease-relevant features.

**5 Folds**: Five independent train-validation splits are created. For each fold: 80% of unique patients (training set) and 20% (validation set) are used. Models are trained from scratch on each fold, and metrics are averaged to produce unbiased performance estimates.

## Optimization Details

**Optimizer**: AdamW with learning rate 1e-4 and weight decay 1e-4. The weight decay term provides mild L2 regularization to prevent overfitting.

**Learning Rate Scheduling**: If validation AUC plateaus for 3 consecutive epochs, the learning rate is reduced by 50%. This adaptive schedule prevents the optimizer from getting stuck in suboptimal regions.

**Early Stopping**: Training halts if validation AUC does not improve for 7 consecutive epochs. This prevents wasteful computation and reduces overfitting.

**Batch Size**: 8 samples per batch balances memory efficiency with stable gradient estimates.

**Epochs**: Maximum 40 epochs, though early stopping typically terminates around epoch 20-30.

## Output Artifacts

```
trained_models_checkpoints/
├── Public_Dataset/
│   ├── resnet18_fold_1.pth
│   ├── resnet18_fold_2.pth
│   └── ...
└── BD_Dataset/
  ├── resnet18_fold_1.pth
  ├── resnet18_fold_2.pth
  └── ...

deep_learning_results/
├── Public_Dataset/
│   ├── fold_1_predictions.csv
│   ├── fold_2_predictions.csv
│   └── final_cross_validation_results.csv
└── BD_Dataset/
  ├── fold_1_predictions.csv
  ├── fold_2_predictions.csv
  └── final_cross_validation_results.csv
```

## Evaluation Metrics

For each fold, the following metrics are computed on held-out validation data:

- **Accuracy**: Fraction of correct classifications (TP+TN)/(TP+TN+FP+FN)
- **Precision**: True positives among predicted positives (TP)/(TP+FP) - avoids false alarms
- **Sensitivity (Recall)**: True positives among actual positives (TP)/(TP+FN) - ensures disease detection
- **Specificity**: True negatives among actual negatives (TN)/(TN+FP) - ensures healthy classification
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve, summarizing performance across thresholds

The final reported metrics are the mean and standard deviation across all folds.

## Running the Pipeline

```bash
# Public dataset
python train.py --dataset Public_Dataset

# HandPD dataset
python train.py --dataset HandPD

# BD dataset
python train.py --dataset BD_Dataset
```

Prerequisites: Requires `preprocessed_images/Public_Dataset/metadata.csv`, `preprocessed_images/HandPD/metadata.csv`, and/or `preprocessed_images/BD_Dataset/metadata.csv` from the preprocessing stage.

Expected runtime: 1-2 hours on GPU, 5-10 hours on CPU.

## Reproducibility

All random seeds are fixed (SEED=42) including NumPy, PyTorch, and CUDA to ensure deterministic results. CuDNN is configured for deterministic operations at the cost of some performance.
