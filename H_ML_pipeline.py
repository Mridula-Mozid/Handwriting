"""
========================================================================
HANDCRAFTED FEATURE + CLASSICAL ML PIPELINE
FOR PARKINSON'S HANDWRITING ANALYSIS
========================================================================

GOALS
-----
1. Evaluate classical ML approaches
2. Compare against deep learning models
3. Build publication-grade baseline systems
4. Prevent patient-level leakage
5. Use statistically safer feature reduction

MODELS
------
1. SVM
2. Random Forest
3. XGBoost

FEATURES
--------
Loaded from:
feature_extraction.py

EVALUATION
----------
- StratifiedGroupKFold
- Patient-level splitting
- Accuracy
- Precision
- Sensitivity
- Specificity
- F1-score
- ROC-AUC

OUTPUTS
-------
- Fold-wise metrics
- Final averaged metrics
- Confusion matrices
- Results CSV

========================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# =========================================================================
# CONFIGURATION
# =========================================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_DIR = PROJECT_ROOT / "extracted_features"

RESULTS_DIR = PROJECT_ROOT / "ml_results"

RESULTS_DIR.mkdir(exist_ok=True)

# =========================================================================
# LOAD FEATURES
# =========================================================================

print("\n================================================")
print("LOADING FEATURES")
print("================================================")

features = np.load(
    FEATURE_DIR / "features.npy"
)

labels = np.load(
    FEATURE_DIR / "labels.npy"
)

patient_ids = np.load(
    FEATURE_DIR / "patient_ids.npy"
)

print(f"\nFeature Shape: {features.shape}")

# =========================================================================
# PCA COMPONENT SAFETY
# =========================================================================

"""
Very important.

PCA components must NEVER exceed:
- number of samples
- number of features

Otherwise:
- instability
- overfitting
- numerical issues
"""

safe_pca_components = min(

    64,
    features.shape[0] - 1,
    features.shape[1]
)

print(f"\nUsing PCA Components: {safe_pca_components}")

# =========================================================================
# CROSS VALIDATION
# =========================================================================

sgkf = StratifiedGroupKFold(

    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# =========================================================================
# MODELS
# =========================================================================

models = {

    # =====================================================================
    # SVM
    # =====================================================================

    "SVM": Pipeline([

        (
            'scaler',
            StandardScaler()
        ),

        (
            'pca',
            PCA(
                n_components=safe_pca_components,
                random_state=SEED
            )
        ),

        (
            'clf',
            SVC(

                kernel='rbf',

                probability=True,

                class_weight='balanced',

                random_state=SEED
            )
        )
    ]),

    # =====================================================================
    # RANDOM FOREST
    # =====================================================================

    "RandomForest": Pipeline([

        (
            'pca',
            PCA(
                n_components=safe_pca_components,
                random_state=SEED
            )
        ),

        (
            'clf',
            RandomForestClassifier(

                n_estimators=300,

                max_depth=10,

                class_weight='balanced',

                random_state=SEED
            )
        )
    ]),

    # =====================================================================
    # XGBOOST
    # =====================================================================

    "XGBoost": Pipeline([

        (
            'pca',
            PCA(
                n_components=safe_pca_components,
                random_state=SEED
            )
        ),

        (
            'clf',
            XGBClassifier(

                n_estimators=200,

                learning_rate=0.03,

                max_depth=4,

                subsample=0.8,

                colsample_bytree=0.8,

                eval_metric='logloss',

                random_state=SEED
            )
        )
    ])
}

# =========================================================================
# STORE FINAL RESULTS
# =========================================================================

all_model_results = []

# =========================================================================
# MAIN EVALUATION LOOP
# =========================================================================

for model_name, model in models.items():

    print("\n================================================")
    print(f"MODEL: {model_name}")
    print("================================================")

    fold_metrics = []

    fold_num = 1

    for train_idx, test_idx in sgkf.split(

        features,
        labels,
        patient_ids
    ):

        print(f"\nFold {fold_num}")

        # -----------------------------------------------------------------
        # SPLIT
        # -----------------------------------------------------------------

        X_train = features[train_idx]
        X_test = features[test_idx]

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # -----------------------------------------------------------------
        # TRAIN
        # -----------------------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # -----------------------------------------------------------------
        # PREDICT
        # -----------------------------------------------------------------

        preds = model.predict(X_test)

        probs = model.predict_proba(X_test)[:, 1]

        # -----------------------------------------------------------------
        # METRICS
        # -----------------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            preds
        )

        precision = precision_score(
            y_test,
            preds,
            zero_division=0
        )

        sensitivity = recall_score(
            y_test,
            preds,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            preds,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            probs
        )

        cm = confusion_matrix(
            y_test,
            preds
        )

        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp + 1e-8)

        metrics = {

            "accuracy": accuracy,

            "precision": precision,

            "sensitivity": sensitivity,

            "specificity": specificity,

            "f1_score": f1,

            "roc_auc": auc
        }

        fold_metrics.append(metrics)

        # -----------------------------------------------------------------
        # PRINT
        # -----------------------------------------------------------------

        for k, v in metrics.items():

            print(f"{k}: {v:.4f}")

        # -----------------------------------------------------------------
        # SAVE CONFUSION MATRIX
        # -----------------------------------------------------------------

        plt.figure(figsize=(5, 4))

        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            cmap="Blues",

            xticklabels=["Healthy", "Parkinson"],

            yticklabels=["Healthy", "Parkinson"]
        )

        plt.title(
            f"{model_name} Fold {fold_num}"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()

        save_cm_path = (

            RESULTS_DIR
            / f"{model_name}_fold_{fold_num}_cm.png"
        )

        plt.savefig(
            save_cm_path,
            dpi=300
        )

        plt.close()

        fold_num += 1

    # =========================================================================
    # FINAL MODEL RESULTS
    # =========================================================================

    metrics_df = pd.DataFrame(fold_metrics)

    mean_metrics = metrics_df.mean()

    std_metrics = metrics_df.std()

    print("\n================================================")
    print(f"FINAL RESULTS: {model_name}")
    print("================================================\n")

    for metric in mean_metrics.index:

        print(

            f"{metric}: "
            f"{mean_metrics[metric]:.4f} "
            f"± "
            f"{std_metrics[metric]:.4f}"
        )

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------

    for metric in mean_metrics.index:

        all_model_results.append({

            "model": model_name,

            "metric": metric,

            "mean": mean_metrics[metric],

            "std": std_metrics[metric]
        })

# =========================================================================
# SAVE FINAL RESULTS CSV
# =========================================================================

results_df = pd.DataFrame(all_model_results)

results_csv_path = (
    RESULTS_DIR
    / "classical_ml_results.csv"
)

results_df.to_csv(
    results_csv_path,
    index=False
)

print("\n================================================")
print("CLASSICAL ML PIPELINE COMPLETE")
print("================================================")

print(f"\nSaved Results:\n{results_csv_path}")