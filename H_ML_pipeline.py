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

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV

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

import argparse

parser = argparse.ArgumentParser(description='Run classical ML pipeline on handcrafted features')
parser.add_argument('--dataset', type=str, default=None,
                    help='Dataset tag used when features were saved (e.g., Public_Dataset or BD_Dataset)')
args = parser.parse_args()

FEATURE_DIR = PROJECT_ROOT / "handcrafted_features_classical_ml" / (args.dataset if args.dataset else "default")

RESULTS_DIR = PROJECT_ROOT / "classical_ml_results" / (args.dataset if args.dataset else "default")

DATASET_TAG = args.dataset if args.dataset else "default"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================================
# LOAD FEATURES
# =========================================================================

print("\n================================================")
print("LOADING FEATURES")
print("================================================")
print(f"Dataset: {DATASET_TAG}")
print(f"Feature Directory: {FEATURE_DIR}")

features = np.load(FEATURE_DIR / "handcrafted_features.npy")
labels = np.load(FEATURE_DIR / "class_labels.npy")
patient_ids = np.load(FEATURE_DIR / "patient_identifiers.npy")

print(f"\nFeature Shape: {features.shape}")

# =========================================================================
# CROSS VALIDATION
# =========================================================================

sgkf = StratifiedGroupKFold(

    n_splits=5,
    shuffle=True,
    random_state=SEED
)

cv_splits = list(sgkf.split(features, labels, patient_ids))

# =========================================================================
# MODELS
# =========================================================================

models = {

    # =====================================================================
    # SVM
    # =====================================================================

    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(
            score_func=mutual_info_classif,
            k=64
        )),
        ('clf', CalibratedClassifierCV(
            estimator=SVC(
                kernel='rbf',
                C=3,
                gamma='scale',
                class_weight='balanced',
                random_state=SEED
            ),
            method='sigmoid',
            cv=3
        ))
    ]),

    # =====================================================================
# RANDOM FOREST
# =====================================================================

"RandomForest": Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(
        score_func=mutual_info_classif,
        k=64
    )),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=SEED
    ))
]),

# =====================================================================
# XGBOOST
# =====================================================================

"XGBoost": Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(
        score_func=mutual_info_classif,
        k=64
    )),
    ('clf', XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=SEED
    ))
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
    print(f"[{DATASET_TAG}][{model_name}] START")
    print("================================================")

    fold_metrics = []

    fold_num = 1

    for train_idx, test_idx in cv_splits:

        print(f"\n[{DATASET_TAG}][{model_name}][Fold {fold_num}] START")

        # -----------------------------------------------------------------
        # SPLIT
        # -----------------------------------------------------------------

        X_train = features[train_idx]
        X_test = features[test_idx]

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        print("\nTrain label distribution:")
        print(pd.Series(y_train).value_counts())

        print("\nTest label distribution:")
        print(pd.Series(y_test).value_counts())

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

        try:
            auc = roc_auc_score(
                y_test,
                probs
            )
        except ValueError:
            auc = np.nan

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

            print(f"[{DATASET_TAG}][{model_name}][Fold {fold_num}] {k}: {v:.4f}")

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
    print(f"[{DATASET_TAG}][{model_name}] FINAL RESULTS")
    print("================================================\n")

    for metric in mean_metrics.index:

        print(

            f"[{DATASET_TAG}][{model_name}] {metric}: "
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
print(f"Dataset: {DATASET_TAG}")

print(f"\nSaved Results:\n{results_csv_path}")