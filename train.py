"""
========================================================================
PARKINSON'S HANDWRITING TRAINING PIPELINE
FINAL RESEARCH-GRADE VERSION (UPDATED)
========================================================================

Key Improvements:
- Proper grayscale adaptation of pretrained ResNet-18
- Conv1 initialized using pretrained RGB weights
- Conv1 is trainable (scientifically correct)
- Layer freezing strategy improved
- Fully aligned with thesis architecture figure
========================================================================
"""

import os
import copy
import random
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =========================================================================
# REPRODUCIBILITY
# =========================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================================================
# PATHS
# =========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

import argparse

parser = argparse.ArgumentParser(description='Train ResNet-18 for handwriting datasets')
parser.add_argument('--dataset', type=str, default=None,
                    help='Dataset tag (e.g., Public_Dataset or BD_Dataset). If provided reads preprocessed_images/<dataset>/metadata.csv')
args = parser.parse_args()

DATA_ROOT = PROJECT_ROOT / "preprocessed_images"
if args.dataset:
    DATA_ROOT = DATA_ROOT / args.dataset

METADATA_PATH = DATA_ROOT / "metadata.csv"

MODEL_SAVE_DIR = PROJECT_ROOT / "trained_models_checkpoints" / (args.dataset if args.dataset else "default")

RESULTS_DIR = PROJECT_ROOT / "deep_learning_results" / (args.dataset if args.dataset else "default")

DATASET_TAG = args.dataset if args.dataset else "default"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================================
# CONFIG
# =========================================================================

USE_BINARY_IMAGES = False

IMG_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 40

LEARNING_RATE = 1e-4

PATIENCE = 7

NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = ["Healthy", "Parkinson"]

# =========================================================================
# LOAD METADATA
# =========================================================================

metadata_df = pd.read_csv(METADATA_PATH)

metadata_df["label"] = metadata_df["class"].map({
    "healthy": 0,
    "parkinson": 1
})

if USE_BINARY_IMAGES:
    metadata_df["image_path"] = metadata_df["binary_path"]
else:
    metadata_df["image_path"] = metadata_df["gray_path"]

# =========================================================================
# DATASET
# =========================================================================

class HandwritingDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["image_path"]

        label = row["label"]

        patient_id = row["patient_id"]

        try:

            image = Image.open(image_path).convert("L")

        except Exception:

            image = Image.new(
                "L",
                (IMG_SIZE, IMG_SIZE)
            )

        if self.transform:

            image = self.transform(image)

        return image, label, patient_id

# =========================================================================
# TRANSFORMS
# =========================================================================

train_transform = transforms.Compose([

    transforms.RandomRotation(5),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# =========================================================================
# MODEL
# =========================================================================

def build_model():

    # ================================================================
    # LOAD IMAGENET PRETRAINED RESNET-18
    # ================================================================

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # ================================================================
    # SAVE ORIGINAL RGB CONV1 WEIGHTS
    # ================================================================

    original_conv1_weights = (
        model.conv1.weight.data.clone()
    )

    # ================================================================
    # REPLACE INPUT LAYER FOR GRAYSCALE INPUT
    # ================================================================

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # ================================================================
    # INITIALIZE NEW CONV1 USING
    # AVERAGED RGB PRETRAINED WEIGHTS
    # ================================================================

    model.conv1.weight.data = (
        original_conv1_weights.mean(
            dim=1,
            keepdim=True
        )
    )

    # ================================================================
    # REPLACE CLASSIFICATION HEAD
    # ================================================================

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    # ================================================================
    # FREEZE ALL PARAMETERS
    # ================================================================

    for param in model.parameters():

        param.requires_grad = False

    # ================================================================
    # UNFREEZE INPUT LAYER
    # ================================================================

    for param in model.conv1.parameters():

        param.requires_grad = True

    # ================================================================
    # UNFREEZE FINAL RESIDUAL BLOCK
    # ================================================================

    for param in model.layer4.parameters():

        param.requires_grad = True

    # ================================================================
    # UNFREEZE CLASSIFICATION HEAD
    # ================================================================

    for param in model.fc.parameters():

        param.requires_grad = True

    return model

# =========================================================================
# TRAINING
# =========================================================================

def train_one_fold(
    model,
    train_loader,
    val_loader,
    fold_num
):

    class_weights = torch.tensor(
        [1.0, 1.2]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=3,
        factor=0.5
    )

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    best_auc = 0

    patience_counter = 0

    for epoch in range(EPOCHS):

        # ============================================================
        # TRAINING
        # ============================================================

        model.train()

        running_loss = 0

        for images, labels, _ in train_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = (
            running_loss / len(train_loader)
        )

        # ============================================================
        # VALIDATION
        # ============================================================

        model.eval()

        preds = []

        probs = []

        true_labels = []

        with torch.no_grad():

            for images, labels, _ in val_loader:

                images = images.to(DEVICE)

                labels = labels.to(DEVICE)

                outputs = model(images)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                predictions = outputs.argmax(dim=1)

                preds.extend(
                    predictions.cpu().numpy()
                )

                probs.extend(
                    probabilities[:, 1].cpu().numpy()
                )

                true_labels.extend(
                    labels.cpu().numpy()
                )

        val_acc = accuracy_score(
            true_labels,
            preds
        )

        val_auc = roc_auc_score(
            true_labels,
            probs
        )

        scheduler.step(val_auc)

        print(
            f"[{DATASET_TAG}][Fold {fold_num}][Epoch {epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"AUC: {val_auc:.4f}"
        )

        # ============================================================
        # SAVE BEST MODEL
        # ============================================================

        if val_auc > best_auc:

            best_auc = val_auc

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        # ============================================================
        # EARLY STOPPING
        # ============================================================

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.\n")

            break

    model.load_state_dict(
        best_model_weights
    )

    return model

# =========================================================================
# EVALUATION
# =========================================================================

def evaluate_model(model, test_loader):

    model.eval()

    preds = []

    probs = []

    true_labels = []

    patient_ids = []

    with torch.no_grad():

        for images, labels, pids in test_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = outputs.argmax(dim=1)

            preds.extend(
                predictions.cpu().numpy()
            )

            probs.extend(
                probabilities[:, 1].cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

            patient_ids.extend(pids)

    # ================================================================
    # METRICS
    # ================================================================

    accuracy = accuracy_score(
        true_labels,
        preds
    )

    precision = precision_score(
        true_labels,
        preds,
        zero_division=0
    )

    sensitivity = recall_score(
        true_labels,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        preds,
        zero_division=0
    )

    auc = roc_auc_score(
        true_labels,
        probs
    )

    cm = confusion_matrix(
        true_labels,
        preds
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (
        tn + fp + 1e-8
    )

    metrics = {

        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": auc
    }

    prediction_df = pd.DataFrame({

        "patient_id": patient_ids,

        "true_label": true_labels,

        "predicted_label": preds,

        "probability_PD": probs
    })

    return metrics, cm, prediction_df

# =========================================================================
# CROSS VALIDATION
# =========================================================================

def run_cross_validation():

    print("\n================================================")
    print("STARTING DEEP LEARNING TRAINING")
    print("================================================")
    print(f"Dataset: {DATASET_TAG}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Model Output Dir: {MODEL_SAVE_DIR}")
    print(f"Results Output Dir: {RESULTS_DIR}\n")

    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    X = metadata_df["image_path"]

    y = metadata_df["label"]

    groups = metadata_df["patient_id"]

    all_metrics = []

    fold_num = 1

    for train_idx, test_idx in sgkf.split(
        X,
        y,
        groups
    ):

        print("\n================================================")
        print(f"[{DATASET_TAG}][Fold {fold_num}] START")
        print("================================================\n")

        train_df = metadata_df.iloc[train_idx]

        test_df = metadata_df.iloc[test_idx]

        # ============================================================
        # DATASETS
        # ============================================================

        train_dataset = HandwritingDataset(
            train_df,
            transform=train_transform
        )

        test_dataset = HandwritingDataset(
            test_df,
            transform=test_transform
        )

        # ============================================================
        # DATALOADERS
        # ============================================================

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # ============================================================
        # BUILD MODEL
        # ============================================================

        model = build_model().to(DEVICE)

        # ============================================================
        # TRAIN
        # ============================================================

        model = train_one_fold(
            model,
            train_loader,
            test_loader,
            fold_num
        )

        # ============================================================
        # EVALUATE
        # ============================================================

        metrics, cm, prediction_df = evaluate_model(
            model,
            test_loader
        )

        all_metrics.append(metrics)

        print("\nFold Metrics:\n")

        for k, v in metrics.items():

            print(f"[{DATASET_TAG}][Fold {fold_num}] {k}: {v:.4f}")

        # ============================================================
        # SAVE PREDICTIONS
        # ============================================================

        prediction_csv_path = (
            RESULTS_DIR
            / f"fold_{fold_num}_predictions.csv"
        )

        prediction_df.to_csv(
            prediction_csv_path,
            index=False
        )

        # ============================================================
        # CONFUSION MATRIX
        # ============================================================

        plt.figure(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES
        )

        plt.title(
            f"Fold {fold_num} Confusion Matrix"
        )

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.tight_layout()

        cm_save_path = (
            RESULTS_DIR
            / f"fold_{fold_num}_cm.png"
        )

        plt.savefig(cm_save_path)

        plt.close()

        # ============================================================
        # SAVE MODEL
        # ============================================================

        model_path = (
            MODEL_SAVE_DIR
            / f"resnet18_fold_{fold_num}.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(f"\n[{DATASET_TAG}][Fold {fold_num}] Saved model:\n{model_path}")

        fold_num += 1

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )

    mean_metrics = metrics_df.mean()

    std_metrics = metrics_df.std()

    print("\n================================================")
    print("FINAL RESULTS")
    print("================================================\n")
    print(f"Dataset: {DATASET_TAG}\n")

    for metric in mean_metrics.index:

        print(
            f"{metric}: "
            f"{mean_metrics[metric]:.4f} ± "
            f"{std_metrics[metric]:.4f}"
        )

    # ================================================================
    # SAVE FINAL RESULTS CSV
    # ================================================================

    final_results_df = pd.DataFrame({

        "metric": mean_metrics.index,

        "mean": mean_metrics.values,

        "std": std_metrics.values
    })

    final_results_path = (
        RESULTS_DIR
        / "final_cross_validation_results.csv"
    )

    final_results_df.to_csv(
        final_results_path,
        index=False
    )

    print(
        f"\nSaved final results:\n"
        f"{final_results_path}"
    )

# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":

    run_cross_validation()