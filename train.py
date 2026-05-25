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
import json
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
import torchvision.transforms.functional as TF

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

STANDARD_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "handwriting"
STANDARD_DATASET_ROOT = STANDARD_OUTPUT_ROOT / (args.dataset if args.dataset else "default")
STANDARD_PREDICTIONS_DIR = STANDARD_DATASET_ROOT / "predictions"
STANDARD_EMBEDDINGS_DIR = STANDARD_DATASET_ROOT / "embeddings"
STANDARD_GRADCAM_DIR = STANDARD_DATASET_ROOT / "gradcam"
STANDARD_METRICS_DIR = STANDARD_DATASET_ROOT / "metrics"
STANDARD_LOGS_DIR = STANDARD_DATASET_ROOT / "logs"
STANDARD_FOLDS_DIR = STANDARD_DATASET_ROOT / "folds"

DATASET_TAG = args.dataset if args.dataset else "default"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STANDARD_PREDICTIONS_DIR, exist_ok=True)
os.makedirs(STANDARD_EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(STANDARD_GRADCAM_DIR, exist_ok=True)
os.makedirs(STANDARD_METRICS_DIR, exist_ok=True)
os.makedirs(STANDARD_LOGS_DIR, exist_ok=True)
os.makedirs(STANDARD_FOLDS_DIR, exist_ok=True)

# =========================================================================
# CONFIG
# =========================================================================

USE_BINARY_IMAGES = False

IMG_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 40

LEARNING_RATE = 5e-5

PATIENCE = 10

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

if "master_patient_id" not in metadata_df.columns:
    metadata_df["master_patient_id"] = metadata_df["patient_id"]
else:
    metadata_df["master_patient_id"] = metadata_df["master_patient_id"].fillna(
        metadata_df["patient_id"]
    )

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

        master_patient_id = row["master_patient_id"] if "master_patient_id" in row else patient_id

        image_name = row["filename"] if "filename" in row else Path(image_path).name

        try:

            image = Image.open(image_path).convert("L")

        except Exception:

            image = Image.new(
                "L",
                (IMG_SIZE, IMG_SIZE)
            )

        if self.transform:

            image = self.transform(image)

        metadata = {
            "patient_id": patient_id,
            "master_patient_id": master_patient_id,
            "image_name": image_name,
            "image_path": image_path,
        }

        return image, label, metadata

# =========================================================================
# TRANSFORMS
# =========================================================================

train_transform = transforms.Compose([

    transforms.RandomRotation(3),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03)
    ),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08
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
# EXPORT / ANALYSIS HELPERS
# =========================================================================


def safe_roc_auc_score(true_labels, probabilities):

    try:
        if len(np.unique(true_labels)) < 2:
            return np.nan
        return roc_auc_score(true_labels, probabilities)
    except ValueError:
        return np.nan


def forward_penultimate(model, inputs):

    features = model.conv1(inputs)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)

    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)

    features = model.avgpool(features)
    features = torch.flatten(features, 1)

    return features


def infer_logits(model, images, use_tta=False):

    outputs = model(images)

    if use_tta:
        rotated_images = TF.rotate(images, angle=3)
        rotated_outputs = model(rotated_images)
        outputs = (outputs + rotated_outputs) / 2

    return outputs


def collect_probabilities(model, data_loader, use_tta=False):

    model.eval()

    true_labels = []
    probabilities = []

    with torch.no_grad():

        for images, labels, _ in data_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = infer_logits(
                model,
                images,
                use_tta=use_tta
            )

            batch_probabilities = torch.softmax(outputs, dim=1)[:, 1]

            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(batch_probabilities.cpu().numpy())

    return np.asarray(true_labels), np.asarray(probabilities)


def optimize_threshold(true_labels, probabilities):

    best_threshold = 0.5
    best_balanced_accuracy = -1.0
    best_youden_index = -1.0

    for threshold in np.arange(0.30, 0.701, 0.01):

        predicted_labels = (probabilities >= threshold).astype(int)

        cm = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        balanced_accuracy = (sensitivity + specificity) / 2
        youden_index = sensitivity + specificity - 1

        if (
            balanced_accuracy > best_balanced_accuracy
            or (
                np.isclose(balanced_accuracy, best_balanced_accuracy)
                and youden_index > best_youden_index
            )
        ):
            best_threshold = float(np.round(threshold, 2))
            best_balanced_accuracy = balanced_accuracy
            best_youden_index = youden_index

    return best_threshold


def label_distribution_dict(labels):

    healthy_count = int(np.sum(np.array(labels) == 0))
    pd_count = int(np.sum(np.array(labels) == 1))

    return {
        "healthy_count": healthy_count,
        "pd_count": pd_count,
        "patient_count": int(len(labels)),
    }


def serialize_ids(values):

    return ";".join(map(str, values))


def batch_metadata_rows(batch_metadata):

    batch_size = len(batch_metadata["patient_id"])

    rows = []

    for index in range(batch_size):

        row = {}

        for key, values in batch_metadata.items():

            if isinstance(values, (list, tuple)):
                row[key] = values[index]
            elif hasattr(values, "tolist"):
                converted = values.tolist()
                row[key] = converted[index] if isinstance(converted, list) else converted
            else:
                row[key] = values

        rows.append(row)

    return rows


def export_run_metadata():

    run_metadata = {
        "seed": SEED,
        "dataset": DATASET_TAG,
        "metadata_path": str(METADATA_PATH),
        "preprocessed_root": str(DATA_ROOT),
        "model_type": "resnet18",
        "modality": "handwriting",
        "threshold_selection": "balanced_accuracy_with_youden_tiebreak",
        "optimized_threshold": None,
        "use_binary_images": USE_BINARY_IMAGES,
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "group_split": "StratifiedGroupKFold",
        "n_splits": 5,
    }

    run_metadata_path = STANDARD_LOGS_DIR / "run_metadata.json"

    with open(run_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    return run_metadata_path

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

    for param in model.layer3.parameters():

        param.requires_grad = True

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
    train_df,
    fold_num
):

    label_counts = train_df["label"].value_counts().to_dict()
    healthy_count = max(int(label_counts.get(0, 0)), 1)
    pd_count = max(int(label_counts.get(1, 0)), 1)

    class_weights = torch.tensor(
        [1.0 / healthy_count, 1.0 / pd_count],
        dtype=torch.float32,
        device=DEVICE
    )
    class_weights = class_weights / class_weights.mean()

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

                outputs = infer_logits(
                    model,
                    images,
                    use_tta=False
                )

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

        try:
            val_auc = roc_auc_score(
                true_labels,
                probs
            )
        except ValueError:
            val_auc = np.nan

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

    val_true_labels, val_probabilities = collect_probabilities(
        model,
        val_loader,
        use_tta=True
    )

    best_threshold = optimize_threshold(
        val_true_labels,
        val_probabilities
    )

    return model, best_threshold

# =========================================================================
# EVALUATION
# =========================================================================

def evaluate_model(model, test_loader, best_threshold):

    model.eval()

    preds = []

    probs = []

    true_labels = []

    patient_ids = []

    master_patient_ids = []

    image_names = []

    image_paths = []

    embedding_vectors = []

    prediction_rows = []

    embedding_rows = []

    with torch.no_grad():

        for images, labels, metadata in test_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = infer_logits(
                model,
                images,
                use_tta=True
            )

            embeddings = forward_penultimate(model, images)

            probabilities = torch.softmax(outputs, dim=1)

            probs.extend(
                probabilities[:, 1].cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

            batch_rows = batch_metadata_rows(metadata)

            for row_index, sample_meta in enumerate(batch_rows):

                patient_id = sample_meta.get("patient_id")

                master_patient_id = sample_meta.get("master_patient_id", patient_id)

                image_name = sample_meta.get("image_name")

                image_path = sample_meta.get("image_path")

                patient_ids.append(patient_id)

                master_patient_ids.append(master_patient_id)

                image_names.append(image_name)

                image_paths.append(image_path)

                probability_pd = float(probabilities[row_index, 1].cpu().item())

                predicted_class = int(probability_pd >= best_threshold)

                preds.append(predicted_class)

                prediction_rows.append({

                    "patient_id": patient_id,

                    "master_patient_id": master_patient_id,

                    "true_label": int(labels[row_index].cpu().item()),

                    "predicted_probability": probability_pd,

                    "predicted_class": predicted_class,

                    "threshold_used": best_threshold,

                    "optimized_threshold": best_threshold,

                    "modality": "handwriting",

                    "dataset_name": DATASET_TAG,

                    "image_name": image_name,

                    "image_path": image_path,

                })

                embedding_vector = embeddings[row_index].cpu().numpy().astype(float)

                embedding_vectors.append(embedding_vector)

                embedding_rows.append({

                    "patient_id": patient_id,

                    "master_patient_id": master_patient_id,

                    "true_label": int(labels[row_index].cpu().item()),

                    "fold": None,

                    "embedding_vector": json.dumps(embedding_vector.tolist()),

                })

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

    auc = safe_roc_auc_score(
        true_labels,
        probs
    )

    cm = confusion_matrix(
        true_labels,
        preds,
        labels=[0, 1]
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

        "master_patient_id": master_patient_ids,

        "fold": None,

        "true_label": true_labels,

        "predicted_probability": probs,

        "predicted_class": preds,

        "threshold_used": best_threshold,

        "optimized_threshold": best_threshold,

        "modality": "handwriting",

        "dataset_name": DATASET_TAG,

        "image_name": image_names,

        "image_path": image_paths,

        "predicted_label": preds,

        "probability_PD": probs
    })

    embedding_df = pd.DataFrame(embedding_rows)

    embedding_matrix = np.vstack(embedding_vectors) if embedding_vectors else np.empty((0, 0))

    return metrics, cm, prediction_df, embedding_df, embedding_matrix


def export_fold_trace(fold_trace_rows):

    fold_trace_df = pd.DataFrame(fold_trace_rows)

    fold_trace_path = STANDARD_FOLDS_DIR / "fold_split_trace.csv"

    fold_trace_df.to_csv(fold_trace_path, index=False)

    return fold_trace_path


def export_standardized_fold_outputs(
    fold_num,
    train_df,
    test_df,
    metrics,
    cm,
    prediction_df,
    embedding_df,
    embedding_matrix,
    optimized_threshold,
):

    dataset_predictions_dir = STANDARD_PREDICTIONS_DIR
    dataset_embeddings_dir = STANDARD_EMBEDDINGS_DIR
    dataset_metrics_dir = STANDARD_METRICS_DIR

    prediction_df = prediction_df.copy()
    embedding_df = embedding_df.copy()

    prediction_df["fold"] = fold_num
    embedding_df["fold"] = fold_num
    prediction_df["optimized_threshold"] = optimized_threshold

    prediction_path = dataset_predictions_dir / f"fold_{fold_num}_oof_predictions.csv"
    embedding_csv_path = dataset_embeddings_dir / f"fold_{fold_num}_embeddings.csv"
    embedding_npy_path = dataset_embeddings_dir / f"fold_{fold_num}_embeddings.npy"

    prediction_df.to_csv(prediction_path, index=False)
    embedding_df.to_csv(embedding_csv_path, index=False)
    np.save(embedding_npy_path, embedding_matrix)

    train_labels = train_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    fold_metrics_row = {
        "dataset_name": DATASET_TAG,
        "modality": "handwriting",
        "fold": fold_num,
        "threshold_used": optimized_threshold,
        "optimized_threshold": optimized_threshold,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "train_healthy_count": int(np.sum(np.array(train_labels) == 0)),
        "train_pd_count": int(np.sum(np.array(train_labels) == 1)),
        "train_patient_count": int(len(train_df)),
        "val_healthy_count": int(np.sum(np.array(test_labels) == 0)),
        "val_pd_count": int(np.sum(np.array(test_labels) == 1)),
        "val_patient_count": int(len(test_df)),
    }

    fold_metrics_path = dataset_metrics_dir / f"fold_{fold_num}_metrics.csv"
    pd.DataFrame([fold_metrics_row]).to_csv(fold_metrics_path, index=False)

    return {
        "prediction_path": prediction_path,
        "embedding_csv_path": embedding_csv_path,
        "embedding_npy_path": embedding_npy_path,
        "fold_metrics_path": fold_metrics_path,
        "fold_metrics_row": fold_metrics_row,
    }


def export_aggregate_metrics(metrics_df):

    aggregate_rows = []

    for column in metrics_df.columns:
        if not pd.api.types.is_numeric_dtype(metrics_df[column]):
            continue

        values = metrics_df[column].dropna().to_numpy(dtype=float)

        aggregate_rows.append({
            "metric": column,
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "variance": float(np.var(values, ddof=1)) if len(values) > 1 else np.nan,
        })

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_path = STANDARD_METRICS_DIR / "aggregate_metrics.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    return aggregate_path

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

    export_run_metadata()

    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    X = metadata_df[["image_path"]]

    y = metadata_df["label"]

    groups = metadata_df["patient_id"]

    all_metrics = []

    standardized_fold_metrics = []

    fold_trace_rows = []

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
        # LABEL DISTRIBUTION CHECK
        # ============================================================

        print("Train label distribution:")
        print(train_df["label"].value_counts())

        print("\nTest label distribution:")
        print(test_df["label"].value_counts())

        train_patient_ids = sorted(train_df["patient_id"].astype(str).unique().tolist())
        test_patient_ids = sorted(test_df["patient_id"].astype(str).unique().tolist())

        fold_trace_rows.append({
            "dataset_name": DATASET_TAG,
            "modality": "handwriting",
            "fold": fold_num,
            "train_patient_ids": serialize_ids(train_patient_ids),
            "validation_patient_ids": serialize_ids(test_patient_ids),
            "train_patient_count": int(train_df["patient_id"].nunique()),
            "validation_patient_count": int(test_df["patient_id"].nunique()),
            "train_sample_count": int(len(train_df)),
            "validation_sample_count": int(len(test_df)),
            "train_healthy_count": int((train_df["label"] == 0).sum()),
            "train_pd_count": int((train_df["label"] == 1).sum()),
            "validation_healthy_count": int((test_df["label"] == 0).sum()),
            "validation_pd_count": int((test_df["label"] == 1).sum()),
        })

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
            train_df,
            fold_num
        )

        model, best_threshold = model

        # ============================================================
        # EVALUATE
        # ============================================================

        metrics, cm, prediction_df, embedding_df, embedding_matrix = evaluate_model(
            model,
            test_loader,
            best_threshold
        )

        all_metrics.append(metrics)

        print("\nFold Metrics:\n")

        for k, v in metrics.items():

            print(f"[{DATASET_TAG}][Fold {fold_num}] {k}: {v:.4f}")

        standardized_export = export_standardized_fold_outputs(
            fold_num=fold_num,
            train_df=train_df,
            test_df=test_df,
            metrics=metrics,
            cm=cm,
            prediction_df=prediction_df,
            embedding_df=embedding_df,
            embedding_matrix=embedding_matrix,
            optimized_threshold=best_threshold,
        )

        standardized_fold_metrics.append(standardized_export["fold_metrics_row"])

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

    export_fold_trace(fold_trace_rows)

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

    standardized_fold_metrics_df = pd.DataFrame(standardized_fold_metrics)

    standardized_fold_metrics_path = STANDARD_METRICS_DIR / "fold_metrics.csv"

    standardized_fold_metrics_df.to_csv(
        standardized_fold_metrics_path,
        index=False
    )

    export_aggregate_metrics(
        standardized_fold_metrics_df.drop(columns=["fold"], errors="ignore")
    )

    print(
        f"\nSaved final results:\n"
        f"{final_results_path}"
    )

    print(
        f"Standardized exports saved under:\n"
        f"{STANDARD_DATASET_ROOT}"
    )

# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":

    run_cross_validation()