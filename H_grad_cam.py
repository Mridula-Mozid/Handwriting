import os
import cv2
import random
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import torch.nn as nn
from torchvision import transforms, models
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parent

import argparse

parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for a trained model and dataset')
parser.add_argument('--dataset', type=str, default=None, help='Dataset tag (e.g., Public_Dataset or BD_Dataset)')
parser.add_argument('--model-fold', type=int, default=1, help='Which fold model to load (default: 1)')
args = parser.parse_args()

DATASET_TAG = args.dataset if args.dataset else "default"

MODEL_PATH = PROJECT_ROOT / "trained_models_checkpoints" / DATASET_TAG / f"resnet18_fold_{args.model_fold}.pth"

preproc_root = PROJECT_ROOT / "preprocessed_images"
if args.dataset:
    preproc_root = preproc_root / args.dataset

METADATA_PATH = preproc_root / "metadata.csv"

metadata_df = pd.read_csv(METADATA_PATH)

metadata_df["label"] = metadata_df["class"].map({
    "healthy": 0,
    "parkinson": 1
})

metadata_df["image_path"] = metadata_df["gray_path"]

if "master_patient_id" not in metadata_df.columns:
    metadata_df["master_patient_id"] = metadata_df["patient_id"]
else:
    metadata_df["master_patient_id"] = metadata_df["master_patient_id"].fillna(
        metadata_df["patient_id"]
    )

STANDARD_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "handwriting" / "gradcam" / DATASET_TAG / f"fold_{args.model_fold}"
STANDARD_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_FOLDERS = {
    "parkinson": preproc_root / "grayscale" / "parkinson",
    "healthy": preproc_root / "grayscale" / "healthy",
}

OUTPUT_DIR = PROJECT_ROOT / "model_interpretability_visualizations" / DATASET_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n================================================")
print("STARTING GRAD-CAM GENERATION")
print("================================================")
print(f"Dataset: {DATASET_TAG}")
print(f"Model Path: {MODEL_PATH}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Standardized Output Directory: {STANDARD_OUTPUT_ROOT}")

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

# ============================================================
# BUILD MODEL
# ============================================================

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

# ============================================================
# GRAD CAM CLASS
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):

        output = self.model(input_tensor)

        if class_idx is None:

            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        target = output[:, class_idx]

        target.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        )

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(
            cam,
            (IMG_SIZE, IMG_SIZE)
        )

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return cam

# ============================================================
# LOAD MODEL
# ============================================================

model = build_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)

model.eval()

# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.layer4[-1]

gradcam = GradCAM(
    model,
    target_layer
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "handwriting"
    / DATASET_TAG
    / "predictions"
    / f"fold_{args.model_fold}_oof_predictions.csv"
)

pred_df = pd.read_csv(PREDICTIONS_PATH)

BEST_THRESHOLD = float(
    pred_df["optimized_threshold"].iloc[0]
)

print("Using threshold:", BEST_THRESHOLD)

# ============================================================
# VALIDATION SPLIT / SELECTION
# ============================================================

def safe_component(value):
    return str(value).replace(os.sep, "_").replace(" ", "_")


def resolve_validation_dataframe():

    trace_path = PROJECT_ROOT / "outputs" / "handwriting" / DATASET_TAG / "folds" / "fold_split_trace.csv"

    if trace_path.exists():

        trace_df = pd.read_csv(trace_path)
        fold_row = trace_df.loc[trace_df["fold"] == args.model_fold]

        if not fold_row.empty:
            validation_ids = str(fold_row.iloc[0]["validation_patient_ids"]).split(";")
            return metadata_df[metadata_df["patient_id"].astype(str).isin(validation_ids)].copy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = metadata_df.index.values
    y = metadata_df["label"].values
    groups = metadata_df["patient_id"].values

    for fold_index, (_, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        if fold_index == args.model_fold:
            return metadata_df.iloc[test_idx].copy()

    raise ValueError(f"Could not resolve validation split for fold {args.model_fold}.")


def predict_row(row):

    original = cv2.imread(str(row["image_path"]), cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise FileNotFoundError(f"Could not read image: {row['image_path']}")

    original_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    pil_image = Image.fromarray(original_resized)
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    probability_pd = float(probabilities[:, 1].item())
    predicted_class = int(probability_pd >= BEST_THRESHOLD)

    return {
        "source_index": int(row.name),
        "patient_id": row["patient_id"],
        "master_patient_id": row["master_patient_id"],
        "fold": args.model_fold,
        "true_label": int(row["label"]),
        "predicted_probability": probability_pd,
        "predicted_class": predicted_class,
        "threshold_used": 0.5,
        "modality": "handwriting",
        "dataset_name": DATASET_TAG,
        "image_name": row["filename"],
        "image_path": row["image_path"],
        "original_resized": original_resized,
        "input_tensor": input_tensor,
    }


def select_examples(validation_predictions_df):

    examples = []

    correct_df = validation_predictions_df.loc[
        validation_predictions_df["predicted_class"] == validation_predictions_df["true_label"]
    ]
    incorrect_df = validation_predictions_df.loc[
        validation_predictions_df["predicted_class"] != validation_predictions_df["true_label"]
    ]

    if not correct_df.empty:
        healthy_correct = correct_df.loc[correct_df["true_label"] == 0].sort_values("predicted_probability")
        if not healthy_correct.empty:
            examples.append(("best_healthy_correct", healthy_correct.iloc[0]))

        pd_correct = correct_df.loc[correct_df["true_label"] == 1].sort_values("predicted_probability", ascending=False)
        if not pd_correct.empty:
            examples.append(("best_parkinson_correct", pd_correct.iloc[0]))

    if not incorrect_df.empty:
        healthy_wrong = incorrect_df.loc[incorrect_df["true_label"] == 0].sort_values("predicted_probability", ascending=False)
        if not healthy_wrong.empty:
            examples.append(("worst_healthy_wrong", healthy_wrong.iloc[0]))

        pd_wrong = incorrect_df.loc[incorrect_df["true_label"] == 1].sort_values("predicted_probability")
        if not pd_wrong.empty:
            examples.append(("worst_parkinson_wrong", pd_wrong.iloc[0]))

    uncertain_df = validation_predictions_df.iloc[(validation_predictions_df["predicted_probability"] - 0.5).abs().argsort()]
    if not uncertain_df.empty:
        examples.append(("most_uncertain", uncertain_df.iloc[0]))

    return examples


def generate_gradcam_artifact(sample_row, example_role):

    original_resized = sample_row["original_resized"]
    input_tensor = sample_row["input_tensor"]

    cam = gradcam.generate_cam(input_tensor)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    original_bgr = cv2.cvtColor(
        original_resized,
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap,
        0.4,
        0
    )

    base_name = (
        f"fold_{args.model_fold}_{example_role}_"
        f"{safe_component(sample_row['patient_id'])}_"
        f"{safe_component(sample_row['image_name'])}_"
        f"p{sample_row['predicted_probability']:.3f}"
    )

    legacy_save_path = OUTPUT_DIR / f"{base_name}_gradcam.png"
    legacy_preview_path = OUTPUT_DIR / f"{base_name}_preview.png"
    standardized_save_path = STANDARD_OUTPUT_ROOT / f"{base_name}_gradcam.png"
    standardized_preview_path = STANDARD_OUTPUT_ROOT / f"{base_name}_preview.png"

    for save_path in [legacy_save_path, standardized_save_path]:
        cv2.imwrite(str(save_path), overlay)

    for preview_path in [legacy_preview_path, standardized_preview_path]:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(original_resized, cmap='gray')
        plt.title(f"Original - {sample_row['dataset_name']}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title("Grad-CAM")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(preview_path, bbox_inches="tight")
        plt.close()

    return {
        "example_role": example_role,
        "patient_id": sample_row["patient_id"],
        "master_patient_id": sample_row["master_patient_id"],
        "fold": args.model_fold,
        "true_label": sample_row["true_label"],
        "predicted_probability": sample_row["predicted_probability"],
        "predicted_class": sample_row["predicted_class"],
        "threshold_used": sample_row["threshold_used"],
        "modality": "handwriting",
        "dataset_name": DATASET_TAG,
        "image_name": sample_row["image_name"],
        "image_path": sample_row["image_path"],
        "legacy_gradcam_path": str(legacy_save_path),
        "legacy_preview_path": str(legacy_preview_path),
        "standard_gradcam_path": str(standardized_save_path),
        "standard_preview_path": str(standardized_preview_path),
    }


validation_df = resolve_validation_dataframe().copy()
validation_prediction_records = []

for _, row in validation_df.iterrows():
    prediction_record = predict_row(row)
    validation_prediction_records.append({
        key: value for key, value in prediction_record.items() if key not in {"original_resized", "input_tensor"}
    })

validation_predictions_df = pd.DataFrame(validation_prediction_records)
validation_predictions_standard_path = STANDARD_OUTPUT_ROOT / "validation_predictions.csv"
validation_predictions_legacy_path = OUTPUT_DIR / f"fold_{args.model_fold}_validation_predictions.csv"
validation_predictions_df.to_csv(validation_predictions_standard_path, index=False)
validation_predictions_df.to_csv(validation_predictions_legacy_path, index=False)

selected_examples = select_examples(validation_predictions_df)
gradcam_metadata_rows = []

for example_role, sample_series in selected_examples:
    sample_row = predict_row(validation_df.loc[sample_series["source_index"]])
    gradcam_metadata_rows.append(
        generate_gradcam_artifact(sample_row, example_role)
    )

gradcam_metadata_df = pd.DataFrame(gradcam_metadata_rows)
gradcam_metadata_path = STANDARD_OUTPUT_ROOT / "gradcam_metadata.csv"
gradcam_metadata_df.to_csv(gradcam_metadata_path, index=False)

print("\nSaved validation predictions:")
print(validation_predictions_standard_path)
print("Saved Grad-CAM metadata:")
print(gradcam_metadata_path)