"""Generate side-by-side Public vs BD dataset comparison artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

DL_RESULTS_DIR = PROJECT_ROOT / "deep_learning_results"
CLASSICAL_RESULTS_DIR = PROJECT_ROOT / "classical_ml_results"

VISUALIZATION_DIR = PROJECT_ROOT.parent / "Visualization"
SUMMARY_DIR = PROJECT_ROOT / "results"

VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["Public_Dataset", "BD_Dataset"]
METRICS = ["accuracy", "sensitivity", "specificity", "roc_auc"]


def load_dl_metrics(dataset: str) -> dict:
    csv_path = DL_RESULTS_DIR / dataset / "final_cross_validation_results.csv"
    df = pd.read_csv(csv_path)
    metric_map = dict(zip(df["metric"], df["mean"]))
    return {metric: float(metric_map.get(metric, np.nan)) for metric in METRICS}


def load_best_classical_metrics(dataset: str) -> tuple[str, dict]:
    csv_path = CLASSICAL_RESULTS_DIR / dataset / "classical_ml_results.csv"
    df = pd.read_csv(csv_path)

    roc = df[df["metric"] == "roc_auc"].copy()
    best_model = roc.sort_values("mean", ascending=False).iloc[0]["model"]

    best_df = df[df["model"] == best_model]
    metric_map = dict(zip(best_df["metric"], best_df["mean"]))

    metrics = {metric: float(metric_map.get(metric, np.nan)) for metric in METRICS}
    return best_model, metrics


def build_summary() -> pd.DataFrame:
    rows = []

    for dataset in DATASETS:
        dl_metrics = load_dl_metrics(dataset)
        best_classical_model, classical_metrics = load_best_classical_metrics(dataset)

        for metric in METRICS:
            rows.append(
                {
                    "dataset": dataset,
                    "pipeline": "DeepLearning_ResNet18",
                    "model": "ResNet18",
                    "metric": metric,
                    "mean": dl_metrics[metric],
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "pipeline": "ClassicalML_Best",
                    "model": best_classical_model,
                    "metric": metric,
                    "mean": classical_metrics[metric],
                }
            )

    return pd.DataFrame(rows)


def plot_comparison(summary_df: pd.DataFrame) -> None:
    metric_titles = {
        "accuracy": "Accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "roc_auc": "ROC-AUC",
    }

    datasets = ["Public_Dataset", "BD_Dataset"]

    dl_public = []
    dl_bd = []
    cl_public = []
    cl_bd = []

    for metric in METRICS:
        dl_public.append(
            summary_df[
                (summary_df["dataset"] == datasets[0])
                & (summary_df["pipeline"] == "DeepLearning_ResNet18")
                & (summary_df["metric"] == metric)
            ]["mean"].iloc[0]
        )
        dl_bd.append(
            summary_df[
                (summary_df["dataset"] == datasets[1])
                & (summary_df["pipeline"] == "DeepLearning_ResNet18")
                & (summary_df["metric"] == metric)
            ]["mean"].iloc[0]
        )
        cl_public.append(
            summary_df[
                (summary_df["dataset"] == datasets[0])
                & (summary_df["pipeline"] == "ClassicalML_Best")
                & (summary_df["metric"] == metric)
            ]["mean"].iloc[0]
        )
        cl_bd.append(
            summary_df[
                (summary_df["dataset"] == datasets[1])
                & (summary_df["pipeline"] == "ClassicalML_Best")
                & (summary_df["metric"] == metric)
            ]["mean"].iloc[0]
        )

    x = np.arange(len(METRICS))
    width = 0.35

    plt.figure(figsize=(13, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(x - width / 2, dl_public, width=width, label="Public", color="#2a9d8f")
    ax1.bar(x + width / 2, dl_bd, width=width, label="BD", color="#e76f51")
    ax1.set_title("Deep Learning (ResNet-18)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([metric_titles[m] for m in METRICS], rotation=20)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(x - width / 2, cl_public, width=width, label="Public", color="#457b9d")
    ax2.bar(x + width / 2, cl_bd, width=width, label="BD", color="#f4a261")
    ax2.set_title("Best Classical ML Model", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([metric_titles[m] for m in METRICS], rotation=20)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    plt.suptitle(
        "Public_Dataset vs BD_Dataset Performance Comparison",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    png_path = VISUALIZATION_DIR / "09_Public_vs_BD_Performance_Comparison.png"
    pdf_path = VISUALIZATION_DIR / "09_Public_vs_BD_Performance_Comparison.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved comparison chart:")
    print(png_path)
    print(pdf_path)


def main() -> None:
    summary_df = build_summary()

    summary_csv_path = SUMMARY_DIR / "dataset_comparison_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("Saved comparison CSV:")
    print(summary_csv_path)

    plot_comparison(summary_df)


if __name__ == "__main__":
    main()
