import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "final_results"
)

OUTPUT_DIR = RESULTS_DIR / "figures"


def create_performance_figure():

    metrics_path = (
        RESULTS_DIR
        / "final_model_metrics.csv"
    )

    df = pd.read_csv(metrics_path)

    metric_names = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "Binary F1",
        "Macro F1"
    ]

    baseline = [
        float(df.loc[0, metric])
        for metric in metric_names
    ]

    augmented = [
        float(df.loc[1, metric])
        for metric in metric_names
    ]

    x = range(len(metric_names))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    baseline_positions = [
        i - width / 2
        for i in x
    ]

    augmented_positions = [
        i + width / 2
        for i in x
    ]

    ax.bar(
        baseline_positions,
        baseline,
        width,
        label="Baseline CNN"
    )

    ax.bar(
        augmented_positions,
        augmented,
        width,
        label="CNN with Training Augmentation"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Baseline vs Augmented CNN Performance"
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        metric_names,
        rotation=20
    )

    ax.set_ylim(
        0,
        1.0
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "baseline_vs_augmented_performance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Performance figure saved:"
    )
    print(output_path)


def create_confusion_matrix_figure():

    confusion_path = (
        RESULTS_DIR
        / "confusion_matrix_comparison.csv"
    )

    df = pd.read_csv(
        confusion_path
    )

    categories = [
        "TN",
        "FP",
        "FN",
        "TP"
    ]

    baseline = [
        int(
            df.loc[
                df["Category"] == category,
                "Baseline"
            ].iloc[0]
        )
        for category in categories
    ]

    augmented = [
        int(
            df.loc[
                df["Category"] == category,
                "Augmented"
            ].iloc[0]
        )
        for category in categories
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = range(
        len(categories)
    )

    width = 0.35

    baseline_positions = [
        i - width / 2
        for i in x
    ]

    augmented_positions = [
        i + width / 2
        for i in x
    ]

    ax.bar(
        baseline_positions,
        baseline,
        width,
        label="Baseline CNN"
    )

    ax.bar(
        augmented_positions,
        augmented,
        width,
        label="CNN with Training Augmentation"
    )

    ax.set_xlabel(
        "Prediction Category"
    )

    ax.set_ylabel(
        "Number of Samples"
    )

    ax.set_title(
        "Confusion Category Comparison on Independent EXP2"
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        [
            "True Negative",
            "False Positive",
            "False Negative",
            "True Positive"
        ]
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "baseline_vs_augmented_confusion_categories.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Confusion comparison figure saved:"
    )
    print(output_path)


def main():

    print("=" * 80)
    print("FINAL PERFORMANCE FIGURE GENERATION")
    print("=" * 80)

    create_performance_figure()

    create_confusion_matrix_figure()

    print()
    print("=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "Figures directory:"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()