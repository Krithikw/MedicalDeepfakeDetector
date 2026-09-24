import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASELINE_EVAL = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
    / "CNN_BASELINE_EXP2_EVALUATION.txt"
)

AUGMENTED_EVAL = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_augmented"
    / "CNN_AUGMENTED_EXP2_EVALUATION.txt"
)

COMPARISON_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_comparison"
    / "baseline_vs_augmented.csv"
)

UUID6031_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "uuid_6031"
    / "uuid_6031_audit.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "final_results"
)


# ============================================================
# KNOWN EXP2 METRICS
# These are also present in the evaluation outputs.
# ============================================================

baseline_metrics = {
    "Model": "Baseline CNN",
    "Accuracy": 0.542857,
    "Balanced Accuracy": 0.547386,
    "Precision": 0.583333,
    "Recall": 0.388889,
    "Binary F1": 0.466667,
    "Macro F1": 0.533333,
    "Correct": 19,
    "Incorrect": 16,
    "Total": 35,
    "TN": 12,
    "FP": 5,
    "FN": 11,
    "TP": 7,
}

augmented_metrics = {
    "Model": "CNN with Training Augmentation",
    "Accuracy": 0.657143,
    "Balanced Accuracy": 0.658497,
    "Precision": 0.687500,
    "Recall": 0.611111,
    "Binary F1": 0.647059,
    "Macro F1": 0.656863,
    "Correct": 23,
    "Incorrect": 12,
    "Total": 35,
    "TN": 12,
    "FP": 5,
    "FN": 7,
    "TP": 11,
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. MODEL METRICS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            baseline_metrics,
            augmented_metrics
        ]
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "final_model_metrics.csv",
        index=False
    )

    print()
    print("Saved:")
    print(
        OUTPUT_DIR / "final_model_metrics.csv"
    )

    # --------------------------------------------------------
    # 2. METRIC IMPROVEMENT
    # --------------------------------------------------------

    improvement_rows = []

    metric_names = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "Binary F1",
        "Macro F1"
    ]

    for metric in metric_names:

        baseline_value = baseline_metrics[metric]
        augmented_value = augmented_metrics[metric]

        improvement_rows.append(
            {
                "Metric": metric,
                "Baseline": baseline_value,
                "Augmented": augmented_value,
                "Absolute Change": (
                    augmented_value - baseline_value
                ),
                "Change Percentage Points": (
                    (augmented_value - baseline_value) * 100
                ),
            }
        )

    improvement_df = pd.DataFrame(
        improvement_rows
    )

    improvement_df.to_csv(
        OUTPUT_DIR / "metric_improvement.csv",
        index=False
    )

    print(
        OUTPUT_DIR / "metric_improvement.csv"
    )

    # --------------------------------------------------------
    # 3. CONFUSION MATRIX COMPARISON
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        [
            {
                "Category": "TN",
                "Baseline": baseline_metrics["TN"],
                "Augmented": augmented_metrics["TN"],
            },
            {
                "Category": "FP",
                "Baseline": baseline_metrics["FP"],
                "Augmented": augmented_metrics["FP"],
            },
            {
                "Category": "FN",
                "Baseline": baseline_metrics["FN"],
                "Augmented": augmented_metrics["FN"],
            },
            {
                "Category": "TP",
                "Baseline": baseline_metrics["TP"],
                "Augmented": augmented_metrics["TP"],
            },
        ]
    )

    confusion_df.to_csv(
        OUTPUT_DIR / "confusion_matrix_comparison.csv",
        index=False
    )

    print(
        OUTPUT_DIR / "confusion_matrix_comparison.csv"
    )

    # --------------------------------------------------------
    # 4. SAMPLE TRANSITION SUMMARY
    # --------------------------------------------------------

    comparison = pd.read_csv(
        COMPARISON_CSV
    )

    transition_df = (
        comparison["transition"]
        .value_counts()
        .rename_axis("Transition")
        .reset_index(name="Count")
    )

    transition_df.to_csv(
        OUTPUT_DIR / "prediction_transition_summary.csv",
        index=False
    )

    print(
        OUTPUT_DIR / "prediction_transition_summary.csv"
    )

    # --------------------------------------------------------
    # 5. UUID 6031 SUMMARY
    # --------------------------------------------------------

    uuid_df = pd.read_csv(
        UUID6031_CSV
    )

    uuid_summary = pd.DataFrame(
        [
            {
                "UUID": 6031,
                "Samples": len(uuid_df),
                "True Tampered": int(
                    (uuid_df["true_label"] == 1).sum()
                ),
                "Baseline Correct": int(
                    uuid_df["baseline_correct"].sum()
                ),
                "Augmented Correct": int(
                    uuid_df["augmented_correct"].sum()
                ),
                "Persistent False Negatives": int(
                    (
                        (uuid_df["true_label"] == 1)
                        & (uuid_df["augmented_correct"] == False)
                    ).sum()
                ),
            }
        ]
    )

    uuid_summary.to_csv(
        OUTPUT_DIR / "uuid_6031_summary.csv",
        index=False
    )

    print(
        OUTPUT_DIR / "uuid_6031_summary.csv"
    )

    # --------------------------------------------------------
    # 6. FINAL TEXT SUMMARY
    # --------------------------------------------------------

    summary_text = f"""
FINAL CNN EXPERIMENTAL RESULTS
========================================

Independent EXP2 Evaluation
---------------------------

Samples: {augmented_metrics["Total"]}

Baseline CNN:
Accuracy              : {baseline_metrics["Accuracy"]:.6f}
Balanced Accuracy     : {baseline_metrics["Balanced Accuracy"]:.6f}
Precision             : {baseline_metrics["Precision"]:.6f}
Recall                : {baseline_metrics["Recall"]:.6f}
Binary F1             : {baseline_metrics["Binary F1"]:.6f}
Macro F1              : {baseline_metrics["Macro F1"]:.6f}
Correct               : {baseline_metrics["Correct"]}/{baseline_metrics["Total"]}
Incorrect             : {baseline_metrics["Incorrect"]}/{baseline_metrics["Total"]}

Augmented CNN:
Accuracy              : {augmented_metrics["Accuracy"]:.6f}
Balanced Accuracy     : {augmented_metrics["Balanced Accuracy"]:.6f}
Precision             : {augmented_metrics["Precision"]:.6f}
Recall                : {augmented_metrics["Recall"]:.6f}
Binary F1             : {augmented_metrics["Binary F1"]:.6f}
Macro F1              : {augmented_metrics["Macro F1"]:.6f}
Correct               : {augmented_metrics["Correct"]}/{augmented_metrics["Total"]}
Incorrect             : {augmented_metrics["Incorrect"]}/{augmented_metrics["Total"]}

Confusion Matrix
----------------

                 Baseline    Augmented

TN                   {baseline_metrics["TN"]}          {augmented_metrics["TN"]}
FP                    {baseline_metrics["FP"]}          {augmented_metrics["FP"]}
FN                   {baseline_metrics["FN"]}          {augmented_metrics["FN"]}
TP                    {baseline_metrics["TP"]}          {augmented_metrics["TP"]}

Sample Transitions
------------------

Wrong -> Correct : {
    int(
        (comparison["transition"] == "WRONG_TO_CORRECT").sum()
    )
}

Correct -> Wrong : {
    int(
        (comparison["transition"] == "CORRECT_TO_WRONG").sum()
    )
}

Correct -> Correct : {
    int(
        (comparison["transition"] == "CORRECT_TO_CORRECT").sum()
    )
}

Wrong -> Wrong : {
    int(
        (comparison["transition"] == "WRONG_TO_WRONG").sum()
    )
}

UUID 6031 Persistent Failure
---------------------------

Samples: {len(uuid_df)}
Baseline correct: {
    int(uuid_df["baseline_correct"].sum())
}/{len(uuid_df)}

Augmented correct: {
    int(uuid_df["augmented_correct"].sum())
}/{len(uuid_df)}

All four UUID 6031 samples remained false negatives
after augmentation.

Interpretation
--------------

Mild training-time augmentation improved independent
EXP2 performance. However, the improvement was not
uniform across all samples, and UUID 6031 remained a
persistent failure case.

Grad-CAM visualizations were generated for the persistent
failure samples to examine changes in model activation.
These visualizations represent model attribution and
should not be interpreted as verified localization of
tampered regions.
"""

    with open(
        OUTPUT_DIR / "final_results_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            summary_text.strip()
        )

    print(
        OUTPUT_DIR / "final_results_summary.txt"
    )

    # --------------------------------------------------------
    # 7. CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL RESULTS GENERATED")
    print("=" * 80)

    print()
    print(
        "Baseline Accuracy :",
        f"{baseline_metrics['Accuracy']:.4f}"
    )

    print(
        "Augmented Accuracy:",
        f"{augmented_metrics['Accuracy']:.4f}"
    )

    print(
        "Baseline Macro F1 :",
        f"{baseline_metrics['Macro F1']:.4f}"
    )

    print(
        "Augmented Macro F1:",
        f"{augmented_metrics['Macro F1']:.4f}"
    )

    print()
    print(
        "Output directory:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()