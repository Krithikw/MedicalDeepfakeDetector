"""
Medical Deepfake Detector
CNN Baseline — EXP2 Error Analysis

Purpose:
    Analyze individual predictions from the frozen CNN baseline
    on the independent EXP2 test set.

Classes:
    0 = Authentic
    1 = Tampered

This script DOES NOT retrain the model.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data.create_torch_dataset import ROIDataset
from src.models.cnn_baseline import CNNBaseline


# ============================================================
# PATHS
# ============================================================

ROI_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "roi_dataset"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "cnn_baseline_best.pt"
)

ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
)

ANALYSIS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTIONS_FILE = (
    ANALYSIS_ROOT
    / "EXP2_CNN_PREDICTIONS.csv"
)

SUMMARY_FILE = (
    ANALYSIS_ROOT
    / "EXP2_CNN_ERROR_ANALYSIS.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    model = CNNBaseline()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return (
        model,
        checkpoint,
    )


# ============================================================
# LOAD EXP2 DATASET
# ============================================================

def load_test_dataset():

    test_csv = (
        ROI_ROOT
        / "test_roi_metadata.csv"
    )

    if not test_csv.exists():

        raise FileNotFoundError(
            f"Test metadata not found:\n"
            f"{test_csv}"
        )

    dataset = ROIDataset(
        test_csv
    )

    return dataset


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    dataset,
    device,
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    probabilities = np.asarray(
        all_probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    return (
        probabilities,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDICAL DEEPFAKE DETECTOR")
    print("CNN BASELINE — EXP2 ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING CNN CHECKPOINT")
    print("-" * 70)

    (
        model,
        checkpoint,
    ) = load_model(
        CHECKPOINT_PATH,
        device,
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    print(
        f"Checkpoint epoch: {checkpoint_epoch}"
    )

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING EXP2 TEST DATA")
    print("-" * 70)

    dataset = load_test_dataset()

    metadata = dataset.data.copy()

    print(
        f"Test samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GENERATING PREDICTIONS")
    print("-" * 70)

    (
        probabilities,
        predictions,
    ) = generate_predictions(
        model,
        dataset,
        device,
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    targets = (
        metadata[
            "binary_label"
        ]
        .to_numpy()
        .astype(np.int64)
    )

    # --------------------------------------------------------
    # Create prediction table
    # --------------------------------------------------------

    results = metadata[
        [
            "experiment",
            "uuid",
            "slice",
            "x",
            "y",
            "type",
            "binary_label",
            "roi_size",
            "pixel_spacing_x",
            "pixel_spacing_y",
            "dicom",
            "roi_path",
        ]
    ].copy()

    results[
        "predicted_label"
    ] = predictions

    results[
        "tampered_probability"
    ] = probabilities

    results[
        "authentic_probability"
    ] = 1.0 - probabilities

    results[
        "confidence"
    ] = np.maximum(
        probabilities,
        1.0 - probabilities
    )

    results[
        "correct"
    ] = (
        targets == predictions
    )

    # --------------------------------------------------------
    # Prediction category
    # --------------------------------------------------------

    def prediction_category(row):

        true_label = int(
            row["binary_label"]
        )

        predicted_label = int(
            row["predicted_label"]
        )

        if (
            true_label == 0
            and predicted_label == 0
        ):

            return "TN"

        if (
            true_label == 0
            and predicted_label == 1
        ):

            return "FP"

        if (
            true_label == 1
            and predicted_label == 0
        ):

            return "FN"

        if (
            true_label == 1
            and predicted_label == 1
        ):

            return "TP"

        return "UNKNOWN"

    results[
        "prediction_category"
    ] = results.apply(
        prediction_category,
        axis=1
    )

    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    results = results.sort_values(
        by=[
            "correct",
            "confidence",
        ],
        ascending=[
            True,
            False,
        ]
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    results.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        targets,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            targets,
            predictions
        )
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0,
    )

    binary_f1 = f1_score(
        targets,
        predictions,
        average="binary",
        zero_division=0,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    confusion = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    tn = confusion[0, 0]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    tp = confusion[1, 1]

    # --------------------------------------------------------
    # Confidence analysis
    # --------------------------------------------------------

    correct_mask = (
        results["correct"]
        .to_numpy()
    )

    correct_confidence = (
        results.loc[
            correct_mask,
            "confidence"
        ]
        .to_numpy()
    )

    incorrect_confidence = (
        results.loc[
            ~correct_mask,
            "confidence"
        ]
        .to_numpy()
    )

    if len(correct_confidence) > 0:

        mean_correct_confidence = (
            correct_confidence.mean()
        )

    else:

        mean_correct_confidence = 0.0

    if len(incorrect_confidence) > 0:

        mean_incorrect_confidence = (
            incorrect_confidence.mean()
        )

    else:

        mean_incorrect_confidence = 0.0

    # --------------------------------------------------------
    # Error counts
    # --------------------------------------------------------

    error_count = int(
        np.sum(
            targets != predictions
        )
    )

    correct_count = int(
        np.sum(
            targets == predictions
        )
    )

    # --------------------------------------------------------
    # Per-type analysis
    # --------------------------------------------------------

    type_lines = []

    type_order = [
        "FB",
        "FM",
        "TB",
        "TM",
    ]

    for annotation_type in type_order:

        subset = results[
            results["type"]
            == annotation_type
        ]

        if len(subset) == 0:

            continue

        type_targets = (
            subset[
                "binary_label"
            ]
            .to_numpy()
            .astype(np.int64)
        )

        type_predictions = (
            subset[
                "predicted_label"
            ]
            .to_numpy()
            .astype(np.int64)
        )

        type_accuracy = (
            accuracy_score(
                type_targets,
                type_predictions
            )
        )

        type_correct = int(
            np.sum(
                type_targets
                == type_predictions
            )
        )

        type_errors = int(
            len(subset)
            - type_correct
        )

        type_lines.append(
            (
                annotation_type,
                len(subset),
                type_correct,
                type_errors,
                type_accuracy,
            )
        )

    # --------------------------------------------------------
    # UUID / scan-group analysis
    # --------------------------------------------------------

    uuid_summary = (
        results
        .groupby("uuid")
        .agg(
            samples=(
                "uuid",
                "size"
            ),
            correct=(
                "correct",
                "sum"
            ),
            mean_confidence=(
                "confidence",
                "mean"
            ),
        )
        .reset_index()
    )

    uuid_summary[
        "errors"
    ] = (
        uuid_summary["samples"]
        - uuid_summary["correct"]
    )

    uuid_summary[
        "accuracy"
    ] = (
        uuid_summary["correct"]
        / uuid_summary["samples"]
    )

    uuid_summary = uuid_summary.sort_values(
        by=[
            "errors",
            "samples",
        ],
        ascending=[
            False,
            False,
        ]
    )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "MEDICAL DEEPFAKE DETECTOR"
    )

    report_lines.append(
        "CNN BASELINE — EXP2 ERROR ANALYSIS"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")

    report_lines.append(
        "CHECKPOINT"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        str(CHECKPOINT_PATH)
    )

    report_lines.append(
        f"Checkpoint epoch: {checkpoint_epoch}"
    )

    report_lines.append("")

    report_lines.append(
        "OVERALL EXP2 PERFORMANCE"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        f"Test samples: {len(targets)}"
    )

    report_lines.append(
        f"Correct predictions: {correct_count}"
    )

    report_lines.append(
        f"Incorrect predictions: {error_count}"
    )

    report_lines.append(
        f"Accuracy: {accuracy:.6f}"
    )

    report_lines.append(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.6f}"
    )

    report_lines.append(
        f"Precision: {precision:.6f}"
    )

    report_lines.append(
        f"Recall: {recall:.6f}"
    )

    report_lines.append(
        f"Binary F1: {binary_f1:.6f}"
    )

    report_lines.append(
        f"Macro F1: {macro_f1:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "CONFUSION MATRIX"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        "                    Predicted"
    )

    report_lines.append(
        "                  Authentic  Tampered"
    )

    report_lines.append(
        f"Actual Authentic     {tn:3d}       {fp:3d}"
    )

    report_lines.append(
        f"Actual Tampered      {fn:3d}       {tp:3d}"
    )

    report_lines.append("")

    report_lines.append(
        f"TN: {tn}"
    )

    report_lines.append(
        f"FP: {fp}"
    )

    report_lines.append(
        f"FN: {fn}"
    )

    report_lines.append(
        f"TP: {tp}"
    )

    report_lines.append("")

    report_lines.append(
        "CONFIDENCE ANALYSIS"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        f"Mean confidence — correct: "
        f"{mean_correct_confidence:.6f}"
    )

    report_lines.append(
        f"Mean confidence — incorrect: "
        f"{mean_incorrect_confidence:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "PER-TYPE ANALYSIS"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        "Type    Samples    Correct    Errors    Accuracy"
    )

    for (
        annotation_type,
        sample_count,
        type_correct,
        type_errors,
        type_accuracy,
    ) in type_lines:

        report_lines.append(
            f"{annotation_type:<7}"
            f"{sample_count:>7}"
            f"{type_correct:>11}"
            f"{type_errors:>10}"
            f"{type_accuracy:>11.4f}"
        )

    report_lines.append("")

    report_lines.append(
        "ERROR BREAKDOWN"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        "False positives: "
        f"{fp}"
    )

    report_lines.append(
        "False negatives: "
        f"{fn}"
    )

    report_lines.append("")

    report_lines.append(
        "FALSE NEGATIVE CASES"
    )

    report_lines.append(
        "-" * 70
    )

    false_negatives = results[
        results[
            "prediction_category"
        ]
        == "FN"
    ].sort_values(
        by="tampered_probability"
    )

    if len(false_negatives) == 0:

        report_lines.append(
            "None"
        )

    else:

        for _, row in false_negatives.iterrows():

            report_lines.append(
                f"UUID={row['uuid']} | "
                f"slice={row['slice']} | "
                f"type={row['type']} | "
                f"P(tampered)="
                f"{row['tampered_probability']:.4f}"
            )

    report_lines.append("")

    report_lines.append(
        "FALSE POSITIVE CASES"
    )

    report_lines.append(
        "-" * 70
    )

    false_positives = results[
        results[
            "prediction_category"
        ]
        == "FP"
    ].sort_values(
        by="tampered_probability",
        ascending=False
    )

    if len(false_positives) == 0:

        report_lines.append(
            "None"
        )

    else:

        for _, row in false_positives.iterrows():

            report_lines.append(
                f"UUID={row['uuid']} | "
                f"slice={row['slice']} | "
                f"type={row['type']} | "
                f"P(tampered)="
                f"{row['tampered_probability']:.4f}"
            )

    report_lines.append("")

    report_lines.append(
        "PER-UUID / SCAN-GROUP SUMMARY"
    )

    report_lines.append(
        "-" * 70
    )

    for _, row in uuid_summary.iterrows():

        report_lines.append(
            f"UUID={row['uuid']} | "
            f"samples={int(row['samples'])} | "
            f"correct={int(row['correct'])} | "
            f"errors={int(row['errors'])} | "
            f"accuracy={row['accuracy']:.4f} | "
            f"mean_confidence="
            f"{row['mean_confidence']:.4f}"
        )

    report_lines.append("")

    report_lines.append(
        "FILES"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        f"Per-sample predictions:\n"
        f"{PREDICTIONS_FILE}"
    )

    report_lines.append(
        f"\nAnalysis report:\n"
        f"{SUMMARY_FILE}"
    )

    report_lines.append("")

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "EXP2 CNN ERROR ANALYSIS COMPLETE"
    )

    report_lines.append(
        "STATUS: PASS"
    )

    report_lines.append(
        "=" * 70
    )

    report_text = "\n".join(
        report_lines
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    SUMMARY_FILE.write_text(
        report_text + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print()
    print(report_text)

    print()
    print(
        "Analysis files saved successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()