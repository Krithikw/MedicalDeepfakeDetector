"""
Medical Deepfake Detector
CNN Baseline — Independent EXP2 Evaluation

IMPORTANT:
    This script evaluates the already-trained CNN checkpoint
    on EXP2 only.

    EXP2 is the independent held-out test experiment.

Classes:
    0 = Authentic
    1 = Tampered
"""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
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
    / "cnn_augmented_best.pt"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FILE = (
    RESULTS_ROOT
    / "CNN_AUGMENTED_EXP2_EVALUATION.txt"
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
# LOAD CHECKPOINT
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"CNN checkpoint not found:\n"
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
# TEST DATASET
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

    test_dataset = ROIDataset(
        test_csv
    )

    return test_dataset


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    test_loader,
    device,
):

    all_targets = []

    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            all_targets.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    targets = np.asarray(
        all_targets
    ).astype(np.int64)

    probabilities = np.asarray(
        all_probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    return (
        targets,
        probabilities,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDICAL DEEPFAKE DETECTOR")
    print("CNN BASELINE — INDEPENDENT EXP2 EVALUATION")
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
    # Checkpoint
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING BEST CHECKPOINT")
    print("-" * 70)

    print(
        f"Checkpoint:\n{CHECKPOINT_PATH}"
    )

    (
        model,
        checkpoint,
    ) = load_model(
        CHECKPOINT_PATH,
        device,
    )

    print(
        f"Training epoch of checkpoint: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    checkpoint_metrics = checkpoint.get(
        "validation_metrics",
        {}
    )

    if checkpoint_metrics:

        print(
            f"Checkpoint validation Macro F1: "
            f"{checkpoint_metrics.get('macro_f1', 0.0):.6f}"
        )

    print()
    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Load EXP2
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING INDEPENDENT EXP2 TEST SET")
    print("-" * 70)

    test_dataset = load_test_dataset()

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Verify test labels
    # --------------------------------------------------------

    test_labels = (
        test_dataset.data[
            "binary_label"
        ]
        .to_numpy()
        .astype(np.int64)
    )

    authentic_count = np.sum(
        test_labels == 0
    )

    tampered_count = np.sum(
        test_labels == 1
    )

    print(
        f"Authentic (0): {authentic_count}"
    )

    print(
        f"Tampered  (1): {tampered_count}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("EVALUATING EXP2")
    print("-" * 70)

    (
        targets,
        probabilities,
        predictions,
    ) = evaluate_model(
        model,
        test_loader,
        device,
    )

    # --------------------------------------------------------
    # Metrics
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

    true_negative = confusion[0, 0]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]
    true_positive = confusion[1, 1]

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXP2 TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Test samples          : {len(targets)}"
    )

    print(
        f"Correct predictions   : "
        f"{np.sum(targets == predictions)}"
    )

    print(
        f"Incorrect predictions : "
        f"{np.sum(targets != predictions)}"
    )

    print()
    print(
        f"Accuracy              : {accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy     : "
        f"{balanced_accuracy:.6f}"
    )

    print(
        f"Precision             : {precision:.6f}"
    )

    print(
        f"Recall                : {recall:.6f}"
    )

    print(
        f"Binary F1             : {binary_f1:.6f}"
    )

    print(
        f"Macro F1              : {macro_f1:.6f}"
    )

    print()
    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print()
    print("                    Predicted")
    print("                  Authentic  Tampered")
    print(
        f"Actual Authentic     {true_negative:3d}       "
        f"{false_positive:3d}"
    )
    print(
        f"Actual Tampered      {false_negative:3d}       "
        f"{true_positive:3d}"
    )

    print()
    print(
        "TN (Authentic → Authentic): "
        f"{true_negative}"
    )

    print(
        "FP (Authentic → Tampered): "
        f"{false_positive}"
    )

    print(
        "FN (Tampered → Authentic): "
        f"{false_negative}"
    )

    print(
        "TP (Tampered → Tampered): "
        f"{true_positive}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_text = f"""
MEDICAL DEEPFAKE DETECTOR
CNN BASELINE — INDEPENDENT EXP2 EVALUATION
============================================================

Checkpoint:
{CHECKPOINT_PATH}

Checkpoint epoch:
{checkpoint.get("epoch", "unknown")}

Checkpoint validation Macro F1:
{checkpoint_metrics.get("macro_f1", 0.0):.6f}

TEST SET
------------------------------------------------------------
Experiment: EXP2
Samples: {len(targets)}

Authentic (0): {authentic_count}
Tampered  (1): {tampered_count}

METRICS
------------------------------------------------------------
Accuracy:
{accuracy:.6f}

Balanced Accuracy:
{balanced_accuracy:.6f}

Precision:
{precision:.6f}

Recall:
{recall:.6f}

Binary F1:
{binary_f1:.6f}

Macro F1:
{macro_f1:.6f}

CONFUSION MATRIX
------------------------------------------------------------
                    Predicted
                  Authentic  Tampered

Actual Authentic     {true_negative:3d}       {false_positive:3d}
Actual Tampered      {false_negative:3d}       {true_positive:3d}

TN: {true_negative}
FP: {false_positive}
FN: {false_negative}
TP: {true_positive}

============================================================
TEST SET:
EXP2 ONLY

EXP2 was not used during CNN training.
============================================================
"""

    RESULTS_FILE.write_text(
        results_text.strip() + "\n",
        encoding="utf-8"
    )

    print()
    print("-" * 70)
    print("RESULTS SAVED")
    print("-" * 70)

    print(
        RESULTS_FILE
    )

    print()
    print("=" * 70)
    print("CNN BASELINE EXP2 EVALUATION COMPLETE")
    print("STATUS: PASS")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()