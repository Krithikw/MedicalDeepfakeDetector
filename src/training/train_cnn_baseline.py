"""
Medical Deepfake Detector
CNN Baseline Training Pipeline

Training split:
    EXP1 only

Validation split:
    EXP1 only

Test split:
    EXP2 only
    NOT USED DURING TRAINING

Classes:
    0 = Authentic
    1 = Tampered
"""

from pathlib import Path
import random
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
# PROJECT PATHS
# ============================================================

ROI_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "roi_dataset"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "cnn_baseline_best.pt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16

NUM_EPOCHS = 40

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 8

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DATASET CREATION
# ============================================================

def create_datasets():

    train_csv = (
        ROI_ROOT
        / "train_roi_metadata.csv"
    )

    val_csv = (
        ROI_ROOT
        / "val_roi_metadata.csv"
    )

    test_csv = (
        ROI_ROOT
        / "test_roi_metadata.csv"
    )

    train_dataset = ROIDataset(
        train_csv
    )

    val_dataset = ROIDataset(
        val_csv
    )

    test_dataset = ROIDataset(
        test_csv
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# CLASS WEIGHT
# ============================================================

def calculate_pos_weight(train_dataset):

    labels = train_dataset.data[
        "binary_label"
    ].to_numpy()

    negative_count = np.sum(
        labels == 0
    )

    positive_count = np.sum(
        labels == 1
    )

    if positive_count == 0:

        raise ValueError(
            "Training dataset contains no positive samples."
        )

    if negative_count == 0:

        raise ValueError(
            "Training dataset contains no negative samples."
        )

    pos_weight = (
        negative_count
        / positive_count
    )

    return (
        negative_count,
        positive_count,
        pos_weight,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    targets = targets.astype(
        np.int64
    )

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

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    binary_f1 = f1_score(
        targets,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "binary_f1": binary_f1,
    }


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    total_samples = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    epoch_loss = (
        running_loss
        / total_samples
    )

    return epoch_loss


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    total_samples = 0

    all_targets = []

    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            probabilities = torch.sigmoid(
                logits
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            all_targets.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    epoch_loss = (
        running_loss
        / total_samples
    )

    all_targets = np.asarray(
        all_targets
    )

    all_probabilities = np.asarray(
        all_probabilities
    )

    metrics = calculate_metrics(
        all_targets,
        all_probabilities,
    )

    return (
        epoch_loss,
        metrics,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_metrics,
):

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }

    torch.save(
        checkpoint,
        BEST_CHECKPOINT
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("MEDICAL DEEPFAKE DETECTOR")
    print("CNN BASELINE TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    print()
    print(f"Random seed: {SEED}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: {torch.version.cuda}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING DATASETS")
    print("-" * 70)

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets()

    print(
        f"Train samples      : {len(train_dataset)}"
    )

    print(
        f"Validation samples : {len(val_dataset)}"
    )

    print(
        f"Test samples       : {len(test_dataset)}"
    )

    print()
    print("IMPORTANT:")
    print("EXP2 test data will NOT be used during training.")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    (
        negative_count,
        positive_count,
        pos_weight_value,
    ) = calculate_pos_weight(
        train_dataset
    )

    print()
    print("-" * 70)
    print("TRAINING CLASS DISTRIBUTION")
    print("-" * 70)

    print(
        f"Authentic (0): {negative_count}"
    )

    print(
        f"Tampered  (1): {positive_count}"
    )

    print(
        f"Positive class weight: {pos_weight_value:.6f}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    print()
    print("DataLoaders created successfully.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING CNN")
    print("-" * 70)

    model = CNNBaseline()

    model = model.to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TRAINING CONFIGURATION")
    print("-" * 70)

    print(
        f"Batch size              : {BATCH_SIZE}"
    )

    print(
        f"Maximum epochs          : {NUM_EPOCHS}"
    )

    print(
        f"Learning rate           : {LEARNING_RATE}"
    )

    print(
        f"Weight decay            : {WEIGHT_DECAY}"
    )

    print(
        f"Early stopping patience : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        "Validation metric       : Macro F1"
    )

    print()
    print("-" * 70)
    print("TRAINING")
    print("-" * 70)

    best_validation_f1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            validation_loss,
            validation_metrics,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_f1 = (
            validation_metrics[
                "macro_f1"
            ]
        )

        improved = (
            current_f1
            > best_validation_f1
        )

        if improved:

            best_validation_f1 = (
                current_f1
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                validation_metrics,
            )

            checkpoint_status = "  <-- BEST"

        else:

            epochs_without_improvement += 1

            checkpoint_status = ""

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: "
            f"{validation_metrics['accuracy']:.4f} | "
            f"Val Bal Acc: "
            f"{validation_metrics['balanced_accuracy']:.4f} | "
            f"Val Macro F1: "
            f"{validation_metrics['macro_f1']:.4f} | "
            f"Val Recall: "
            f"{validation_metrics['recall']:.4f}"
            f"{checkpoint_status}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_validation_f1:.6f}"
    )

    print()
    print(
        "Best checkpoint saved to:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()
    print(
        "NOTE: EXP2 test data has not been evaluated."
    )

    print(
        "The independent test evaluation will be performed"
    )

    print(
        "in a separate step after training is finalized."
    )

    print()
    print("=" * 70)
    print("CNN BASELINE TRAINING PIPELINE COMPLETE")
    print("STATUS: PASS")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()