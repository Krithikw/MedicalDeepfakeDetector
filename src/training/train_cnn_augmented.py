from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

BEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "cnn_augmented_best.pt"
)

# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_baseline import CNNBaseline


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

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CONTROLLED TRAINING AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomAffine(
        degrees=5,
        translate=(4 / 128, 4 / 128),
    ),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
    ),

])


# ============================================================
# DATASET
# ============================================================

class AugmentedROIDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.csv_file = Path(csv_file)

        if not self.csv_file.exists():

            raise FileNotFoundError(
                f"Metadata file not found:\n{self.csv_file}"
            )

        self.data = pd.read_csv(self.csv_file)

        required_columns = [
            "roi_path",
            "binary_label",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.data.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        roi_path = Path(
            str(row["roi_path"])
        )

        if not roi_path.is_absolute():

            roi_path = PROJECT_ROOT / roi_path

        if not roi_path.exists():

            raise FileNotFoundError(
                f"ROI file not found:\n{roi_path}"
            )

        image = np.load(
            roi_path
        ).astype(np.float32)

        if image.shape != (128, 128):

            raise ValueError(
                f"Unexpected ROI shape at {roi_path}: "
                f"{image.shape}"
            )

        if not np.isfinite(image).all():

            raise ValueError(
                f"ROI contains non-finite values:\n{roi_path}"
            )

        image = torch.from_numpy(image)

        image = image.unsqueeze(0)

        if self.transform is not None:

            image = self.transform(image)

        label = torch.tensor(
            float(row["binary_label"]),
            dtype=torch.float32
        )

        return image, label


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {

        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),

        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "binary_f1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),
    }


# ============================================================
# TRAINING
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

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

    return (
        running_loss
        / len(loader.dataset)
    )


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

    all_labels = []
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

            running_loss += (
                loss.item()
                * images.size(0)
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    labels = np.asarray(
        all_labels
    )

    probabilities = np.asarray(
        all_probabilities
    )

    metrics = calculate_metrics(
        labels,
        probabilities
    )

    metrics["loss"] = (
        running_loss
        / len(loader.dataset)
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MEDICAL DEEPFAKE DETECTOR")
    print("CONTROLLED AUGMENTED CNN EXPERIMENT")
    print("=" * 70)

    print("\nDevice:")
    print(device)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    train_csv = (
        ROI_ROOT
        / "train_roi_metadata.csv"
    )

    val_csv = (
        ROI_ROOT
        / "val_roi_metadata.csv"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = AugmentedROIDataset(
        train_csv,
        transform=train_transform,
    )

    val_dataset = AugmentedROIDataset(
        val_csv,
        transform=None,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    train_labels = (
        train_dataset.data["binary_label"]
        .astype(int)
        .to_numpy()
    )

    negative_count = np.sum(
        train_labels == 0
    )

    positive_count = np.sum(
        train_labels == 1
    )

    pos_weight_value = (
        negative_count
        / positive_count
    )

    print("\nDataset:")
    print(
        f"Train      : {len(train_dataset)}"
    )
    print(
        f"Validation : {len(val_dataset)}"
    )

    print("\nClass distribution:")
    print(
        f"Authentic  : {negative_count}"
    )
    print(
        f"Tampered   : {positive_count}"
    )

    print(
        f"Positive weight: {pos_weight_value:.6f}"
    )

    print("\nAugmentation:")
    print(
        "RandomAffine: ±5° rotation, ±4 pixel translation"
    )
    print(
        "ColorJitter: brightness ±8%, contrast ±8%"
    )
    print(
        "Validation augmentation: NONE"
    )
    print(
        "EXP2 augmentation: NONE"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNNBaseline()

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    pos_weight = torch.tensor(
        pos_weight_value,
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
    # Training
    # --------------------------------------------------------

    best_macro_f1 = -np.inf

    best_epoch = 0

    patience_counter = 0

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

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

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"\nEpoch {epoch:02d}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss       : {train_loss:.6f}"
        )

        print(
            f"Val Loss         : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Val Accuracy     : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Balanced Acc : "
            f"{val_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Val Precision    : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Val Recall       : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Val Macro F1     : "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Binary F1    : "
            f"{val_metrics['binary_f1']:.4f}"
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "validation_metrics":
                        val_metrics,

                    "seed":
                        SEED,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "augmentation":
                        "RandomAffine ±5 degrees, ±4 pixels; "
                        "ColorJitter brightness/contrast ±8%",
                },
                BEST_CHECKPOINT,
            )

            print(
                "  -> Best augmented checkpoint saved."
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement "
                f"({patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUGMENTED TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch    : {best_epoch}"
    )

    print(
        f"Best Macro F1 : {best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint    : {BEST_CHECKPOINT}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "EXP2 was NOT used during training."
    )

    print(
        "The original CNN baseline checkpoint was NOT modified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()