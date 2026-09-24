from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.create_torch_dataset import ROIDataset
from models.cnn_baseline import CNNBaseline


# ============================================================
# PATHS
# ============================================================

TEST_CSV = PROJECT_ROOT / "outputs" / "roi_dataset" / "test_roi_metadata.csv"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "cnn_augmented_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_augmented"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "EXP2_CNN_AUGMENTED_PREDICTIONS.csv"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
THRESHOLD = 0.5


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERATE AUGMENTED EXP2 PREDICTIONS")
    print("=" * 70)

    print()
    print("Test CSV:")
    print(TEST_CSV)

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    if not TEST_CSV.exists():
        print()
        print("ERROR: Test CSV not found.")
        sys.exit(1)

    if not CHECKPOINT.exists():
        print()
        print("ERROR: Augmented checkpoint not found.")
        sys.exit(1)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = ROIDataset(
        TEST_CSV
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print()
    print("EXP2 samples:", len(test_dataset))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNNBaseline()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print(
                "Checkpoint epoch:",
                checkpoint["epoch"]
            )

        if "val_macro_f1" in checkpoint:
            print(
                "Checkpoint validation Macro F1:",
                checkpoint["val_macro_f1"]
            )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = []

    sample_index = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).detach().cpu().numpy().reshape(-1)

            labels_np = (
                labels.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            predicted_np = (
                probabilities >= THRESHOLD
            ).astype(int)

            batch_size = len(probabilities)

            for i in range(batch_size):

                metadata_row = test_dataset.data.iloc[
                    sample_index
                ]

                row = metadata_row.to_dict()

                row["true_label"] = int(
                    labels_np[i]
                )

                row["predicted_label"] = int(
                    predicted_np[i]
                )

                row["p_tampered"] = float(
                    probabilities[i]
                )

                row["confidence"] = float(
                    max(
                        probabilities[i],
                        1.0 - probabilities[i]
                    )
                )

                row["correct"] = int(
                    labels_np[i]
                    == predicted_np[i]
                )

                if (
                    labels_np[i] == 0
                    and predicted_np[i] == 0
                ):
                    category = "TN"

                elif (
                    labels_np[i] == 0
                    and predicted_np[i] == 1
                ):
                    category = "FP"

                elif (
                    labels_np[i] == 1
                    and predicted_np[i] == 0
                ):
                    category = "FN"

                else:
                    category = "TP"

                row["prediction_category"] = category

                predictions.append(row)

                sample_index += 1

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        predictions
    )

    prediction_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    correct = int(
        prediction_df["correct"].sum()
    )

    incorrect = len(prediction_df) - correct

    print()
    print("=" * 70)
    print("PREDICTION GENERATION COMPLETE")
    print("=" * 70)

    print()
    print("Total samples:", len(prediction_df))
    print("Correct:", correct)
    print("Incorrect:", incorrect)

    print()
    print("Confusion categories:")

    print(
        prediction_df[
            "prediction_category"
        ].value_counts()
        .sort_index()
    )

    print()
    print("Saved:")
    print(OUTPUT_CSV)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
