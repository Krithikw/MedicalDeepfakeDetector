"""
Medical Deepfake Detector
CNN Baseline — EXP2 Error Visualization

Purpose:
    Visually inspect all independent EXP2 test ROIs and CNN predictions.

Creates:
    1. Complete EXP2 prediction montage
    2. False-negative montage
    3. False-positive montage
    4. UUID 6031 focused montage
    5. Individual high-resolution error images

No model training is performed.
No test data is modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
    / "EXP2_CNN_PREDICTIONS.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
    / "visualizations"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

MAX_COLUMNS = 5

IMAGE_SIZE = 128


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    data = pd.read_csv(
        PREDICTIONS_FILE
    )

    required_columns = [
        "experiment",
        "uuid",
        "slice",
        "x",
        "y",
        "type",
        "binary_label",
        "predicted_label",
        "tampered_probability",
        "confidence",
        "correct",
        "prediction_category",
        "roi_path",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns in prediction file: "
            f"{missing_columns}"
        )

    return data


# ============================================================
# LABEL HELPERS
# ============================================================

def label_name(label):

    if int(label) == 0:

        return "Authentic"

    return "Tampered"


def prediction_category_name(category):

    names = {
        "TN": "TRUE NEGATIVE",
        "FP": "FALSE POSITIVE",
        "FN": "FALSE NEGATIVE",
        "TP": "TRUE POSITIVE",
    }

    return names.get(
        str(category),
        str(category)
    )


# ============================================================
# LOAD ROI
# ============================================================

def load_roi(path_string):

    roi_path = Path(
        str(path_string)
    )

    if not roi_path.exists():

        raise FileNotFoundError(
            f"ROI not found:\n{roi_path}"
        )

    image = np.load(
        roi_path
    ).astype(np.float32)

    if image.shape != (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):

        raise ValueError(
            f"Unexpected ROI shape "
            f"{image.shape}:\n{roi_path}"
        )

    return image


# ============================================================
# CREATE TITLE
# ============================================================

def make_title(row):

    true_label = label_name(
        row["binary_label"]
    )

    predicted_label = label_name(
        row["predicted_label"]
    )

    probability = float(
        row["tampered_probability"]
    )

    confidence = float(
        row["confidence"]
    )

    category = (
        prediction_category_name(
            row["prediction_category"]
        )
    )

    return (
        f"UUID {int(row['uuid'])} | "
        f"slice {int(row['slice'])}\n"
        f"{row['type']} | "
        f"True: {true_label} | "
        f"Pred: {predicted_label}\n"
        f"P(tampered)={probability:.3f} | "
        f"Conf={confidence:.3f}\n"
        f"{category}"
    )


# ============================================================
# CREATE MONTAGE
# ============================================================

def create_montage(
    data,
    output_path,
    title,
):

    if len(data) == 0:

        print(
            f"\nNo samples available for: "
            f"{title}"
        )

        return

    number_of_images = len(data)

    columns = min(
        MAX_COLUMNS,
        number_of_images
    )

    rows = int(
        np.ceil(
            number_of_images
            / columns
        )
    )

    figure_width = (
        columns * 3.2
    )

    figure_height = (
        rows * 3.4
        + 0.7
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            figure_width,
            figure_height,
        )
    )

    axes = np.asarray(
        axes
    ).reshape(
        -1
    )

    for axis, (_, row) in zip(
        axes,
        data.iterrows()
    ):

        image = load_roi(
            row["roi_path"]
        )

        axis.imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axis.set_title(
            make_title(row),
            fontsize=8
        )

        axis.axis("off")

    # Hide unused axes

    for axis in axes[
        number_of_images:
    ]:

        axis.axis("off")

    figure.suptitle(
        title,
        fontsize=14,
        fontweight="bold"
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"Saved:\n{output_path}"
    )


# ============================================================
# CREATE INDIVIDUAL ERROR IMAGE
# ============================================================

def save_individual_image(
    row,
    output_directory,
    index,
):

    image = load_roi(
        row["roi_path"]
    )

    uuid_value = int(
        row["uuid"]
    )

    slice_value = int(
        row["slice"]
    )

    category = str(
        row["prediction_category"]
    )

    annotation_type = str(
        row["type"]
    )

    probability = float(
        row["tampered_probability"]
    )

    output_name = (
        f"{index:02d}_"
        f"{category}_"
        f"UUID{uuid_value}_"
        f"slice{slice_value}_"
        f"{annotation_type}_"
        f"P{probability:.3f}.png"
    )

    output_path = (
        output_directory
        / output_name
    )

    figure, axis = plt.subplots(
        figsize=(6, 6)
    )

    axis.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    axis.set_title(
        make_title(row),
        fontsize=11
    )

    axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDICAL DEEPFAKE DETECTOR")
    print("CNN BASELINE — EXP2 ERROR VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING CNN PREDICTIONS")
    print("-" * 70)

    data = load_predictions()

    print(
        f"Total samples: {len(data)}"
    )

    # --------------------------------------------------------
    # Verify experiment
    # --------------------------------------------------------

    experiments = (
        data["experiment"]
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Experiments present: "
        f"{experiments}"
    )

    if experiments != ["EXP2"]:

        raise RuntimeError(
            "Expected EXP2-only predictions."
        )

    # --------------------------------------------------------
    # Sort for visualization
    # --------------------------------------------------------

    # First show errors, then correct predictions.
    #
    # Within errors:
    #     false negatives first
    #     then false positives
    #
    # Within each group:
    #     highest confidence first

    category_order = {
        "FN": 0,
        "FP": 1,
        "TP": 2,
        "TN": 3,
    }

    data = data.copy()

    data["_category_order"] = (
        data["prediction_category"]
        .map(category_order)
    )

    data = data.sort_values(
        by=[
            "_category_order",
            "confidence",
        ],
        ascending=[
            True,
            False,
        ]
    )

    # --------------------------------------------------------
    # Split groups
    # --------------------------------------------------------

    false_negatives = data[
        data[
            "prediction_category"
        ] == "FN"
    ].copy()

    false_positives = data[
        data[
            "prediction_category"
        ] == "FP"
    ].copy()

    true_positives = data[
        data[
            "prediction_category"
        ] == "TP"
    ].copy()

    true_negatives = data[
        data[
            "prediction_category"
        ] == "TN"
    ].copy()

    # --------------------------------------------------------
    # Print counts
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("PREDICTION GROUPS")
    print("-" * 70)

    print(
        f"False negatives : "
        f"{len(false_negatives)}"
    )

    print(
        f"False positives : "
        f"{len(false_positives)}"
    )

    print(
        f"True positives  : "
        f"{len(true_positives)}"
    )

    print(
        f"True negatives  : "
        f"{len(true_negatives)}"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    fn_directory = (
        OUTPUT_ROOT
        / "false_negatives"
    )

    fp_directory = (
        OUTPUT_ROOT
        / "false_positives"
    )

    fn_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    fp_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create complete montage
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING COMPLETE EXP2 MONTAGE")
    print("-" * 70)

    create_montage(
        data,
        OUTPUT_ROOT
        / "EXP2_ALL_PREDICTIONS.png",
        "EXP2 — ALL CNN PREDICTIONS",
    )

    # --------------------------------------------------------
    # False-negative montage
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING FALSE-NEGATIVE MONTAGE")
    print("-" * 70)

    create_montage(
        false_negatives,
        OUTPUT_ROOT
        / "EXP2_FALSE_NEGATIVES.png",
        "EXP2 — FALSE NEGATIVES",
    )

    # --------------------------------------------------------
    # False-positive montage
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING FALSE-POSITIVE MONTAGE")
    print("-" * 70)

    create_montage(
        false_positives,
        OUTPUT_ROOT
        / "EXP2_FALSE_POSITIVES.png",
        "EXP2 — FALSE POSITIVES",
    )

    # --------------------------------------------------------
    # True-positive montage
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING TRUE-POSITIVE MONTAGE")
    print("-" * 70)

    create_montage(
        true_positives,
        OUTPUT_ROOT
        / "EXP2_TRUE_POSITIVES.png",
        "EXP2 — TRUE POSITIVES",
    )

    # --------------------------------------------------------
    # True-negative montage
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING TRUE-NEGATIVE MONTAGE")
    print("-" * 70)

    create_montage(
        true_negatives,
        OUTPUT_ROOT
        / "EXP2_TRUE_NEGATIVES.png",
        "EXP2 — TRUE NEGATIVES",
    )

    # --------------------------------------------------------
    # UUID 6031 analysis
    # --------------------------------------------------------

    uuid_6031 = data[
        data["uuid"].astype(int)
        == 6031
    ].copy()

    print()
    print("-" * 70)
    print("UUID 6031 ANALYSIS")
    print("-" * 70)

    print(
        f"UUID 6031 samples: "
        f"{len(uuid_6031)}"
    )

    if len(uuid_6031) > 0:

        for _, row in uuid_6031.iterrows():

            print(
                f"slice={int(row['slice'])} | "
                f"type={row['type']} | "
                f"true={label_name(row['binary_label'])} | "
                f"pred={label_name(row['predicted_label'])} | "
                f"P(tampered)="
                f"{float(row['tampered_probability']):.6f}"
            )

        create_montage(
            uuid_6031,
            OUTPUT_ROOT
            / "EXP2_UUID6031.png",
            "EXP2 — UUID 6031",
        )

    # --------------------------------------------------------
    # Save individual false negatives
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("SAVING INDIVIDUAL FALSE NEGATIVES")
    print("-" * 70)

    for index, (_, row) in enumerate(
        false_negatives.iterrows(),
        start=1
    ):

        save_individual_image(
            row,
            fn_directory,
            index,
        )

    print(
        f"Saved {len(false_negatives)} "
        f"false-negative images."
    )

    # --------------------------------------------------------
    # Save individual false positives
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("SAVING INDIVIDUAL FALSE POSITIVES")
    print("-" * 70)

    for index, (_, row) in enumerate(
        false_positives.iterrows(),
        start=1
    ):

        save_individual_image(
            row,
            fp_directory,
            index,
        )

    print(
        f"Saved {len(false_positives)} "
        f"false-positive images."
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CNN EXP2 ERROR VISUALIZATION COMPLETE")
    print("=" * 70)

    print()
    print("Visualization directory:")
    print(
        OUTPUT_ROOT
    )

    print()
    print("Created files include:")

    print(
        "  EXP2_ALL_PREDICTIONS.png"
    )

    print(
        "  EXP2_FALSE_NEGATIVES.png"
    )

    print(
        "  EXP2_FALSE_POSITIVES.png"
    )

    print(
        "  EXP2_TRUE_POSITIVES.png"
    )

    print(
        "  EXP2_TRUE_NEGATIVES.png"
    )

    print(
        "  EXP2_UUID6031.png"
    )

    print()
    print(
        "Individual error images:"
    )

    print(
        f"  {fn_directory}"
    )

    print(
        f"  {fp_directory}"
    )

    print()
    print("=" * 70)
    print(
        "STATUS: PASS"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()