from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPARISON_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_comparison"
    / "baseline_vs_augmented.csv"
)

METADATA_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "roi_dataset"
    / "test_roi_metadata.csv"
)

BASELINE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_comparison"
    / "baseline_gradcam_changed"
)

AUGMENTED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_comparison"
    / "augmented_gradcam"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_comparison"
    / "clean_gradcam_comparison.png"
)


def find_roi(metadata, uuid, slice_number):

    matches = metadata[
        (metadata["uuid"].astype(str) == str(uuid))
        &
        (metadata["slice"].astype(int) == int(slice_number))
    ]

    if len(matches) == 0:
        return None

    return Path(matches.iloc[0]["roi_path"])


def load_roi(path):

    return np.load(path).astype(np.float32)


def find_gradcam(directory, uuid, slice_number):

    files = list(
        directory.glob(
            f"UUID{uuid}_SLICE{slice_number}_*.png"
        )
    )

    if not files:
        return None

    return files[0]


def extract_overlay(path):

    """
    The saved Grad-CAM figure has 3 panels:
        1. Original ROI
        2. Grad-CAM overlay
        3. Heatmap

    This function crops the middle panel only.
    """

    image = Image.open(path).convert("RGB")

    width, height = image.size

    # Middle third of the figure.
    left = width // 3
    right = (2 * width) // 3

    cropped = image.crop(
        (left, 0, right, height)
    )

    return np.asarray(cropped)


def main():

    print("=" * 80)
    print("CLEAN BASELINE VS AUGMENTED GRAD-CAM COMPARISON")
    print("=" * 80)

    comparison = pd.read_csv(
        COMPARISON_CSV
    )

    changed = comparison[
        comparison["baseline_predicted_label"]
        != comparison["augmented_predicted_label"]
    ].copy()

    metadata = pd.read_csv(
        METADATA_CSV
    )

    print()
    print("Changed samples:", len(changed))

    n = len(changed)

    fig, axes = plt.subplots(
        n,
        3,
        figsize=(12, 3.7 * n)
    )

    if n == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for i, (_, row) in enumerate(
        changed.iterrows()
    ):

        uuid = int(row["UUID"])
        slice_number = int(row["slice"])

        # ----------------------------------------------------------
        # ROI
        # ----------------------------------------------------------

        roi_path = find_roi(
            metadata,
            uuid,
            slice_number
        )

        if roi_path is None:
            print(
                "WARNING: ROI not found:",
                uuid,
                slice_number
            )
            continue

        roi = load_roi(
            roi_path
        )

        # ----------------------------------------------------------
        # Grad-CAM paths
        # ----------------------------------------------------------

        baseline_path = find_gradcam(
            BASELINE_DIR,
            uuid,
            slice_number
        )

        augmented_path = find_gradcam(
            AUGMENTED_DIR,
            uuid,
            slice_number
        )

        # ----------------------------------------------------------
        # Original ROI
        # ----------------------------------------------------------

        axes[i, 0].imshow(
            roi,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i, 0].set_title(
            "Original ROI",
            fontsize=11
        )

        axes[i, 0].axis("off")

        # ----------------------------------------------------------
        # Baseline Grad-CAM
        # ----------------------------------------------------------

        if baseline_path is not None:

            baseline_overlay = extract_overlay(
                baseline_path
            )

            axes[i, 1].imshow(
                baseline_overlay
            )

        else:

            axes[i, 1].text(
                0.5,
                0.5,
                "Baseline Grad-CAM\nnot found",
                ha="center",
                va="center"
            )

        axes[i, 1].set_title(
            "Baseline Grad-CAM",
            fontsize=11
        )

        axes[i, 1].axis("off")

        # ----------------------------------------------------------
        # Augmented Grad-CAM
        # ----------------------------------------------------------

        if augmented_path is not None:

            augmented_overlay = extract_overlay(
                augmented_path
            )

            axes[i, 2].imshow(
                augmented_overlay
            )

        else:

            axes[i, 2].text(
                0.5,
                0.5,
                "Augmented Grad-CAM\nnot found",
                ha="center",
                va="center"
            )

        axes[i, 2].set_title(
            "Augmented Grad-CAM",
            fontsize=11
        )

        axes[i, 2].axis("off")

        # ----------------------------------------------------------
        # Case information
        # ----------------------------------------------------------

        true_name = (
            "Tampered"
            if int(row["true_label"]) == 1
            else "Authentic"
        )

        baseline_name = (
            "Tampered"
            if int(row["baseline_predicted_label"]) == 1
            else "Authentic"
        )

        augmented_name = (
            "Tampered"
            if int(row["augmented_predicted_label"]) == 1
            else "Authentic"
        )

        info = (
            f"UUID {uuid} | Slice {slice_number}\n"
            f"True: {true_name}\n"
            f"Baseline: {baseline_name} "
            f"(P={float(row['baseline_p_tampered']):.3f})\n"
            f"Augmented: {augmented_name} "
            f"(P={float(row['augmented_p_tampered']):.3f})\n"
            f"{row['transition']}"
        )

        axes[i, 0].set_ylabel(
            info,
            fontsize=8,
            rotation=0,
            labelpad=55,
            va="center"
        )

    # --------------------------------------------------------------
    # Figure title
    # --------------------------------------------------------------

    fig.suptitle(
        "Baseline vs Augmented CNN — Grad-CAM Comparison",
        fontsize=16,
        y=0.995
    )

    plt.subplots_adjust(
        left=0.20,
        right=0.98,
        top=0.98,
        bottom=0.01,
        hspace=0.55,
        wspace=0.08
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("=" * 80)
    print("CLEAN COMPARISON COMPLETE")
    print("=" * 80)

    print()
    print("Saved:")
    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()