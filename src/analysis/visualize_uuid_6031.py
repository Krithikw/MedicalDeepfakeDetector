from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "uuid_6031"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "uuid_6031_visual_audit.png"
)


def find_roi(metadata, slice_number):

    matches = metadata[
        (metadata["uuid"].astype(str) == "6031")
        &
        (metadata["slice"].astype(int) == int(slice_number))
    ]

    if len(matches) == 0:
        return None

    return Path(matches.iloc[0]["roi_path"])


def find_gradcam(directory, slice_number):

    files = list(
        directory.glob(
            f"UUID6031_SLICE{slice_number}_*.png"
        )
    )

    if not files:
        return None

    return files[0]


def extract_overlay(path):

    image = Image.open(path).convert("RGB")

    width, height = image.size

    left = width // 3
    right = (2 * width) // 3

    cropped = image.crop(
        (left, 0, right, height)
    )

    return np.asarray(cropped)


def main():

    print("=" * 80)
    print("UUID 6031 VISUAL + GRAD-CAM AUDIT")
    print("=" * 80)

    metadata = pd.read_csv(
        METADATA_CSV
    )

    slices = [51, 64, 65, 71]

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(10, 13)
    )

    for i, slice_number in enumerate(slices):

        roi_path = find_roi(
            metadata,
            slice_number
        )

        baseline_path = find_gradcam(
            BASELINE_DIR,
            slice_number
        )

        augmented_path = find_gradcam(
            AUGMENTED_DIR,
            slice_number
        )

        # Original ROI
        if roi_path is not None:

            roi = np.load(
                roi_path
            ).astype(np.float32)

            axes[i, 0].imshow(
                roi,
                cmap="gray",
                vmin=0,
                vmax=1
            )

        else:

            axes[i, 0].text(
                0.5,
                0.5,
                "ROI not found",
                ha="center",
                va="center"
            )

        axes[i, 0].set_title(
            "Original ROI"
        )

        axes[i, 0].axis("off")

        # Baseline Grad-CAM
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
                "Grad-CAM not found",
                ha="center",
                va="center"
            )

        axes[i, 1].set_title(
            "Baseline Grad-CAM"
        )

        axes[i, 1].axis("off")

        # Augmented Grad-CAM
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
                "Grad-CAM not found",
                ha="center",
                va="center"
            )

        axes[i, 2].set_title(
            "Augmented Grad-CAM"
        )

        axes[i, 2].axis("off")

        # Slice label
        axes[i, 0].set_ylabel(
            f"UUID 6031\nSlice {slice_number}\n"
            "True: TAMPERED\n"
            "FN → FN",
            rotation=0,
            labelpad=55,
            va="center",
            fontsize=9
        )

    fig.suptitle(
        "UUID 6031 Persistent Failure — Visual and Grad-CAM Audit",
        fontsize=15
    )

    plt.subplots_adjust(
        left=0.20,
        right=0.98,
        top=0.94,
        bottom=0.03,
        hspace=0.35,
        wspace=0.08
    )

    OUTPUT_DIR.mkdir(
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
    print("Saved:")
    print(OUTPUT_PATH)

    print()
    print("=" * 80)
    print("VISUAL AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()