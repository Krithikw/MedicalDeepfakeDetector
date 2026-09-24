import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "uuid_6031"
)

BASELINE_DIR = INPUT_DIR / "baseline_gradcam"
AUGMENTED_DIR = INPUT_DIR / "augmented_gradcam"

OUTPUT_PATH = (
    INPUT_DIR
    / "uuid_6031_gradcam_comparison.png"
)


SLICES = [51, 64, 65, 71]


def extract_panels(image_path):

    image = np.array(Image.open(image_path))

    height, width = image.shape[:2]

    # The generated Grad-CAM image contains 3 horizontal panels.
    panel_width = width // 3

    original = image[:, :panel_width]
    gradcam_overlay = image[
        :,
        panel_width:2 * panel_width
    ]

    return original, gradcam_overlay


def main():

    print("=" * 80)
    print("UUID 6031 GRAD-CAM COMPARISON")
    print("=" * 80)

    fig, axes = plt.subplots(
        len(SLICES),
        3,
        figsize=(12, 15)
    )

    for row_index, slice_number in enumerate(SLICES):

        baseline_path = (
            BASELINE_DIR
            / f"UUID6031_SLICE{slice_number}_baseline.png"
        )

        augmented_path = (
            AUGMENTED_DIR
            / f"UUID6031_SLICE{slice_number}_augmented.png"
        )

        print()
        print(f"Processing slice {slice_number}")

        baseline_original, baseline_cam = extract_panels(
            baseline_path
        )

        augmented_original, augmented_cam = extract_panels(
            augmented_path
        )

        # Original ROI
        axes[row_index, 0].imshow(
            baseline_original
        )

        axes[row_index, 0].set_title(
            f"Slice {slice_number}\nOriginal ROI"
        )

        axes[row_index, 0].axis("off")

        # Baseline Grad-CAM
        axes[row_index, 1].imshow(
            baseline_cam
        )

        axes[row_index, 1].set_title(
            "Baseline CNN\nGrad-CAM"
        )

        axes[row_index, 1].axis("off")

        # Augmented Grad-CAM
        axes[row_index, 2].imshow(
            augmented_cam
        )

        axes[row_index, 2].set_title(
            "Augmented CNN\nGrad-CAM"
        )

        axes[row_index, 2].axis("off")

    fig.suptitle(
        "UUID 6031 Persistent Failure — Grad-CAM Comparison",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
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
    print("COMPARISON COMPLETE")
    print("=" * 80)

    print()
    print("Saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()