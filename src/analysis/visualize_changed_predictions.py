from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPARISON_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_comparison"
    / "baseline_vs_augmented.csv"
)

TEST_METADATA_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "roi_dataset"
    / "test_roi_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_comparison"
)

OUTPUT_FIGURE = OUTPUT_DIR / "changed_predictions_visual_audit.png"


# ============================================================
# HELPERS
# ============================================================

def find_roi_path(metadata, uuid, slice_number):
    """
    Find the ROI path using UUID + slice from the original
    test metadata.
    """

    matches = metadata[
        (metadata["uuid"].astype(str) == str(uuid))
        & (metadata["slice"].astype(int) == int(slice_number))
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No ROI found for UUID={uuid}, slice={slice_number}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: Multiple ROIs found for "
            f"UUID={uuid}, slice={slice_number}. Using first."
        )

    return Path(matches.iloc[0]["roi_path"])


def load_roi(path):
    roi = np.load(path).astype(np.float32)

    if roi.shape != (128, 128):
        raise ValueError(
            f"Unexpected ROI shape {roi.shape} for {path}"
        )

    return roi


def short_transition(row):
    return (
        f"{row['baseline_predicted_label']} → "
        f"{row['augmented_predicted_label']}"
    )


def label_name(value):
    return "Tampered" if int(value) == 1 else "Authentic"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CHANGED PREDICTIONS VISUAL AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load comparison
    # --------------------------------------------------------

    comparison = pd.read_csv(COMPARISON_CSV)

    changed = comparison[
        comparison["baseline_predicted_label"]
        != comparison["augmented_predicted_label"]
    ].copy()

    print()
    print(f"Changed predictions: {len(changed)}")

    # --------------------------------------------------------
    # Load original test metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(TEST_METADATA_CSV)

    print(f"Test metadata rows: {len(metadata)}")

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    n = len(changed)

    if n == 0:
        print("No changed predictions found.")
        return

    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(20, 8),
    )

    axes = np.array(axes).reshape(-1)

    # --------------------------------------------------------
    # Plot each changed prediction
    # --------------------------------------------------------

    for i, (_, row) in enumerate(changed.iterrows()):

        ax = axes[i]

        uuid = row["UUID"]
        slice_number = int(row["slice"])

        roi_path = find_roi_path(
            metadata,
            uuid,
            slice_number,
        )

        roi = load_roi(roi_path)

        ax.imshow(
            roi,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        true_label = label_name(row["true_label"])

        baseline_label = label_name(
            row["baseline_predicted_label"]
        )

        augmented_label = label_name(
            row["augmented_predicted_label"]
        )

        baseline_p = float(
            row["baseline_p_tampered"]
        )

        augmented_p = float(
            row["augmented_p_tampered"]
        )

        transition = row["transition"]

        ax.set_title(
            f"UUID {uuid} | Slice {slice_number}\n"
            f"True: {true_label}\n"
            f"Baseline: {baseline_label} "
            f"(P={baseline_p:.3f})\n"
            f"Augmented: {augmented_label} "
            f"(P={augmented_p:.3f})\n"
            f"{transition}",
            fontsize=9,
        )

        ax.axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Baseline vs Augmented CNN — Changed Predictions Visual Audit",
        fontsize=16,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 70)
    print("VISUAL AUDIT COMPLETE")
    print("=" * 70)
    print(f"Saved: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()