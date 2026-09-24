from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
    / "EXP2_CNN_PREDICTIONS.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
)

OUTPUT_PATH = OUTPUT_DIR / "failed_roi_visual_audit.png"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(PREDICTION_CSV)

# Keep only incorrect predictions
failed = df[df["correct"] == False].copy()

print("=" * 70)
print("FAILED ROI VISUAL AUDIT")
print("=" * 70)
print(f"Total EXP2 samples      : {len(df)}")
print(f"Incorrect predictions   : {len(failed)}")


# ============================================================
# PREPARE FIGURE
# ============================================================

n = len(failed)

cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, rows * 4)
)

axes = np.array(axes).reshape(-1)


# ============================================================
# DISPLAY EACH FAILED ROI
# ============================================================

for i, (_, row) in enumerate(failed.iterrows()):

    ax = axes[i]

    roi_path = Path(row["roi_path"])

    if not roi_path.is_absolute():
        roi_path = PROJECT_ROOT / roi_path

    try:
        roi = np.load(roi_path)

        # Handle possible singleton dimensions
        roi = np.squeeze(roi)

        ax.imshow(
            roi,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        true_label = int(row["binary_label"])
        pred_label = int(row["predicted_label"])
        probability = float(row["p_tampered"])

        uuid = row["UUID"]
        slice_num = row["slice"]

        true_text = "Authentic" if true_label == 0 else "Tampered"
        pred_text = "Authentic" if pred_label == 0 else "Tampered"

        ax.set_title(
            f"UUID {uuid} | Slice {slice_num}\n"
            f"True: {true_text} | Pred: {pred_text}\n"
            f"P(Tampered): {probability:.4f}",
            fontsize=9
        )

        ax.axis("off")

    except Exception as e:

        ax.text(
            0.5,
            0.5,
            f"ERROR\n{e}",
            ha="center",
            va="center"
        )

        ax.axis("off")


# ============================================================
# HIDE UNUSED AXES
# ============================================================

for j in range(n, len(axes)):
    axes[j].axis("off")


# ============================================================
# SAVE
# ============================================================

plt.suptitle(
    "EXP2 CNN — Misclassified ROI Visual Audit",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print(f"Saved visual audit:")
print(OUTPUT_PATH)
print("=" * 70)