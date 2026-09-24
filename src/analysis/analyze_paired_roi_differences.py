import os
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "roi_dataset", "train"
)

TEST_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "roi_dataset", "test"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "analysis",
    "cnn_baseline",
    "paired_6031"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


PAIRS = [
    (51, 119, 315),
    (64, 240, 290),
    (65, 365, 395),
    (71, 413, 336),
]


print("=" * 70)
print("MEDICAL DEEPFAKE DETECTOR")
print("UUID 6031 — PAIRED EXP1 vs EXP2 DIFFERENCE ANALYSIS")
print("=" * 70)

summary_rows = []

for slice_id, x, y in PAIRS:

    exp1_name = f"EXP1_6031_{slice_id}_{x}_{y}.npy"
    exp2_name = f"EXP2_6031_{slice_id}_{x}_{y}.npy"

    exp1_path = os.path.join(TRAIN_DIR, exp1_name)
    exp2_path = os.path.join(TEST_DIR, exp2_name)

    exp1 = np.load(exp1_path).astype(np.float32)
    exp2 = np.load(exp2_path).astype(np.float32)

    diff = np.abs(exp1 - exp2)

    print()
    print("-" * 70)
    print(f"SLICE {slice_id}")
    print("-" * 70)

    print(f"Shape       : {exp1.shape}")
    print(f"EXP1 mean   : {exp1.mean():.6f}")
    print(f"EXP2 mean   : {exp2.mean():.6f}")
    print(f"EXP1 std    : {exp1.std():.6f}")
    print(f"EXP2 std    : {exp2.std():.6f}")
    print(f"Max diff    : {diff.max():.6f}")
    print(f"Mean diff   : {diff.mean():.6f}")

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]

    print()
    print("Meaningful difference thresholds:")

    for threshold in thresholds:
        count = int((diff > threshold).sum())
        percentage = 100.0 * count / diff.size

        print(
            f"  > {threshold:.3f}: "
            f"{count:5d} pixels "
            f"({percentage:6.2f}%)"
        )

    summary_rows.append(
        {
            "slice": slice_id,
            "mean_diff": float(diff.mean()),
            "max_diff": float(diff.max()),
            "pixels_gt_001": int((diff > 0.001).sum()),
            "pixels_gt_005": int((diff > 0.005).sum()),
            "pixels_gt_01": int((diff > 0.01).sum()),
            "pixels_gt_02": int((diff > 0.02).sum()),
            "pixels_gt_05": int((diff > 0.05).sum()),
        }
    )

    # ---------------------------------------------------------------
    # Save individual difference map
    # ---------------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"UUID6031_slice{slice_id}_difference.png"
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(exp1, cmap="gray", vmin=0, vmax=1)
    plt.title(f"EXP1 — Slice {slice_id}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(exp2, cmap="gray", vmin=0, vmax=1)
    plt.title(f"EXP2 — Slice {slice_id}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="hot", vmin=0, vmax=max(0.05, diff.max()))
    plt.title(f"|EXP1 − EXP2| — Slice {slice_id}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved       : {output_path}")


# -------------------------------------------------------------------
# Save numerical summary
# -------------------------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "UUID6031_paired_difference_summary.csv"
)

import csv

with open(summary_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary_rows)


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Output directory : {OUTPUT_DIR}")
print(f"Summary CSV      : {summary_path}")
print("STATUS           : PASS")
print("=" * 70)