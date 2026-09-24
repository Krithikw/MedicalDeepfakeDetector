import os
import numpy as np
import pandas as pd

print("=" * 70)
print("MEDICAL DEEPFAKE DETECTOR")
print("EXP1 VISUAL FEATURE DISTRIBUTION ANALYSIS")
print("=" * 70)

metadata_paths = [
    "outputs/roi_dataset/train_roi_metadata.csv",
    "outputs/roi_dataset/val_roi_metadata.csv"
]

records = []

for metadata_path in metadata_paths:

    if not os.path.exists(metadata_path):
        print("ERROR: Metadata file not found:", metadata_path)
        continue

    df = pd.read_csv(metadata_path)

    print("\nLoading:", metadata_path)
    print("Samples:", len(df))

    for _, row in df.iterrows():

        roi_path = row.get("roi_path")

        if pd.isna(roi_path) or not os.path.exists(roi_path):
            print("WARNING: ROI not found:", roi_path)
            continue

        img = np.load(roi_path).astype(np.float32)
        flat = img.ravel()

        mean_intensity = float(img.mean())
        std_intensity = float(img.std())

        near_black_fraction = float(np.mean(img < 0.05))
        dark_fraction = float(np.mean(img < 0.10))
        bright_fraction = float(np.mean(img > 0.75))
        very_bright_fraction = float(np.mean(img > 0.90))

        p10 = float(np.percentile(flat, 10))
        p25 = float(np.percentile(flat, 25))
        p50 = float(np.percentile(flat, 50))
        p75 = float(np.percentile(flat, 75))
        p90 = float(np.percentile(flat, 90))

        contrast = p90 - p10

        dx = np.abs(img[:, 1:] - img[:, :-1])
        dy = np.abs(img[1:, :] - img[:-1, :])

        edge_density_005 = float(np.mean(dx > 0.05))
        edge_density_010 = float(np.mean(dx > 0.10))

        edge_density_combined = float(
            (np.mean(dx > 0.05) + np.mean(dy > 0.05)) / 2
        )

        records.append({
            "experiment": "EXP1",
            "uuid": int(row["uuid"]),
            "slice": int(row["slice"]),
            "type": row["type"],
            "binary_label": int(row["binary_label"]),
            "split": os.path.basename(os.path.dirname(roi_path)),
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "near_black_fraction": near_black_fraction,
            "dark_fraction": dark_fraction,
            "bright_fraction": bright_fraction,
            "very_bright_fraction": very_bright_fraction,
            "p10": p10,
            "p25": p25,
            "median": p50,
            "p75": p75,
            "p90": p90,
            "contrast_p90_p10": contrast,
            "edge_density_005": edge_density_005,
            "edge_density_010": edge_density_010,
            "edge_density_combined": edge_density_combined
        })

out = pd.DataFrame(records)

out_path = "outputs/analysis/cnn_baseline/EXP1_visual_features.csv"
out.to_csv(out_path, index=False)

print("\n" + "-" * 70)
print("EXP1 DATASET SUMMARY")
print("-" * 70)

print("Total samples:", len(out))
print("\nBy split:")
print(out["split"].value_counts().to_string())

print("\nBy type:")
print(out["type"].value_counts().to_string())

print("\nBy label:")
print(out["binary_label"].value_counts().to_string())

features = [
    "mean_intensity",
    "std_intensity",
    "near_black_fraction",
    "dark_fraction",
    "bright_fraction",
    "very_bright_fraction",
    "contrast_p90_p10",
    "edge_density_005",
    "edge_density_010",
    "edge_density_combined"
]

print("\n" + "-" * 70)
print("EXP1 VISUAL FEATURE SUMMARY")
print("-" * 70)

print(
    out[features]
    .describe()
    .round(4)
    .to_string()
)

print("\n" + "-" * 70)
print("EXP1 VISUAL FEATURES BY TYPE")
print("-" * 70)

print(
    out.groupby("type")[features]
    .mean()
    .round(4)
    .to_string()
)

print("\n" + "=" * 70)
print("EXP1 VISUAL FEATURE EXTRACTION COMPLETE")
print("=" * 70)
print("Output:", os.path.abspath(out_path))
print("STATUS: PASS")
