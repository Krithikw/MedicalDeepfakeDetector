import os
import numpy as np
import pandas as pd

print("=" * 70)
print("MEDICAL DEEPFAKE DETECTOR")
print("CNN BASELINE - VISUAL SHORTCUT ANALYSIS")
print("=" * 70)

pred_path = "outputs/analysis/cnn_baseline/EXP2_CNN_PREDICTIONS.csv"
roi_dir = "outputs/roi_dataset/test"

df = pd.read_csv(pred_path)

print("\nPrediction columns:")
print(list(df.columns))

print("\nSamples:", len(df))

records = []

for _, row in df.iterrows():

    roi_path = row.get("roi_path")

    if pd.isna(roi_path) or not os.path.exists(roi_path):
        filename = (
            f"EXP2_{int(row['uuid'])}_{int(row['slice'])}_"
            f"{int(row['x'])}_{int(row['y'])}.npy"
        )
        roi_path = os.path.join(roi_dir, filename)

    if not os.path.exists(roi_path):
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

    # Simple edge-density estimate using neighboring pixel differences.
    dx = np.abs(img[:, 1:] - img[:, :-1])
    dy = np.abs(img[1:, :] - img[:-1, :])

    edge_density_01 = float(np.mean(dx > 0.05))
    edge_density_02 = float(np.mean(dx > 0.10))
    edge_density_combined = float(
        (np.mean(dx > 0.05) + np.mean(dy > 0.05)) / 2
    )

    records.append({
        "uuid": int(row["uuid"]),
        "slice": int(row["slice"]),
        "type": row["type"],
        "true_label": int(row["binary_label"]),
        "predicted_label": int(row["predicted_label"]),
        "p_tampered": float(row["tampered_probability"]),
        "correct": int(row["binary_label"]) == int(row["predicted_label"]),
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
        "edge_density_005": edge_density_01,
        "edge_density_010": edge_density_02,
        "edge_density_combined": edge_density_combined
    })

out = pd.DataFrame(records)

out_path = "outputs/analysis/cnn_baseline/EXP2_visual_shortcut_features.csv"
out.to_csv(out_path, index=False)

print("\n" + "-" * 70)
print("OVERALL VISUAL FEATURE SUMMARY")
print("-" * 70)

features = [
    "mean_intensity",
    "std_intensity",
    "near_black_fraction",
    "dark_fraction",
    "bright_fraction",
    "very_bright_fraction",
    "contrast_p90_p10",
    "edge_density_combined"
]

print(
    out[features]
    .describe()
    .round(4)
    .to_string()
)

print("\n" + "-" * 70)
print("CORRECT vs INCORRECT")
print("-" * 70)

print(
    out.groupby("correct")[features]
    .mean()
    .round(4)
    .to_string()
)

print("\n" + "-" * 70)
print("FALSE NEGATIVES")
print("-" * 70)

fn = out[(out.true_label == 1) & (out.predicted_label == 0)]

if len(fn):
    print(
        fn[
            ["uuid", "slice", "type", "p_tampered"] + features
        ].round(4).to_string(index=False)
    )
else:
    print("None")

print("\n" + "-" * 70)
print("FALSE POSITIVES")
print("-" * 70)

fp = out[(out.true_label == 0) & (out.predicted_label == 1)]

if len(fp):
    print(
        fp[
            ["uuid", "slice", "type", "p_tampered"] + features
        ].round(4).to_string(index=False)
    )
else:
    print("None")

print("\n" + "-" * 70)
print("PER-TYPE VISUAL FEATURES")
print("-" * 70)

print(
    out.groupby("type")[features]
    .mean()
    .round(4)
    .to_string()
)

print("\n" + "-" * 70)
print("CORRELATION WITH P(TAMPERED)")
print("-" * 70)

corr = out[features + ["p_tampered"]].corr()["p_tampered"].drop("p_tampered")
print(corr.sort_values(ascending=False).round(4).to_string())

print("\n" + "=" * 70)
print("VISUAL SHORTCUT ANALYSIS COMPLETE")
print("=" * 70)
print("Output:", os.path.abspath(out_path))
print("STATUS: PASS")
