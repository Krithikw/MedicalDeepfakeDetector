import pandas as pd
import numpy as np

print("=" * 80)
print("MEDICAL DEEPFAKE DETECTOR")
print("EXP1 vs EXP2 VISUAL DISTRIBUTION COMPARISON")
print("=" * 80)

exp1_path = r"outputs\analysis\cnn_baseline\EXP1_visual_features.csv"
exp2_path = r"outputs\analysis\cnn_baseline\EXP2_visual_shortcut_features.csv"

exp1 = pd.read_csv(exp1_path)
exp2 = pd.read_csv(exp2_path)

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

print("\nDATASET SIZES")
print("-" * 80)
print("EXP1:", len(exp1))
print("EXP2:", len(exp2))

print("\nTYPE DISTRIBUTION")
print("-" * 80)

type_table = pd.DataFrame({
    "EXP1": exp1["type"].value_counts(),
    "EXP2": exp2["type"].value_counts()
}).fillna(0).astype(int)

print(type_table.to_string())

print("\nLABEL DISTRIBUTION")
print("-" * 80)

label_table = pd.DataFrame({
    "EXP1": exp1["binary_label"].value_counts(),
    "EXP2": exp2["true_label"].value_counts()
}).fillna(0).astype(int)

print(label_table.to_string())

print("\nOVERALL VISUAL FEATURE COMPARISON")
print("-" * 80)

summary = pd.DataFrame({
    "EXP1_mean": exp1[features].mean(),
    "EXP1_median": exp1[features].median(),
    "EXP2_mean": exp2[features].mean(),
    "EXP2_median": exp2[features].median()
})

summary["mean_difference_EXP2_minus_EXP1"] = (
    summary["EXP2_mean"] - summary["EXP1_mean"]
)

summary["percent_difference"] = (
    100 * summary["mean_difference_EXP2_minus_EXP1"]
    / summary["EXP1_mean"].replace(0, np.nan)
)

print(summary.round(4).to_string())

print("\nTYPE-SPECIFIC COMPARISON")
print("-" * 80)

for t in ["FB", "FM", "TB", "TM"]:

    a = exp1[exp1["type"] == t]
    b = exp2[exp2["type"] == t]

    print("\nTYPE:", t)
    print("EXP1 samples:", len(a))
    print("EXP2 samples:", len(b))

    if len(a) == 0 or len(b) == 0:
        print("Not available in both experiments.")
        continue

    comparison = pd.DataFrame({
        "EXP1_mean": a[features].mean(),
        "EXP2_mean": b[features].mean()
    })

    comparison["difference"] = (
        comparison["EXP2_mean"] - comparison["EXP1_mean"]
    )

    print(comparison.round(4).to_string())

print("\n" + "=" * 80)
print("EXP1 vs EXP2 DISTRIBUTION COMPARISON COMPLETE")
print("=" * 80)
