import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

print("=" * 90)
print("MEDICAL DEEPFAKE DETECTOR")
print("EXP1 vs EXP2 STATISTICAL VISUAL FEATURE ANALYSIS")
print("=" * 90)

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

results = []

for t in ["FB", "FM", "TM"]:

    a = exp1[exp1["type"] == t]
    b = exp2[exp2["type"] == t]

    print("\n" + "-" * 90)
    print("TYPE:", t)
    print("EXP1 n =", len(a), "| EXP2 n =", len(b))
    print("-" * 90)

    for feature in features:

        x = a[feature].dropna().values
        y = b[feature].dropna().values

        if len(x) == 0 or len(y) == 0:
            continue

        u, p = mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )

        # Probability that a randomly selected EXP2 value
        # is greater than a randomly selected EXP1 value.
        effect = u / (len(x) * len(y))

        results.append({
            "type": t,
            "feature": feature,
            "EXP1_n": len(x),
            "EXP2_n": len(y),
            "EXP1_median": np.median(x),
            "EXP2_median": np.median(y),
            "U": u,
            "p_value": p,
            "effect_probability_EXP2_gt_EXP1": effect
        })

        print(
            f"{feature:25s} "
            f"EXP1 median={np.median(x):.4f} "
            f"EXP2 median={np.median(y):.4f} "
            f"p={p:.4f} "
            f"effect={effect:.4f}"
        )

out = pd.DataFrame(results)

out_path = r"outputs\analysis\cnn_baseline\EXP1_EXP2_statistical_comparison.csv"
out.to_csv(out_path, index=False)

print("\n" + "=" * 90)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 90)
print("Output:", out_path)
print("STATUS: PASS")
