import pandas as pd
import numpy as np
from math import erf, sqrt

print("=" * 90)
print("MEDICAL DEEPFAKE DETECTOR")
print("EXP1 vs EXP2 STATISTICAL VISUAL FEATURE ANALYSIS")
print("PURE PYTHON IMPLEMENTATION — NO SCIPY")
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


def normal_cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def mann_whitney_u(x, y):
    """
    Mann–Whitney U using average ranks.

    Returns:
        U1
        two-sided normal-approximation p-value
        rank-biserial style probability P(X > Y) with ties handled
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n1 = len(x)
    n2 = len(y)

    combined = np.concatenate([x, y])

    order = np.argsort(combined, kind="mergesort")
    sorted_values = combined[order]

    ranks = np.zeros(len(combined), dtype=float)

    i = 0

    while i < len(sorted_values):

        j = i + 1

        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1

        average_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = average_rank

        i = j

    rank_sum_x = np.sum(ranks[:n1])

    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0

    mean_u = n1 * n2 / 2.0

    # Tie correction
    _, counts = np.unique(combined, return_counts=True)

    tie_sum = np.sum(counts**3 - counts)

    if len(combined) > 1:
        variance_u = (
            n1 * n2 / 12.0
        ) * (
            (len(combined) + 1)
            - tie_sum / (len(combined) * (len(combined) - 1))
        )
    else:
        variance_u = 0.0

    if variance_u > 0:
        z = (u1 - mean_u) / sqrt(variance_u)

        # Two-sided normal approximation
        p_value = 2.0 * (1.0 - normal_cdf(abs(z)))

    else:
        p_value = 1.0

    probability = u1 / (n1 * n2)

    return u1, p_value, probability


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

        u, p, effect = mann_whitney_u(x, y)

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
