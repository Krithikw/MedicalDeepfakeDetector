from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
    / "failed_roi_audit.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
)

OUTPUT_CSV = OUTPUT_DIR / "failed_roi_pattern_summary.csv"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(AUDIT_CSV)

print("=" * 75)
print("FAILED ROI PATTERN SUMMARY")
print("=" * 75)

print(f"Total failed samples: {len(df)}")


# ============================================================
# ERROR TYPE
# ============================================================

df["error_type"] = "OTHER"

df.loc[
    (df["true_label"] == 1) & (df["predicted_label"] == 0),
    "error_type"
] = "FN"

df.loc[
    (df["true_label"] == 0) & (df["predicted_label"] == 1),
    "error_type"
] = "FP"


print("\nError counts:")
print(df["error_type"].value_counts())


# ============================================================
# SUMMARY BY ERROR TYPE
# ============================================================

summary = (
    df.groupby("error_type")
    .agg(
        samples=("error_type", "size"),
        mean_roi_intensity=("roi_mean", "mean"),
        mean_roi_std=("roi_std", "mean"),
        mean_zero_fraction=("zero_fraction", "mean"),
        mean_near_zero_fraction=("near_zero_fraction", "mean"),
        mean_nonzero_fraction=("nonzero_fraction", "mean"),
        mean_dark_fraction=("dark_fraction", "mean"),
        mean_bright_fraction=("bright_fraction", "mean"),
        mean_very_bright_fraction=("very_bright_fraction", "mean"),
        mean_p_tampered=("p_tampered", "mean"),
        min_p_tampered=("p_tampered", "min"),
        max_p_tampered=("p_tampered", "max"),
    )
    .reset_index()
)


print("\n" + "=" * 75)
print("SUMMARY BY ERROR TYPE")
print("=" * 75)

print(summary.to_string(index=False))


# ============================================================
# UUID SUMMARY
# ============================================================

uuid_summary = (
    df.groupby(["UUID", "error_type"])
    .agg(
        samples=("UUID", "size"),
        mean_roi_intensity=("roi_mean", "mean"),
        mean_roi_std=("roi_std", "mean"),
        mean_zero_fraction=("zero_fraction", "mean"),
        mean_dark_fraction=("dark_fraction", "mean"),
        mean_p_tampered=("p_tampered", "mean"),
    )
    .reset_index()
    .sort_values(["error_type", "UUID"])
)


print("\n" + "=" * 75)
print("FAILURE SUMMARY BY UUID")
print("=" * 75)

print(uuid_summary.to_string(index=False))


# ============================================================
# FALSE NEGATIVES
# ============================================================

fn = df[df["error_type"] == "FN"][
    [
        "UUID",
        "slice",
        "p_tampered",
        "roi_mean",
        "roi_std",
        "zero_fraction",
        "dark_fraction",
        "bright_fraction",
        "nonzero_fraction",
    ]
].sort_values("p_tampered")


print("\n" + "=" * 75)
print("FALSE NEGATIVES (FN)")
print("=" * 75)

print(fn.to_string(index=False))


# ============================================================
# FALSE POSITIVES
# ============================================================

fp = df[df["error_type"] == "FP"][
    [
        "UUID",
        "slice",
        "p_tampered",
        "roi_mean",
        "roi_std",
        "zero_fraction",
        "dark_fraction",
        "bright_fraction",
        "nonzero_fraction",
    ]
].sort_values("p_tampered", ascending=False)


print("\n" + "=" * 75)
print("FALSE POSITIVES (FP)")
print("=" * 75)

print(fp.to_string(index=False))


# ============================================================
# SAVE SUMMARY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 75)
print("SUMMARY SAVED")
print("=" * 75)

print(OUTPUT_CSV)