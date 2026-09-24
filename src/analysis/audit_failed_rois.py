from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "cnn_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Change this ONLY if your prediction CSV has a different name.
PREDICTION_CSV = OUTPUT_DIR / "EXP2_CNN_PREDICTIONS.csv"

AUDIT_CSV = OUTPUT_DIR / "failed_roi_audit.csv"


# ============================================================
# POSSIBLE ROI LOCATIONS
# ============================================================

POSSIBLE_ROI_DIRS = [
    PROJECT_ROOT / "data" / "rois",
    PROJECT_ROOT / "Data" / "rois",
    PROJECT_ROOT / "data" / "ROI",
    PROJECT_ROOT / "Data" / "ROI",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "Data" / "processed",
]


# ============================================================
# FAILED UUIDs IDENTIFIED DURING BASELINE ANALYSIS
# ============================================================

FAILED_UUIDS = {
    "2031",
    "2220",
    "2590",
    "2592",
    "4709",
    "6031",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_roi_file(uuid, slice_id):
    """
    Search for the ROI corresponding to a UUID and slice.

    Supports several common filename conventions.
    """

    uuid = str(uuid)
    slice_id = str(slice_id)

    patterns = [
        f"*{uuid}*{slice_id}*.npy",
        f"*{uuid}*slice*{slice_id}*.npy",
        f"*{uuid}*_{slice_id}.npy",
        f"{uuid}_{slice_id}.npy",
        f"{uuid}-{slice_id}.npy",
    ]

    for roi_dir in POSSIBLE_ROI_DIRS:
        if not roi_dir.exists():
            continue

        for pattern in patterns:
            matches = list(roi_dir.rglob(pattern))

            if matches:
                return matches[0]

    return None


def find_roi_from_row(row):
    """
    Try to determine the ROI file using the information
    contained in the prediction row.
    """

    # --------------------------------------------------------
    # 1. If the prediction CSV already contains an ROI path
    # --------------------------------------------------------

    possible_path_columns = [
        "roi_path",
        "ROI_path",
        "path",
        "image_path",
        "filepath",
        "file_path",
        "filename",
        "roi_file",
    ]

    for column in possible_path_columns:
        if column in row.index:
            value = row[column]

            if pd.notna(value):
                value = str(value)

                candidate = Path(value)

                if candidate.exists():
                    return candidate

                candidate = PROJECT_ROOT / value

                if candidate.exists():
                    return candidate

    # --------------------------------------------------------
    # 2. Search using UUID + slice
    # --------------------------------------------------------

    uuid = row.get("UUID", row.get("uuid", row.get("Uuid", "")))

    slice_id = row.get(
        "slice",
        row.get(
            "Slice",
            row.get(
                "slice_idx",
                row.get(
                    "slice_index",
                    row.get("SliceIndex", "")
                )
            )
        )
    )

    if pd.isna(uuid) or pd.isna(slice_id):
        return None

    return find_roi_file(uuid, slice_id)


def calculate_roi_statistics(roi):
    """
    Calculate intensity statistics for one ROI.
    """

    roi = np.asarray(roi)

    # Remove NaN/Inf values before calculating statistics.
    finite_values = roi[np.isfinite(roi)]

    if finite_values.size == 0:
        return {
            "roi_shape": str(roi.shape),
            "roi_dtype": str(roi.dtype),
            "roi_min": np.nan,
            "roi_max": np.nan,
            "roi_mean": np.nan,
            "roi_std": np.nan,
            "zero_fraction": np.nan,
            "near_zero_fraction": np.nan,
            "nonzero_fraction": np.nan,
            "dark_fraction": np.nan,
            "bright_fraction": np.nan,
            "very_bright_fraction": np.nan,
        }

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    roi_min = float(np.min(finite_values))
    roi_max = float(np.max(finite_values))
    roi_mean = float(np.mean(finite_values))
    roi_std = float(np.std(finite_values))

    # --------------------------------------------------------
    # Fractions
    # --------------------------------------------------------

    zero_fraction = float(
        np.mean(finite_values == 0)
    )

    near_zero_fraction = float(
        np.mean(np.abs(finite_values) <= 0.01)
    )

    nonzero_fraction = float(
        np.mean(finite_values != 0)
    )

    # --------------------------------------------------------
    # Percentile-based intensity fractions
    #
    # These are useful even when ROIs are not normalized.
    # --------------------------------------------------------

    p10 = np.percentile(finite_values, 10)
    p90 = np.percentile(finite_values, 90)
    p95 = np.percentile(finite_values, 95)

    dark_fraction = float(
        np.mean(finite_values <= p10)
    )

    bright_fraction = float(
        np.mean(finite_values >= p90)
    )

    very_bright_fraction = float(
        np.mean(finite_values >= p95)
    )

    return {
        "roi_shape": str(roi.shape),
        "roi_dtype": str(roi.dtype),
        "roi_min": roi_min,
        "roi_max": roi_max,
        "roi_mean": roi_mean,
        "roi_std": roi_std,
        "zero_fraction": zero_fraction,
        "near_zero_fraction": near_zero_fraction,
        "nonzero_fraction": nonzero_fraction,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "very_bright_fraction": very_bright_fraction,
    }


def load_roi(roi_path):
    """
    Load an ROI from .npy.
    """

    roi = np.load(roi_path, allow_pickle=True)

    # Handle object arrays containing another array.
    if isinstance(roi, np.ndarray) and roi.dtype == object:
        try:
            roi = roi.item()
        except Exception:
            pass

    # Handle dictionaries saved inside .npy.
    if isinstance(roi, dict):

        possible_keys = [
            "roi",
            "image",
            "array",
            "data",
            "pixels",
        ]

        for key in possible_keys:
            if key in roi:
                roi = roi[key]
                break

    return np.asarray(roi)


def normalise_column_names(df):
    """
    Make column lookup easier without changing the original data.
    """

    mapping = {}

    for column in df.columns:
        clean = str(column).strip().lower()

        if clean == "uuid":
            mapping[column] = "UUID"

        elif clean in {"slice", "slice_idx", "slice_index"}:
            mapping[column] = "slice"

        elif clean in {
        "true_label",
        "binary_label",
        "label",
        "target",
        "y_true",
}:
            mapping[column] = "true_label"

        elif clean in {
            "predicted_label",
            "prediction",
            "pred",
            "y_pred",
        }:
            mapping[column] = "predicted_label"

        elif clean in {
            "p(tampered)",
            "p_tampered",
            "prob_tampered",
            "tampered_probability",
            "probability",
        }:
            mapping[column] = "p_tampered"

    return df.rename(columns=mapping)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("FAILED ROI AUDIT")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Prediction CSV:")
    print(PREDICTION_CSV)

    if not PREDICTION_CSV.exists():

        print()
        print("ERROR: Prediction CSV was not found.")
        print()
        print("Expected:")
        print(PREDICTION_CSV)
        print()
        print("If your prediction CSV has a different name,")
        print("change PREDICTION_CSV near the top of this script.")
        print()

        sys.exit(1)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    df = pd.read_csv(PREDICTION_CSV)

    df = normalise_column_names(df)

    print()
    print("Prediction columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required = [
        "UUID",
        "slice",
        "true_label",
        "predicted_label",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print()
        print("ERROR: Missing required columns:")
        print(missing)
        print()
        print("Available columns:")
        print(list(df.columns))
        print()

        sys.exit(1)

    # --------------------------------------------------------
    # Keep failed UUIDs
    # --------------------------------------------------------

    df["UUID"] = df["UUID"].astype(str)

    failed_df = df[
        df["UUID"].isin(FAILED_UUIDS)
    ].copy()

    # Also include all incorrectly classified samples,
    # so the audit is not limited to the six known UUIDs.
    failed_predictions = df[
        df["true_label"] != df["predicted_label"]
    ].copy()

    audit_source = pd.concat(
        [failed_df, failed_predictions],
        ignore_index=True
    )

    audit_source = audit_source.drop_duplicates()

    print()
    print("Total EXP2 samples:", len(df))
    print("Known failed UUID samples:", len(failed_df))
    print("Total incorrect predictions:", len(failed_predictions))

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    results = []

    for _, row in audit_source.iterrows():

        uuid = str(row["UUID"])
        slice_id = row["slice"]

        print()
        print("-" * 70)
        print(f"UUID: {uuid}")
        print(f"Slice: {slice_id}")

        true_label = row["true_label"]
        predicted_label = row["predicted_label"]

        p_tampered = row.get(
            "p_tampered",
            np.nan
        )

        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label}")

        if pd.notna(p_tampered):
            print(
                f"P(tampered): {float(p_tampered):.6f}"
            )
        else:
            print("P(tampered): N/A")

        # ----------------------------------------------------
        # Find ROI
        # ----------------------------------------------------

        roi_path = find_roi_from_row(row)

        if roi_path is None:

            print("ROI file: NOT FOUND")

            result = row.to_dict()

            result["roi_path"] = ""
            result.update({
                "roi_shape": "",
                "roi_dtype": "",
                "roi_min": np.nan,
                "roi_max": np.nan,
                "roi_mean": np.nan,
                "roi_std": np.nan,
                "zero_fraction": np.nan,
                "near_zero_fraction": np.nan,
                "nonzero_fraction": np.nan,
                "dark_fraction": np.nan,
                "bright_fraction": np.nan,
                "very_bright_fraction": np.nan,
            })

            results.append(result)

            continue

        print("ROI file:", roi_path)

        try:

            roi = load_roi(roi_path)

            stats = calculate_roi_statistics(roi)

            print("ROI shape:", stats["roi_shape"])
            print("ROI dtype:", stats["roi_dtype"])
            print(
                f"Min: {stats['roi_min']:.6f}"
            )
            print(
                f"Max: {stats['roi_max']:.6f}"
            )
            print(
                f"Mean: {stats['roi_mean']:.6f}"
            )
            print(
                f"Std: {stats['roi_std']:.6f}"
            )
            print(
                f"Zero fraction: "
                f"{stats['zero_fraction']:.6f}"
            )
            print(
                f"Near-zero fraction: "
                f"{stats['near_zero_fraction']:.6f}"
            )
            print(
                f"Non-zero fraction: "
                f"{stats['nonzero_fraction']:.6f}"
            )

            result = row.to_dict()

            result["roi_path"] = str(roi_path)

            result.update(stats)

            results.append(result)

        except Exception as exc:

            print("ERROR loading ROI:")
            print(exc)

            result = row.to_dict()

            result["roi_path"] = str(roi_path)
            result["roi_error"] = str(exc)

            results.append(result)

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    if not results:

        print()
        print("No audit results were generated.")
        print()
        sys.exit(1)

    audit_df = pd.DataFrame(results)

    # Put important columns first.
    preferred_columns = [
        "UUID",
        "slice",
        "true_label",
        "predicted_label",
        "p_tampered",
        "roi_path",
        "roi_shape",
        "roi_dtype",
        "roi_min",
        "roi_max",
        "roi_mean",
        "roi_std",
        "zero_fraction",
        "near_zero_fraction",
        "nonzero_fraction",
        "dark_fraction",
        "bright_fraction",
        "very_bright_fraction",
    ]

    existing_preferred = [
        column
        for column in preferred_columns
        if column in audit_df.columns
    ]

    remaining = [
        column
        for column in audit_df.columns
        if column not in existing_preferred
    ]

    audit_df = audit_df[
        existing_preferred + remaining
    ]

    audit_df.to_csv(
        AUDIT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print()
    print("Saved:")
    print(AUDIT_CSV)

    print()
    print("Rows audited:", len(audit_df))

    print()
    print("Audited UUIDs:")

    for uuid in sorted(
        audit_df["UUID"].astype(str).unique()
    ):
        count = int(
            (audit_df["UUID"].astype(str) == uuid).sum()
        )

        print(
            f"  {uuid}: {count} sample(s)"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()