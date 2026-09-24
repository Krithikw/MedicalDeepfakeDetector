from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_baseline"
    / "EXP2_CNN_PREDICTIONS.csv"
)

AUGMENTED_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_augmented"
    / "EXP2_CNN_AUGMENTED_PREDICTIONS.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "cnn_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "baseline_vs_augmented.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BASELINE vs AUGMENTED EXP2 COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"Baseline prediction CSV not found:\n{BASELINE_CSV}"
        )

    if not AUGMENTED_CSV.exists():
        raise FileNotFoundError(
            f"Augmented prediction CSV not found:\n{AUGMENTED_CSV}"
        )

    # --------------------------------------------------------
    # Load prediction files
    # --------------------------------------------------------

    baseline = pd.read_csv(
        BASELINE_CSV
    )

    augmented = pd.read_csv(
        AUGMENTED_CSV
    )

    print()
    print("Baseline samples:", len(baseline))
    print("Augmented samples:", len(augmented))

    # --------------------------------------------------------
    # Normalize important column names
    # --------------------------------------------------------

    baseline = baseline.rename(
        columns={
            "uuid": "UUID",
            "binary_label": "true_label",
            "tampered_probability": "p_tampered",
        }
    )

    augmented = augmented.rename(
        columns={
            "uuid": "UUID",
        }
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required = [
        "UUID",
        "slice",
        "true_label",
        "predicted_label",
        "p_tampered",
        "correct",
        "prediction_category",
    ]

    for name, df in [
        ("BASELINE", baseline),
        ("AUGMENTED", augmented),
    ]:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"{name} prediction CSV is missing: {missing}"
            )

    # --------------------------------------------------------
    # Stable sample identity
    #
    # UUID + slice + ROI path ensures that the exact same
    # EXP2 sample is compared.
    # --------------------------------------------------------

    def create_key(df):

        return (
            df["UUID"].astype(str).str.strip()
            + "|"
            + df["slice"].astype(str).str.strip()
            + "|"
            + df["roi_path"].astype(str).str.strip()
        )

    baseline["comparison_key"] = create_key(
        baseline
    )

    augmented["comparison_key"] = create_key(
        augmented
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    baseline_duplicates = (
        baseline["comparison_key"]
        .duplicated()
        .sum()
    )

    augmented_duplicates = (
        augmented["comparison_key"]
        .duplicated()
        .sum()
    )

    print()
    print(
        "Baseline duplicate keys:",
        baseline_duplicates
    )

    print(
        "Augmented duplicate keys:",
        augmented_duplicates
    )

    if baseline_duplicates > 0:
        raise RuntimeError(
            "Duplicate sample keys detected in baseline CSV."
        )

    if augmented_duplicates > 0:
        raise RuntimeError(
            "Duplicate sample keys detected in augmented CSV."
        )

    # --------------------------------------------------------
    # Check that both experiments contain the same samples
    # --------------------------------------------------------

    baseline_keys = set(
        baseline["comparison_key"]
    )

    augmented_keys = set(
        augmented["comparison_key"]
    )

    only_baseline = baseline_keys - augmented_keys
    only_augmented = augmented_keys - baseline_keys

    print()
    print(
        "Samples only in baseline:",
        len(only_baseline)
    )

    print(
        "Samples only in augmented:",
        len(only_augmented)
    )

    if only_baseline or only_augmented:

        print()
        print("WARNING: The two prediction files do not contain")
        print("exactly the same sample keys.")

    # --------------------------------------------------------
    # Select baseline columns
    # --------------------------------------------------------

    baseline_compare = baseline[
        [
            "comparison_key",
            "UUID",
            "slice",
            "true_label",
            "predicted_label",
            "p_tampered",
            "confidence",
            "correct",
            "prediction_category",
        ]
    ].copy()

    baseline_compare = baseline_compare.rename(
        columns={
            "predicted_label":
                "baseline_predicted_label",

            "p_tampered":
                "baseline_p_tampered",

            "confidence":
                "baseline_confidence",

            "correct":
                "baseline_correct",

            "prediction_category":
                "baseline_category",
        }
    )

    # --------------------------------------------------------
    # Select augmented columns
    # --------------------------------------------------------

    augmented_compare = augmented[
        [
            "comparison_key",
            "predicted_label",
            "p_tampered",
            "confidence",
            "correct",
            "prediction_category",
        ]
    ].copy()

    augmented_compare = augmented_compare.rename(
        columns={
            "predicted_label":
                "augmented_predicted_label",

            "p_tampered":
                "augmented_p_tampered",

            "confidence":
                "augmented_confidence",

            "correct":
                "augmented_correct",

            "prediction_category":
                "augmented_category",
        }
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    comparison = baseline_compare.merge(
        augmented_compare,
        on="comparison_key",
        how="inner",
        validate="one_to_one"
    )

    print()
    print(
        "Samples compared:",
        len(comparison)
    )

    # --------------------------------------------------------
    # Transition classification
    # --------------------------------------------------------

    def classify_transition(row):

        baseline_correct = (
            int(row["baseline_correct"]) == 1
        )

        augmented_correct = (
            int(row["augmented_correct"]) == 1
        )

        if (
            not baseline_correct
            and augmented_correct
        ):
            return "WRONG_TO_CORRECT"

        if (
            baseline_correct
            and not augmented_correct
        ):
            return "CORRECT_TO_WRONG"

        if (
            baseline_correct
            and augmented_correct
        ):
            return "CORRECT_TO_CORRECT"

        return "WRONG_TO_WRONG"

    comparison["transition"] = comparison.apply(
        classify_transition,
        axis=1
    )

    # --------------------------------------------------------
    # Save complete comparison
    # --------------------------------------------------------

    comparison.to_csv(
        COMPARISON_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Overall transition summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL TRANSITION SUMMARY")
    print("=" * 70)

    transition_counts = (
        comparison["transition"]
        .value_counts()
    )

    for transition in [
        "WRONG_TO_CORRECT",
        "CORRECT_TO_WRONG",
        "CORRECT_TO_CORRECT",
        "WRONG_TO_WRONG",
    ]:

        print(
            f"{transition:20s}: "
            f"{int(transition_counts.get(transition, 0))}"
        )

    # --------------------------------------------------------
    # Changed predictions
    # --------------------------------------------------------

    changed = comparison[
        comparison["baseline_predicted_label"]
        != comparison["augmented_predicted_label"]
    ].copy()

    print()
    print("=" * 70)
    print("CHANGED PREDICTIONS")
    print("=" * 70)

    print()
    print(
        "Number of changed predictions:",
        len(changed)
    )

    if len(changed) > 0:

        display_columns = [
            "UUID",
            "slice",
            "true_label",
            "baseline_predicted_label",
            "augmented_predicted_label",
            "baseline_p_tampered",
            "augmented_p_tampered",
            "baseline_category",
            "augmented_category",
            "transition",
        ]

        print()

        print(
            changed[
                display_columns
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # UUID 6031
    # --------------------------------------------------------

    uuid_6031 = comparison[
        comparison["UUID"].astype(str) == "6031"
    ].copy()

    print()
    print("=" * 70)
    print("UUID 6031 ANALYSIS")
    print("=" * 70)

    if uuid_6031.empty:

        print()
        print("UUID 6031 was not found in EXP2.")

    else:

        display_columns = [
            "UUID",
            "slice",
            "true_label",
            "baseline_predicted_label",
            "augmented_predicted_label",
            "baseline_p_tampered",
            "augmented_p_tampered",
            "baseline_category",
            "augmented_category",
            "transition",
        ]

        print()

        print(
            uuid_6031[
                display_columns
            ].to_string(index=False)
        )

        print()

        print(
            "UUID 6031 baseline correct:",
            int(
                uuid_6031["baseline_correct"].sum()
            ),
            "/",
            len(uuid_6031)
        )

        print(
            "UUID 6031 augmented correct:",
            int(
                uuid_6031["augmented_correct"].sum()
            ),
            "/",
            len(uuid_6031)
        )

    # --------------------------------------------------------
    # Baseline errors corrected
    # --------------------------------------------------------

    baseline_errors = comparison[
        comparison["baseline_correct"] == 0
    ].copy()

    augmented_fixed = baseline_errors[
        baseline_errors["augmented_correct"] == 1
    ].copy()

    print()
    print("=" * 70)
    print("BASELINE ERRORS FIXED BY AUGMENTATION")
    print("=" * 70)

    print()
    print(
        "Baseline errors:",
        len(baseline_errors)
    )

    print(
        "Errors corrected:",
        len(augmented_fixed)
    )

    if len(augmented_fixed) > 0:

        display_columns = [
            "UUID",
            "slice",
            "true_label",
            "baseline_predicted_label",
            "augmented_predicted_label",
            "baseline_p_tampered",
            "augmented_p_tampered",
            "transition",
        ]

        print()

        print(
            augmented_fixed[
                display_columns
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # New errors
    # --------------------------------------------------------

    new_errors = comparison[
        comparison["baseline_correct"] == 1
    ].copy()

    new_errors = new_errors[
        new_errors["augmented_correct"] == 0
    ].copy()

    print()
    print("=" * 70)
    print("NEW ERRORS INTRODUCED BY AUGMENTATION")
    print("=" * 70)

    print()
    print(
        "New errors:",
        len(new_errors)
    )

    if len(new_errors) > 0:

        display_columns = [
            "UUID",
            "slice",
            "true_label",
            "baseline_predicted_label",
            "augmented_predicted_label",
            "baseline_p_tampered",
            "augmented_p_tampered",
            "transition",
        ]

        print()

        print(
            new_errors[
                display_columns
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    print()
    print("Saved:")
    print(COMPARISON_CSV)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()