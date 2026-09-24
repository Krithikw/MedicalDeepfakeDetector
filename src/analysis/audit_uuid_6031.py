from pathlib import Path

import pandas as pd


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
    / "uuid_6031"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "uuid_6031_audit.csv"
)


def main():

    print("=" * 80)
    print("UUID 6031 PERSISTENT FAILURE AUDIT")
    print("=" * 80)

    baseline = pd.read_csv(BASELINE_CSV)
    augmented = pd.read_csv(AUGMENTED_CSV)

    baseline = baseline[
        baseline["uuid"].astype(str) == "6031"
    ].copy()

    augmented = augmented[
        augmented["uuid"].astype(str) == "6031"
    ].copy()

    print()
    print("Baseline UUID 6031 samples:", len(baseline))
    print("Augmented UUID 6031 samples:", len(augmented))

    baseline = baseline[
        [
            "uuid",
            "slice",
            "binary_label",
            "predicted_label",
            "tampered_probability",
            "correct",
            "prediction_category",
        ]
    ].rename(
        columns={
            "binary_label": "true_label",
            "predicted_label": "baseline_predicted",
            "tampered_probability": "baseline_p_tampered",
            "correct": "baseline_correct",
            "prediction_category": "baseline_category",
        }
    )

    augmented = augmented[
        [
            "uuid",
            "slice",
            "true_label",
            "predicted_label",
            "p_tampered",
            "correct",
            "prediction_category",
        ]
    ].rename(
        columns={
            "predicted_label": "augmented_predicted",
            "p_tampered": "augmented_p_tampered",
            "correct": "augmented_correct",
            "prediction_category": "augmented_category",
        }
    )

    comparison = baseline.merge(
        augmented,
        on=["uuid", "slice", "true_label"],
        how="outer",
    )

    comparison = comparison.sort_values(
        "slice"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("-" * 80)
    print("UUID 6031 RESULTS")
    print("-" * 80)

    for _, row in comparison.iterrows():

        print()
        print(
            f"Slice {int(row['slice'])}"
        )

        print(
            f"  True label              : "
            f"{'Tampered' if int(row['true_label']) == 1 else 'Authentic'}"
        )

        print(
            f"  Baseline prediction     : "
            f"{'Tampered' if int(row['baseline_predicted']) == 1 else 'Authentic'}"
        )

        print(
            f"  Baseline P(tampered)   : "
            f"{float(row['baseline_p_tampered']):.6f}"
        )

        print(
            f"  Augmented prediction   : "
            f"{'Tampered' if int(row['augmented_predicted']) == 1 else 'Authentic'}"
        )

        print(
            f"  Augmented P(tampered)  : "
            f"{float(row['augmented_p_tampered']):.6f}"
        )

        print(
            f"  Baseline result        : "
            f"{row['baseline_category']}"
        )

        print(
            f"  Augmented result       : "
            f"{row['augmented_category']}"
        )

    print()
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)

    print(
        "Baseline correct:",
        int(baseline["baseline_correct"].sum()),
        "/",
        len(baseline)
    )

    print(
        "Augmented correct:",
        int(augmented["augmented_correct"].sum()),
        "/",
        len(augmented)
    )

    print()
    print("Saved:")
    print(OUTPUT_CSV)

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()