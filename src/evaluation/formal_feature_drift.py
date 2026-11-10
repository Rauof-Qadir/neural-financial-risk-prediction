
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILES = {
    "train": ROOT / "data/interim/train.parquet",
    "validation": ROOT / "data/interim/validation.parquet",
    "test": ROOT / "data/interim/test.parquet",
}

OUTPUT_DIR = ROOT / "reports/formal_feature_drift"

BATCH_SIZE = 100_000
SAMPLE_PER_BATCH = 5_000
MAX_SAMPLE_PER_SPLIT = 200_000
MAX_FRAUD_SAMPLE_PER_SPLIT = 50_000
RANDOM_SEED = 42

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_FEATURES = ["type"]

REQUIRED_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + ["isFraud"]
)

COMPARISONS = [
    ("train", "validation"),
    ("train", "test"),
]


# =========================================================
# PSI
# =========================================================

def calculate_psi(reference, comparison, bins=10):
    """
    Calculate PSI using quantile bins from reference data.
    Returns PSI and the number of usable bins.
    """

    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]

    if len(reference) == 0 or len(comparison) == 0:
        return np.nan, 0

    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        np.quantile(reference, quantiles)
    )

    if len(edges) < 2:
        return 0.0, 1

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    comp_counts, _ = np.histogram(comparison, bins=edges)

    # Smoothing prevents division by zero.
    epsilon = 1e-6

    ref_pct = (
        ref_counts + epsilon
    ) / (
        ref_counts.sum() + epsilon * len(ref_counts)
    )

    comp_pct = (
        comp_counts + epsilon
    ) / (
        comp_counts.sum() + epsilon * len(comp_counts)
    )

    psi = np.sum(
        (comp_pct - ref_pct)
        * np.log(comp_pct / ref_pct)
    )

    return float(psi), len(ref_counts)



def calculate_zero_inflated_psi(reference, comparison, nonzero_bins=5):
    """
    PSI for a zero-heavy numeric feature.

    Bin 1: exact zero values.
    Remaining bins: non-zero values using reference-derived quantiles.

    Also returns conditional PSI for non-zero observations.
    """

    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]

    if len(reference) == 0 or len(comparison) == 0:
        return {
            "psi": np.nan,
            "psi_bins": 0,
            "reference_zero_rate": np.nan,
            "comparison_zero_rate": np.nan,
            "zero_rate_difference_pp": np.nan,
            "reference_nonzero_count": 0,
            "comparison_nonzero_count": 0,
            "nonzero_psi": np.nan,
            "nonzero_psi_bins": 0,
        }

    ref_zero = reference == 0
    comp_zero = comparison == 0

    ref_nonzero = reference[~ref_zero]
    comp_nonzero = comparison[~comp_zero]

    # Derive non-zero bin boundaries from reference data only.
    if len(ref_nonzero) > 0:
        quantiles = np.linspace(0, 1, nonzero_bins + 1)
        edges = np.unique(
            np.quantile(ref_nonzero, quantiles)
        )

        # If reference non-zero values are constant,
        # use one non-zero bin rather than reporting zero PSI
        # for a collapsed quantile structure.
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
    else:
        edges = np.array([-np.inf, np.inf])

    ref_nonzero_counts, _ = np.histogram(
        ref_nonzero, bins=edges
    )
    comp_nonzero_counts, _ = np.histogram(
        comp_nonzero, bins=edges
    )

    # First bin represents exact zeros.
    ref_counts = np.concatenate([
        [np.sum(ref_zero)],
        ref_nonzero_counts,
    ]).astype(float)

    comp_counts = np.concatenate([
        [np.sum(comp_zero)],
        comp_nonzero_counts,
    ]).astype(float)

    epsilon = 1e-6

    ref_pct = (
        ref_counts + epsilon
    ) / (
        ref_counts.sum() + epsilon * len(ref_counts)
    )

    comp_pct = (
        comp_counts + epsilon
    ) / (
        comp_counts.sum() + epsilon * len(comp_counts)
    )

    total_psi = np.sum(
        (comp_pct - ref_pct)
        * np.log(comp_pct / ref_pct)
    )

    # Conditional PSI: compare only non-zero observations.
    # Require enough observations and at least two distinct
    # reference values to avoid misleading tiny-sample results.
    if (
        len(ref_nonzero) >= 20
        and len(comp_nonzero) >= 20
        and len(np.unique(ref_nonzero)) >= 2
    ):
        nonzero_psi, nonzero_psi_bins = calculate_psi(
            ref_nonzero,
            comp_nonzero,
            bins=nonzero_bins,
        )
    else:
        nonzero_psi = np.nan
        nonzero_psi_bins = 0

    reference_zero_rate = float(np.mean(ref_zero))
    comparison_zero_rate = float(np.mean(comp_zero))

    return {
        "psi": float(total_psi),
        "psi_bins": len(ref_counts),
        "reference_zero_rate": reference_zero_rate,
        "comparison_zero_rate": comparison_zero_rate,
        "zero_rate_difference_pp": (
            comparison_zero_rate - reference_zero_rate
        ) * 100,
        "reference_nonzero_count": int(len(ref_nonzero)),
        "comparison_nonzero_count": int(len(comp_nonzero)),
        "nonzero_psi": (
            float(nonzero_psi)
            if np.isfinite(nonzero_psi)
            else np.nan
        ),
        "nonzero_psi_bins": int(nonzero_psi_bins),
    }



def calculate_categorical_psi(reference, comparison):
    """Calculate PSI for categorical distributions."""

    ref_counts = reference.value_counts()
    comp_counts = comparison.value_counts()

    categories = sorted(
        set(ref_counts.index).union(comp_counts.index)
    )

    if not categories:
        return np.nan

    ref_counts = ref_counts.reindex(
        categories, fill_value=0
    ).to_numpy(dtype=float)

    comp_counts = comp_counts.reindex(
        categories, fill_value=0
    ).to_numpy(dtype=float)

    epsilon = 1e-6

    ref_pct = (
        ref_counts + epsilon
    ) / (
        ref_counts.sum() + epsilon * len(categories)
    )

    comp_pct = (
        comp_counts + epsilon
    ) / (
        comp_counts.sum() + epsilon * len(categories)
    )

    psi = np.sum(
        (comp_pct - ref_pct)
        * np.log(comp_pct / ref_pct)
    )

    return float(psi)


# =========================================================
# DRIFT INTERPRETATION
# =========================================================

def interpret_psi(value):
    if pd.isna(value):
        return "NOT_AVAILABLE"

    if value < 0.10:
        return "LOW"

    if value < 0.25:
        return "MODERATE"

    return "HIGH"


def interpret_ks(value):
    if pd.isna(value):
        return "NOT_AVAILABLE"

    if value < 0.05:
        return "SMALL"

    if value < 0.10:
        return "MODERATE"

    return "LARGE"


# =========================================================
# SAMPLE COLLECTION
# =========================================================

def collect_samples(split_name, file_path):

    parquet_file = pq.ParquetFile(file_path)

    available = parquet_file.schema.names

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in available
    ]

    if missing:
        raise ValueError(
            f"{split_name} is missing columns: {missing}"
        )

    population_parts = []
    fraud_parts = []

    total_rows = 0
    total_fraud = 0

    print(f"\nCollecting samples from {split_name}...")

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=REQUIRED_COLUMNS,
        ),
        start=1,
    ):

        df = batch.to_pandas()

        df["isFraud"] = pd.to_numeric(
            df["isFraud"], errors="coerce"
        )

        df = df.dropna(subset=REQUIRED_COLUMNS)

        if df.empty:
            continue

        df["isFraud"] = df["isFraud"].astype(int)

        total_rows += len(df)

        fraud_df = df[df["isFraud"] == 1]

        total_fraud += len(fraud_df)

        # Random representative sample from every batch.
        sample_n = min(
            SAMPLE_PER_BATCH,
            len(df),
        )

        population_sample = df.sample(
            n=sample_n,
            random_state=RANDOM_SEED + batch_number,
        )

        population_parts.append(population_sample)

        # Fraud population is much smaller.
        if not fraud_df.empty:
            fraud_parts.append(fraud_df.copy())

    population = pd.concat(
        population_parts,
        ignore_index=True,
    )

    if len(population) > MAX_SAMPLE_PER_SPLIT:
        population = population.sample(
            n=MAX_SAMPLE_PER_SPLIT,
            random_state=RANDOM_SEED,
        )

    if fraud_parts:
        fraud = pd.concat(
            fraud_parts,
            ignore_index=True,
        )
    else:
        fraud = pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )

    if len(fraud) > MAX_FRAUD_SAMPLE_PER_SPLIT:
        fraud = fraud.sample(
            n=MAX_FRAUD_SAMPLE_PER_SPLIT,
            random_state=RANDOM_SEED,
        )

    print(
        f"  Population rows: {total_rows:,} | "
        f"Fraud rows: {total_fraud:,}"
    )

    print(
        f"  Population sample: {len(population):,} | "
        f"Fraud sample: {len(fraud):,}"
    )

    return {
        "population": population,
        "fraud": fraud,
        "total_rows": total_rows,
        "total_fraud": total_fraud,
    }


# =========================================================
# NUMERIC FEATURE DRIFT
# =========================================================


def numeric_drift(
    reference,
    comparison,
    feature,
    comparison_name,
    population_type,
):

    ref_values = pd.to_numeric(
        reference[feature], errors="coerce"
    ).dropna().to_numpy(dtype=float)

    comp_values = pd.to_numeric(
        comparison[feature], errors="coerce"
    ).dropna().to_numpy(dtype=float)

    ref_values = ref_values[np.isfinite(ref_values)]
    comp_values = comp_values[np.isfinite(comp_values)]

    if len(ref_values) == 0 or len(comp_values) == 0:
        return None

    # Separate zero-heavy balance feature from ordinary PSI.
    if feature == "newbalanceOrig":
        psi_result = calculate_zero_inflated_psi(
            ref_values,
            comp_values,
            nonzero_bins=5,
        )

        psi = psi_result["psi"]
        number_of_bins = psi_result["psi_bins"]

    else:
        psi, number_of_bins = calculate_psi(
            ref_values,
            comp_values,
        )

        psi_result = {
            "reference_zero_rate": float(
                np.mean(ref_values == 0)
            ),
            "comparison_zero_rate": float(
                np.mean(comp_values == 0)
            ),
            "zero_rate_difference_pp": float(
                (np.mean(comp_values == 0)
                 - np.mean(ref_values == 0)) * 100
            ),
            "reference_nonzero_count": int(
                np.sum(ref_values != 0)
            ),
            "comparison_nonzero_count": int(
                np.sum(comp_values != 0)
            ),
            "nonzero_psi": np.nan,
            "nonzero_psi_bins": 0,
        }

    ks_result = ks_2samp(
        ref_values,
        comp_values,
        alternative="two-sided",
        method="auto",
    )

    ks_statistic = float(ks_result.statistic)
    ks_pvalue = float(ks_result.pvalue)

    return {
        "comparison": comparison_name,
        "population": population_type,
        "feature": feature,

        "reference_sample_size": len(ref_values),
        "comparison_sample_size": len(comp_values),

        "reference_mean": float(np.mean(ref_values)),
        "comparison_mean": float(np.mean(comp_values)),

        "reference_median": float(np.median(ref_values)),
        "comparison_median": float(np.median(comp_values)),

        "psi": psi,
        "psi_interpretation": interpret_psi(psi),
        "psi_bins": number_of_bins,

        "reference_zero_rate": psi_result["reference_zero_rate"],
        "comparison_zero_rate": psi_result["comparison_zero_rate"],
        "zero_rate_difference_pp": psi_result[
            "zero_rate_difference_pp"
        ],

        "reference_nonzero_count": psi_result[
            "reference_nonzero_count"
        ],
        "comparison_nonzero_count": psi_result[
            "comparison_nonzero_count"
        ],

        "nonzero_psi": psi_result["nonzero_psi"],
        "nonzero_psi_bins": psi_result["nonzero_psi_bins"],

        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "ks_interpretation": interpret_ks(ks_statistic),
        "statistically_significant_005": ks_pvalue < 0.05,
    }



# =========================================================
# CATEGORICAL FEATURE DRIFT
# =========================================================

def categorical_drift(
    reference,
    comparison,
    feature,
    comparison_name,
):

    ref = reference[feature].astype(str)
    comp = comparison[feature].astype(str)

    psi = calculate_categorical_psi(ref, comp)

    categories = sorted(
        set(ref.unique()).union(comp.unique())
    )

    rows = []

    for category in categories:

        ref_rate = float((ref == category).mean())
        comp_rate = float((comp == category).mean())

        rows.append({
            "comparison": comparison_name,
            "feature": feature,
            "category": category,
            "reference_rate": ref_rate,
            "comparison_rate": comp_rate,
            "absolute_rate_difference": abs(
                comp_rate - ref_rate
            ),
            "overall_feature_psi": psi,
            "psi_interpretation": interpret_psi(psi),
        })

    return rows


# =========================================================
# MAIN
# =========================================================

def run_audit():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 65)
    print("FORMAL FEATURE DRIFT AUDIT")
    print("=" * 65)

    datasets = {}

    for split_name, file_path in INPUT_FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        datasets[split_name] = collect_samples(
            split_name,
            file_path,
        )

    numeric_results = []
    categorical_results = []

    for ref_name, comp_name in COMPARISONS:

        print(
            f"\nComparing {ref_name.upper()} "
            f"vs {comp_name.upper()}"
        )

        ref_data = datasets[ref_name]
        comp_data = datasets[comp_name]

        # Population-level numeric drift.
        for feature in NUMERIC_FEATURES:

            result = numeric_drift(
                ref_data["population"],
                comp_data["population"],
                feature,
                f"{ref_name}_vs_{comp_name}",
                "population",
            )

            if result:
                numeric_results.append(result)

        # Fraud-only numeric drift.
        for feature in NUMERIC_FEATURES:

            result = numeric_drift(
                ref_data["fraud"],
                comp_data["fraud"],
                feature,
                f"{ref_name}_vs_{comp_name}",
                "fraud_only",
            )

            if result:
                numeric_results.append(result)

        # Population-level categorical drift.
        for feature in CATEGORICAL_FEATURES:

            rows = categorical_drift(
                ref_data["population"],
                comp_data["population"],
                feature,
                f"{ref_name}_vs_{comp_name}",
            )

            categorical_results.extend(rows)

    # -----------------------------------------------------
    # Save numerical results
    # -----------------------------------------------------

    numeric_df = pd.DataFrame(numeric_results)

    numeric_df.to_csv(
        OUTPUT_DIR / "numeric_feature_drift.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Save categorical results
    # -----------------------------------------------------

    categorical_df = pd.DataFrame(
        categorical_results
    )

    categorical_df.to_csv(
        OUTPUT_DIR / "categorical_feature_drift.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Create ranked summary
    # -----------------------------------------------------

    if not numeric_df.empty:

        ranked = numeric_df.sort_values(
            by=["comparison", "psi"],
            ascending=[True, False],
        )

        ranked.to_csv(
            OUTPUT_DIR / "drift_ranking.csv",
            index=False,
        )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "audit_name": "Formal Feature Drift Analysis",
        "reference_split": "train",
        "comparisons": [
            "train_vs_validation",
            "train_vs_test",
        ],
        "batch_size": BATCH_SIZE,
        "sample_per_batch": SAMPLE_PER_BATCH,
        "max_population_sample_per_split": MAX_SAMPLE_PER_SPLIT,
        "max_fraud_sample_per_split": MAX_FRAUD_SAMPLE_PER_SPLIT,
        "random_seed": RANDOM_SEED,
        "sampling_method": (
            "Random sample from each sequential Parquet batch, "
            "followed by a capped sample."
        ),
        "psi_method": (
            "Standard numeric features: up to ten quantile bins "
            "derived from reference sample. newbalanceOrig: "
            "separate exact-zero bin plus up to five reference-derived "
            "quantile bins for non-zero observations. Conditional "
            "non-zero PSI is reported separately when sample-size "
            "requirements are met. Epsilon smoothing is used."
        ),
        "ks_method": "Two-sided two-sample KS test on sampled observations.",
        "important_notes": [
            "PSI and KS values are sample-based estimates.",
            "KS p-values can be highly sensitive to sample size.",
            "Statistical significance does not imply business importance.",
            "Fraud-only comparisons use observed fraud labels retrospectively.",
            "Test data is used for descriptive analysis only.",
            "No model retraining or threshold tuning was performed.",
            "PSI interpretation bands are heuristics, not universal standards.",
        ],
        "split_counts": {
            name: {
                "transactions": data["total_rows"],
                "fraud": data["total_fraud"],
            }
            for name, data in datasets.items()
        },
    }

    with open(
        OUTPUT_DIR / "formal_drift_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Terminal output
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FORMAL DRIFT AUDIT COMPLETED")
    print("=" * 65)

    if not numeric_df.empty:

        print("\nNUMERIC FEATURE DRIFT:")

        print(
            numeric_df[
                [
                    "comparison",
                    "population",
                    "feature",
                    "psi",
                    "psi_interpretation",
                    "ks_statistic",
                    "ks_pvalue",
                ]
            ]
            .sort_values(
                ["comparison", "population", "psi"],
                ascending=[True, True, False],
            )
            .to_string(index=False)
        )

    if not categorical_df.empty:

        print("\nCATEGORICAL FEATURE DRIFT:")

        print(
            categorical_df.to_string(index=False)
        )

    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_audit()
