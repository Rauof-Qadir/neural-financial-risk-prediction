
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILES = {
    "train": PROJECT_ROOT / "data/interim/train.parquet",
    "validation": PROJECT_ROOT / "data/interim/validation.parquet",
    "test": PROJECT_ROOT / "data/interim/test.parquet",
}

FALSE_NEGATIVES_FILE = (
    PROJECT_ROOT / "reports/error_analysis/neural_false_negatives.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports/segment_level_fraud_drift"

BATCH_SIZE = 100_000

AMOUNT_EDGES = [0, 10_000, 50_000, 100_000, 250_000, np.inf]

AMOUNT_LABELS = [
    "under_10k",
    "10k_to_50k",
    "50k_to_100k",
    "100k_to_250k",
    "above_250k",
]

REQUIRED_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

EXPECTED_SPLIT_COUNTS = {
    "train": {"rows": 6_082_007, "fraud": 5_781},
    "validation": {"rows": 191_147, "fraud": 1_180},
    "test": {"rows": 89_466, "fraud": 1_252},
}

EXPECTED_NEURAL_FALSE_NEGATIVES = 212
EXPECTED_CASHOUT_FALSE_NEGATIVES = 211
EXPECTED_TEST_CASHOUT_FRAUD = 626


# ============================================================
# HELPERS
# ============================================================

def amount_band(series: pd.Series) -> pd.Series:
    """Assign transactions to fixed, interpretable amount bands."""

    result = pd.cut(
        series,
        bins=AMOUNT_EDGES,
        labels=AMOUNT_LABELS,
        right=False,
        include_lowest=True,
    )

    return result.astype("object").fillna("out_of_range")


def combine_partial_results(
    partial_results: list[pd.DataFrame],
    group_columns: list[str],
) -> pd.DataFrame:
    """Combine batch-level sums without loading the original data."""

    if not partial_results:
        return pd.DataFrame()

    combined = pd.concat(partial_results, ignore_index=True)

    value_columns = [
        column
        for column in combined.columns
        if column not in group_columns
    ]

    return (
        combined.groupby(group_columns, dropna=False, as_index=False)[
            value_columns
        ]
        .sum()
    )


def validate_split_counts(
    split_name: str,
    total_rows: int,
    total_fraud: int,
) -> None:
    expected = EXPECTED_SPLIT_COUNTS[split_name]

    if total_rows != expected["rows"]:
        raise ValueError(
            f"{split_name}: expected {expected['rows']:,} rows, "
            f"found {total_rows:,}. Check the dataset/split."
        )

    if total_fraud != expected["fraud"]:
        raise ValueError(
            f"{split_name}: expected {expected['fraud']:,} fraud rows, "
            f"found {total_fraud:,}. Check the dataset/split."
        )


# ============================================================
# STREAMING SPLIT ANALYSIS
# ============================================================

def analyze_split(
    split_name: str,
    parquet_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    print(f"\nProcessing {split_name}: {parquet_path.name}")

    parquet_file = pq.ParquetFile(parquet_path)

    available_columns = set(parquet_file.schema.names)
    missing_columns = set(REQUIRED_COLUMNS) - available_columns

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing columns: {sorted(missing_columns)}"
        )

    segment_partials = []
    balance_partials = []

    total_rows = 0
    total_fraud = 0
    batch_number = 0

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=REQUIRED_COLUMNS,
    ):
        batch_number += 1
        df = batch.to_pandas()

        df["isFraud"] = df["isFraud"].astype("int8")
        df["amount_band"] = amount_band(df["amount"])

        total_rows += len(df)
        total_fraud += int(df["isFraud"].sum())

        # ----------------------------------------------------
        # A. Fraud prevalence by transaction type + amount band
        # ----------------------------------------------------

        df["fraud_amount"] = df["amount"].where(
            df["isFraud"] == 1, 0.0
        )

        segment_group = (
            df.groupby(
                ["type", "amount_band"],
                observed=True,
                dropna=False,
            )
            .agg(
                transaction_count=("isFraud", "size"),
                fraud_count=("isFraud", "sum"),
                total_amount=("amount", "sum"),
                fraud_amount=("fraud_amount", "sum"),
            )
            .reset_index()
        )

        segment_partials.append(segment_group)

        # ----------------------------------------------------
        # B. CASH_OUT balance behavior by fraud status
        # ----------------------------------------------------

        cashout = df.loc[df["type"] == "CASH_OUT"].copy()

        if not cashout.empty:
            cashout["amount_to_origin_ratio"] = np.where(
                cashout["oldbalanceOrg"] > 0,
                cashout["amount"] / cashout["oldbalanceOrg"],
                np.nan,
            )

            cashout["newbalance_zero"] = (
                cashout["newbalanceOrig"] == 0
            ).astype("int8")

            cashout["ratio_valid_count"] = (
                cashout["amount_to_origin_ratio"].notna()
            ).astype("int8")

            balance_group = (
                cashout.groupby(
                    ["amount_band", "isFraud"],
                    observed=True,
                    dropna=False,
                )
                .agg(
                    transaction_count=("isFraud", "size"),
                    amount_sum=("amount", "sum"),
                    oldbalance_sum=("oldbalanceOrg", "sum"),
                    newbalance_sum=("newbalanceOrig", "sum"),
                    newbalance_zero_count=("newbalance_zero", "sum"),
                    ratio_sum=("amount_to_origin_ratio", "sum"),
                    ratio_count=("ratio_valid_count", "sum"),
                )
                .reset_index()
            )

            balance_partials.append(balance_group)

        if batch_number % 10 == 0:
            print(
                f"  Batches: {batch_number:,} | "
                f"Rows processed: {total_rows:,}"
            )

        del df

    # --------------------------------------------------------
    # Combine segment-level batch aggregates
    # --------------------------------------------------------

    segments = combine_partial_results(
        segment_partials,
        ["type", "amount_band"],
    )

    segments["split"] = split_name
    segments["fraud_rate"] = (
        segments["fraud_count"]
        / segments["transaction_count"]
    )

    segments["fraud_rate_pct"] = segments["fraud_rate"] * 100

    segments["fraud_amount_share"] = np.where(
        segments["total_amount"] > 0,
        segments["fraud_amount"] / segments["total_amount"],
        np.nan,
    )

    # --------------------------------------------------------
    # Combine CASH_OUT balance aggregates
    # --------------------------------------------------------

    balances = combine_partial_results(
        balance_partials,
        ["amount_band", "isFraud"],
    )

    if not balances.empty:
        balances["split"] = split_name

        balances["fraud_status"] = balances["isFraud"].map(
            {0: "legitimate", 1: "fraud"}
        )

        balances["mean_amount"] = (
            balances["amount_sum"] / balances["transaction_count"]
        )

        balances["mean_oldbalanceOrg"] = (
            balances["oldbalance_sum"] / balances["transaction_count"]
        )

        balances["mean_newbalanceOrig"] = (
            balances["newbalance_sum"] / balances["transaction_count"]
        )

        balances["newbalanceOrig_zero_rate"] = (
            balances["newbalance_zero_count"]
            / balances["transaction_count"]
        )

        balances["mean_amount_to_origin_balance_ratio"] = np.where(
            balances["ratio_count"] > 0,
            balances["ratio_sum"] / balances["ratio_count"],
            np.nan,
        )

    validate_split_counts(
        split_name,
        total_rows,
        total_fraud,
    )

    metadata = {
        "split": split_name,
        "rows": total_rows,
        "fraud_count": total_fraud,
        "fraud_rate": total_fraud / total_rows,
        "batches_processed": batch_number,
    }

    print(
        f"  Completed: {total_rows:,} rows | "
        f"{total_fraud:,} fraud | "
        f"Fraud rate: {metadata['fraud_rate']:.6%}"
    )

    return segments, balances, metadata


# ============================================================
# FROZEN NEURAL MODEL ERROR ANALYSIS
# ============================================================

def analyze_frozen_cashout_performance(
    test_segments: pd.DataFrame,
) -> pd.DataFrame:

    if not FALSE_NEGATIVES_FILE.exists():
        raise FileNotFoundError(
            f"Missing frozen model error report: {FALSE_NEGATIVES_FILE}"
        )

    print("\nAnalyzing frozen neural network CASH_OUT errors...")

    false_negatives = pd.read_csv(FALSE_NEGATIVES_FILE)

    required_fn_columns = {"type", "amount"}
    missing = required_fn_columns - set(false_negatives.columns)

    if missing:
        raise ValueError(
            "The neural false-negative report is missing columns: "
            f"{sorted(missing)}"
        )

    if len(false_negatives) != EXPECTED_NEURAL_FALSE_NEGATIVES:
        raise ValueError(
            f"Expected {EXPECTED_NEURAL_FALSE_NEGATIVES} neural false "
            f"negatives, found {len(false_negatives)}. "
            "Do not continue until the source report is verified."
        )

    cashout_fn = false_negatives.loc[
        false_negatives["type"] == "CASH_OUT"
    ].copy()

    if len(cashout_fn) != EXPECTED_CASHOUT_FALSE_NEGATIVES:
        raise ValueError(
            f"Expected {EXPECTED_CASHOUT_FALSE_NEGATIVES} CASH_OUT "
            f"false negatives, found {len(cashout_fn)}."
        )

    cashout_fn["amount_band"] = amount_band(cashout_fn["amount"])

    missed = (
        cashout_fn.groupby("amount_band", observed=True)
        .size()
        .rename("missed_fraud_count")
        .reset_index()
    )

    test_cashout = test_segments.loc[
        test_segments["split"] == "test"
    ]

    test_cashout = test_cashout.loc[
        (test_cashout["type"] == "CASH_OUT")
        & (test_cashout["fraud_count"] > 0)
    ].copy()

    test_cashout = test_cashout[
        ["amount_band", "transaction_count", "fraud_count", "fraud_rate"]
    ].rename(
        columns={
            "transaction_count": "cashout_transaction_count",
            "fraud_count": "total_fraud_count",
            "fraud_rate": "segment_fraud_rate",
        }
    )

    performance = test_cashout.merge(
        missed,
        on="amount_band",
        how="left",
    )

    performance["missed_fraud_count"] = (
        performance["missed_fraud_count"]
        .fillna(0)
        .astype(int)
    )

    performance["detected_fraud_count"] = (
        performance["total_fraud_count"]
        - performance["missed_fraud_count"]
    )

    performance["recall"] = (
        performance["detected_fraud_count"]
        / performance["total_fraud_count"]
    )

    performance["recall_pct"] = performance["recall"] * 100

    performance["miss_rate"] = (
        performance["missed_fraud_count"]
        / performance["total_fraud_count"]
    )

    performance["miss_rate_pct"] = performance["miss_rate"] * 100

    performance["model_threshold"] = 0.1444527953863144
    performance["model_status"] = "frozen_existing_model"

    total_fraud = int(performance["total_fraud_count"].sum())
    total_missed = int(performance["missed_fraud_count"].sum())
    total_detected = int(performance["detected_fraud_count"].sum())

    if total_fraud != EXPECTED_TEST_CASHOUT_FRAUD:
        raise ValueError(
            f"Expected {EXPECTED_TEST_CASHOUT_FRAUD} test CASH_OUT "
            f"frauds, found {total_fraud}."
        )

    if total_missed != EXPECTED_CASHOUT_FALSE_NEGATIVES:
        raise ValueError(
            f"Expected {EXPECTED_CASHOUT_FALSE_NEGATIVES} missed "
            f"CASH_OUT frauds, found {total_missed}."
        )

    if total_detected + total_missed != total_fraud:
        raise AssertionError("CASH_OUT fraud counts do not reconcile.")

    return performance.sort_values("amount_band")


# ============================================================
# CROSS-SPLIT DRIFT COMPARISON
# ============================================================

def build_split_comparison(
    segment_data: pd.DataFrame,
) -> pd.DataFrame:

    focus = segment_data.loc[
        segment_data["type"].isin(["CASH_OUT", "TRANSFER"])
    ].copy()

    comparison = focus.pivot_table(
        index=["type", "amount_band"],
        columns="split",
        values="fraud_rate",
        aggfunc="first",
        observed=True,
    ).reset_index()

    for column in ["train", "validation", "test"]:
        if column not in comparison.columns:
            comparison[column] = np.nan

    comparison["validation_minus_train_pp"] = (
        comparison["validation"] - comparison["train"]
    ) * 100

    comparison["test_minus_train_pp"] = (
        comparison["test"] - comparison["train"]
    ) * 100

    comparison["test_to_train_rate_ratio"] = np.where(
        comparison["train"] > 0,
        comparison["test"] / comparison["train"],
        np.nan,
    )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEGMENT-LEVEL FRAUD DRIFT INVESTIGATION")
    print("=" * 70)

    print("\nImportant safeguards:")
    print("- No model training")
    print("- No threshold tuning")
    print("- No final test rerun")
    print("- Test data used for retrospective analysis only")
    print(f"- Batch size: {BATCH_SIZE:,}")

    all_segments = []
    all_balances = []
    split_metadata = []

    for split_name, parquet_path in SPLIT_FILES.items():

        segments, balances, metadata = analyze_split(
            split_name,
            parquet_path,
        )

        all_segments.append(segments)
        all_balances.append(balances)
        split_metadata.append(metadata)

    segment_data = pd.concat(all_segments, ignore_index=True)
    balance_data = pd.concat(all_balances, ignore_index=True)

    # --------------------------------------------------------
    # Report 1: All split-level segment distributions
    # --------------------------------------------------------

    segment_data = segment_data.sort_values(
        ["type", "amount_band", "split"]
    )

    segment_data.to_csv(
        OUTPUT_DIR / "segment_fraud_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Report 2: CASH_OUT model failure by amount segment
    # --------------------------------------------------------

    model_performance = analyze_frozen_cashout_performance(
        segment_data
    )

    model_performance.to_csv(
        OUTPUT_DIR / "cashout_amount_segment_model_performance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Report 3: CASH_OUT balance behavior
    # --------------------------------------------------------

    balance_data = balance_data.sort_values(
        ["split", "amount_band", "isFraud"]
    )

    balance_data.to_csv(
        OUTPUT_DIR / "cashout_balance_behavior.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Report 4: Train/validation/test fraud-rate comparison
    # --------------------------------------------------------

    split_comparison = build_split_comparison(segment_data)

    split_comparison.to_csv(
        OUTPUT_DIR / "segment_split_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary and reconciliation
    # --------------------------------------------------------

    total_cashout_fraud = int(
        model_performance["total_fraud_count"].sum()
    )

    total_cashout_detected = int(
        model_performance["detected_fraud_count"].sum()
    )

    total_cashout_missed = int(
        model_performance["missed_fraud_count"].sum()
    )

    summary = {
        "project": "Neural Financial Risk Prediction System",
        "analysis": "Segment-Level Fraud Drift Investigation",
        "created_by": "segment_level_fraud_drift.py",
        "batch_size": BATCH_SIZE,
        "amount_band_definition": {
            "under_10k": "[0, 10000)",
            "10k_to_50k": "[10000, 50000)",
            "50k_to_100k": "[50000, 100000)",
            "100k_to_250k": "[100000, 250000)",
            "above_250k": "[250000, infinity)",
        },
        "split_metadata": split_metadata,
        "frozen_model": {
            "model": "Existing PyTorch MLP",
            "threshold": 0.1444527953863144,
            "retrained": False,
            "threshold_retuned": False,
            "final_test_rerun": False,
        },
        "cashout_test_reconciliation": {
            "total_fraud": total_cashout_fraud,
            "detected_fraud": total_cashout_detected,
            "missed_fraud": total_cashout_missed,
            "recall": (
                total_cashout_detected / total_cashout_fraud
                if total_cashout_fraud else None
            ),
            "reconciled": (
                total_cashout_detected + total_cashout_missed
                == total_cashout_fraud
            ),
        },
        "interpretation_limits": [
            "Segment differences are descriptive, not causal evidence.",
            "Test results are retrospective and were not used for tuning.",
            "Fraud prevalence changes do not alone prove concept drift.",
            "Synthetic PaySim patterns should not be generalized "
            "as real banking behavior.",
            "Amount bands use fixed boundaries across all splits.",
        ],
        "reports": [
            "segment_fraud_distribution.csv",
            "cashout_amount_segment_model_performance.csv",
            "cashout_balance_behavior.csv",
            "segment_split_comparison.csv",
        ],
    }

    with open(
        OUTPUT_DIR / "segment_drift_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CASH_OUT MODEL PERFORMANCE BY AMOUNT BAND")
    print("=" * 70)

    print(
        model_performance[
            [
                "amount_band",
                "total_fraud_count",
                "detected_fraud_count",
                "missed_fraud_count",
                "recall_pct",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("CASH_OUT FRAUD RATE: TRAIN vs VALIDATION vs TEST")
    print("=" * 70)

    cashout_comparison = split_comparison.loc[
        split_comparison["type"] == "CASH_OUT"
    ]

    print(
        cashout_comparison[
            [
                "amount_band",
                "train",
                "validation",
                "test",
                "test_to_train_rate_ratio",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("RECONCILIATION")
    print("=" * 70)

    print(f"Test CASH_OUT fraud: {total_cashout_fraud:,}")
    print(f"Detected by frozen MLP: {total_cashout_detected:,}")
    print(f"Missed by frozen MLP: {total_cashout_missed:,}")
    print(
        "Overall CASH_OUT recall: "
        f"{total_cashout_detected / total_cashout_fraud:.4%}"
    )

    print("\nAll reports saved to:")
    print(OUTPUT_DIR)

    print("\nSEGMENT-LEVEL INVESTIGATION COMPLETED.")


if __name__ == "__main__":
    main()
