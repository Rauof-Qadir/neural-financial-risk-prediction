
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILES = {
    "train": ROOT / "data" / "interim" / "train.parquet",
    "validation": ROOT / "data" / "interim" / "validation.parquet",
    "test": ROOT / "data" / "interim" / "test.parquet",
}

OUTPUT_DIR = ROOT / "reports" / "temporal_drift_audit"
BATCH_SIZE = 100_000

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

AMOUNT_EDGES = [-np.inf, 10_000, 50_000, 100_000, 250_000, np.inf]
AMOUNT_LABELS = [
    "Under 10K",
    "10K-50K",
    "50K-100K",
    "100K-250K",
    "Above 250K",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def combine_grouped_frames(frames, group_columns):
    """Combine small batch-level aggregates into final aggregates."""
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    return (
        combined
        .groupby(group_columns, dropna=False, as_index=False)
        .sum(numeric_only=True)
    )


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def add_amount_band(df):
    df["amount_band"] = pd.cut(
        df["amount"],
        bins=AMOUNT_EDGES,
        labels=AMOUNT_LABELS,
        right=False,
    )
    return df


# ---------------------------------------------------------
# Main audit
# ---------------------------------------------------------

def run_audit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_rows = []
    type_frames = []
    amount_frames = []
    time_frames = []
    numeric_frames = []

    total_processed = 0

    print("\nStarting temporal fraud pattern and drift audit...")
    print(f"Batch size: {BATCH_SIZE:,}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for split_name, file_path in INPUT_FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

        parquet_file = pq.ParquetFile(file_path)

        available_columns = parquet_file.schema.names
        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in available_columns
        ]

        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: {missing_columns}"
            )

        print(f"Processing {split_name}: {file_path.name}")

        split_count = 0
        split_fraud = 0
        split_amount_sum = 0.0
        split_fraud_amount_sum = 0.0

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=REQUIRED_COLUMNS,
        ):
            df = batch.to_pandas()

            # Ensure numeric columns are valid numeric values.
            numeric_columns = [
                "step",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud",
            ]

            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=REQUIRED_COLUMNS)

            if df.empty:
                continue

            df["isFraud"] = df["isFraud"].astype("int8")

            # Derived business features.
            df["origin_balance_error"] = (
                df["oldbalanceOrg"]
                - df["amount"]
                - df["newbalanceOrig"]
            )

            df["destination_balance_error"] = (
                df["oldbalanceDest"]
                + df["amount"]
                - df["newbalanceDest"]
            )

            df["amount_to_origin_balance"] = np.where(
                df["oldbalanceOrg"] > 0,
                df["amount"] / df["oldbalanceOrg"],
                np.nan,
            )

            df = add_amount_band(df)

            # ---------------------------------------------
            # 1. Split-level summary
            # ---------------------------------------------

            batch_count = len(df)
            batch_fraud = int(df["isFraud"].sum())

            split_count += batch_count
            split_fraud += batch_fraud
            split_amount_sum += float(df["amount"].sum())
            split_fraud_amount_sum += float(
                df.loc[df["isFraud"] == 1, "amount"].sum()
            )

            # ---------------------------------------------
            # 2. Fraud by transaction type
            # ---------------------------------------------

            type_group = (
                df.groupby("type", observed=False)
                .agg(
                    transactions=("isFraud", "size"),
                    fraud_count=("isFraud", "sum"),
                )
                .reset_index()
            )

            type_group["split"] = split_name
            type_frames.append(type_group)

            # ---------------------------------------------
            # 3. Fraud by transaction amount band
            # ---------------------------------------------

            amount_group = (
                df.groupby("amount_band", observed=False)
                .agg(
                    transactions=("isFraud", "size"),
                    fraud_count=("isFraud", "sum"),
                )
                .reset_index()
            )

            amount_group["split"] = split_name
            amount_frames.append(amount_group)

            # ---------------------------------------------
            # 4. Fraud over time
            # ---------------------------------------------

            # Each time bucket covers 20 simulation steps.
            df["time_period"] = (
                (df["step"].astype(int) - 1) // 20
            ) * 20 + 1

            time_group = (
                df.groupby("time_period")
                .agg(
                    transactions=("isFraud", "size"),
                    fraud_count=("isFraud", "sum"),
                )
                .reset_index()
            )

            time_group["split"] = split_name
            time_frames.append(time_group)

            # ---------------------------------------------
            # 5. Numeric feature distribution by target
            # ---------------------------------------------

            features = [
                "step",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "origin_balance_error",
                "destination_balance_error",
                "amount_to_origin_balance",
            ]

            for target_value, target_df in df.groupby("isFraud"):
                for feature in features:
                    values = target_df[feature].dropna()

                    if values.empty:
                        continue

                    numeric_frames.append({
                        "split": split_name,
                        "target": int(target_value),
                        "feature": feature,
                        "count": int(values.count()),
                        "sum": float(values.sum()),
                        "sum_sq": float(
                            np.square(values.to_numpy(dtype="float64")).sum()
                        ),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    })

            total_processed += batch_count

        split_rows.append({
            "split": split_name,
            "transactions": split_count,
            "fraud_count": split_fraud,
            "non_fraud_count": split_count - split_fraud,
            "fraud_rate": safe_divide(split_fraud, split_count),
            "average_transaction_amount": safe_divide(
                split_amount_sum, split_count
            ),
            "gross_fraud_transaction_amount": split_fraud_amount_sum,
        })

        print(
            f"  Rows: {split_count:,} | "
            f"Fraud: {split_fraud:,} | "
            f"Fraud rate: {safe_divide(split_fraud, split_count):.4%}"
        )

    # -----------------------------------------------------
    # Finalize reports
    # -----------------------------------------------------

    split_summary = pd.DataFrame(split_rows)

    split_summary["fraud_rate_percent"] = (
        split_summary["fraud_rate"] * 100
    )

    split_summary.to_csv(
        OUTPUT_DIR / "split_fraud_summary.csv",
        index=False,
    )

    # Fraud by transaction type.
    type_summary = combine_grouped_frames(
        type_frames,
        ["split", "type"],
    )

    type_summary["fraud_rate"] = (
        type_summary["fraud_count"]
        / type_summary["transactions"].replace(0, np.nan)
    )

    type_summary.to_csv(
        OUTPUT_DIR / "fraud_by_transaction_type.csv",
        index=False,
    )

    # Fraud by amount band.
    amount_summary = combine_grouped_frames(
        amount_frames,
        ["split", "amount_band"],
    )

    amount_summary["fraud_rate"] = (
        amount_summary["fraud_count"]
        / amount_summary["transactions"].replace(0, np.nan)
    )

    amount_summary.to_csv(
        OUTPUT_DIR / "fraud_by_amount_band.csv",
        index=False,
    )

    # Fraud over time.
    time_summary = combine_grouped_frames(
        time_frames,
        ["split", "time_period"],
    )

    time_summary["fraud_rate"] = (
        time_summary["fraud_count"]
        / time_summary["transactions"].replace(0, np.nan)
    )

    time_summary.to_csv(
        OUTPUT_DIR / "fraud_over_time.csv",
        index=False,
    )

    # Numeric feature summaries.
    numeric_summary = pd.DataFrame(numeric_frames)

    if not numeric_summary.empty:
        numeric_summary = (
            numeric_summary
            .groupby(
                ["split", "target", "feature"],
                as_index=False,
            )
            .agg(
                count=("count", "sum"),
                sum=("sum", "sum"),
                sum_sq=("sum_sq", "sum"),
                minimum=("minimum", "min"),
                maximum=("maximum", "max"),
            )
        )

        numeric_summary["mean"] = (
            numeric_summary["sum"] / numeric_summary["count"]
        )

        variance = (
            numeric_summary["sum_sq"] / numeric_summary["count"]
            - numeric_summary["mean"] ** 2
        ).clip(lower=0)

        numeric_summary["std"] = np.sqrt(variance)

        numeric_summary = numeric_summary.drop(
            columns=["sum", "sum_sq"]
        )

        numeric_summary["target_label"] = (
            numeric_summary["target"]
            .map({0: "Non-fraud", 1: "Fraud"})
        )

        numeric_summary.to_csv(
            OUTPUT_DIR / "feature_distribution_summary.csv",
            index=False,
        )

    # -----------------------------------------------------
    # Machine-readable overview
    # -----------------------------------------------------

    report = {
        "audit_name": "Temporal Fraud Pattern and Feature Drift Audit",
        "method": "Batch-based descriptive analysis of original Parquet splits",
        "batch_size": BATCH_SIZE,
        "total_rows_processed": total_processed,
        "split_summary": split_rows,
        "important_notes": [
            "This is descriptive analysis, not causal proof.",
            "The test split is used retrospectively for analysis only.",
            "No model was retrained and no threshold was changed.",
            "Gross fraud transaction amount is not verified financial loss.",
            "Amount bands are fixed across all splits for comparability.",
            "Time periods represent 20 simulation steps each.",
        ],
    }

    with open(
        OUTPUT_DIR / "temporal_drift_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("TEMPORAL DRIFT AUDIT COMPLETED")
    print("=" * 60)

    print(f"Total rows processed: {total_processed:,}")
    print("\nSplit fraud summary:")
    print(
        split_summary[
            [
                "split",
                "transactions",
                "fraud_count",
                "fraud_rate_percent",
            ]
        ].to_string(index=False)
    )

    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_audit()
