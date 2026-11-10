
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

INPUT_DIR = Path("data/interim")
OUTPUT_DIR = Path("data/processed")
MODEL_DIR = Path("models/baseline")

BATCH_SIZE = 100_000

RAW_NUMERIC = ["step", "amount", "oldbalanceOrg", "oldbalanceDest"]
FEATURES = [
    "step",
    "log_amount",
    "log_oldbalanceOrg",
    "log_oldbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]
CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TARGET = "isFraud"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def transform_numeric(df):
    values = df[RAW_NUMERIC].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite numeric values found.")

    if (values[:, 1:] < 0).any():
        raise ValueError("Negative amount or balance found.")

    transformed = values.copy()
    transformed[:, 1:] = np.log1p(transformed[:, 1:])

    return transformed


def fit_scaler():
    print("\nFitting v2 scaler on training data only...")
    scaler = StandardScaler()

    parquet = pq.ParquetFile(INPUT_DIR / "train.parquet")

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=RAW_NUMERIC,
    ):
        df = batch.to_pandas()
        scaler.partial_fit(transform_numeric(df))

    joblib.dump(scaler, MODEL_DIR / "preprocessor_scaler_v2.joblib")
    print("V2 scaler fitted.")

    return scaler


def transform_split(split, scaler):
    print(f"\nProcessing {split}...")

    parquet = pq.ParquetFile(INPUT_DIR / f"{split}.parquet")
    output_path = OUTPUT_DIR / f"{split}_features_v2.parquet"

    writer = None
    total_rows = 0

    try:
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas()

            numeric = scaler.transform(
                transform_numeric(df)
            ).astype(np.float32)

            result = pd.DataFrame(numeric, columns=[
                "step",
                "log_amount",
                "log_oldbalanceOrg",
                "log_oldbalanceDest",
            ])

            transaction_type = pd.Categorical(
                df["type"],
                categories=CATEGORIES,
            )

            encoded = pd.get_dummies(
                transaction_type,
                prefix="type",
                dtype=np.float32,
            )

            result = pd.concat(
                [result, encoded.reset_index(drop=True)],
                axis=1,
            )

            result = result[FEATURES]
            result[TARGET] = df[TARGET].to_numpy(dtype=np.int8)

            table = pa.Table.from_pandas(
                result,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)
            total_rows += len(result)

    finally:
        if writer is not None:
            writer.close()

    print(f"Rows: {total_rows:,}")
    print(f"Saved: {output_path}")

    return total_rows


def main():
    print("\n========== FEATURE ENGINEERING V2 ==========")

    scaler = fit_scaler()

    counts = {}

    for split in ["train", "validation", "test"]:
        counts[split] = transform_split(split, scaler)

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "transformation": {
            "step": "standard scaling",
            "amount": "log1p then standard scaling",
            "oldbalanceOrg": "log1p then standard scaling",
            "oldbalanceDest": "log1p then standard scaling",
        },
        "scaler_fitted_on": "training only",
        "row_counts": counts,
    }

    with open(OUTPUT_DIR / "feature_metadata_v2.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nV2 feature engineering complete!")


if __name__ == "__main__":
    main()
