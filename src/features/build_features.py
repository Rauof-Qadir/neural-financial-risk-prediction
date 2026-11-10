
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
MODEL_DIR = Path("models")

BATCH_SIZE = 100_000

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

CATEGORIES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

TARGET = "isFraud"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_input_path(split):
    return INPUT_DIR / f"{split}.parquet"


def fit_scaler():

    print("\nFitting scaler on training data only...")

    scaler = StandardScaler()

    parquet_file = pq.ParquetFile(
        get_input_path("train")
    )

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=NUMERIC_FEATURES,
    ):

        df = batch.to_pandas()

        values = df[NUMERIC_FEATURES].to_numpy(
            dtype=np.float64
        )

        scaler.partial_fit(values)

    joblib.dump(
        scaler,
        MODEL_DIR / "preprocessor_scaler.joblib",
    )

    print("Scaler fitted successfully!")

    return scaler


def transform_split(split, scaler):

    print(f"\nProcessing {split} dataset...")

    parquet_file = pq.ParquetFile(
        get_input_path(split)
    )

    output_path = (
        OUTPUT_DIR / f"{split}_features.parquet"
    )

    writer = None
    total_rows = 0

    try:

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            df = batch.to_pandas()

            numeric = scaler.transform(
                df[NUMERIC_FEATURES].to_numpy(
                    dtype=np.float64
                )
            ).astype(np.float32)

            result = pd.DataFrame(
                numeric,
                columns=NUMERIC_FEATURES,
            )

            # One-hot encoding with fixed categories.
            # This guarantees identical feature columns
            # across train, validation and test.

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

            result[TARGET] = (
                df[TARGET].to_numpy(dtype=np.int8)
            )

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

        print(f"Rows processed: {total_rows:,}")
        print(f"Saved: {output_path}")

    finally:

        if writer is not None:
            writer.close()

    return total_rows


def build_features():

    print("\n========== FEATURE ENGINEERING ==========")

    scaler = fit_scaler()

    statistics = {}

    for split in ["train", "validation", "test"]:

        statistics[split] = transform_split(
            split,
            scaler,
        )

    feature_names = (
        NUMERIC_FEATURES
        + [f"type_{category}" for category in CATEGORIES]
    )

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORIES,
        "feature_names": feature_names,
        "target": TARGET,
        "statistics": statistics,
        "scaler_fitted_on": "train_only",
    }

    with open(
        OUTPUT_DIR / "feature_metadata.json",
        "w",
    ) as file:

        json.dump(metadata, file, indent=4)

    print("\nFeature columns:")
    print(feature_names)

    print("\n========== FEATURE ENGINEERING COMPLETE ==========")


if __name__ == "__main__":
    build_features()
