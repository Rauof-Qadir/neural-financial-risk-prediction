
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DATA_DIR = Path("data/processed")

EXPECTED_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

forbidden_features = [
    "newbalanceOrig",
    "newbalanceDest",
    "nameOrig",
    "nameDest",
    "isFlaggedFraud",
]

for split in ["train", "validation", "test"]:

    path = DATA_DIR / f"{split}_features.parquet"

    print(f"\nChecking {split.upper()}...")

    parquet = pq.ParquetFile(path)

    columns = parquet.schema.names

    assert columns[:-1] == EXPECTED_FEATURES
    assert columns[-1] == "isFraud"

    assert not any(
        feature in columns
        for feature in forbidden_features
    )

    print("Rows:", f"{parquet.metadata.num_rows:,}")
    print("Columns:", len(columns))
    print("Feature schema: PASS")

    sample = pd.read_parquet(
        path,
        columns=EXPECTED_FEATURES + ["isFraud"],
    )

    assert sample.isnull().sum().sum() == 0
    assert sample["isFraud"].isin([0, 1]).all()

    print("Missing values: PASS")
    print("Target validation: PASS")

print("\nALL FEATURE VALIDATION CHECKS PASSED!")
