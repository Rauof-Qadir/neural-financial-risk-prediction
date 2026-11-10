
from pathlib import Path

import pandas as pd

DATA_PATH = Path(
    "data/raw/transactions.csv"
)


def inspect_dataset():

    print("\nLoading financial dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET OVERVIEW ==========")

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTransaction types:")
    print(df["type"].value_counts())

    print("\nFraud distribution:")
    print(df["isFraud"].value_counts())

    print("\nFraud percentage:")
    print(df["isFraud"].value_counts(normalize=True) * 100)

    print("\nTime range:")
    print(df["step"].min(), "to", df["step"].max())

    print("\n========== INSPECTION COMPLETE ==========")


if __name__ == "__main__":
    inspect_dataset()
