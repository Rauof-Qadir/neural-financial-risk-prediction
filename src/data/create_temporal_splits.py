
from pathlib import Path
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RAW_PATH = Path(
    "data/raw/transactions.csv"
)

OUTPUT_DIR = Path("data/interim")

CHUNK_SIZE = 250_000

TRAIN_END = 520
VALIDATION_END = 631
TEST_END = 743

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_temporal_splits():

    print("\nStarting temporal data splitting...")

    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    writers = {}
    statistics = {
        "train": {"rows": 0, "fraud": 0},
        "validation": {"rows": 0, "fraud": 0},
        "test": {"rows": 0, "fraud": 0},
    }

    dtype = {
        "step": "int16",
        "type": "str",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
        "nameOrig": "str",
        "nameDest": "str",
    }

    try:
        reader = pd.read_csv(
            RAW_PATH,
            chunksize=CHUNK_SIZE,
            dtype=dtype
        )

        for chunk_number, chunk in enumerate(reader, 1):

            splits = {
                "train": chunk[
                    chunk["step"] <= TRAIN_END
                ],
                "validation": chunk[
                    (chunk["step"] > TRAIN_END)
                    & (chunk["step"] <= VALIDATION_END)
                ],
                "test": chunk[
                    (chunk["step"] > VALIDATION_END)
                    & (chunk["step"] <= TEST_END)
                ],
            }

            for split_name, split_df in splits.items():

                if split_df.empty:
                    continue

                table = pa.Table.from_pandas(
                    split_df,
                    preserve_index=False
                )

                output_path = (
                    OUTPUT_DIR / f"{split_name}.parquet"
                )

                if split_name not in writers:
                    writers[split_name] = (
                        pq.ParquetWriter(
                            output_path,
                            table.schema,
                            compression="snappy"
                        )
                    )

                writers[split_name].write_table(table)

                statistics[split_name]["rows"] += len(
                    split_df
                )

                statistics[split_name]["fraud"] += int(
                    split_df["isFraud"].sum()
                )

            print(
                f"Processed chunk: {chunk_number}"
            )

    finally:
        for writer in writers.values():
            writer.close()

    total_rows = sum(
        item["rows"] for item in statistics.values()
    )

    print("\n========== TEMPORAL SPLIT REPORT ==========")

    for name, values in statistics.items():

        rows = values["rows"]
        fraud = values["fraud"]

        fraud_rate = (
            fraud / rows * 100 if rows else 0
        )

        print(f"\n{name.upper()}")
        print("Rows:", f"{rows:,}")
        print("Fraud:", f"{fraud:,}")
        print("Fraud rate:", f"{fraud_rate:.4f}%")

    print("\nTotal split rows:", f"{total_rows:,}")

    report = {
        "boundaries": {
            "train": [1, TRAIN_END],
            "validation": [
                TRAIN_END + 1,
                VALIDATION_END
            ],
            "test": [
                VALIDATION_END + 1,
                TEST_END
            ],
        },
        "statistics": statistics,
        "total_rows": total_rows,
    }

    with open(
        OUTPUT_DIR / "split_report.json",
        "w"
    ) as file:

        json.dump(report, file, indent=4)

    print("\nTemporal splitting completed!")


if __name__ == "__main__":
    create_temporal_splits()
