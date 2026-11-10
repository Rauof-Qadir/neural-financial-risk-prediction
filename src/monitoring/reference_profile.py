
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "processed" / "train_features_v2.parquet"
)

OUTPUT_PATH = (
    ROOT / "reports" / "monitoring" / "training_reference_profile.json"
)

NUMERIC_FEATURES = [
    "step",
    "log_amount",
    "log_oldbalanceOrg",
    "log_oldbalanceDest",
]

TYPE_FEATURES = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

MAX_SAMPLE = 200_000
SAMPLE_PER_BATCH = 5_000
BATCH_SIZE = 100_000
RANDOM_SEED = 42


def main():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training feature file not found: {TRAIN_PATH}"
        )

    parquet_file = pq.ParquetFile(TRAIN_PATH)

    required_columns = NUMERIC_FEATURES + TYPE_FEATURES

    available_columns = parquet_file.schema.names

    missing = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing:
        raise ValueError(
            f"Missing training feature columns: {missing}"
        )

    rng = np.random.default_rng(RANDOM_SEED)

    numeric_samples = {
        feature: []
        for feature in NUMERIC_FEATURES
    }

    type_counts = {
        category: 0
        for category in TRANSACTION_TYPES
    }

    total_rows = 0
    sampled_rows = 0

    print("Building training reference profile...")
    print(f"Training file: {TRAIN_PATH}")
    print(f"Maximum sample: {MAX_SAMPLE:,}")

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=required_columns,
    ):

        frame = batch.to_pandas()

        total_rows += len(frame)

        # ----------------------------------------------
        # Sample numeric features
        # ----------------------------------------------

        remaining = MAX_SAMPLE - sampled_rows

        if remaining > 0:

            take = min(
                SAMPLE_PER_BATCH,
                remaining,
                len(frame),
            )

            if take > 0:

                indices = rng.choice(
                    len(frame),
                    size=take,
                    replace=False,
                )

                for feature in NUMERIC_FEATURES:

                    values = frame[feature].to_numpy(
                        dtype=np.float64
                    )

                    numeric_samples[feature].append(
                        values[indices]
                    )

                sampled_rows += take

        # ----------------------------------------------
        # Count transaction types
        # ----------------------------------------------

        type_matrix = frame[
            TYPE_FEATURES
        ].to_numpy(dtype=np.float32)

        valid_rows = (
            np.isfinite(type_matrix).all(axis=1)
            & (type_matrix.sum(axis=1) == 1)
            & (type_matrix.max(axis=1) == 1)
        )

        if valid_rows.any():

            type_codes = np.argmax(
                type_matrix[valid_rows],
                axis=1,
            )

            counts = np.bincount(
                type_codes,
                minlength=len(TRANSACTION_TYPES),
            )

            for index, category in enumerate(
                TRANSACTION_TYPES
            ):
                type_counts[category] += int(
                    counts[index]
                )

        print(
            f"\rProcessed {total_rows:,} rows",
            end="",
        )

    print("\n")

    # ----------------------------------------------
    # Build numeric quantile bins
    # ----------------------------------------------

    numeric_profiles = {}

    for feature in NUMERIC_FEATURES:

        values = np.concatenate(
            numeric_samples[feature]
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise RuntimeError(
                f"No valid reference values for {feature}"
            )

        quantiles = np.quantile(
            values,
            np.linspace(0, 1, 11),
        )

        # Remove duplicate edges caused by tied values.
        internal_edges = np.unique(quantiles[1:-1])

        edges = np.concatenate([
            [-np.inf],
            internal_edges,
            [np.inf],
        ])

        counts, _ = np.histogram(
            values,
            bins=edges,
        )

        numeric_profiles[feature] = {
            "edges": [
                float(value)
                for value in edges
            ],
            "counts": [
                int(value)
                for value in counts
            ],
            "sample_size": int(len(values)),
        }

    # ----------------------------------------------
    # Build transaction type distribution
    # ----------------------------------------------

    total_type_rows = sum(type_counts.values())

    if total_type_rows == 0:
        raise RuntimeError(
            "No valid transaction type observations found."
        )

    type_distribution = {
        category: count / total_type_rows
        for category, count in type_counts.items()
    }

    profile = {
        "profile_version": "1.0",
        "source": "training_features_v2",
        "total_training_rows": total_rows,
        "numeric_sample_rows": sampled_rows,
        "numeric_features": numeric_profiles,
        "transaction_type_counts": type_counts,
        "transaction_type_distribution": type_distribution,
        "transaction_type_rows": total_type_rows,
        "note": (
            "Training-only reference. Numeric bins use a "
            "deterministic seeded sample. This profile is "
            "not a real-bank approved monitoring baseline."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            profile,
            file,
            indent=2,
        )

    print("REFERENCE PROFILE CREATED")
    print(f"Training rows: {total_rows:,}")
    print(f"Numeric sample: {sampled_rows:,}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nTransaction type distribution:")

    for category, proportion in type_distribution.items():

        print(
            f"{category:12s}: {proportion:.4%}"
        )


if __name__ == "__main__":
    main()
