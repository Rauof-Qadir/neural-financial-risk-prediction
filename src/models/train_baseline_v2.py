
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pyarrow.parquet as pq

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("reports/experiments")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

TARGET = "isFraud"
COLUMNS = FEATURES + [TARGET]

BATCH_SIZE = 100_000
EPOCHS = 5
FRAUD_WEIGHT = 50.0
RANDOM_STATE = 42


def load_validation():
    table = pq.read_table(
        DATA_DIR / "validation_features_v2.parquet",
        columns=COLUMNS,
    )
    df = table.to_pandas()

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int8)

    return X, y


def train(model):
    parquet = pq.ParquetFile(
        DATA_DIR / "train_features_v2.parquet"
    )

    rng = np.random.default_rng(RANDOM_STATE)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        rows_seen = 0

        for batch in parquet.iter_batches(
            batch_size=BATCH_SIZE,
            columns=COLUMNS,
        ):
            df = batch.to_pandas()

            X = df[FEATURES].to_numpy(dtype=np.float32)
            y = df[TARGET].to_numpy(dtype=np.int8)

            indices = rng.permutation(len(y))
            X = X[indices]
            y = y[indices]

            weights = np.where(
                y == 1,
                FRAUD_WEIGHT,
                1.0,
            ).astype(np.float32)

            model.partial_fit(
                X,
                y,
                classes=np.array([0, 1]),
                sample_weight=weights,
            )

            rows_seen += len(y)

        print(f"Rows processed: {rows_seen:,}")

    return model


def choose_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y, probabilities
    )

    eligible = np.flatnonzero(recall[:-1] >= 0.90)

    if len(eligible) == 0:
        return 0.5

    # Prefer the eligible threshold with the highest precision.
    best = eligible[np.argmax(precision[eligible])]

    return float(thresholds[best])


def main():
    print("\n========== BASELINE V2 TRAINING ==========")

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="constant",
        eta0=0.001,
        average=True,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()
    train(model)
    training_seconds = time.perf_counter() - start

    print("\nEvaluating validation data...")

    X_val, y_val = load_validation()
    probabilities = model.predict_proba(X_val)[:, 1]

    ap = average_precision_score(y_val, probabilities)
    roc_auc = roc_auc_score(y_val, probabilities)

    threshold = choose_threshold(y_val, probabilities)
    predictions = (probabilities >= threshold).astype(np.int8)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    report = {
        "model": "SGD Logistic Regression V2",
        "feature_version": "v2_log1p",
        "fraud_sample_weight": FRAUD_WEIGHT,
        "epochs": EPOCHS,
        "learning_rate": 0.001,
        "training_seconds": training_seconds,
        "validation_average_precision": float(ap),
        "validation_roc_auc": float(roc_auc),
        "validation_threshold": threshold,
        "validation_precision": float(precision),
        "validation_recall": float(recall),
        "validation_f1": float(f1),
        "validation_alert_rate": float(predictions.mean()),
        "validation_confusion_matrix": confusion_matrix(
            y_val, predictions
        ).tolist(),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "probability_median": float(np.median(probabilities)),
        "exact_zero_count": int((probabilities == 0).sum()),
        "exact_one_count": int((probabilities == 1).sum()),
    }

    joblib.dump(
        model,
        MODEL_DIR / "logistic_regression_v2.joblib",
    )

    with open(
        REPORT_DIR / "baseline_v2_results.json",
        "w",
    ) as file:
        json.dump(report, file, indent=4)

    print("\n========== BASELINE V2 RESULTS ==========")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nModel and report saved.")
    print("\n========== V2 COMPLETE ==========")


if __name__ == "__main__":
    main()
