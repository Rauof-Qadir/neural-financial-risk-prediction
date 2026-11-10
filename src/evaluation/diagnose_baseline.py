
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
)

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/baseline/logistic_regression.joblib")

FEATURES = [
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


def main():
    print("\n========== BASELINE DIAGNOSTICS ==========")

    model = joblib.load(MODEL_PATH)

    table = pq.read_table(
        DATA_DIR / "validation_features.parquet"
    )

    df = table.to_pandas()

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["isFraud"].to_numpy(dtype=np.int8)

    probabilities = model.predict_proba(X)[:, 1]

    print("\n1. PROBABILITY DISTRIBUTION")
    print("Minimum:", probabilities.min())
    print("Maximum:", probabilities.max())
    print("Mean:", probabilities.mean())
    print("Median:", np.median(probabilities))

    print("\nProbability percentiles:")
    for percentile in [0, 25, 50, 75, 90, 95, 99, 99.9, 100]:
        value = np.percentile(probabilities, percentile)
        print(f"{percentile:>5}%: {value:.10f}")

    print("\nExactly zero:", np.sum(probabilities == 0))
    print("Exactly one:", np.sum(probabilities == 1))

    print("\n2. PERFORMANCE AT DIFFERENT THRESHOLDS")

    for threshold in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y, predictions, zero_division=0
        )
        recall = recall_score(
            y, predictions, zero_division=0
        )
        alert_rate = predictions.mean()

        print(
            f"Threshold={threshold:.3f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"Alerts={alert_rate:.2%}"
        )

    print("\n3. BEST F1 THRESHOLD")

    precision, recall, thresholds = precision_recall_curve(
        y, probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_index])

    print("Best F1:", f"{f1_scores[best_index]:.4f}")
    print("Threshold:", f"{best_threshold:.10f}")
    print("Precision:", f"{precision[best_index]:.4f}")
    print("Recall:", f"{recall[best_index]:.4f}")

    print("\n4. MODEL PARAMETERS")
    print("Intercept:", model.intercept_)
    print("Largest absolute coefficient:", np.abs(model.coef_).max())
    print("Coefficient values:")

    for name, coefficient in zip(FEATURES, model.coef_[0]):
        print(f"{name:20s}: {coefficient:.6f}")

    print("\n5. VALIDATION PR-AUC")
    print(f"{average_precision_score(y, probabilities):.6f}")

    print("\n6. CONFUSION MATRIX AT BEST F1")
    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print(confusion_matrix(y, predictions))

    print("\n========== DIAGNOSTICS COMPLETE ==========")


if __name__ == "__main__":
    main()
