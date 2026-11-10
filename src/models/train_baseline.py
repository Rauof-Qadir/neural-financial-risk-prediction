
from pathlib import Path
import json
import joblib
import numpy as np
import pyarrow.parquet as pq

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------- CONFIGURATION ----------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("reports/experiments")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100_000
EPOCHS = 3
MIN_RECALL = 0.90
RANDOM_STATE = 42

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

TARGET = "isFraud"
COLUMNS = FEATURES + [TARGET]


def load_validation_data():
    """Validation set is small enough to load into memory."""
    path = DATA_DIR / "validation_features.parquet"
    table = pq.read_table(path, columns=COLUMNS)

    df = table.to_pandas()

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int8)

    return X, y


def count_training_classes():
    """Count classes without loading the full training dataset."""
    parquet = pq.ParquetFile(DATA_DIR / "train_features.parquet")

    positives = 0
    total = 0

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[TARGET],
    ):
        y = batch.column(TARGET).to_numpy()
        positives += int(y.sum())
        total += len(y)

    negatives = total - positives

    return negatives, positives


def train_model(model, positive_weight):
    """Train incrementally using memory-efficient batches."""
    path = DATA_DIR / "train_features.parquet"
    parquet = pq.ParquetFile(path)

    rng = np.random.default_rng(RANDOM_STATE)
    classes = np.array([0, 1])

    for epoch in range(EPOCHS):
        print(f"\nTraining epoch {epoch + 1}/{EPOCHS}")

        rows_seen = 0

        for batch in parquet.iter_batches(
            batch_size=BATCH_SIZE,
            columns=COLUMNS,
        ):
            df = batch.to_pandas()

            X = df[FEATURES].to_numpy(dtype=np.float32)
            y = df[TARGET].to_numpy(dtype=np.int8)

            # Shuffle rows inside each batch.
            indices = rng.permutation(len(y))
            X = X[indices]
            y = y[indices]

            # Give rare fraud cases more learning importance.
            sample_weights = np.where(
                y == 1,
                positive_weight,
                1.0,
            ).astype(np.float32)

            model.partial_fit(
                X,
                y,
                classes=classes,
                sample_weight=sample_weights,
            )

            rows_seen += len(y)

        print(f"Rows processed: {rows_seen:,}")

    return model


def select_threshold(y_true, probabilities):
    """Select the highest-precision threshold meeting minimum recall."""
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    eligible = np.where(recall[:-1] >= MIN_RECALL)[0]

    if len(eligible) == 0:
        return 0.5

    best_index = eligible[
        np.argmax(precision[eligible])
    ]

    return float(thresholds[best_index])


def main():
    print("\n========== LOGISTIC REGRESSION BASELINE ==========")

    # 1. Calculate class imbalance.
    negatives, positives = count_training_classes()

    positive_weight = negatives / positives

    print(f"\nNormal transactions: {negatives:,}")
    print(f"Fraud transactions: {positives:,}")
    print(f"Fraud class weight: {positive_weight:.2f}")

    # 2. Create scalable logistic regression.
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.00001,
        learning_rate="optimal",
        average=True,
        random_state=RANDOM_STATE,
    )

    # 3. Train on training data only.
    model = train_model(model, positive_weight)

    # 4. Evaluate on validation data.
    print("\nEvaluating on validation dataset...")

    X_val, y_val = load_validation_data()
    probabilities = model.predict_proba(X_val)[:, 1]

    ap = average_precision_score(y_val, probabilities)
    roc_auc = roc_auc_score(y_val, probabilities)

    threshold = select_threshold(y_val, probabilities)
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

    alert_rate = float(predictions.mean())
    matrix = confusion_matrix(y_val, predictions).tolist()

    # 5. Save model and experiment results.
    model_path = MODEL_DIR / "logistic_regression.joblib"
    joblib.dump(model, model_path)

    report = {
        "model": "SGDClassifier Logistic Regression",
        "epochs": EPOCHS,
        "positive_class_weight": positive_weight,
        "validation_average_precision": float(ap),
        "validation_roc_auc": float(roc_auc),
        "validation_threshold": threshold,
        "validation_precision": float(precision),
        "validation_recall": float(recall),
        "validation_f1": float(f1),
        "validation_alert_rate": alert_rate,
        "validation_confusion_matrix": matrix,
        "minimum_recall_target": MIN_RECALL,
        "training_rows": negatives + positives,
        "validation_rows": len(y_val),
    }

    report_path = REPORT_DIR / "baseline_results.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    # 6. Print results.
    print("\n========== BASELINE RESULTS ==========")
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"ROC-AUC:                    {roc_auc:.4f}")
    print(f"Selected threshold:         {threshold:.6f}")
    print(f"Precision:                  {precision:.4f}")
    print(f"Recall:                     {recall:.4f}")
    print(f"F1-score:                   {f1:.4f}")
    print(f"Alert rate:                 {alert_rate:.2%}")
    print(f"Confusion matrix:           {matrix}")

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")

    print("\n========== BASELINE COMPLETE ==========")


if __name__ == "__main__":
    main()
