
import json
import joblib

from pathlib import Path
from sklearn.metrics import precision_score, recall_score

from src.evaluation.model_diagnostics import (
    load_validation_data,
    threshold_for_recall,
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/baseline/logistic_regression_v2.joblib"
MANIFEST_PATH = ROOT / "reports/model_freeze/model_freeze_manifest.json"

RECALL_TARGET = 0.90


def main():
    print("\n========== BASELINE POLICY FREEZE ==========")

    X, y, feature_cols = load_validation_data()

    print("\nLoading Logistic Regression baseline...")
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X)[:, 1]

    # Select threshold using validation data only.
    threshold = threshold_for_recall(
        y,
        probabilities,
        RECALL_TARGET,
    )

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    alert_rate = float(predictions.mean())

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    manifest["baseline_decision_policy"] = {
        "selection_method": (
            "Highest precision among thresholds "
            "achieving at least target recall"
        ),
        "recall_target": RECALL_TARGET,
        "frozen_probability_threshold": float(threshold),
    }

    manifest["baseline_validation_performance"] = {
        "precision": float(precision),
        "recall": float(recall),
        "alert_rate": alert_rate,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print(f"\nFrozen baseline threshold: {threshold:.10f}")
    print(f"Validation precision: {precision:.4f}")
    print(f"Validation recall: {recall:.4f}")
    print(f"Validation alert rate: {alert_rate:.4%}")

    print("\nUpdated manifest:")
    print(MANIFEST_PATH)

    print("\n========== BASELINE POLICY FROZEN ==========")


if __name__ == "__main__":
    main()
