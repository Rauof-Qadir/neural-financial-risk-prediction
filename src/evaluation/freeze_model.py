
import hashlib
import json

from pathlib import Path

import numpy as np

from src.evaluation.model_diagnostics import (
    load_validation_data,
    load_neural_model,
    neural_predictions,
    threshold_for_recall,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "models/neural_network/financial_risk_mlp.pt"
)

BASELINE_PATH = (
    ROOT
    / "models/baseline/logistic_regression_v2.joblib"
)

VALIDATION_PATH = (
    ROOT
    / "data/processed/validation_features_v2.parquet"
)

EXPERIMENT_PATH = (
    ROOT
    / "reports/experiments/neural_network_results.json"
)

OUTPUT_DIR = ROOT / "reports/model_freeze"

RECALL_TARGET = 0.90
ALERT_CAP = 0.01


# ============================================================
# FILE HASH
# ============================================================

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========== MODEL FREEZE ==========")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load validation data only.
    X, y, feature_cols = load_validation_data()

    # Load selected neural network.
    model = load_neural_model(
        input_features=len(feature_cols)
    )

    # Generate validation probabilities.
    probabilities = neural_predictions(
        model,
        X,
    )

    # Freeze threshold using validation recall requirement.
    threshold = threshold_for_recall(
        y,
        probabilities,
        RECALL_TARGET,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    # Validation metrics at frozen threshold.
    tp = int(
        ((y == 1) & (predictions == 1)).sum()
    )

    fp = int(
        ((y == 0) & (predictions == 1)).sum()
    )

    tn = int(
        ((y == 0) & (predictions == 0)).sum()
    )

    fn = int(
        ((y == 1) & (predictions == 0)).sum()
    )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    alert_rate = float(
        predictions.mean()
    )

    # Secondary top-1% alert budget analysis.
    alert_count = max(
        1,
        int(np.floor(len(y) * ALERT_CAP)),
    )

    top_indices = np.argsort(
        -probabilities,
        kind="stable",
    )[:alert_count]

    top_predictions = np.zeros(
        len(y),
        dtype=np.int8,
    )

    top_predictions[top_indices] = 1

    top_tp = int(
        ((y == 1) & (top_predictions == 1)).sum()
    )

    top_fp = int(
        ((y == 0) & (top_predictions == 1)).sum()
    )

    top_fn = int(
        ((y == 1) & (top_predictions == 0)).sum()
    )

    top_precision = top_tp / max(
        top_tp + top_fp,
        1,
    )

    top_recall = top_tp / max(
        top_tp + top_fn,
        1,
    )

    # Read experiment information.
    with open(
        EXPERIMENT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        experiment = json.load(file)

    # Artifact fingerprints.
    model_hash = calculate_sha256(
        MODEL_PATH
    )

    baseline_hash = calculate_sha256(
        BASELINE_PATH
    )

    # Freeze manifest.
    manifest = {
        "project": "Neural Financial Risk Prediction System",

        "status": "FROZEN_BEFORE_FINAL_TEST",

        "selected_model": {
            "name": "PyTorch FinancialRiskMLP",
            "architecture": [9, 64, 32, 1],
            "dropout": 0.20,
            "best_epoch": experiment.get(
                "best_epoch",
                7,
            ),
            "artifact": str(
                MODEL_PATH.relative_to(ROOT)
            ),
            "sha256": model_hash,
        },

        "baseline_model": {
            "name": "Logistic Regression V2",
            "artifact": str(
                BASELINE_PATH.relative_to(ROOT)
            ),
            "sha256": baseline_hash,
        },

        "validation_data": {
            "artifact": str(
                VALIDATION_PATH.relative_to(ROOT)
            ),
            "rows": int(len(y)),
            "fraud_transactions": int(y.sum()),
            "fraud_prevalence": float(y.mean()),
        },

        "feature_columns": feature_cols,

        "decision_policy": {
            "selection_method": (
                "Highest precision among thresholds "
                "achieving at least target recall"
            ),
            "recall_target": RECALL_TARGET,
            "frozen_probability_threshold": float(
                threshold
            ),
        },

        "validation_performance": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": float(precision),
            "recall": float(recall),
            "alert_rate": alert_rate,
        },

        "secondary_alert_budget": {
            "policy": "Top risk-ranked transactions",
            "alert_cap": ALERT_CAP,
            "alert_count": int(alert_count),
            "precision": float(top_precision),
            "recall": float(top_recall),
        },

        "final_test": {
            "status": "NOT_EVALUATED",
            "threshold_must_not_be_retuned": True,
        },
    }

    output_path = (
        OUTPUT_DIR
        / "model_freeze_manifest.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    print("\nSelected model: PyTorch FinancialRiskMLP")

    print(f"Frozen threshold: {threshold:.10f}")
    print(f"Validation precision: {precision:.4f}")
    print(f"Validation recall: {recall:.4f}")
    print(f"Validation alert rate: {alert_rate:.4%}")

    print("\nSecondary top-1% policy:")
    print(f"Precision: {top_precision:.4f}")
    print(f"Recall: {top_recall:.4f}")

    print("\nModel SHA-256:")
    print(model_hash)

    print("\nFreeze manifest saved:")
    print(output_path)

    print("\n========== MODEL FROZEN ==========")


if __name__ == "__main__":
    main()
