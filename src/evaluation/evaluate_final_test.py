
import hashlib
import json

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

from src.evaluation.model_diagnostics import (
    load_neural_model,
    neural_predictions,
)


ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data/processed/test_features_v2.parquet"

MANIFEST_PATH = (
    ROOT / "reports/model_freeze/model_freeze_manifest.json"
)

NEURAL_MODEL_PATH = (
    ROOT / "models/neural_network/financial_risk_mlp.pt"
)

BASELINE_MODEL_PATH = (
    ROOT / "models/baseline/logistic_regression_v2.joblib"
)

OUTPUT_DIR = ROOT / "reports/final_evaluation"


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_model_hash(file_path, expected_hash, model_name):
    actual_hash = calculate_sha256(file_path)

    if actual_hash != expected_hash:
        raise ValueError(
            f"{model_name} hash mismatch! "
            "The frozen model artifact has changed."
        )

    print(f"{model_name} SHA-256 verified.")


def evaluate_model(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),

        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),

        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),

        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),

        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),

        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),

        "alert_rate": float(predictions.mean()),

        "alert_count": int(predictions.sum()),

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },

        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
    }


def main():
    print("\n========== FINAL UNSEEN TEST EVALUATION ==========")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # 1. Load frozen experiment manifest
    # ------------------------------------------------

    print("\nLoading frozen model manifest...")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest["final_test"]["status"] != "NOT_EVALUATED":
        raise ValueError(
            "Final test has already been evaluated. "
            "Do not overwrite the original evaluation."
        )

    feature_cols = manifest["feature_columns"]

    neural_threshold = manifest[
        "decision_policy"
    ]["frozen_probability_threshold"]

    baseline_threshold = manifest[
        "baseline_decision_policy"
    ]["frozen_probability_threshold"]

    print(f"Neural threshold: {neural_threshold:.10f}")
    print(f"Baseline threshold: {baseline_threshold:.10f}")

    # ------------------------------------------------
    # 2. Verify model artifacts
    # ------------------------------------------------

    print("\nVerifying frozen model artifacts...")

    verify_model_hash(
        NEURAL_MODEL_PATH,
        manifest["selected_model"]["sha256"],
        "Neural Network",
    )

    verify_model_hash(
        BASELINE_MODEL_PATH,
        manifest["baseline_model"]["sha256"],
        "Logistic Regression",
    )

    # ------------------------------------------------
    # 3. Load untouched test dataset
    # ------------------------------------------------

    print("\nLoading final test dataset...")

    df = pd.read_parquet(TEST_PATH)

    if "isFraud" not in df.columns:
        raise ValueError("Target column 'isFraud' not found.")

    missing_features = [
        col for col in feature_cols if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing test features: {missing_features}"
        )

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["isFraud"].to_numpy(dtype=np.int8)

    if not np.isfinite(X).all():
        raise ValueError("Test features contain NaN or infinity.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    print(f"Test rows: {len(y):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Fraud transactions: {int(y.sum()):,}")
    print(f"Fraud prevalence: {y.mean():.4%}")

    # ------------------------------------------------
    # 4. Load both frozen models
    # ------------------------------------------------

    print("\nLoading frozen models...")

    neural_model = load_neural_model(
        input_features=len(feature_cols)
    )

    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    # ------------------------------------------------
    # 5. Generate test probabilities
    # ------------------------------------------------

    print("\nGenerating Neural Network predictions...")

    neural_probabilities = neural_predictions(
        neural_model,
        X,
    )

    print("Generating Logistic Regression predictions...")

    baseline_probabilities = baseline_model.predict_proba(X)[:, 1]

    # ------------------------------------------------
    # 6. Evaluate with frozen thresholds
    # ------------------------------------------------

    print("\nEvaluating Neural Network...")

    neural_results = evaluate_model(
        y,
        neural_probabilities,
        neural_threshold,
    )

    print("\nEvaluating Logistic Regression...")

    baseline_results = evaluate_model(
        y,
        baseline_probabilities,
        baseline_threshold,
    )

    # ------------------------------------------------
    # 7. Compare validation and test prevalence
    # ------------------------------------------------

    validation_prevalence = manifest[
        "validation_data"
    ]["fraud_prevalence"]

    test_prevalence = float(y.mean())

    prevalence_shift = (
        test_prevalence / validation_prevalence
        if validation_prevalence > 0
        else None
    )

    # ------------------------------------------------
    # 8. Build final report
    # ------------------------------------------------

    report = {
        "project": "Neural Financial Risk Prediction System",

        "evaluation_status": "COMPLETED",

        "evaluation_policy": (
            "Untouched temporal test set. "
            "Both thresholds were frozen on validation data. "
            "No test-based threshold tuning."
        ),

        "test_data": {
            "artifact": str(TEST_PATH.relative_to(ROOT)),
            "rows": int(len(y)),
            "fraud_transactions": int(y.sum()),
            "normal_transactions": int((y == 0).sum()),
            "fraud_prevalence": test_prevalence,
            "features": feature_cols,
        },

        "distribution_comparison": {
            "validation_fraud_prevalence": float(
                validation_prevalence
            ),
            "test_fraud_prevalence": test_prevalence,
            "test_to_validation_prevalence_ratio": (
                float(prevalence_shift)
                if prevalence_shift is not None
                else None
            ),
        },

        "neural_network": {
            "model": "PyTorch FinancialRiskMLP",
            "results": neural_results,
        },

        "logistic_regression": {
            "model": "Logistic Regression V2",
            "results": baseline_results,
        },

        "validation_reference": {
            "neural_network": manifest[
                "validation_performance"
            ],
            "logistic_regression": manifest[
                "baseline_validation_performance"
            ],
        },

        "important_note": (
            "These results represent one temporal holdout "
            "from the synthetic PaySim dataset. They do not "
            "guarantee production performance."
        ),
    }

    output_path = OUTPUT_DIR / "final_test_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

# Record that the frozen test evaluation is complete.
    manifest["final_test"]["status"] = "COMPLETED"
    manifest["final_test"]["report"] = str(
        output_path.relative_to(ROOT)
    )

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    # ------------------------------------------------
    # 9. Print final results
    # ------------------------------------------------

    print("\n========== FINAL TEST RESULTS ==========")

    for name, results in [
        ("PYTORCH NEURAL NETWORK", neural_results),
        ("LOGISTIC REGRESSION", baseline_results),
    ]:
        print(f"\n{name}")
        print("-" * 45)

        print(f"PR-AUC:       {results['pr_auc']:.6f}")
        print(f"ROC-AUC:      {results['roc_auc']:.6f}")
        print(f"Precision:    {results['precision']:.4f}")
        print(f"Recall:       {results['recall']:.4f}")
        print(f"F1:           {results['f1']:.4f}")
        print(f"Alert rate:   {results['alert_rate']:.4%}")
        print(f"Alert count:  {results['alert_count']:,}")
        print(f"Brier score:  {results['brier_score']:.6f}")
        print(f"Log loss:     {results['log_loss']:.6f}")

        print("\nConfusion Matrix:")
        print(results["confusion_matrix"])

    print("\n========== DISTRIBUTION SHIFT ==========")

    print(f"Validation fraud rate: {validation_prevalence:.4%}")
    print(f"Test fraud rate:       {test_prevalence:.4%}")

    print(f"Test / Validation:     {prevalence_shift:.2f}x")

    print("\nFinal report saved:")
    print(output_path)

    print("\n========== EVALUATION COMPLETED ==========")


if __name__ == "__main__":
    main()
