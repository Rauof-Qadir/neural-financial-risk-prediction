from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION — FROZEN MODEL, NO RETUNING
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data/processed/validation_features_v2.parquet"
RAW_FILE = ROOT / "data/interim/validation.parquet"
CHECKPOINT = ROOT / "models/neural_network/financial_risk_mlp.pt"

OUTPUT_DIR = ROOT / "reports/robustness_stress_test"

FROZEN_THRESHOLD = 0.1444527953863144

EXPECTED_MODEL_HASH = (
    "ae9d3e357c2e0dcf9abe0600245eede7ebfc0198338009a5c084185f7302189b"
)

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

TYPE_NAMES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

BATCH_SIZE = 8192
RANDOM_SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# LOGGING AND UTILITIES
# ============================================================

def log(message):
    print(f"[ROBUSTNESS] {message}", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(data, path):
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        if isinstance(value, (np.bool_,)):
            return bool(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            default=convert,
            allow_nan=False,
        )


# ============================================================
# MODEL ARCHITECTURE
# State-dict keys: network.0, network.3, network.6
# ============================================================

class FinancialRiskMLP(nn.Module):

    def __init__(self, input_dim=9, dropout=0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# MODEL LOADING AND INTEGRITY CHECK
# ============================================================

def load_frozen_model():

    log("Verifying model SHA-256...")

    actual_hash = sha256_file(CHECKPOINT)

    if actual_hash != EXPECTED_MODEL_HASH:
        raise RuntimeError(
            "Frozen model hash mismatch. "
            "Do not continue with a different checkpoint."
        )

    log(f"Model hash verified: {actual_hash}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model = FinancialRiskMLP(
        input_dim=len(FEATURES),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    log(f"Checkpoint best epoch: {checkpoint.get('best_epoch')}")

    return model, actual_hash


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_data():

    log("Loading validation feature matrix...")
    features_df = pd.read_parquet(FEATURE_FILE)

    log("Loading raw validation transactions...")
    raw_df = pd.read_parquet(RAW_FILE)

    if len(features_df) != len(raw_df):
        raise ValueError("Raw and processed validation row counts differ.")

    missing = [c for c in FEATURES if c not in features_df.columns]

    if missing:
        raise ValueError(f"Missing expected features: {missing}")

    X = features_df[FEATURES].to_numpy(dtype=np.float32)
    y = raw_df["isFraud"].to_numpy(dtype=np.int8)

    if not np.isfinite(X).all():
        raise ValueError("Validation features contain NaN or infinity.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Unexpected target labels.")

    # Confirm that the encoded transaction categories match raw data.
    raw_types = raw_df["type"].astype(str).to_numpy()

    expected_one_hot = np.column_stack([
        (raw_types == category).astype(np.int8)
        for category in TYPE_NAMES
    ])

    actual_one_hot = features_df[TYPE_FEATURES].to_numpy(dtype=np.int8)

    if not np.array_equal(actual_one_hot, expected_one_hot):
        raise ValueError(
            "Transaction type encoding does not match raw validation data."
        )

    one_hot_sums = actual_one_hot.sum(axis=1)

    if not np.all(one_hot_sums == 1):
        raise ValueError("Invalid transaction-type one-hot encoding.")

    log(f"Validation observations: {len(X):,}")
    log(f"Validation frauds: {int(y.sum()):,}")
    log(f"Validation fraud prevalence: {y.mean():.6%}")

    return X, y


# ============================================================
# INPUT CONTRACT VALIDATION
# ============================================================

def validate_input_matrix(X):

    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("Input must be a two-dimensional matrix.")

    if X.shape[1] != len(FEATURES):
        raise ValueError("Unexpected feature count.")

    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinity.")

    one_hot = X[:, 4:9]

    if not np.all((one_hot == 0) | (one_hot == 1)):
        raise ValueError("Transaction type values must be binary.")

    if not np.all(one_hot.sum(axis=1) == 1):
        raise ValueError("Each row must contain exactly one transaction type.")

    return True


def run_input_validation_tests(X):

    log("Running corrupted-input rejection tests...")

    tests = []

    def test(name, matrix):
        try:
            validate_input_matrix(matrix)
            tests.append({
                "test": name,
                "expected": "REJECT",
                "actual": "ACCEPT",
                "passed": False,
            })
        except ValueError as exc:
            tests.append({
                "test": name,
                "expected": "REJECT",
                "actual": "REJECT",
                "passed": True,
                "reason": str(exc),
            })

    nan_input = X[:5].copy()
    nan_input[0, 0] = np.nan
    test("NaN input", nan_input)

    inf_input = X[:5].copy()
    inf_input[0, 1] = np.inf
    test("Infinite input", inf_input)

    invalid_type = X[:5].copy()
    invalid_type[0, 4:9] = 0
    test("All-zero transaction type", invalid_type)

    multiple_types = X[:5].copy()
    multiple_types[0, 4:9] = 1
    test("Multiple active transaction types", multiple_types)

    fractional_type = X[:5].copy()
    fractional_type[0, 4] = 0.5
    test("Fractional one-hot value", fractional_type)

    wrong_width = X[:5, :-1].copy()
    test("Wrong feature count", wrong_width)

    result = {
        "tests": tests,
        "passed": sum(item["passed"] for item in tests),
        "failed": sum(not item["passed"] for item in tests),
    }

    if result["failed"] > 0:
        raise RuntimeError("One or more input validation tests failed.")

    return result


# ============================================================
# INFERENCE
# ============================================================

def predict_probabilities(model, X):

    validate_input_matrix(X)

    probabilities = []

    with torch.inference_mode():

        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.from_numpy(
                np.array(
                    X[start:start + BATCH_SIZE],
                    dtype=np.float32,
                    copy=True,
                )
            ).to(DEVICE)

            logits = model(batch)
            probs = torch.sigmoid(logits)

            probabilities.append(probs.cpu().numpy())

    return np.concatenate(probabilities)


# ============================================================
# METRICS AT THE FROZEN THRESHOLD
# ============================================================

def calculate_metrics(y, probabilities):

    predictions = (
        probabilities >= FROZEN_THRESHOLD
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "alert_rate": float(predictions.mean()),
        "alerts": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "threshold": FROZEN_THRESHOLD,
    }


# ============================================================
# RUN ONE STRESS SCENARIO
# ============================================================

def evaluate_scenario(
    model,
    scenario_name,
    X_scenario,
    y,
    baseline_probabilities,
):

    validate_input_matrix(X_scenario)

    probabilities = predict_probabilities(model, X_scenario)

    metrics = calculate_metrics(y, probabilities)

    probability_change = np.abs(
        probabilities - baseline_probabilities
    )

    metrics.update({
        "scenario": scenario_name,
        "mean_absolute_probability_change": float(
            probability_change.mean()
        ),
        "p95_absolute_probability_change": float(
            np.quantile(probability_change, 0.95)
        ),
        "max_absolute_probability_change": float(
            probability_change.max()
        ),
    })

    return metrics


# ============================================================
# NUMERIC PERTURBATION
# ============================================================

def run_numeric_perturbations(model, X, y, baseline_probabilities):

    log("Running numeric feature perturbation tests...")

    results = []

    shifts = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]

    for feature in NUMERIC_FEATURES:

        feature_index = FEATURES.index(feature)

        for shift in shifts:

            X_changed = X.copy()
            X_changed[:, feature_index] += shift

            name = f"{feature}_shift_{shift:+.2f}_std"

            result = evaluate_scenario(
                model,
                name,
                X_changed,
                y,
                baseline_probabilities,
            )

            result["perturbed_feature"] = feature
            result["perturbation"] = shift

            results.append(result)

            log(
                f"{name}: "
                f"PR-AUC={result['average_precision']:.5f}, "
                f"Recall={result['recall']:.5f}"
            )

    return results


# ============================================================
# GAUSSIAN MEASUREMENT NOISE
# ============================================================

def run_noise_tests(model, X, y, baseline_probabilities):

    log("Running Gaussian noise tests...")

    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    for sigma in [0.05, 0.10, 0.25, 0.50]:

        X_noisy = X.copy()

        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=(len(X), len(NUMERIC_FEATURES)),
        ).astype(np.float32)

        X_noisy[:, :4] += noise

        result = evaluate_scenario(
            model,
            f"gaussian_noise_sigma_{sigma:.2f}",
            X_noisy,
            y,
            baseline_probabilities,
        )

        result["noise_standard_deviation"] = sigma

        results.append(result)

        log(
            f"Noise {sigma:.2f}: "
            f"PR-AUC={result['average_precision']:.5f}, "
            f"Recall={result['recall']:.5f}"
        )

    return results


# ============================================================
# TRANSACTION TYPE COUNTERFACTUAL STRESS
# ============================================================

def run_transaction_type_tests(
    model,
    X,
    y,
    baseline_probabilities,
):

    log("Running transaction-type counterfactual tests...")

    results = []

    for category_index, category in enumerate(TYPE_NAMES):

        X_changed = X.copy()

        # Preserve a valid one-hot representation.
        X_changed[:, 4:9] = 0
        X_changed[:, 4 + category_index] = 1

        result = evaluate_scenario(
            model,
            f"all_transactions_as_{category}",
            X_changed,
            y,
            baseline_probabilities,
        )

        result["forced_transaction_type"] = category

        results.append(result)

        log(
            f"Forced type {category}: "
            f"PR-AUC={result['average_precision']:.5f}, "
            f"Recall={result['recall']:.5f}, "
            f"Alerts={result['alerts']:,}"
        )

    return results


# ============================================================
# EXTREME VALUE CLIPPING STRESS
# ============================================================

def run_extreme_value_tests(
    model,
    X,
    y,
    baseline_probabilities,
):

    log("Running numeric clipping stress tests...")

    results = []

    for limit in [3.0, 5.0, 8.0]:

        X_clipped = X.copy()

        X_clipped[:, :4] = np.clip(
            X_clipped[:, :4],
            -limit,
            limit,
        )

        result = evaluate_scenario(
            model,
            f"numeric_clipping_at_plus_minus_{limit}",
            X_clipped,
            y,
            baseline_probabilities,
        )

        result["clipping_limit"] = limit

        results.append(result)

        log(
            f"Clipping +/-{limit}: "
            f"PR-AUC={result['average_precision']:.5f}, "
            f"Recall={result['recall']:.5f}"
        )

    return results


# ============================================================
# FRAUD PREVALENCE SCENARIOS
# Uses validation TPR/FPR at frozen threshold.
# ============================================================

def run_prevalence_scenarios(y, baseline_probabilities):

    log("Calculating fraud prevalence scenarios...")

    predictions = (
        baseline_probabilities >= FROZEN_THRESHOLD
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    scenarios = []

    for prevalence in [
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.05,
    ]:

        expected_precision = (
            prevalence * tpr
        ) / (
            prevalence * tpr
            + (1 - prevalence) * fpr
        )

        expected_alert_rate = (
            prevalence * tpr
            + (1 - prevalence) * fpr
        )

        population = 100_000

        scenarios.append({
            "assumed_fraud_prevalence": prevalence,
            "validation_tpr_used": tpr,
            "validation_fpr_used": fpr,
            "expected_precision": expected_precision,
            "expected_alert_rate": expected_alert_rate,
            "expected_alerts_per_100k": (
                expected_alert_rate * population
            ),
            "expected_true_alerts_per_100k": (
                prevalence * tpr * population
            ),
            "expected_false_alerts_per_100k": (
                (1 - prevalence) * fpr * population
            ),
            "threshold": FROZEN_THRESHOLD,
        })

    return scenarios


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 70)
    log("FROZEN NEURAL MODEL ROBUSTNESS & STRESS TEST")
    log("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, model_hash = load_frozen_model()

    X, y = load_data()

    input_validation = run_input_validation_tests(X)

    log(
        f"Input validation: "
        f"{input_validation['passed']} passed, "
        f"{input_validation['failed']} failed"
    )

    log("Calculating unperturbed validation predictions...")

    baseline_probabilities = predict_probabilities(model, X)

    baseline_metrics = calculate_metrics(
        y,
        baseline_probabilities,
    )

    log(
        f"Baseline validation PR-AUC: "
        f"{baseline_metrics['average_precision']:.6f}"
    )

    log(
        f"Baseline validation recall: "
        f"{baseline_metrics['recall']:.6f}"
    )

    all_results = []

    all_results.append({
        **baseline_metrics,
        "scenario": "baseline_unperturbed",
        "mean_absolute_probability_change": 0.0,
        "p95_absolute_probability_change": 0.0,
        "max_absolute_probability_change": 0.0,
    })

    all_results.extend(
        run_numeric_perturbations(
            model,
            X,
            y,
            baseline_probabilities,
        )
    )

    all_results.extend(
        run_noise_tests(
            model,
            X,
            y,
            baseline_probabilities,
        )
    )

    all_results.extend(
        run_transaction_type_tests(
            model,
            X,
            y,
            baseline_probabilities,
        )
    )

    all_results.extend(
        run_extreme_value_tests(
            model,
            X,
            y,
            baseline_probabilities,
        )
    )

    prevalence_results = run_prevalence_scenarios(
        y,
        baseline_probabilities,
    )

    # Save detailed results.
    pd.DataFrame(all_results).to_csv(
        OUTPUT_DIR / "stress_scenario_results.csv",
        index=False,
    )

    pd.DataFrame(prevalence_results).to_csv(
        OUTPUT_DIR / "prevalence_scenario_results.csv",
        index=False,
    )

    save_json(
        input_validation,
        OUTPUT_DIR / "input_validation_results.json",
    )

    summary = {
        "project": "Neural Financial Risk Prediction System",
        "phase": "Robustness and Stress Testing",
        "model_sha256": model_hash,
        "model_best_epoch": int(
            torch.load(
                CHECKPOINT,
                map_location="cpu",
                weights_only=True,
            )["best_epoch"]
        ),
        "threshold": FROZEN_THRESHOLD,
        "dataset": "Validation only",
        "validation_rows": len(X),
        "validation_frauds": int(y.sum()),
        "baseline_metrics": baseline_metrics,
        "scenario_count": len(all_results) - 1,
        "prevalence_scenario_count": len(prevalence_results),
        "input_validation_passed": input_validation["passed"],
        "input_validation_failed": input_validation["failed"],
        "model_retrained": False,
        "threshold_retuned": False,
        "final_test_used": False,
        "final_test_report_modified": False,
        "interpretation_warning": (
            "Perturbation and transaction-type experiments are "
            "counterfactual sensitivity tests, not real-world performance "
            "estimates. Labels remain unchanged. Prevalence estimates assume "
            "validation TPR/FPR remain constant across prevalence scenarios."
        ),
    }

    save_json(
        summary,
        OUTPUT_DIR / "robustness_summary.json",
    )

    log("=" * 70)
    log("ROBUSTNESS AUDIT COMPLETED")
    log("=" * 70)

    log(f"Scenarios completed: {len(all_results) - 1}")
    log(f"Input tests passed: {input_validation['passed']}")
    log(f"Reports: {OUTPUT_DIR}")

    log("Frozen model and threshold remained unchanged.")


if __name__ == "__main__":
    main()