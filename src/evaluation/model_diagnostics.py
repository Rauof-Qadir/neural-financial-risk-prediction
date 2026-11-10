
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import calibration
import torch

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from src.models.neural_network import FinancialRiskMLP


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = ROOT / "data/processed/validation_features_v2.parquet"

LOGISTIC_PATH = ROOT / "models/baseline/logistic_regression_v2.joblib"

NEURAL_PATH = ROOT / "models/neural_network/financial_risk_mlp.pt"

OUTPUT_DIR = ROOT / "reports/diagnostics"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET = "isFraud"
RECALL_TARGET = 0.90

ALERT_CAPS = [0.01, 0.02, 0.05]

BATCH_SIZE = 1024
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

def load_validation_data():
    print("\nLoading validation dataset...")

    df = pd.read_parquet(VALIDATION_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    y = df[TARGET].to_numpy(dtype=np.int64)

    feature_cols = [
        col for col in df.columns
        if col != TARGET
    ]

    X = df[feature_cols].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Validation features contain NaN or infinity.")

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Fraud transactions: {int(y.sum()):,}")
    print(f"Fraud prevalence: {y.mean():.4%}")

    return X, y, feature_cols


# ============================================================
# 3. LOAD MODELS
# ============================================================

def load_logistic_model():
    print("\nLoading Logistic Regression V2...")

    model = joblib.load(LOGISTIC_PATH)

    return model


def load_neural_model(input_features):
    print("\nLoading PyTorch Neural Network...")

    model = FinancialRiskMLP(
        input_features=input_features,
        dropout=0.20,
    ).to(DEVICE)

    checkpoint = torch.load(
        NEURAL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # Support different checkpoint formats.
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# 4. GENERATE PROBABILITIES
# ============================================================

def logistic_predictions(model, X):
    probabilities = model.predict_proba(X)[:, 1]

    return probabilities.astype(np.float64)


def neural_predictions(model, X):
    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(X), BATCH_SIZE):
            batch = X[start:start + BATCH_SIZE].copy()

            x_tensor = torch.from_numpy(batch).to(DEVICE)

            logits = model(x_tensor).squeeze(1)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    return np.asarray(probabilities, dtype=np.float64)


# ============================================================
# 5. CALIBRATION DIAGNOSTICS
# ============================================================


def calibration_metrics(y, probabilities, model_name):
    eps = 1e-7

    probabilities = np.clip(
        probabilities,
        eps,
        1 - eps,
    )

    brier = brier_score_loss(y, probabilities)
    cross_entropy = log_loss(y, probabilities)

    # Equal-frequency bins.
    # Each bin contains approximately the same number of rows.
    calibration_df = pd.DataFrame({
        "probability": probabilities,
        "actual": y,
    })

    calibration_df["bin"] = pd.qcut(
        calibration_df["probability"],
        q=10,
        duplicates="drop",
    )

    grouped = (
        calibration_df
        .groupby("bin", observed=True)
        .agg(
            count=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            actual_fraud_rate=("actual", "mean"),
        )
        .reset_index()
    )

    grouped["absolute_gap"] = (
        grouped["mean_predicted_probability"]
        - grouped["actual_fraud_rate"]
    ).abs()

    grouped["model"] = model_name

    ece = np.average(
        grouped["absolute_gap"],
        weights=grouped["count"],
    )

    rows = grouped.to_dict(orient="records")

    # Convert interval labels into strings for JSON compatibility.
    for row in rows:
        row["bin"] = str(row["bin"])

    return {
        "brier_score": float(brier),
        "log_loss": float(cross_entropy),
        "expected_calibration_error": float(ece),
        "calibration_method": "equal_frequency",
        "calibration_bins": len(rows),
    }, rows


# ============================================================
# 6. SELECT THRESHOLD FOR RECALL TARGET
# ============================================================

def threshold_for_recall(y, probabilities, target=0.90):
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(
        y,
        probabilities,
    )

    valid_indices = np.where(
        recall[:-1] >= target
    )[0]

    if len(valid_indices) == 0:
        return 0.0

    # Among thresholds meeting recall target,
    # select the one with the highest precision.
    best_index = valid_indices[
        np.argmax(precision[valid_indices])
    ]

    return float(thresholds[best_index])


# ============================================================
# 7. TOP-K ALERT BUDGET POLICY
# ============================================================

def evaluate_alert_budgets(y, probabilities, model_name):
    results = []

    total = len(y)

    for cap in ALERT_CAPS:
        alert_count = max(
            1,
            int(np.floor(total * cap)),
        )

        # Rank transactions by risk score.
        # Alert only the highest-risk transactions.
        order = np.argsort(
            -probabilities,
            kind="stable",
        )

        predictions = np.zeros(total, dtype=np.int8)

        predictions[order[:alert_count]] = 1

        results.append({
            "model": model_name,
            "alert_cap": cap,
            "alert_count": int(alert_count),
            "actual_alert_rate": float(predictions.mean()),
            "precision": float(
                precision_score(
                    y,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y,
                    predictions,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y,
                    predictions,
                    zero_division=0,
                )
            ),
        })

    return results


# ============================================================
# 8. STANDARD MODEL METRICS
# ============================================================


def evaluate_model(y, probabilities, model_name):
    threshold = threshold_for_recall(
        y,
        probabilities,
        RECALL_TARGET,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    calibration, calibration_rows = calibration_metrics(
        y,
        probabilities,
        model_name,
    )

    # Brier score of a simple prevalence-based model.
    # This model predicts the same fraud probability for every transaction.
    brier_reference = float(
        y.mean() * (1 - y.mean())
    )

    # Positive BSS means our model is better than the reference model.
    brier_skill_score = float(
        1 - calibration["brier_score"] / brier_reference
    )

    results = {
        "model": model_name,

        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),

        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),

        "threshold_for_90pct_recall": threshold,

        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "alert_rate": float(predictions.mean()),

        "true_positives": int(
            ((y == 1) & (predictions == 1)).sum()
        ),

        "false_positives": int(
            ((y == 0) & (predictions == 1)).sum()
        ),

        "true_negatives": int(
            ((y == 0) & (predictions == 0)).sum()
        ),

        "false_negatives": int(
            ((y == 1) & (predictions == 0)).sum()
        ),

        # Calibration metrics
        **calibration,

        # Brier reference comparison
        "brier_reference": brier_reference,
        "brier_skill_score": brier_skill_score,
    }

    return results, calibration_rows


# ============================================================
# 9. INFERENCE LATENCY BENCHMARK
# ============================================================

def benchmark_latency(predict_function, X, model_name):
    batch_size = min(BATCH_SIZE, len(X))

    batch = X[:batch_size].copy()

    # Warmup: exclude initial overhead.
    for _ in range(WARMUP_RUNS):
        predict_function(batch)

    timings = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        predict_function(batch)

        elapsed = time.perf_counter() - start

        timings.append(elapsed)

    timings = np.asarray(timings)

    median_seconds = float(np.median(timings))

    return {
        "model": model_name,
        "device": str(DEVICE),
        "batch_size": int(batch_size),
        "benchmark_runs": BENCHMARK_RUNS,
        "p50_batch_latency_ms": float(
            np.percentile(timings, 50) * 1000
        ),
        "p95_batch_latency_ms": float(
            np.percentile(timings, 95) * 1000
        ),
        "p99_batch_latency_ms": float(
            np.percentile(timings, 99) * 1000
        ),
        "median_latency_per_transaction_ms": float(
            median_seconds * 1000 / batch_size
        ),
        "throughput_transactions_per_second": float(
            batch_size / median_seconds
        ),
    }


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n========== MODEL DIAGNOSTICS ==========")
    print(f"Device: {DEVICE}")

    X, y, feature_cols = load_validation_data()

    logistic_model = load_logistic_model()

    neural_model = load_neural_model(
        input_features=len(feature_cols)
    )

    # Generate validation probabilities.
    print("\nGenerating Logistic Regression predictions...")
    logistic_probs = logistic_predictions(
        logistic_model,
        X,
    )

    print("Generating Neural Network predictions...")
    neural_probs = neural_predictions(
        neural_model,
        X,
    )

    # Main metrics.
    all_results = {}
    calibration_rows = []
    policy_rows = []

    models = {
        "Logistic Regression V2": logistic_probs,
        "PyTorch MLP": neural_probs,
    }

    for name, probabilities in models.items():
        print(f"\nEvaluating: {name}")

        results, bins = evaluate_model(
            y,
            probabilities,
            name,
        )

        all_results[name] = results
        calibration_rows.extend(bins)

        policy_rows.extend(
            evaluate_alert_budgets(
                y,
                probabilities,
                name,
            )
        )

    # Latency benchmarks.
    print("\nBenchmarking inference latency...")

    latency_results = [
        benchmark_latency(
            lambda batch: logistic_predictions(
                logistic_model,
                batch,
            ),
            X,
            "Logistic Regression V2",
        ),
        benchmark_latency(
            lambda batch: neural_predictions(
                neural_model,
                batch,
            ),
            X,
            "PyTorch MLP",
        ),
    ]

    # Save results.
    final_report = {
        "dataset": "PaySim validation V2",
        "validation_rows": int(len(y)),
        "fraud_transactions": int(y.sum()),
        "fraud_prevalence": float(y.mean()),
        "device": str(DEVICE),
        "models": all_results,
        "inference_latency": latency_results,
        "alert_budget_policy": policy_rows,
    }

    report_path = OUTPUT_DIR / "model_diagnostics.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            final_report,
            file,
            indent=4,
        )

    pd.DataFrame(calibration_rows).to_csv(
        OUTPUT_DIR / "calibration_bins.csv",
        index=False,
    )

    pd.DataFrame(policy_rows).to_csv(
        OUTPUT_DIR / "alert_budget_comparison.csv",
        index=False,
    )

    pd.DataFrame(latency_results).to_csv(
        OUTPUT_DIR / "inference_latency.csv",
        index=False,
    )

    # Print summary.
    print("\n========== MODEL COMPARISON ==========")

    for name, result in all_results.items():
        print(f"\n{name}")
        print(f"PR-AUC: {result['pr_auc']:.4f}")
        print(f"ROC-AUC: {result['roc_auc']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1: {result['f1']:.4f}")
        print(f"Alert rate: {result['alert_rate']:.4%}")
        print(f"Brier score: {result['brier_score']:.6f}")
        print(f"Log loss: {result['log_loss']:.6f}")
        print(
            "ECE: "
            f"{result['expected_calibration_error']:.6f}"
        )

    print("\n========== ALERT BUDGET POLICIES ==========")

    for row in policy_rows:
        print(
            f"{row['model']} | "
            f"Cap: {row['alert_cap']:.0%} | "
            f"Precision: {row['precision']:.4f} | "
            f"Recall: {row['recall']:.4f}"
        )

    print("\n========== INFERENCE LATENCY ==========")

    for row in latency_results:
        print(
            f"{row['model']} | "
            f"P50: {row['p50_batch_latency_ms']:.3f} ms | "
            f"P95: {row['p95_batch_latency_ms']:.3f} ms | "
            f"Throughput: "
            f"{row['throughput_transactions_per_second']:.1f} tx/sec"
        )

    print("\nReports saved:")
    print(report_path)
    print(OUTPUT_DIR / "calibration_bins.csv")
    print(OUTPUT_DIR / "alert_budget_comparison.csv")
    print(OUTPUT_DIR / "inference_latency.csv")

    print("\n========== DIAGNOSTICS COMPLETE ==========")


if __name__ == "__main__":
    main()
