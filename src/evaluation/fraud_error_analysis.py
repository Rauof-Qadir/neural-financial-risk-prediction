
import json
import hashlib

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.model_diagnostics import (
    load_neural_model,
    neural_predictions,
)


ROOT = Path(__file__).resolve().parents[2]

RAW_TEST_PATH = ROOT / "data/interim/test.parquet"
FEATURE_TEST_PATH = ROOT / "data/processed/test_features_v2.parquet"

MANIFEST_PATH = ROOT / "reports/model_freeze/model_freeze_manifest.json"

NEURAL_MODEL_PATH = ROOT / "models/neural_network/financial_risk_mlp.pt"

BASELINE_MODEL_PATH = ROOT / "models/baseline/logistic_regression_v2.joblib"

OUTPUT_DIR = ROOT / "reports/error_analysis"


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_hash(path, expected):
    actual = sha256(path)

    if actual != expected:
        raise ValueError(f"Model hash mismatch: {path}")

    print(f"Verified: {path.name}")


def classification_columns(df, prefix):
    actual = df[f"{prefix}_prediction"].to_numpy()
    target = df["isFraud"].to_numpy()

    df[f"{prefix}_error"] = np.select(
        [
            (target == 0) & (actual == 1),
            (target == 1) & (actual == 0),
        ],
        [
            "False Positive",
            "False Negative",
        ],
        default="Correct",
    )

    return df


def group_analysis(df, group_column, prefix="neural"):
    grouped = df.groupby(
        group_column,
        observed=False,
    )

    rows = []

    for group_name, part in grouped:
        tp = int(
            (
                (part["isFraud"] == 1)
                & (part[f"{prefix}_prediction"] == 1)
            ).sum()
        )

        fp = int(
            (
                (part["isFraud"] == 0)
                & (part[f"{prefix}_prediction"] == 1)
            ).sum()
        )

        fn = int(
            (
                (part["isFraud"] == 1)
                & (part[f"{prefix}_prediction"] == 0)
            ).sum()
        )

        fraud_count = int(part["isFraud"].sum())

        rows.append({
            "group": str(group_name),
            "transactions": len(part),
            "fraud_transactions": fraud_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(fraud_count, 1),
            "alert_rate": float(
                part[f"{prefix}_prediction"].mean()
            ),
        })

    return pd.DataFrame(rows)


def main():
    print("\n========== FRAUD ERROR ANALYSIS ==========")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest and verify final test evaluation.

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest["final_test"]["status"] != "COMPLETED":
        raise ValueError("Final test evaluation is not completed.")

    verify_hash(
        NEURAL_MODEL_PATH,
        manifest["selected_model"]["sha256"],
    )

    verify_hash(
        BASELINE_MODEL_PATH,
        manifest["baseline_model"]["sha256"],
    )

    # 2. Load original and processed test data.

    print("\nLoading test datasets...")

    raw = pd.read_parquet(RAW_TEST_PATH)
    features = pd.read_parquet(FEATURE_TEST_PATH)

    if len(raw) != len(features):
        raise ValueError("Raw and processed test row counts differ.")

    y_raw = raw["isFraud"].to_numpy(dtype=np.int8)
    y_features = features["isFraud"].to_numpy(dtype=np.int8)

    if not np.array_equal(y_raw, y_features):
        raise ValueError(
            "Raw and processed test targets/order do not match."
        )

    print(f"Verified test rows: {len(raw):,}")

    # 3. Load models and frozen thresholds.

    neural_threshold = manifest[
        "decision_policy"
    ]["frozen_probability_threshold"]

    baseline_threshold = manifest[
        "baseline_decision_policy"
    ]["frozen_probability_threshold"]

    feature_cols = manifest["feature_columns"]

    X = features[feature_cols].to_numpy(dtype=np.float32)

    neural_model = load_neural_model(
        input_features=len(feature_cols)
    )

    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    # 4. Generate predictions.

    print("\nGenerating predictions...")

    neural_scores = neural_predictions(neural_model, X)

    baseline_scores = baseline_model.predict_proba(X)[:, 1]

    # 5. Build analysis dataframe without customer IDs.

    analysis = raw[
        ["step", "type", "amount", "isFraud"]
    ].copy()

    analysis["neural_score"] = neural_scores
    analysis["neural_prediction"] = (
        neural_scores >= neural_threshold
    ).astype(np.int8)

    analysis["baseline_score"] = baseline_scores
    analysis["baseline_prediction"] = (
        baseline_scores >= baseline_threshold
    ).astype(np.int8)

    analysis = classification_columns(
        analysis,
        "neural",
    )

    analysis = classification_columns(
        analysis,
        "baseline",
    )

    # 6. Overall error summary.

    print("\nCreating error summary...")

    neural_errors = (
        analysis["neural_error"].value_counts().to_dict()
    )

    baseline_errors = (
        analysis["baseline_error"].value_counts().to_dict()
    )

    error_summary = {
        "test_transactions": int(len(analysis)),
        "neural_network": neural_errors,
        "logistic_regression": baseline_errors,
        "analysis_policy": (
            "Descriptive analysis only. "
            "Do not retune using this test set."
        ),
    }

    with open(
        OUTPUT_DIR / "error_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(error_summary, file, indent=4)

    # 7. Analyze transaction types.

    print("Analyzing transaction types...")

    type_results = group_analysis(
        analysis,
        "type",
    )

    type_results.to_csv(
        OUTPUT_DIR / "performance_by_transaction_type.csv",
        index=False,
    )

    # 8. Analyze transaction amount bands.

    print("Analyzing transaction amounts...")

    analysis["amount_band"] = pd.qcut(
        analysis["amount"],
        q=5,
        duplicates="drop",
    )

    amount_results = group_analysis(
        analysis,
        "amount_band",
    )

    amount_results.to_csv(
        OUTPUT_DIR / "performance_by_amount_band.csv",
        index=False,
    )

    # 9. Analyze temporal periods.

    print("Analyzing temporal periods...")

    analysis["time_period"] = pd.qcut(
        analysis["step"],
        q=5,
        duplicates="drop",
    )

    time_results = group_analysis(
        analysis,
        "time_period",
    )

    time_results.to_csv(
        OUTPUT_DIR / "performance_by_time_period.csv",
        index=False,
    )

    # 10. Export false positives and false negatives.

    print("Exporting individual error cases...")

    false_positives = analysis[
        analysis["neural_error"] == "False Positive"
    ].sort_values(
        "neural_score",
        ascending=False,
    )

    false_negatives = analysis[
        analysis["neural_error"] == "False Negative"
    ].sort_values(
        "neural_score",
        ascending=True,
    )

    false_positives.to_csv(
        OUTPUT_DIR / "neural_false_positives.csv",
        index=False,
    )

    false_negatives.to_csv(
        OUTPUT_DIR / "neural_false_negatives.csv",
        index=False,
    )

    # 11. Export highest-confidence error examples.

    false_positives.head(25).to_csv(
        OUTPUT_DIR / "top_25_false_positives.csv",
        index=False,
    )

    false_negatives.head(25).to_csv(
        OUTPUT_DIR / "top_25_missed_frauds.csv",
        index=False,
    )

    # 12. Print results.

    print("\n========== ERROR SUMMARY ==========")

    print("\nNeural Network:")
    for name, count in neural_errors.items():
        print(f"{name}: {count:,}")

    print("\nLogistic Regression:")
    for name, count in baseline_errors.items():
        print(f"{name}: {count:,}")

    print("\nPerformance by transaction type:")
    print(type_results.to_string(index=False))

    print("\nPerformance by amount band:")
    print(amount_results.to_string(index=False))

    print("\nPerformance by time period:")
    print(time_results.to_string(index=False))

    print("\nReports saved in:")
    print(OUTPUT_DIR)

    print("\n========== ERROR ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
