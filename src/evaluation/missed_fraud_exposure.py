
import json

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data/interim/test.parquet"
FEATURE_PATH = ROOT / "data/processed/test_features_v2.parquet"

MANIFEST_PATH = ROOT / "reports/model_freeze/model_freeze_manifest.json"

NEURAL_ERRORS_PATH = (
    ROOT / "reports/error_analysis/neural_false_negatives.csv"
)

OUTPUT_DIR = ROOT / "reports/missed_fraud_exposure"


def main():
    print("\n========== MISSED FRAUD EXPOSURE INTELLIGENCE ==========")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest["final_test"]["status"] != "COMPLETED":
        raise ValueError("Final test evaluation not completed.")

    raw = pd.read_parquet(RAW_PATH)
    features = pd.read_parquet(FEATURE_PATH)

    if len(raw) != len(features):
        raise ValueError("Test dataset row counts differ.")

    if not np.array_equal(
        raw["isFraud"].to_numpy(),
        features["isFraud"].to_numpy(),
    ):
        raise ValueError("Test targets or ordering do not match.")

    # Load error exports created by the previous analysis.
    neural_errors = pd.read_csv(NEURAL_ERRORS_PATH)

    # Load model predictions from the frozen final test report.
    with open(
        ROOT / manifest["final_test"]["report"],
        "r",
        encoding="utf-8",
    ) as file:
        test_report = json.load(file)

    # Reconstruct predictions to compare individual cases.
    # Hashes and thresholds remain unchanged.
    from src.evaluation.model_diagnostics import (
        load_neural_model,
        neural_predictions,
    )
    import joblib

    neural_path = ROOT / "models/neural_network/financial_risk_mlp.pt"
    baseline_path = ROOT / "models/baseline/logistic_regression_v2.joblib"

    import hashlib

    def file_hash(path):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if file_hash(neural_path) != manifest["selected_model"]["sha256"]:
        raise ValueError("Neural model hash mismatch.")

    if file_hash(baseline_path) != manifest["baseline_model"]["sha256"]:
        raise ValueError("Baseline model hash mismatch.")

    X = features[manifest["feature_columns"]].to_numpy(
        dtype=np.float32
    )

    y = raw["isFraud"].to_numpy(dtype=np.int8)

    nn = load_neural_model(input_features=X.shape[1])
    lr = joblib.load(baseline_path)

    nn_scores = neural_predictions(nn, X)
    lr_scores = lr.predict_proba(X)[:, 1]

    nn_threshold = manifest["decision_policy"][
        "frozen_probability_threshold"
    ]

    lr_threshold = manifest["baseline_decision_policy"][
        "frozen_probability_threshold"
    ]

    nn_pred = (nn_scores >= nn_threshold).astype(np.int8)
    lr_pred = (lr_scores >= lr_threshold).astype(np.int8)

    # Identify fraud cases missed by each model.
    nn_missed = (y == 1) & (nn_pred == 0)
    lr_missed = (y == 1) & (lr_pred == 0)

    common_missed = nn_missed & lr_missed
    nn_only_missed = nn_missed & ~lr_missed
    lr_only_missed = lr_missed & ~nn_missed

    analysis = raw[
        ["step", "type", "amount", "isFraud"]
    ].copy()

    analysis["neural_score"] = nn_scores
    analysis["baseline_score"] = lr_scores

    analysis["neural_missed"] = nn_missed
    analysis["baseline_missed"] = lr_missed

    analysis["miss_category"] = np.select(
        [
            common_missed,
            nn_only_missed,
            lr_only_missed,
        ],
        [
            "Both models missed",
            "Only Neural Network missed",
            "Only Logistic Regression missed",
        ],
        default="Detected by both or not applicable",
    )

    missed = analysis[y == 1].copy()

    # Transaction amount bands for missed fraud analysis.
    missed["amount_band"] = pd.cut(
        missed["amount"],
        bins=[-np.inf, 10000, 50000, 100000, 250000, np.inf],
        labels=[
            "Under 10K",
            "10K-50K",
            "50K-100K",
            "100K-250K",
            "Above 250K",
        ],
    )

    # 1. Overall model comparison.
    comparison = []

    for name, mask in [
        ("Neural Network", nn_missed),
        ("Logistic Regression", lr_missed),
        ("Both models", common_missed),
        ("Only Neural Network", nn_only_missed),
        ("Only Logistic Regression", lr_only_missed),
    ]:
        selected = analysis[mask]

        comparison.append({
            "group": name,
            "missed_fraud_count": len(selected),
            "gross_transaction_amount": float(
                selected["amount"].sum()
            ),
            "average_transaction_amount": float(
                selected["amount"].mean()
            ) if len(selected) else 0.0,
            "maximum_transaction_amount": float(
                selected["amount"].max()
            ) if len(selected) else 0.0,
        })

    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        OUTPUT_DIR / "missed_fraud_model_comparison.csv",
        index=False,
    )

    # 2. Neural Network missed fraud by transaction type.
    type_report = (
        analysis[nn_missed]
        .groupby("type", observed=False)
        .agg(
            missed_count=("amount", "size"),
            gross_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
            maximum_amount=("amount", "max"),
        )
        .reset_index()
    )

    type_report.to_csv(
        OUTPUT_DIR / "neural_missed_fraud_by_type.csv",
        index=False,
    )

    # 3. Neural Network missed fraud by amount band.
    amount_report = (
        missed[missed["neural_missed"]]
        .groupby("amount_band", observed=False)
        .agg(
            missed_count=("amount", "size"),
            gross_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
        )
        .reset_index()
    )

    amount_report.to_csv(
        OUTPUT_DIR / "neural_missed_fraud_by_amount.csv",
        index=False,
    )

    # 4. Highest-value missed fraud transactions.
    high_value = (
        analysis[nn_missed]
        .sort_values("amount", ascending=False)
        .head(50)
    )

    high_value.to_csv(
        OUTPUT_DIR / "top_50_missed_fraud_exposure.csv",
        index=False,
    )

    # 5. Compare amount-based loss sensitivity.
    #
    # These are hypothetical exposure-cost rates, not actual losses.
    cost_rows = []

    for rate in [0.01, 0.05, 0.10, 0.25]:
        for model_name, mask in [
            ("Neural Network", nn_missed),
            ("Logistic Regression", lr_missed),
        ]:
            gross_exposure = float(
                analysis.loc[mask, "amount"].sum()
            )

            cost_rows.append({
                "model": model_name,
                "assumed_loss_rate": rate,
                "gross_missed_transaction_amount": gross_exposure,
                "illustrative_loss_sensitivity": (
                    gross_exposure * rate
                ),
            })

    sensitivity = pd.DataFrame(cost_rows)

    sensitivity.to_csv(
        OUTPUT_DIR / "missed_fraud_loss_sensitivity.csv",
        index=False,
    )

    # 6. JSON summary.
    summary = {
        "analysis": "Missed Fraud Exposure Intelligence",
        "test_transactions": int(len(raw)),
        "fraud_transactions": int(y.sum()),
        "neural_network_missed": int(nn_missed.sum()),
        "baseline_missed": int(lr_missed.sum()),
        "missed_by_both": int(common_missed.sum()),
        "only_neural_missed": int(nn_only_missed.sum()),
        "only_baseline_missed": int(lr_only_missed.sum()),
        "neural_gross_missed_amount": float(
            raw.loc[nn_missed, "amount"].sum()
        ),
        "baseline_gross_missed_amount": float(
            raw.loc[lr_missed, "amount"].sum()
        ),
        "interpretation": (
            "Gross transaction amounts are exposure proxies, "
            "not realized or verified bank losses. "
            "Loss sensitivity percentages are illustrative."
        ),
    }

    with open(
        OUTPUT_DIR / "exposure_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== MISSED FRAUD SUMMARY ==========")
    print(json.dumps(summary, indent=4))

    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))

    print("\nMissed fraud by transaction type:")
    print(type_report.to_string(index=False))

    print("\nMissed fraud by amount:")
    print(amount_report.to_string(index=False))

    print("\nLoss sensitivity:")
    print(sensitivity.to_string(index=False))

    print("\nReports saved:")
    print(OUTPUT_DIR)

    print("\n========== ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
