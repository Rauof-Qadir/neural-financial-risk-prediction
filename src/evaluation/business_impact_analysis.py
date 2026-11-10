
import hashlib
import json

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
COST_CONFIG_PATH = ROOT / "config/business_cost_scenarios.json"

NEURAL_MODEL_PATH = ROOT / "models/neural_network/financial_risk_mlp.pt"
BASELINE_MODEL_PATH = ROOT / "models/baseline/logistic_regression_v2.joblib"

OUTPUT_DIR = ROOT / "reports/business_impact"


def calculate_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_hash(path, expected_hash):
    actual_hash = calculate_sha256(path)

    if actual_hash != expected_hash:
        raise ValueError(
            f"Model artifact changed after freeze: {path.name}"
        )


def calculate_error_details(y, predictions, amounts):
    false_positive_mask = (y == 0) & (predictions == 1)
    false_negative_mask = (y == 1) & (predictions == 0)

    true_positive_mask = (y == 1) & (predictions == 1)

    return {
        "true_positives": int(true_positive_mask.sum()),
        "false_positives": int(false_positive_mask.sum()),
        "false_negatives": int(false_negative_mask.sum()),
        "total_alerts": int(predictions.sum()),

        "false_positive_transaction_amount": float(
            amounts[false_positive_mask].sum()
        ),

        "missed_fraud_transaction_amount": float(
            amounts[false_negative_mask].sum()
        ),

        "detected_fraud_transaction_amount": float(
            amounts[true_positive_mask].sum()
        ),

        "false_positive_rate": float(
            false_positive_mask.sum() / max((y == 0).sum(), 1)
        ),

        "missed_fraud_count": int(false_negative_mask.sum()),
    }


def calculate_scenario_cost(details, fp_cost, fn_cost):
    fp_count = details["false_positives"]
    fn_count = details["false_negatives"]

    fp_total_cost = fp_count * fp_cost
    fn_total_cost = fn_count * fn_cost

    return {
        "false_positive_cost": float(fp_total_cost),
        "false_negative_cost": float(fn_total_cost),
        "total_estimated_cost": float(
            fp_total_cost + fn_total_cost
        ),
    }


def main():
    print("\n========== BUSINESS IMPACT ANALYSIS ==========")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen experiment manifest.

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest["final_test"]["status"] != "COMPLETED":
        raise ValueError("Final test evaluation must be completed first.")

    # 2. Verify frozen models.

    print("\nVerifying frozen models...")

    verify_hash(
        NEURAL_MODEL_PATH,
        manifest["selected_model"]["sha256"],
    )

    verify_hash(
        BASELINE_MODEL_PATH,
        manifest["baseline_model"]["sha256"],
    )

    print("Both model hashes verified.")

    # 3. Load cost assumptions.

    with open(COST_CONFIG_PATH, "r", encoding="utf-8") as file:
        cost_config = json.load(file)

    scenarios = cost_config["scenarios"]
    currency = cost_config["currency"]

    # 4. Load test datasets.

    print("\nLoading test datasets...")

    raw = pd.read_parquet(RAW_TEST_PATH)
    features = pd.read_parquet(FEATURE_TEST_PATH)

    if len(raw) != len(features):
        raise ValueError("Raw and feature test row counts differ.")

    y = raw["isFraud"].to_numpy(dtype=np.int8)
    feature_y = features["isFraud"].to_numpy(dtype=np.int8)

    if not np.array_equal(y, feature_y):
        raise ValueError("Test targets or row ordering do not match.")

    amounts = raw["amount"].to_numpy(dtype=np.float64)

    feature_cols = manifest["feature_columns"]
    X = features[feature_cols].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite test features detected.")

    # 5. Load frozen thresholds.

    neural_threshold = manifest[
        "decision_policy"
    ]["frozen_probability_threshold"]

    baseline_threshold = manifest[
        "baseline_decision_policy"
    ]["frozen_probability_threshold"]

    # 6. Load models.

    print("\nLoading models...")

    neural_model = load_neural_model(
        input_features=len(feature_cols)
    )

    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    # 7. Generate predictions.

    print("\nGenerating predictions...")

    neural_scores = neural_predictions(neural_model, X)

    baseline_scores = baseline_model.predict_proba(X)[:, 1]

    neural_predictions_binary = (
        neural_scores >= neural_threshold
    ).astype(np.int8)

    baseline_predictions_binary = (
        baseline_scores >= baseline_threshold
    ).astype(np.int8)

    # 8. Calculate error and exposure details.

    print("\nCalculating error and exposure details...")

    model_details = {
        "neural_network": calculate_error_details(
            y,
            neural_predictions_binary,
            amounts,
        ),

        "logistic_regression": calculate_error_details(
            y,
            baseline_predictions_binary,
            amounts,
        ),
    }

    # 9. Calculate cost scenarios.

    scenario_rows = []

    for scenario_name, assumptions in scenarios.items():
        fp_cost = float(
            assumptions["cost_per_false_positive"]
        )

        fn_cost = float(
            assumptions["cost_per_false_negative"]
        )

        if fp_cost < 0 or fn_cost < 0:
            raise ValueError("Cost assumptions cannot be negative.")

        for model_name, details in model_details.items():
            costs = calculate_scenario_cost(
                details,
                fp_cost,
                fn_cost,
            )

            scenario_rows.append({
                "scenario": scenario_name,
                "model": model_name,
                "currency": currency,
                "cost_per_false_positive": fp_cost,
                "cost_per_false_negative": fn_cost,
                **costs,
            })

    scenario_df = pd.DataFrame(scenario_rows)

    # 10. Calculate savings relative to baseline.

    savings_rows = []

    for scenario_name in scenarios:
        part = scenario_df[
            scenario_df["scenario"] == scenario_name
        ]

        neural_cost = float(
            part.loc[
                part["model"] == "neural_network",
                "total_estimated_cost",
            ].iloc[0]
        )

        baseline_cost = float(
            part.loc[
                part["model"] == "logistic_regression",
                "total_estimated_cost",
            ].iloc[0]
        )

        savings = baseline_cost - neural_cost

        savings_rows.append({
            "scenario": scenario_name,
            "baseline_estimated_cost": baseline_cost,
            "neural_estimated_cost": neural_cost,
            "estimated_savings": savings,
            "savings_percentage": (
                savings / baseline_cost * 100
                if baseline_cost > 0
                else None
            ),
        })

    savings_df = pd.DataFrame(savings_rows)

    # 11. Save CSV reports.

    scenario_df.to_csv(
        OUTPUT_DIR / "cost_scenario_comparison.csv",
        index=False,
    )

    savings_df.to_csv(
        OUTPUT_DIR / "estimated_business_savings.csv",
        index=False,
    )

    # 12. Build JSON report.

    report = {
        "project": "Neural Financial Risk Prediction System",

        "status": "COMPLETED",

        "dataset": {
            "name": "PaySim temporal test set",
            "transactions": int(len(y)),
            "fraud_transactions": int(y.sum()),
            "total_transaction_amount": float(amounts.sum()),
        },

        "assumptions": cost_config,

        "cost_methodology": (
            "Estimated operational cost equals false positives "
            "multiplied by review cost plus false negatives "
            "multiplied by assumed missed-fraud cost."
        ),

        "models": model_details,

        "scenario_comparison": savings_rows,

        "interpretation": {
            "gross_missed_fraud_amount_is_actual_loss": False,
            "costs_are_actual_bank_costs": False,
            "note": (
                "Missed-fraud transaction amount is a gross exposure "
                "proxy, not verified realized financial loss. "
                "Cost outputs depend entirely on illustrative assumptions."
            ),
        },
    }

    with open(
        OUTPUT_DIR / "business_impact_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    # 13. Display results.

    print("\n========== ERROR AND EXPOSURE SUMMARY ==========")

    for model_name, details in model_details.items():
        print(f"\n{model_name.upper()}")

        print(
            f"False positives: "
            f"{details['false_positives']:,}"
        )

        print(
            f"False negatives: "
            f"{details['false_negatives']:,}"
        )

        print(
            f"Total alerts: "
            f"{details['total_alerts']:,}"
        )

        print(
            f"Gross missed-fraud transaction amount: "
            f"{details['missed_fraud_transaction_amount']:,.2f} "
            f"{currency}"
        )

    print("\n========== COST SCENARIO COMPARISON ==========")
    print(scenario_df.to_string(index=False))

    print("\n========== ESTIMATED SAVINGS ==========")
    print(savings_df.to_string(index=False))

    print("\nReports saved:")
    print(OUTPUT_DIR)

    print("\n========== BUSINESS IMPACT ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
