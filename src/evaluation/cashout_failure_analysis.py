
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data/interim/test.parquet"
FEATURE_PATH = ROOT / "data/processed/test_features_v2.parquet"
MANIFEST_PATH = ROOT / "reports/model_freeze/model_freeze_manifest.json"

MODEL_PATH = ROOT / "models/neural_network/financial_risk_mlp.pt"

OUTPUT_DIR = ROOT / "reports/cashout_failure_analysis"


def file_hash(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def describe_group(df, group_name):
    """Generate descriptive statistics for one fraud group."""

    result = {
        "group": group_name,
        "transactions": int(len(df)),
    }

    numeric_columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "neural_score",
        "origin_balance_error",
        "destination_balance_error",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            continue

        values = df[column].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(values) == 0:
            continue

        result[f"{column}_mean"] = float(values.mean())
        result[f"{column}_median"] = float(values.median())
        result[f"{column}_p90"] = float(values.quantile(0.90))

    return result


def main():

    print("\n========== CASH_OUT FAILURE ANALYSIS ==========")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load frozen experiment manifest
    # --------------------------------------------------

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest["final_test"]["status"] != "COMPLETED":
        raise ValueError("Final test evaluation is not completed.")

    # --------------------------------------------------
    # 2. Load test data
    # --------------------------------------------------

    print("\nLoading test data...")

    raw = pd.read_parquet(RAW_PATH)
    features = pd.read_parquet(FEATURE_PATH)

    if len(raw) != len(features):
        raise ValueError("Raw and feature row counts do not match.")

    if not np.array_equal(
        raw["isFraud"].to_numpy(),
        features["isFraud"].to_numpy(),
    ):
        raise ValueError("Raw and feature target ordering mismatch.")

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    missing = [
        column for column in required_columns
        if column not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Required raw columns missing: {missing}"
        )

    # --------------------------------------------------
    # 3. Verify frozen model
    # --------------------------------------------------

    print("Verifying frozen model hash...")

    if file_hash(MODEL_PATH) != manifest["selected_model"]["sha256"]:
        raise ValueError("Neural model hash mismatch.")

    # --------------------------------------------------
    # 4. Load original trained model
    # --------------------------------------------------

    from src.evaluation.model_diagnostics import (
        load_neural_model,
        neural_predictions,
    )

    X = features[
        manifest["feature_columns"]
    ].to_numpy(dtype=np.float32)

    model = load_neural_model(input_features=X.shape[1])

    scores = neural_predictions(model, X)

    threshold = manifest["decision_policy"][
        "frozen_probability_threshold"
    ]

    predictions = (scores >= threshold).astype(np.int8)

    # --------------------------------------------------
    # 5. Build diagnostic dataframe
    # --------------------------------------------------

    df = raw[required_columns].copy()

    df["neural_score"] = scores
    df["neural_prediction"] = predictions

    # Fraud that model correctly detected
    df["detected_fraud"] = (
        (df["isFraud"] == 1)
        & (df["neural_prediction"] == 1)
    )

    # Fraud that model missed
    df["missed_fraud"] = (
        (df["isFraud"] == 1)
        & (df["neural_prediction"] == 0)
    )

    # --------------------------------------------------
    # 6. Calculate balance behavior
    # --------------------------------------------------

    # Expected origin balance after transaction:
    # old origin balance - transaction amount

    df["origin_balance_error"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    # Expected destination balance after transaction:
    # old destination balance + transaction amount

    df["destination_balance_error"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    # Absolute balance inconsistencies
    df["abs_origin_balance_error"] = (
        df["origin_balance_error"].abs()
    )

    df["abs_destination_balance_error"] = (
        df["destination_balance_error"].abs()
    )

    # Relative transaction amount compared to origin balance
    df["amount_to_origin_balance_ratio"] = (
        df["amount"]
        / df["oldbalanceOrg"].replace(0, np.nan)
    )

    # --------------------------------------------------
    # 7. Isolate CASH_OUT fraud
    # --------------------------------------------------

    cashout_fraud = df[
        (df["type"] == "CASH_OUT")
        & (df["isFraud"] == 1)
    ].copy()

    detected = cashout_fraud[
        cashout_fraud["detected_fraud"]
    ].copy()

    missed = cashout_fraud[
        cashout_fraud["missed_fraud"]
    ].copy()

    print("\n========== CASH_OUT FRAUD SUMMARY ==========")

    print("Total CASH_OUT fraud:", len(cashout_fraud))
    print("Detected:", len(detected))
    print("Missed:", len(missed))

    if len(cashout_fraud) > 0:
        print(
            "Recall:",
            round(
                len(detected) / len(cashout_fraud),
                4
            )
        )

    # --------------------------------------------------
    # 8. Compare detected vs missed fraud
    # --------------------------------------------------

    comparison = pd.DataFrame([
        describe_group(detected, "Detected CASH_OUT Fraud"),
        describe_group(missed, "Missed CASH_OUT Fraud"),
    ])

    comparison.to_csv(
        OUTPUT_DIR / "detected_vs_missed_cashout.csv",
        index=False,
    )

    print("\nDetected vs missed comparison:")
    print(comparison.to_string(index=False))

    # --------------------------------------------------
    # 9. Amount-based failure segments
    # --------------------------------------------------

    bins = [
        -np.inf,
        10000,
        50000,
        100000,
        250000,
        np.inf,
    ]

    labels = [
        "Under 10K",
        "10K-50K",
        "50K-100K",
        "100K-250K",
        "Above 250K",
    ]

    cashout_fraud["amount_band"] = pd.cut(
        cashout_fraud["amount"],
        bins=bins,
        labels=labels,
    )

    amount_report = (
        cashout_fraud
        .groupby("amount_band", observed=False)
        .agg(
            fraud_count=("isFraud", "size"),
            detected=("detected_fraud", "sum"),
            missed=("missed_fraud", "sum"),
            total_amount=("amount", "sum"),
            missed_amount=(
                "amount",
                lambda x: x[
                    cashout_fraud.loc[x.index, "missed_fraud"]
                ].sum(),
            ),
        )
        .reset_index()
    )

    amount_report["recall"] = (
        amount_report["detected"]
        / amount_report["fraud_count"].replace(0, np.nan)
    )

    amount_report.to_csv(
        OUTPUT_DIR / "cashout_failure_by_amount.csv",
        index=False,
    )

    # --------------------------------------------------
    # 10. Confidence score analysis
    # --------------------------------------------------

    missed["confidence_band"] = pd.cut(
        missed["neural_score"],
        bins=[-np.inf, 0.01, 0.05, 0.10, threshold, np.inf],
        labels=[
            "Below 1%",
            "1%-5%",
            "5%-10%",
            "10%-Threshold",
            "Above Threshold",
        ],
        include_lowest=True,
        duplicates="drop",
    )

    confidence_report = (
        missed
        .groupby("confidence_band", observed=False)
        .agg(
            missed_count=("isFraud", "size"),
            average_score=("neural_score", "mean"),
            average_amount=("amount", "mean"),
            gross_amount=("amount", "sum"),
        )
        .reset_index()
    )

    confidence_report.to_csv(
        OUTPUT_DIR / "missed_fraud_confidence.csv",
        index=False,
    )

    # --------------------------------------------------
    # 11. Balance behavior comparison
    # --------------------------------------------------

    balance_columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "origin_balance_error",
        "destination_balance_error",
        "abs_origin_balance_error",
        "abs_destination_balance_error",
        "amount_to_origin_balance_ratio",
        "neural_score",
    ]

    balance_rows = []

    for name, group in [
        ("Detected", detected),
        ("Missed", missed),
    ]:

        for column in balance_columns:

            values = group[column].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            if len(values) == 0:
                continue

            balance_rows.append({
                "group": name,
                "feature": column,
                "count": len(values),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p90": float(values.quantile(0.90)),
            })

    balance_report = pd.DataFrame(balance_rows)

    balance_report.to_csv(
        OUTPUT_DIR / "cashout_balance_behavior.csv",
        index=False,
    )

    # --------------------------------------------------
    # 12. Top missed high-value frauds
    # --------------------------------------------------

    top_missed = (
        missed
        .sort_values("amount", ascending=False)
        .head(50)
    )

    top_missed.to_csv(
        OUTPUT_DIR / "top_50_cashout_missed_frauds.csv",
        index=False,
    )

    # --------------------------------------------------
    # 13. Time-based failure analysis
    # --------------------------------------------------

    cashout_fraud["step_band"] = pd.cut(
        cashout_fraud["step"],
        bins=[631, 661, 686, 688, 692, 743],
        labels=[
            "632-661",
            "662-686",
            "687-688",
            "689-692",
            "693-743",
        ],
        include_lowest=True,
    )

    time_report = (
        cashout_fraud
        .groupby("step_band", observed=False)
        .agg(
            fraud_count=("isFraud", "size"),
            detected=("detected_fraud", "sum"),
            missed=("missed_fraud", "sum"),
        )
        .reset_index()
    )

    time_report["recall"] = (
        time_report["detected"]
        / time_report["fraud_count"].replace(0, np.nan)
    )

    time_report.to_csv(
        OUTPUT_DIR / "cashout_failure_by_time.csv",
        index=False,
    )

    # --------------------------------------------------
    # 14. JSON summary
    # --------------------------------------------------

    summary = {
        "analysis": "CASH_OUT Fraud Failure Analysis",
        "test_data_usage": "Retrospective descriptive analysis only",
        "model_hash_verified": True,
        "frozen_threshold": float(threshold),
        "cashout_fraud_count": int(len(cashout_fraud)),
        "detected_cashout_fraud": int(len(detected)),
        "missed_cashout_fraud": int(len(missed)),
        "cashout_recall": (
            float(len(detected) / len(cashout_fraud))
            if len(cashout_fraud) else None
        ),
        "missed_cashout_gross_amount": float(
            missed["amount"].sum()
        ),
        "missed_cashout_average_amount": (
            float(missed["amount"].mean())
            if len(missed) else None
        ),
        "detected_cashout_average_amount": (
            float(detected["amount"].mean())
            if len(detected) else None
        ),
        "interpretation": (
            "Descriptive analysis of frozen test predictions. "
            "No model retraining or threshold tuning performed. "
            "Transaction amounts are exposure proxies, not verified losses."
        ),
    }

    with open(
        OUTPUT_DIR / "cashout_failure_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== AMOUNT SEGMENTS ==========")
    print(amount_report.to_string(index=False))

    print("\n========== CONFIDENCE ANALYSIS ==========")
    print(confidence_report.to_string(index=False))

    print("\n========== TIME ANALYSIS ==========")
    print(time_report.to_string(index=False))

    print("\n========== FINAL SUMMARY ==========")
    print(json.dumps(summary, indent=4))

    print("\nReports saved to:")
    print(OUTPUT_DIR)

    print("\n========== ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
