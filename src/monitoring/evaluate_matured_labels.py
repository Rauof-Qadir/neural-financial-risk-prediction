
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG = (
    ROOT / "logs" / "monitoring" / "predictions.jsonl"
)

OUTPUT_DIR = (
    ROOT / "reports" / "monitoring" / "matured_labels"
)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate predictions against matured fraud labels."
    )

    parser.add_argument(
        "--labels",
        required=True,
        help=(
            "CSV containing request_id,isFraud,"
            "transaction_type,amount"
        ),
    )

    args = parser.parse_args()

    labels_path = Path(args.labels)

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    if not PREDICTION_LOG.exists():
        raise FileNotFoundError(PREDICTION_LOG)

    labels = pd.read_csv(labels_path)

    required = {
        "request_id",
        "isFraud",
        "transaction_type",
        "amount",
    }

    missing = required - set(labels.columns)

    if missing:
        raise ValueError(
            f"Label file missing columns: {sorted(missing)}"
        )

    if labels["request_id"].duplicated().any():
        raise ValueError(
            "Duplicate request_id values in label file."
        )

    if not labels["isFraud"].isin([0, 1]).all():
        raise ValueError(
            "isFraud must contain only 0 or 1."
        )

    if not labels["transaction_type"].isin([
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]).all():
        raise ValueError(
            "Unknown transaction_type in label file."
        )

    if (
        not np.isfinite(
            labels["amount"].to_numpy(dtype=float)
        ).all()
        or (labels["amount"] < 0).any()
    ):
        raise ValueError(
            "amount must be finite and non-negative."
        )

    predictions = pd.read_json(
        PREDICTION_LOG,
        lines=True,
    )

    if predictions["request_id"].duplicated().any():
        raise ValueError(
            "Duplicate request IDs in prediction audit log."
        )

    matched = labels.merge(
        predictions[
            [
                "request_id",
                "risk_probability",
                "risk_decision",
                "threshold",
                "model_version",
            ]
        ],
        on="request_id",
        how="inner",
        validate="one_to_one",
    )

    if matched.empty:
        raise ValueError(
            "No matching prediction and label records found."
        )

    y_true = matched["isFraud"].astype(int).to_numpy()

    y_score = matched[
        "risk_probability"
    ].astype(float).to_numpy()

    y_pred = (
        matched["risk_decision"] == "REVIEW"
    ).astype(int).to_numpy()

    if not np.isfinite(y_score).all():
        raise ValueError("Invalid prediction probabilities.")

    if len(np.unique(y_true)) < 2:
        pr_auc = None
    else:
        pr_auc = float(
            average_precision_score(y_true, y_score)
        )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    missed_fraud = matched[
        (matched["isFraud"] == 1)
        & (matched["risk_decision"] == "NO_REVIEW")
    ]

    # Segment metrics for the known critical weakness.
    cashout_segment = matched[
        (matched["transaction_type"] == "CASH_OUT")
        & (matched["amount"] >= 50_000)
        & (matched["amount"] <= 250_000)
        & (matched["isFraud"] == 1)
    ]

    cashout_detected = int(
        (
            cashout_segment["risk_decision"] == "REVIEW"
        ).sum()
    )

    cashout_total = len(cashout_segment)

    cashout_recall = (
        cashout_detected / cashout_total
        if cashout_total > 0
        else None
    )

    result = {
        "evaluation_type": "matured_delayed_labels",
        "model_versions": sorted(
            matched["model_version"].dropna().unique().tolist()
        ),
        "matched_predictions": int(len(matched)),
        "unmatched_label_rows": int(
            len(labels) - len(matched)
        ),
        "fraud_count": int(y_true.sum()),
        "fraud_prevalence": float(y_true.mean()),
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "missed_fraud_count": int(len(missed_fraud)),
        "missed_fraud_gross_transaction_amount": float(
            missed_fraud["amount"].sum()
        ),
        "critical_cashout_50k_250k": {
            "fraud_count": int(cashout_total),
            "detected_count": cashout_detected,
            "missed_count": int(
                cashout_total - cashout_detected
            ),
            "recall": cashout_recall,
        },
        "governance_note": (
            "Evaluation only. This report does not retrain "
            "the model or modify its frozen threshold. "
            "Gross transaction amount is not realised financial loss."
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / "matured_label_metrics.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print("=" * 65)
    print("MATURED LABEL PERFORMANCE REPORT")
    print("=" * 65)

    print(f"Matched predictions : {len(matched):,}")
    print(f"Fraud prevalence    : {y_true.mean():.4%}")
    print(f"PR-AUC              : {pr_auc}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1                  : {f1:.4f}")

    print("\nCritical CASH_OUT $50K-$250K segment")
    print(f"Fraud cases         : {cashout_total}")
    print(f"Detected            : {cashout_detected}")
    print(f"Recall              : {cashout_recall}")

    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
