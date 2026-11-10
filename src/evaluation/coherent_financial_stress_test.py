
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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


ROOT = Path(__file__).resolve().parents[2]

TRAIN_RAW = ROOT / "data/interim/train.parquet"
VALIDATION_RAW = ROOT / "data/interim/validation.parquet"
VALIDATION_FEATURES = ROOT / "data/processed/validation_features_v2.parquet"
CHECKPOINT = ROOT / "models/neural_network/financial_risk_mlp.pt"
FREEZE_MANIFEST = ROOT / "reports/model_freeze/model_freeze_manifest.json"

OUTPUT = ROOT / "reports/coherent_financial_stress_test"
OUTPUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8192
DEVICE = torch.device("cpu")

EXPECTED_THRESHOLD = 0.1444527953863144
EXPECTED_MODEL_HASH = (
    "ae9d3e357c2e0dcf9abe0600245eede7ebfc0198338009a5c084185f7302189b"
)

MONEY_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

MODEL_RAW_FEATURES = {
    "log_amount": "amount",
    "log_oldbalanceOrg": "oldbalanceOrg",
    "log_oldbalanceDest": "oldbalanceDest",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_frozen_threshold() -> float:
    if not FREEZE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Freeze manifest not found: {FREEZE_MANIFEST}"
        )

    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))

    # The expected threshold is deliberately fixed here.
    # The manifest is checked for auditability, not modified.
    manifest_text = json.dumps(manifest).lower()

    if "completed" not in manifest_text and "frozen" not in manifest_text:
        print(
            "WARNING: Could not identify a simple frozen/completed "
            "status string in the manifest. Continuing with the "
            "explicitly frozen threshold configured in this script."
        )

    return EXPECTED_THRESHOLD


class FinancialRiskMLP(nn.Module):
    """
    Architecture reconstructed from the frozen checkpoint's
    state-dict layer names and dimensions.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def load_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model = FinancialRiskMLP().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_features = checkpoint["input_features"]

    if len(input_features) != 9:
        raise ValueError(
            f"Expected 9 model features, received {len(input_features)}"
        )

    return model, input_features, checkpoint


def calculate_training_log_statistics():
    """
    Reconstruct StandardScaler statistics from the complete raw
    training split using streaming batches.

    This does NOT fit a new scaler on validation data.
    """

    columns = [
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]

    parquet_file = pq.ParquetFile(TRAIN_RAW)

    totals = {column: 0.0 for column in columns}
    squared_totals = {column: 0.0 for column in columns}
    counts = {column: 0 for column in columns}

    print("\nCalculating training-only log feature statistics...")
    print(f"Training rows: {parquet_file.metadata.num_rows:,}")

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=100_000,
            columns=columns,
        ),
        start=1,
    ):
        frame = batch.to_pandas()

        for column in columns:
            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite training values found in {column}"
                )

            if (values < 0).any():
                raise ValueError(
                    f"Negative balance/amount found in {column}"
                )

            logged = np.log1p(values)

            totals[column] += float(np.sum(logged))
            squared_totals[column] += float(np.sum(logged * logged))
            counts[column] += len(logged)

        if batch_number % 10 == 0:
            print(f"  Processed training batches: {batch_number}")

    statistics = {}

    for column in columns:
        count = counts[column]
        mean = totals[column] / count
        variance = squared_totals[column] / count - mean**2
        variance = max(variance, 0.0)
        scale = float(np.sqrt(variance))

        if scale == 0:
            scale = 1.0

        feature_name = {
            "amount": "log_amount",
            "oldbalanceOrg": "log_oldbalanceOrg",
            "oldbalanceDest": "log_oldbalanceDest",
        }[column]

        statistics[feature_name] = {
            "mean": float(mean),
            "scale": float(scale),
            "count": int(count),
        }

    return statistics


def transform_log_feature(
    raw_values: np.ndarray,
    mean: float,
    scale: float,
) -> np.ndarray:
    raw_values = np.asarray(raw_values, dtype=np.float64)

    if not np.isfinite(raw_values).all():
        raise ValueError("Non-finite monetary values encountered.")

    if (raw_values < 0).any():
        raise ValueError("Negative monetary values encountered.")

    return (
        (np.log1p(raw_values) - mean) / scale
    ).astype(np.float32)


def model_probabilities(
    model: nn.Module,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    missing = [
        name for name in feature_names
        if name not in frame.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    matrix = frame[feature_names].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(matrix).all():
        raise ValueError("Model input contains NaN or infinity.")

    outputs = []

    with torch.inference_mode():
        for start in range(0, len(matrix), BATCH_SIZE):
            batch = torch.from_numpy(
                np.array(
                    matrix[start:start + BATCH_SIZE],
                    dtype=np.float32,
                    copy=True,
                )
            ).to(DEVICE)

            logits = model(batch)
            probabilities = torch.sigmoid(logits)

            outputs.append(
                probabilities.cpu().numpy()
            )

    return np.concatenate(outputs)


def metric_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": int(len(y_true)),
        "fraud_count": int(y_true.sum()),
        "fraud_prevalence": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "alert_rate": float(predictions.mean()),
        "alerts": int(predictions.sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def segment_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    results = {}

    for label, name in [(0, "legitimate"), (1, "fraud")]:
        mask = y_true == label
        predictions = probabilities[mask] >= threshold

        results[name] = {
            "rows": int(mask.sum()),
            "alerts": int(predictions.sum()),
            "alert_rate": (
                float(predictions.mean())
                if mask.sum() else None
            ),
            "mean_probability": (
                float(probabilities[mask].mean())
                if mask.sum() else None
            ),
            "median_probability": (
                float(np.median(probabilities[mask]))
                if mask.sum() else None
            ),
        }

    return results


def accounting_audit(raw: pd.DataFrame) -> dict:
    """
    Descriptive accounting reconciliation.

    Origin equation is assessed for CASH_OUT and TRANSFER.
    Destination equation is assessed for CASH_IN, CASH_OUT,
    and TRANSFER. These are diagnostics, not universal banking rules.
    """

    results = {}

    origin_mask = raw["type"].isin(["CASH_OUT", "TRANSFER"])
    destination_mask = raw["type"].isin(
        ["CASH_IN", "CASH_OUT", "TRANSFER"]
    )

    equations = {
        "origin_balance": (
            origin_mask,
            raw["oldbalanceOrg"]
            - raw["amount"]
            - raw["newbalanceOrig"],
        ),
        "destination_balance": (
            destination_mask,
            raw["oldbalanceDest"]
            + raw["amount"]
            - raw["newbalanceDest"],
        ),
    }

    for name, (mask, residual_series) in equations.items():
        residual = residual_series[mask].to_numpy(dtype=np.float64)
        absolute = np.abs(residual)

        results[name] = {
            "eligible_rows": int(mask.sum()),
            "exact_within_0_01": int((absolute <= 0.01).sum()),
            "within_0_01_rate": float(
                np.mean(absolute <= 0.01)
            ) if len(absolute) else None,
            "median_absolute_residual": float(
                np.median(absolute)
            ) if len(absolute) else None,
            "p95_absolute_residual": float(
                np.quantile(absolute, 0.95)
            ) if len(absolute) else None,
            "max_absolute_residual": float(
                np.max(absolute)
            ) if len(absolute) else None,
        }

    return results


def apply_scenario(
    raw: pd.DataFrame,
    base_features: pd.DataFrame,
    statistics: dict,
    factor: float,
    scenario_type: str,
) -> pd.DataFrame:
    """
    Return a new feature frame.

    coherent:
        Scale every monetary field by the same factor.

    independent_amount:
        Scale only amount, leaving account balances unchanged.
    """

    stressed_raw = raw.copy(deep=True)

    if scenario_type == "coherent":
        for column in MONEY_COLUMNS:
            stressed_raw[column] = (
                stressed_raw[column].to_numpy(dtype=np.float64)
                * factor
            )

    elif scenario_type == "independent_amount":
        stressed_raw["amount"] = (
            stressed_raw["amount"].to_numpy(dtype=np.float64)
            * factor
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")

    stressed_features = base_features.copy(deep=True)

    for feature_name, raw_column in MODEL_RAW_FEATURES.items():
        # In independent amount stress, balances remain unchanged.
        if (
            scenario_type == "independent_amount"
            and raw_column != "amount"
        ):
            continue

        stats = statistics[feature_name]

        stressed_features[feature_name] = transform_log_feature(
            stressed_raw[raw_column].to_numpy(),
            stats["mean"],
            stats["scale"],
        )

    return stressed_features


def main():
    print("=" * 72)
    print("COHERENT FINANCIAL TRANSACTION STRESS TEST")
    print("=" * 72)

    for path in [
        TRAIN_RAW,
        VALIDATION_RAW,
        VALIDATION_FEATURES,
        CHECKPOINT,
        FREEZE_MANIFEST,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    model_hash = sha256_file(CHECKPOINT)

    if model_hash != EXPECTED_MODEL_HASH:
        raise RuntimeError(
            "Frozen model hash mismatch. Stopping without evaluation.\n"
            f"Expected: {EXPECTED_MODEL_HASH}\n"
            f"Actual:   {model_hash}"
        )

    threshold = load_frozen_threshold()

    print(f"\nFrozen model hash verified: {model_hash}")
    print(f"Frozen threshold: {threshold:.12f}")
    print(f"Device: {DEVICE}")

    model, feature_names, checkpoint = load_model()

    print(f"Model input features: {feature_names}")

    statistics = calculate_training_log_statistics()

    validation_raw = pd.read_parquet(VALIDATION_RAW)
    validation_features = pd.read_parquet(VALIDATION_FEATURES)

    if len(validation_raw) != len(validation_features):
        raise ValueError(
            "Raw validation and feature validation row counts differ."
        )

    y_raw = validation_raw["isFraud"].to_numpy(dtype=np.int8)
    y_features = validation_features["isFraud"].to_numpy(dtype=np.int8)

    if not np.array_equal(y_raw, y_features):
        raise ValueError(
            "Raw and engineered validation labels/order do not match."
        )

    if not np.isin(y_raw, [0, 1]).all():
        raise ValueError("Unexpected target labels.")

    print(f"\nValidation rows: {len(validation_raw):,}")
    print(f"Validation fraud: {int(y_raw.sum()):,}")

    raw_required = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    missing_raw = [
        column for column in raw_required
        if column not in validation_raw.columns
    ]

    if missing_raw:
        raise ValueError(
            f"Missing raw validation columns: {missing_raw}"
        )

    for column in MONEY_COLUMNS:
        values = validation_raw[column].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"Invalid values in validation column: {column}"
            )

    print("\nRunning frozen-model baseline reconciliation...")

    baseline_probabilities = model_probabilities(
        model,
        validation_features,
        feature_names,
    )

    baseline_metrics = metric_summary(
        y_raw,
        baseline_probabilities,
        threshold,
    )

    baseline_segments = segment_metrics(
        y_raw,
        baseline_probabilities,
        threshold,
    )

    print(json.dumps(baseline_metrics, indent=2))

    # Protect against accidental feature/scaler reconstruction errors.
    if abs(baseline_metrics["pr_auc"] - 0.9119293) > 0.003:
        raise RuntimeError(
            "Baseline PR-AUC did not reconcile with the frozen "
            "validation benchmark. Stop and inspect feature scaling."
        )

    if baseline_metrics["false_positive"] != 606:
        raise RuntimeError(
            "Baseline FP count did not reconcile. Stop."
        )

    if baseline_metrics["false_negative"] != 118:
        raise RuntimeError(
            "Baseline FN count did not reconcile. Stop."
        )

    print("\nBaseline accounting consistency audit...")

    baseline_accounting = accounting_audit(validation_raw)

    rows = []
    scenario_details = {}

    factors = [0.80, 0.90, 1.10, 1.20]

    scenario_definitions = [
        ("coherent", factor)
        for factor in factors
    ] + [
        ("independent_amount", factor)
        for factor in factors
    ]

    baseline_alerts = (
        baseline_probabilities >= threshold
    ).astype(np.int8)

    print("\nRunning stress scenarios...")

    for scenario_type, factor in scenario_definitions:
        scenario_name = f"{scenario_type}_x{factor:.2f}"

        print(f"\nScenario: {scenario_name}")

        stressed_features = apply_scenario(
            validation_raw,
            validation_features,
            statistics,
            factor,
            scenario_type,
        )

        probabilities = model_probabilities(
            model,
            stressed_features,
            feature_names,
        )

        metrics = metric_summary(
            y_raw,
            probabilities,
            threshold,
        )

        segments = segment_metrics(
            y_raw,
            probabilities,
            threshold,
        )

        stressed_alerts = (
            probabilities >= threshold
        ).astype(np.int8)

        changed_alerts = (
            stressed_alerts != baseline_alerts
        )

        probability_change = (
            probabilities - baseline_probabilities
        )

        row = {
            "scenario": scenario_name,
            "scenario_type": scenario_type,
            "monetary_factor": factor,
            **metrics,
            "alert_flips": int(changed_alerts.sum()),
            "alert_flip_rate": float(changed_alerts.mean()),
            "new_alerts": int(
                ((baseline_alerts == 0) & (stressed_alerts == 1)).sum()
            ),
            "removed_alerts": int(
                ((baseline_alerts == 1) & (stressed_alerts == 0)).sum()
            ),
            "mean_absolute_probability_change": float(
                np.mean(np.abs(probability_change))
            ),
            "p95_absolute_probability_change": float(
                np.quantile(np.abs(probability_change), 0.95)
            ),
            "max_absolute_probability_change": float(
                np.max(np.abs(probability_change))
            ),
            "legitimate_alert_rate": (
                segments["legitimate"]["alert_rate"]
            ),
            "fraud_alert_rate": (
                segments["fraud"]["alert_rate"]
            ),
        }

        rows.append(row)

        scenario_details[scenario_name] = {
            "metrics": metrics,
            "segments": segments,
            "alert_stability": {
                "alert_flips": row["alert_flips"],
                "alert_flip_rate": row["alert_flip_rate"],
                "new_alerts": row["new_alerts"],
                "removed_alerts": row["removed_alerts"],
                "mean_absolute_probability_change": (
                    row["mean_absolute_probability_change"]
                ),
                "p95_absolute_probability_change": (
                    row["p95_absolute_probability_change"]
                ),
            },
        }

        print(
            f"PR-AUC: {metrics['pr_auc']:.6f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Alerts: {metrics['alerts']:,} | "
            f"Alert flips: {row['alert_flips']:,}"
        )

    results_frame = pd.DataFrame(rows)

    results_frame.to_csv(
        OUTPUT / "stress_scenario_comparison.csv",
        index=False,
    )

    print("\nCreating accounting consistency comparison...")

    accounting_rows = []

    for factor in [0.80, 1.20]:
        coherent_raw = validation_raw.copy(deep=True)

        for column in MONEY_COLUMNS:
            coherent_raw[column] = (
                coherent_raw[column].to_numpy(dtype=np.float64)
                * factor
            )

        audit = accounting_audit(coherent_raw)

        for equation, values in audit.items():
            accounting_rows.append({
                "scenario": f"coherent_x{factor:.2f}",
                "equation": equation,
                **values,
            })

    accounting_comparison = {
        "original_validation": baseline_accounting,
        "coherent_scenarios": accounting_rows,
        "interpretation": (
            "A common positive multiplier preserves the relative "
            "accounting equations and zero/non-zero structure. "
            "Absolute residuals scale with the multiplier, so compare "
            "relative residual behaviour as well as absolute values."
        ),
    }

    pd.DataFrame(accounting_rows).to_csv(
        OUTPUT / "accounting_consistency_stress.csv",
        index=False,
    )

    summary = {
        "status": "COMPLETED",
        "experiment": "coherent_financial_transaction_stress_test",
        "data_scope": "validation_only",
        "final_test_accessed": False,
        "model_retrained": False,
        "threshold_retuned": False,
        "frozen_threshold": threshold,
        "frozen_model_sha256": model_hash,
        "training_log_statistics": statistics,
        "validation_baseline": {
            "metrics": baseline_metrics,
            "segments": baseline_segments,
        },
        "accounting_consistency": accounting_comparison,
        "scenarios": scenario_details,
        "scenario_count": len(scenario_definitions),
        "limitations": [
            "PaySim is synthetic data, not production banking traffic.",
            "Coherent common scaling preserves accounting relationships "
            "but does not simulate every real-world economic shock.",
            "Independent amount-only stress intentionally breaks some "
            "amount-to-balance relationships.",
            "The model's response to counterfactual inputs is not proof "
            "of causal behaviour.",
            "The experiment uses the frozen validation threshold and "
            "does not estimate production performance.",
            "Transaction-level scaling is a sensitivity experiment, "
            "not a forecast of future monetary distributions.",
        ],
    }

    (OUTPUT / "coherent_stress_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("COHERENT FINANCIAL STRESS TEST COMPLETED")
    print("=" * 72)

    print(f"Output directory: {OUTPUT}")
    print("\nGenerated files:")
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file():
            print(f"  {path.name}")

    print("\nFrozen model and threshold were not changed.")
    print("Final test was not accessed.")
    print("No training or threshold tuning was performed.")


if __name__ == "__main__":
    main()
