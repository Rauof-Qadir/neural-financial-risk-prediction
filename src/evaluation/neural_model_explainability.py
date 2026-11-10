
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
import torch

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.neural_network import FinancialRiskMLP


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/neural_network/financial_risk_mlp.pt"

TRAIN_FEATURES_PATH = (
    ROOT / "data/processed/train_features_v2.parquet"
)

TEST_FEATURES_PATH = (
    ROOT / "data/processed/test_features_v2.parquet"
)

RAW_TEST_PATH = ROOT / "data/interim/test.parquet"

OUTPUT_DIR = ROOT / "reports/neural_model_explainability"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_THRESHOLD = 0.1444527953863144

EXPECTED_MODEL_SHA256 = (
    "ae9d3e357c2e0dcf9abe0600245eede7ebfc0198338009a5c084185f7302189b"
)

EXPECTED_TEST_FRAUDS = 1252
EXPECTED_FALSE_NEGATIVES = 212

GLOBAL_EXPLANATION_SAMPLE = 500
BACKGROUND_SAMPLE_SIZE = 128
EXPLANATION_BATCH_SIZE = 32
RANDOM_SEED = 42

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

DEVICE = torch.device("cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.set_num_threads(4)


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def log(message: str) -> None:
    print(f"[EXPLAINABILITY] {message}", flush=True)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def load_model():
    log("Verifying frozen model checkpoint...")

    actual_hash = calculate_sha256(MODEL_PATH)

    if actual_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Model hash mismatch. Expected frozen model was not found. "
            "Stopping explainability analysis."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    checkpoint_features = checkpoint["input_features"]

    if checkpoint_features != FEATURES:
        raise RuntimeError(
            f"Feature order mismatch.\n"
            f"Checkpoint: {checkpoint_features}\n"
            f"Expected: {FEATURES}"
        )

    model = FinancialRiskMLP(
        input_features=len(FEATURES),
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    log(f"Model SHA256 verified: {actual_hash}")
    log(f"Best training epoch: {checkpoint['best_epoch']}")
    log(
        "Best validation PR-AUC: "
        f"{checkpoint['best_validation_pr_auc']:.6f}"
    )

    return model, actual_hash, checkpoint


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def predict_probabilities(
    model: torch.nn.Module,
    features: np.ndarray,
) -> np.ndarray:

    model.eval()

    probabilities = []

    with torch.no_grad():
        for start in range(0, len(features), 8192):

            batch = torch.tensor(
                features[start:start + 8192],
                dtype=torch.float32,
                device=DEVICE,
            )

            logits = model(batch).squeeze(1)

            probabilities.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    return np.concatenate(probabilities)


# ============================================================
# 3. TRAINING BACKGROUND SAMPLE
# ============================================================

def load_background_sample() -> np.ndarray:

    log("Selecting training-only SHAP background sample...")

    parquet = pq.ParquetFile(TRAIN_FEATURES_PATH)

    if parquet.metadata.num_row_groups == 0:
        raise RuntimeError("Training feature file has no row groups.")

    rng = np.random.default_rng(RANDOM_SEED)

    row_group_count = parquet.metadata.num_row_groups

    selected_groups = rng.choice(
        row_group_count,
        size=min(5, row_group_count),
        replace=False,
    )

    collected = []

    for group_id in selected_groups:

        table = parquet.read_row_group(
            int(group_id),
            columns=FEATURES,
        )

        frame = table.to_pandas()

        if len(frame) == 0:
            continue

        values = frame[FEATURES].to_numpy(dtype=np.float32)

        take = min(64, len(values))

        indices = rng.choice(
            len(values),
            size=take,
            replace=False,
        )

        collected.append(values[indices])

    if not collected:
        raise RuntimeError("Could not construct SHAP background sample.")

    background = np.vstack(collected)

    if len(background) > BACKGROUND_SAMPLE_SIZE:

        indices = rng.choice(
            len(background),
            size=BACKGROUND_SAMPLE_SIZE,
            replace=False,
        )

        background = background[indices]

    log(f"Background observations: {len(background)}")

    return background


# ============================================================
# 4. LOAD TEST DATA AND RECONCILE ROWS
# ============================================================

def load_test_data():

    log("Loading held-out test features...")

    feature_frame = pd.read_parquet(TEST_FEATURES_PATH)

    raw_frame = pd.read_parquet(
        RAW_TEST_PATH,
        columns=[
            "step",
            "type",
            "amount",
            "isFraud",
        ],
    )

    if len(feature_frame) != len(raw_frame):
        raise RuntimeError(
            "Raw test and processed test row counts do not match."
        )

    y_features = feature_frame["isFraud"].to_numpy(dtype=np.int8)
    y_raw = raw_frame["isFraud"].to_numpy(dtype=np.int8)

    if not np.array_equal(y_features, y_raw):
        raise RuntimeError(
            "Raw and processed test target ordering does not match."
        )

    X = feature_frame[FEATURES].to_numpy(dtype=np.float32)
    y = y_features

    if int(y.sum()) != EXPECTED_TEST_FRAUDS:
        raise RuntimeError(
            f"Unexpected test fraud count: {int(y.sum())}"
        )

    log(f"Test observations: {len(X):,}")
    log(f"Test fraud observations: {int(y.sum()):,}")

    return X, y, raw_frame


# ============================================================
# 5. FROZEN PREDICTIONS AND FALSE NEGATIVES
# ============================================================

def identify_errors(model, X, y, raw_frame):

    log("Reproducing predictions using the frozen model...")

    probabilities = predict_probabilities(model, X)

    predictions = (
        probabilities >= FROZEN_THRESHOLD
    ).astype(np.int8)

    false_negative_mask = (
        (y == 1) & (predictions == 0)
    )

    false_positive_mask = (
        (y == 0) & (predictions == 1)
    )

    false_negative_count = int(false_negative_mask.sum())

    if false_negative_count != EXPECTED_FALSE_NEGATIVES:
        raise RuntimeError(
            "Frozen prediction reconciliation failed. "
            f"Expected {EXPECTED_FALSE_NEGATIVES} false negatives, "
            f"found {false_negative_count}."
        )

    if int(false_positive_mask.sum()) != 216:
        raise RuntimeError(
            "Frozen prediction reconciliation failed: "
            "false-positive count differs from the final evaluation."
        )

    metrics = {
        "test_rows": int(len(y)),
        "test_frauds": int(y.sum()),
        "threshold": FROZEN_THRESHOLD,
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
        "false_positives": int(false_positive_mask.sum()),
        "false_negatives": false_negative_count,
        "alerts": int(predictions.sum()),
        "warning": (
            "Retrospective reproduction of frozen test predictions. "
            "No model or threshold tuning was performed."
        ),
    }

    save_json(
        metrics,
        OUTPUT_DIR / "prediction_reconciliation.json",
    )

    log(
        f"Prediction reconciliation passed: "
        f"FP={metrics['false_positives']}, "
        f"FN={metrics['false_negatives']}"
    )

    details = raw_frame.copy()

    details["fraud_probability"] = probabilities
    details["prediction"] = predictions
    details["actual"] = y
    details["threshold"] = FROZEN_THRESHOLD

    details["error_type"] = np.select(
        [
            (y == 1) & (predictions == 0),
            (y == 0) & (predictions == 1),
        ],
        [
            "FALSE_NEGATIVE",
            "FALSE_POSITIVE",
        ],
        default="CORRECT",
    )

    return probabilities, predictions, details, false_negative_mask


# ============================================================
# 6. SHAP EXPLAINER
# ============================================================

def normalize_shap_output(values) -> np.ndarray:

    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values)

    if values.ndim == 3:

        if values.shape[-1] == 1:
            values = values[:, :, 0]

        elif values.shape[1] == 1:
            values = values[:, 0, :]

    if values.ndim != 2:
        raise RuntimeError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    if values.shape[1] != len(FEATURES):
        raise RuntimeError(
            f"Unexpected SHAP feature dimension: {values.shape}"
        )

    return values.astype(np.float64)


def explain_batches(explainer, X, explainer_type):

    all_values = []

    for start in range(0, len(X), EXPLANATION_BATCH_SIZE):

        batch = torch.tensor(
            X[start:start + EXPLANATION_BATCH_SIZE],
            dtype=torch.float32,
            device=DEVICE,
        )

        if explainer_type == "deep":
            values = explainer.shap_values(
                batch,
                check_additivity=False,
            )
        else:
            values = explainer.shap_values(
                batch,
                nsamples=100,
            )

        all_values.append(
            normalize_shap_output(values)
        )

        completed = min(
            start + EXPLANATION_BATCH_SIZE,
            len(X),
        )

        log(
            f"Explained {completed:,}/{len(X):,} observations"
        )

    return np.vstack(all_values)


def build_explainer(model, background):

    background_tensor = torch.tensor(
        background,
        dtype=torch.float32,
        device=DEVICE,
    )

    log("Initializing SHAP DeepExplainer...")

    try:
        explainer = shap.DeepExplainer(
            model,
            background_tensor,
        )

        return explainer, "DeepExplainer"

    except Exception as error:

        warnings.warn(
            f"DeepExplainer initialization failed: {error}. "
            "Trying GradientExplainer."
        )

        explainer = shap.GradientExplainer(
            model,
            background_tensor,
        )

        return explainer, "GradientExplainer"


# ============================================================
# 7. GLOBAL FEATURE IMPORTANCE
# ============================================================

def create_global_importance(
    X_global,
    shap_values,
):

    log("Creating global feature importance report...")

    rows = []

    for index, feature in enumerate(FEATURES):

        contribution = shap_values[:, index]

        rows.append({
            "feature": feature,
            "mean_absolute_shap": float(
                np.mean(np.abs(contribution))
            ),
            "mean_signed_shap": float(
                np.mean(contribution)
            ),
            "median_signed_shap": float(
                np.median(contribution)
            ),
            "positive_contribution_rate": float(
                np.mean(contribution > 0)
            ),
            "negative_contribution_rate": float(
                np.mean(contribution < 0)
            ),
        })

    importance = pd.DataFrame(rows).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        OUTPUT_DIR / "global_feature_importance.csv",
        index=False,
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_global,
        feature_names=FEATURES,
        show=False,
        max_display=len(FEATURES),
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "global_shap_summary.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_global,
        feature_names=FEATURES,
        plot_type="bar",
        show=False,
        max_display=len(FEATURES),
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "global_shap_bar.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    log("Global importance reports saved.")

    return importance


# ============================================================
# 8. FALSE-NEGATIVE FEATURE EXPLANATIONS
# ============================================================

def create_false_negative_explanations(
    false_negative_details,
    false_negative_shap,
):

    log("Building explanations for missed fraud transactions...")

    result = false_negative_details.reset_index(drop=True).copy()

    for index, feature in enumerate(FEATURES):

        result[f"shap_{feature}"] = (
            false_negative_shap[:, index]
        )

    absolute_values = np.abs(false_negative_shap)

    top_indices = np.argmax(
        absolute_values,
        axis=1,
    )

    result["top_absolute_contributor"] = [
        FEATURES[index] for index in top_indices
    ]

    result["top_contribution_value"] = [
        float(false_negative_shap[row, index])
        for row, index in enumerate(top_indices)
    ]

    result.to_csv(
        OUTPUT_DIR / "false_negative_shap_explanations.csv",
        index=False,
    )

    summary_rows = []

    for index, feature in enumerate(FEATURES):

        contribution = false_negative_shap[:, index]

        summary_rows.append({
            "feature": feature,
            "mean_absolute_shap": float(
                np.mean(np.abs(contribution))
            ),
            "mean_signed_shap": float(
                np.mean(contribution)
            ),
            "positive_contribution_rate": float(
                np.mean(contribution > 0)
            ),
            "negative_contribution_rate": float(
                np.mean(contribution < 0)
            ),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "false_negative_feature_summary.csv",
        index=False,
    )

    return result, summary


# ============================================================
# 9. CASH_OUT $50K-$250K FAILURE INVESTIGATION
# ============================================================

def create_cashout_failure_report(
    false_negative_explanations,
):

    log("Investigating missed CASH_OUT frauds in $50K-$250K...")

    cashout = false_negative_explanations[
        (false_negative_explanations["type"] == "CASH_OUT")
        & (
            false_negative_explanations["amount"] >= 50_000
        )
        & (
            false_negative_explanations["amount"] < 250_000
        )
    ].copy()

    if len(cashout) == 0:

        log("No missed CASH_OUT transactions in this amount band.")

        return {
            "missed_cashout_count": 0,
            "amount_band": "$50,000 to less than $250,000",
        }

    cashout.to_csv(
        OUTPUT_DIR / "cashout_50k_250k_missed_fraud_explanations.csv",
        index=False,
    )

    rows = []

    for feature in FEATURES:

        column = f"shap_{feature}"

        values = cashout[column].to_numpy()

        rows.append({
            "feature": feature,
            "mean_absolute_shap": float(
                np.mean(np.abs(values))
            ),
            "mean_signed_shap": float(
                np.mean(values)
            ),
            "positive_contribution_rate": float(
                np.mean(values > 0)
            ),
            "negative_contribution_rate": float(
                np.mean(values < 0)
            ),
        })

    segment_importance = pd.DataFrame(rows).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    segment_importance.to_csv(
        OUTPUT_DIR / "cashout_50k_250k_feature_summary.csv",
        index=False,
    )

    summary = {
        "segment": "CASH_OUT fraud, $50,000 to less than $250,000",
        "missed_fraud_count": int(len(cashout)),
        "mean_amount": float(cashout["amount"].mean()),
        "median_amount": float(cashout["amount"].median()),
        "mean_fraud_probability": float(
            cashout["fraud_probability"].mean()
        ),
        "frozen_threshold": FROZEN_THRESHOLD,
        "top_features_by_mean_absolute_shap": (
            segment_importance.head(5).to_dict("records")
        ),
        "interpretation_warning": (
            "SHAP describes the frozen model's prediction behavior. "
            "It does not establish causal reasons for fraud or "
            "prove that a feature caused a missed detection."
        ),
    }

    save_json(
        summary,
        OUTPUT_DIR / "cashout_50k_250k_summary.json",
    )

    return summary


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():

    log("=" * 65)
    log("NEURAL FINANCIAL RISK - MODEL EXPLAINABILITY")
    log("=" * 65)

    model, model_hash, checkpoint = load_model()

    X, y, raw_frame = load_test_data()

    (
        probabilities,
        predictions,
        details,
        false_negative_mask,
    ) = identify_errors(
        model,
        X,
        y,
        raw_frame,
    )

    background = load_background_sample()

    rng = np.random.default_rng(RANDOM_SEED)

    global_count = min(
        GLOBAL_EXPLANATION_SAMPLE,
        len(X),
    )

    global_indices = rng.choice(
        len(X),
        size=global_count,
        replace=False,
    )

    X_global = X[global_indices]

    X_false_negatives = X[false_negative_mask]

    log(f"Global explanation sample: {len(X_global)}")
    log(f"False-negative explanations: {len(X_false_negatives)}")

    explainer, explainer_type = build_explainer(
        model,
        background,
    )

    log(f"Using {explainer_type}")

    log("Calculating global SHAP values...")

    global_shap = explain_batches(
        explainer,
        X_global,
        "deep" if explainer_type == "DeepExplainer" else "gradient",
    )

    global_importance = create_global_importance(
        X_global,
        global_shap,
    )

    log("Calculating false-negative SHAP values...")

    false_negative_shap = explain_batches(
        explainer,
        X_false_negatives,
        "deep" if explainer_type == "DeepExplainer" else "gradient",
    )

    false_negative_details = details.loc[
        false_negative_mask
    ].reset_index(drop=True)

    fn_explanations, fn_summary = (
        create_false_negative_explanations(
            false_negative_details,
            false_negative_shap,
        )
    )

    cashout_summary = create_cashout_failure_report(
        fn_explanations
    )

    # --------------------------------------------------------
    # FINAL METADATA
    # --------------------------------------------------------

    metadata = {
        "project": "Neural Financial Risk Prediction System",
        "analysis": "Model explainability and feature-level failure investigation",
        "model": "Frozen PyTorch MLP",
        "model_sha256": model_hash,
        "best_training_epoch": int(checkpoint["best_epoch"]),
        "frozen_threshold": FROZEN_THRESHOLD,
        "explainer": explainer_type,
        "explanation_output": (
            "Model logits; positive SHAP values push the logit upward."
        ),
        "background_source": "Training feature dataset only",
        "background_size": int(len(background)),
        "global_explanation_sample_size": int(len(X_global)),
        "false_negative_explanation_count": int(
            len(X_false_negatives)
        ),
        "global_top_5_features": (
            global_importance.head(5).to_dict("records")
        ),
        "false_negative_top_5_features": (
            fn_summary.head(5).to_dict("records")
        ),
        "cashout_50k_250k_analysis": cashout_summary,
        "scientific_limitations": [
            "SHAP explains model behavior, not causal effects.",
            "Global explanations use a deterministic test sample.",
            "Background observations come from training data.",
            "Test explanations are retrospective and are not used for tuning.",
            "Synthetic PaySim findings are not proof of real-world bank behavior.",
            "SHAP magnitudes are on the model logit scale.",
        ],
        "threshold_retuned": False,
        "model_retrained": False,
        "final_test_report_modified": False,
    }

    save_json(
        metadata,
        OUTPUT_DIR / "explainability_summary.json",
    )

    log("=" * 65)
    log("EXPLAINABILITY ANALYSIS COMPLETED")
    log("=" * 65)

    log(f"Output directory: {OUTPUT_DIR}")

    log("\nTop global features:")

    print(
        global_importance[
            ["feature", "mean_absolute_shap", "mean_signed_shap"]
        ].head(5).to_string(index=False)
    )

    log("\nTop false-negative features:")

    print(
        fn_summary[
            ["feature", "mean_absolute_shap", "mean_signed_shap"]
        ].head(5).to_string(index=False)
    )

    log(
        "\nMissed CASH_OUT frauds in $50K-$250K: "
        f"{cashout_summary.get('missed_fraud_count', 0)}"
    )

    log("No model training or threshold tuning was performed.")


if __name__ == "__main__":
    main()
