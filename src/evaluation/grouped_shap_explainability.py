
from __future__ import annotations

import hashlib
import json
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
# 1. CONFIGURATION
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

OUTPUT_DIR = ROOT / "reports/grouped_shap_explainability"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_THRESHOLD = 0.1444527953863144

EXPECTED_MODEL_SHA256 = (
    "ae9d3e357c2e0dcf9abe0600245eede7ebfc0198338009a5c084185f7302189b"
)

EXPECTED_TEST_FRAUDS = 1252
EXPECTED_FALSE_POSITIVES = 216
EXPECTED_FALSE_NEGATIVES = 212
EXPECTED_CASHOUT_MISSED_50K_250K = 164

# The five original transaction types become one conceptual feature.
TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

ORIGINAL_FEATURES = [
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

GROUPED_FEATURES = [
    "step",
    "log_amount",
    "log_oldbalanceOrg",
    "log_oldbalanceDest",
    "transaction_type",
]

NUMERIC_FEATURES = GROUPED_FEATURES[:4]

GLOBAL_SAMPLE_SIZE = 500
BACKGROUND_SAMPLE_SIZE = 128

# Five conceptual features: minimum evaluations = 2 * 5 + 1.
# 101 evaluations provide multiple permutation paths.
MAX_EVALS = 101

RANDOM_SEED = 42
DEVICE = torch.device("cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.set_num_threads(4)


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def log(message: str) -> None:
    print(f"[GROUPED SHAP] {message}", flush=True)


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(data: dict, path: Path) -> None:

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            default=str,
        )


def load_frozen_model():

    log("Verifying frozen model hash...")

    actual_hash = sha256_file(MODEL_PATH)

    if actual_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Frozen model hash mismatch. "
            "Stopping analysis."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint["input_features"] != ORIGINAL_FEATURES:
        raise RuntimeError(
            "Checkpoint feature order does not match "
            "the expected original model features."
        )

    model = FinancialRiskMLP(
        input_features=len(ORIGINAL_FEATURES),
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    log(f"Model hash verified: {actual_hash}")
    log(f"Best training epoch: {checkpoint['best_epoch']}")

    return model, actual_hash, checkpoint


# ============================================================
# 3. GROUPED FEATURE TRANSFORMATION
# ============================================================

def original_features_to_grouped(
    frame: pd.DataFrame,
) -> np.ndarray:

    one_hot = frame[
        ORIGINAL_FEATURES[4:]
    ].to_numpy(dtype=np.float32)

    row_sums = one_hot.sum(axis=1)

    if not np.all(row_sums == 1):
        raise RuntimeError(
            "Invalid transaction one-hot encoding. "
            "Every row must contain exactly one active type."
        )

    category_codes = np.argmax(
        one_hot,
        axis=1,
    ).astype(np.float32)

    numeric = frame[
        NUMERIC_FEATURES
    ].to_numpy(dtype=np.float32)

    return np.column_stack(
        [numeric, category_codes]
    ).astype(np.float32)


def grouped_features_to_original(
    grouped_values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        grouped_values,
        dtype=np.float32,
    )

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.shape[1] != len(GROUPED_FEATURES):
        raise ValueError(
            f"Expected {len(GROUPED_FEATURES)} grouped features, "
            f"received {values.shape[1]}."
        )

    category_codes = np.rint(
        values[:, 4]
    ).astype(int)

    if np.any(category_codes < 0) or np.any(
        category_codes >= len(TRANSACTION_TYPES)
    ):
        raise RuntimeError(
            "Invalid transaction type code encountered."
        )

    one_hot = np.eye(
        len(TRANSACTION_TYPES),
        dtype=np.float32,
    )[category_codes]

    return np.column_stack(
        [values[:, :4], one_hot]
    ).astype(np.float32)


# ============================================================
# 4. MODEL WRAPPER
# ============================================================

class GroupedModelPredictor:

    """
    Accepts five conceptual features.

    Reconstructs the original nine model features
    while preserving valid one-hot transaction types.

    Returns logits, not probabilities.
    """

    def __init__(self, model):

        self.model = model
        self.model.eval()

    def __call__(self, grouped_values):

        grouped_values = np.asarray(
            grouped_values,
            dtype=np.float32,
        )

        if grouped_values.ndim == 1:
            grouped_values = grouped_values.reshape(1, -1)

        original_values = grouped_features_to_original(
            grouped_values
        )

        outputs = []

        with torch.no_grad():

            for start in range(
                0,
                len(original_values),
                8192,
            ):

                batch = torch.tensor(
                    original_values[
                        start:start + 8192
                    ],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                logits = self.model(
                    batch
                ).squeeze(1)

                outputs.append(
                    logits.cpu().numpy()
                )

        return np.concatenate(outputs)


# ============================================================
# 5. TRAINING BACKGROUND SAMPLE
# ============================================================

def load_training_background():

    log("Loading training-only background sample...")

    parquet = pq.ParquetFile(
        TRAIN_FEATURES_PATH
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    group_count = parquet.num_row_groups

    selected_groups = rng.choice(
        group_count,
        size=min(5, group_count),
        replace=False,
    )

    samples = []

    for group_id in selected_groups:

        table = parquet.read_row_group(
            int(group_id),
            columns=ORIGINAL_FEATURES,
        )

        frame = table.to_pandas()

        if frame.empty:
            continue

        grouped = original_features_to_grouped(
            frame
        )

        take = min(64, len(grouped))

        indices = rng.choice(
            len(grouped),
            size=take,
            replace=False,
        )

        samples.append(
            grouped[indices]
        )

    if not samples:
        raise RuntimeError(
            "Unable to construct training background."
        )

    background = np.vstack(samples)

    if len(background) > BACKGROUND_SAMPLE_SIZE:

        indices = rng.choice(
            len(background),
            size=BACKGROUND_SAMPLE_SIZE,
            replace=False,
        )

        background = background[indices]

    log(
        f"Training background observations: {len(background)}"
    )

    return background


# ============================================================
# 6. LOAD AND VALIDATE TEST DATA
# ============================================================

def load_test_data():

    log("Loading test feature and raw transaction data...")

    feature_frame = pd.read_parquet(
        TEST_FEATURES_PATH
    )

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
            "Raw and processed test row counts differ."
        )

    y = feature_frame[
        "isFraud"
    ].to_numpy(dtype=np.int8)

    raw_y = raw_frame[
        "isFraud"
    ].to_numpy(dtype=np.int8)

    if not np.array_equal(y, raw_y):
        raise RuntimeError(
            "Raw and processed target ordering differs."
        )

    X_grouped = original_features_to_grouped(
        feature_frame
    )

    encoded_types = np.array(
        TRANSACTION_TYPES
    )[
        X_grouped[:, 4].astype(int)
    ]

    raw_types = raw_frame[
        "type"
    ].astype(str).to_numpy()

    if not np.array_equal(
        encoded_types,
        raw_types,
    ):
        raise RuntimeError(
            "Transaction type mapping does not match "
            "the raw test data."
        )

    if int(y.sum()) != EXPECTED_TEST_FRAUDS:
        raise RuntimeError(
            "Unexpected test fraud count."
        )

    log(f"Test rows: {len(y):,}")
    log(f"Test frauds: {int(y.sum()):,}")

    return X_grouped, y, raw_frame


# ============================================================
# 7. PREDICTION RECONCILIATION
# ============================================================

def reconcile_predictions(
    predictor,
    X_grouped,
    y,
    raw_frame,
):

    log("Reproducing frozen predictions...")

    logits = predictor(X_grouped)

    probabilities = 1.0 / (
        1.0 + np.exp(
            -np.clip(logits, -50, 50)
        )
    )

    predictions = (
        probabilities >= FROZEN_THRESHOLD
    ).astype(np.int8)

    fn_mask = (
        (y == 1) & (predictions == 0)
    )

    fp_mask = (
        (y == 0) & (predictions == 1)
    )

    false_negatives = int(fn_mask.sum())
    false_positives = int(fp_mask.sum())

    if false_negatives != EXPECTED_FALSE_NEGATIVES:
        raise RuntimeError(
            f"Expected {EXPECTED_FALSE_NEGATIVES} FN, "
            f"found {false_negatives}."
        )

    if false_positives != EXPECTED_FALSE_POSITIVES:
        raise RuntimeError(
            f"Expected {EXPECTED_FALSE_POSITIVES} FP, "
            f"found {false_positives}."
        )

    details = raw_frame.copy()

    details["fraud_probability"] = probabilities
    details["prediction"] = predictions
    details["actual"] = y

    details["error_type"] = np.select(
        [
            fn_mask,
            fp_mask,
        ],
        [
            "FALSE_NEGATIVE",
            "FALSE_POSITIVE",
        ],
        default="CORRECT",
    )

    metrics = {
        "test_rows": int(len(y)),
        "test_frauds": int(y.sum()),
        "threshold": FROZEN_THRESHOLD,
        "average_precision": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
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
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "alerts": int(predictions.sum()),
        "model_retrained": False,
        "threshold_retuned": False,
    }

    save_json(
        metrics,
        OUTPUT_DIR / "prediction_reconciliation.json",
    )

    log(
        f"Reconciliation passed: FP={false_positives}, "
        f"FN={false_negatives}"
    )

    return (
        logits,
        probabilities,
        predictions,
        details,
        fn_mask,
    )


# ============================================================
# 8. BUILD GROUPED PERMUTATION SHAP EXPLAINER
# ============================================================

def build_grouped_explainer(
    predictor,
    background,
):

    log("Initializing grouped PermutationExplainer...")

    masker = shap.maskers.Independent(
        background,
        max_samples=len(background),
    )

    explainer = shap.Explainer(
        predictor,
        masker,
        algorithm="permutation",
        feature_names=GROUPED_FEATURES,
        output_names=["fraud_logit"],
    )

    return explainer


def calculate_shap(
    explainer,
    X,
    label,
):

    log(
        f"Calculating grouped SHAP values for {label}: "
        f"{len(X)} observations..."
    )

    explanation = explainer(
        X,
        max_evals=MAX_EVALS,
        batch_size=256,
    )

    values = np.asarray(
        explanation.values,
        dtype=np.float64,
    )

    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]

    if values.shape != (
        len(X),
        len(GROUPED_FEATURES),
    ):
        raise RuntimeError(
            f"Unexpected grouped SHAP shape: {values.shape}"
        )

    return explanation, values


# ============================================================
# 9. ADDITIVITY / EXPLANATION QUALITY CHECK
# ============================================================

def calculate_additivity(
    predictor,
    X,
    explanation,
):

    predictions = predictor(X)

    base_values = np.asarray(
        explanation.base_values,
        dtype=np.float64,
    ).reshape(-1)

    shap_values = np.asarray(
        explanation.values,
        dtype=np.float64,
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    reconstructed = (
        base_values + shap_values.sum(axis=1)
    )

    residuals = predictions - reconstructed

    return {
        "mean_absolute_additivity_error": float(
            np.mean(np.abs(residuals))
        ),
        "max_absolute_additivity_error": float(
            np.max(np.abs(residuals))
        ),
        "median_absolute_additivity_error": float(
            np.median(np.abs(residuals))
        ),
        "output_scale": "logit",
    }


# ============================================================
# 10. FEATURE IMPORTANCE REPORTS
# ============================================================

def create_importance_report(
    shap_values,
    filename,
):

    rows = []

    for index, feature in enumerate(
        GROUPED_FEATURES
    ):

        contributions = shap_values[:, index]

        rows.append({
            "feature": feature,
            "mean_absolute_shap": float(
                np.mean(
                    np.abs(contributions)
                )
            ),
            "mean_signed_shap": float(
                np.mean(contributions)
            ),
            "median_signed_shap": float(
                np.median(contributions)
            ),
            "positive_contribution_rate": float(
                np.mean(contributions > 0)
            ),
            "negative_contribution_rate": float(
                np.mean(contributions < 0)
            ),
        })

    result = pd.DataFrame(rows).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    result.to_csv(
        OUTPUT_DIR / filename,
        index=False,
    )

    return result


# ============================================================
# 11. GLOBAL SHAP VISUALIZATIONS
# ============================================================

def create_global_plots(
    explanation,
    X,
):

    log("Creating grouped SHAP plots...")

    shap_values = np.asarray(
        explanation.values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        feature_names=GROUPED_FEATURES,
        show=False,
        max_display=len(GROUPED_FEATURES),
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "grouped_shap_summary.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        feature_names=GROUPED_FEATURES,
        plot_type="bar",
        show=False,
        max_display=len(GROUPED_FEATURES),
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "grouped_shap_bar.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 12. FALSE-NEGATIVE EXPLANATIONS
# ============================================================

def create_false_negative_report(
    fn_details,
    fn_shap,
):

    result = fn_details.reset_index(
        drop=True
    ).copy()

    for index, feature in enumerate(
        GROUPED_FEATURES
    ):

        result[f"shap_{feature}"] = (
            fn_shap[:, index]
        )

    abs_values = np.abs(fn_shap)

    top_indices = np.argmax(
        abs_values,
        axis=1,
    )

    result["top_absolute_contributor"] = [
        GROUPED_FEATURES[index]
        for index in top_indices
    ]

    result["top_contribution_value"] = [
        float(fn_shap[row, index])
        for row, index in enumerate(top_indices)
    ]

    result.to_csv(
        OUTPUT_DIR / "grouped_false_negative_explanations.csv",
        index=False,
    )

    summary = create_importance_report(
        fn_shap,
        "grouped_false_negative_feature_summary.csv",
    )

    return result, summary


# ============================================================
# 13. CASH_OUT $50K-$250K FAILURE ANALYSIS
# ============================================================

def create_cashout_segment_report(
    fn_explanations,
):

    log("Analyzing CASH_OUT missed frauds: $50K-$250K...")

    segment = fn_explanations[
        (fn_explanations["type"] == "CASH_OUT")
        & (fn_explanations["amount"] >= 50_000)
        & (fn_explanations["amount"] < 250_000)
    ].copy()

    if len(segment) != EXPECTED_CASHOUT_MISSED_50K_250K:
        raise RuntimeError(
            "Expected 164 missed CASH_OUT frauds "
            f"in the $50K-$250K segment; found {len(segment)}."
        )

    segment.to_csv(
        OUTPUT_DIR / "grouped_cashout_50k_250k_explanations.csv",
        index=False,
    )

    segment_shap = segment[
        [
            f"shap_{feature}"
            for feature in GROUPED_FEATURES
        ]
    ].to_numpy(dtype=np.float64)

    importance = create_importance_report(
        segment_shap,
        "grouped_cashout_50k_250k_feature_summary.csv",
    )

    summary = {
        "segment": "CASH_OUT fraud, $50K to less than $250K",
        "missed_fraud_count": int(len(segment)),
        "mean_amount": float(segment["amount"].mean()),
        "median_amount": float(segment["amount"].median()),
        "mean_fraud_probability": float(
            segment["fraud_probability"].mean()
        ),
        "frozen_threshold": FROZEN_THRESHOLD,
        "top_features": importance.head(5).to_dict("records"),
        "interpretation": (
            "Grouped permutation SHAP contributions explain "
            "the frozen model's logit behavior for this segment."
        ),
    }

    save_json(
        summary,
        OUTPUT_DIR / "grouped_cashout_50k_250k_summary.json",
    )

    return summary


# ============================================================
# 14. MAIN
# ============================================================

def main():

    log("=" * 70)
    log("GROUPED TRANSACTION-TYPE SHAP EXPLAINABILITY")
    log("=" * 70)

    model, model_hash, checkpoint = load_frozen_model()

    predictor = GroupedModelPredictor(model)

    X_test, y_test, raw_frame = load_test_data()

    (
        logits,
        probabilities,
        predictions,
        details,
        fn_mask,
    ) = reconcile_predictions(
        predictor,
        X_test,
        y_test,
        raw_frame,
    )

    background = load_training_background()

    # Validate that the grouped category is always a valid code.
    category_codes = X_test[:, 4].astype(int)

    if not np.isin(
        category_codes,
        np.arange(len(TRANSACTION_TYPES)),
    ).all():
        raise RuntimeError(
            "Invalid transaction category code."
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    global_size = min(
        GLOBAL_SAMPLE_SIZE,
        len(X_test),
    )

    global_indices = rng.choice(
        len(X_test),
        size=global_size,
        replace=False,
    )

    X_global = X_test[global_indices]

    X_fn = X_test[fn_mask]

    explainer = build_grouped_explainer(
        predictor,
        background,
    )

    # --------------------------------------------------------
    # GLOBAL SHAP
    # --------------------------------------------------------

    global_explanation, global_values = calculate_shap(
        explainer,
        X_global,
        "global sample",
    )

    global_importance = create_importance_report(
        global_values,
        "grouped_global_feature_importance.csv",
    )

    create_global_plots(
        global_explanation,
        X_global,
    )

    global_additivity = calculate_additivity(
        predictor,
        X_global,
        global_explanation,
    )

    # --------------------------------------------------------
    # FALSE-NEGATIVE SHAP
    # --------------------------------------------------------

    fn_explanation, fn_values = calculate_shap(
        explainer,
        X_fn,
        "all false negatives",
    )

    fn_details = details.loc[
        fn_mask
    ].reset_index(drop=True)

    fn_explanations, fn_importance = (
        create_false_negative_report(
            fn_details,
            fn_values,
        )
    )

    fn_additivity = calculate_additivity(
        predictor,
        X_fn,
        fn_explanation,
    )

    # --------------------------------------------------------
    # CASH_OUT FAILURE SEGMENT
    # --------------------------------------------------------

    cashout_summary = create_cashout_segment_report(
        fn_explanations
    )

    # --------------------------------------------------------
    # CATEGORY MAPPING
    # --------------------------------------------------------

    save_json(
        {
            "feature_names": GROUPED_FEATURES,
            "transaction_type_codes": {
                str(index): category
                for index, category in enumerate(
                    TRANSACTION_TYPES
                )
            },
            "original_one_hot_features": ORIGINAL_FEATURES[4:],
            "grouping_method": (
                "Five one-hot transaction type columns "
                "combined into one categorical feature."
            ),
        },
        OUTPUT_DIR / "feature_grouping_metadata.json",
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    summary = {
        "project": "Neural Financial Risk Prediction System",
        "analysis": "Grouped transaction-type permutation SHAP",
        "model_sha256": model_hash,
        "model_best_epoch": int(checkpoint["best_epoch"]),
        "frozen_threshold": FROZEN_THRESHOLD,
        "explainer": "SHAP PermutationExplainer",
        "conceptual_features": GROUPED_FEATURES,
        "original_model_features": ORIGINAL_FEATURES,
        "transaction_type_grouping": TRANSACTION_TYPES,
        "background_source": "Training features only",
        "background_size": int(len(background)),
        "global_sample_size": int(len(X_global)),
        "false_negative_count": int(len(X_fn)),
        "max_evals_per_observation": MAX_EVALS,
        "global_additivity_check": global_additivity,
        "false_negative_additivity_check": fn_additivity,
        "global_top_features": (
            global_importance.head(5).to_dict("records")
        ),
        "false_negative_top_features": (
            fn_importance.head(5).to_dict("records")
        ),
        "cashout_50k_250k": cashout_summary,
        "model_retrained": False,
        "threshold_retuned": False,
        "final_test_report_modified": False,
        "limitations": [
            "SHAP explains model behavior, not causality.",
            "Global explanations use a deterministic test sample.",
            "The background sample comes only from training features.",
            "Transaction type is kept as one valid categorical feature.",
            "Independent masking of numeric features can still create "
            "unrealistic combinations among correlated numeric inputs.",
            "The categorical integer code is a label, not a numeric scale.",
            "SHAP values are measured on the model logit scale.",
            "PaySim is synthetic and does not establish real-bank behavior.",
        ],
    }

    save_json(
        summary,
        OUTPUT_DIR / "grouped_shap_summary.json",
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    log("=" * 70)
    log("GROUPED SHAP ANALYSIS COMPLETED")
    log("=" * 70)

    log("\nGlobal feature importance:")

    print(
        global_importance[
            [
                "feature",
                "mean_absolute_shap",
                "mean_signed_shap",
            ]
        ].to_string(index=False)
    )

    log("\nFalse-negative feature importance:")

    print(
        fn_importance[
            [
                "feature",
                "mean_absolute_shap",
                "mean_signed_shap",
            ]
        ].to_string(index=False)
    )

    log("\nGlobal additivity check:")

    print(
        json.dumps(
            global_additivity,
            indent=4,
        )
    )

    log("\nFalse-negative additivity check:")

    print(
        json.dumps(
            fn_additivity,
            indent=4,
        )
    )

    log(
        "\nMissed CASH_OUT frauds in $50K-$250K: "
        f"{cashout_summary['missed_fraud_count']}"
    )

    log(f"\nOutput directory: {OUTPUT_DIR}")

    log("Model and threshold remained frozen.")


if __name__ == "__main__":
    main()
