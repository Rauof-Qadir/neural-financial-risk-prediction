
import hashlib
import json
import logging
import uuid

from pathlib import Path
import time

from src.monitoring.monitor import prediction_monitor

import joblib
import numpy as np
import torch

from src.models.neural_network import FinancialRiskMLP


logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "neural_network"
    / "financial_risk_mlp.pt"
)

SCALER_PATH = (
    ROOT_DIR
    / "models"
    / "baseline"
    / "preprocessor_scaler_v2.joblib"
)

MANIFEST_PATH = (
    ROOT_DIR
    / "reports"
    / "model_freeze"
    / "model_freeze_manifest.json"
)

METADATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "feature_metadata_v2.json"
)

EXPECTED_MODEL_HASH = (
    "ae9d3e357c2e0dcf9abe0600245eede7"
    "ebfc0198338009a5c084185f7302189b"
)

CATEGORIES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

DEPLOYMENT_MODE = "SHADOW_ONLY"


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 without loading the entire file into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class FinancialRiskService:

    def __init__(self):
        logger.info("Initializing Financial Risk Service...")

        required_files = [
            MODEL_PATH,
            SCALER_PATH,
            MANIFEST_PATH,
            METADATA_PATH,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {path}"
                )

        # --------------------------------------------------
        # 1. Verify frozen model
        # --------------------------------------------------

        self.model_hash = calculate_sha256(MODEL_PATH)

        if self.model_hash.lower() != EXPECTED_MODEL_HASH.lower():
            raise RuntimeError(
                "Frozen model hash verification failed. "
                "Model artifact may have changed."
            )

        # --------------------------------------------------
        # 2. Load freeze manifest
        # --------------------------------------------------

        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            self.manifest = json.load(file)

        selected_model = self.manifest["selected_model"]

        if selected_model["sha256"].lower() != self.model_hash.lower():
            raise RuntimeError(
                "Model hash does not match the freeze manifest."
            )

        self.threshold = float(
            self.manifest["decision_policy"][
                "frozen_probability_threshold"
            ]
        )

        if not 0 < self.threshold < 1:
            raise RuntimeError("Invalid frozen decision threshold.")

        self.features = self.manifest["feature_columns"]

        # --------------------------------------------------
        # 3. Verify feature metadata
        # --------------------------------------------------

        with METADATA_PATH.open("r", encoding="utf-8") as file:
            self.feature_metadata = json.load(file)

        metadata_features = self.feature_metadata["features"]

        if self.features != metadata_features:
            raise RuntimeError(
                "Feature order mismatch between manifest and metadata."
            )

        # --------------------------------------------------
        # 4. Load training-fitted scaler
        # --------------------------------------------------

        self.scaler = joblib.load(SCALER_PATH)

        if getattr(self.scaler, "n_features_in_", None) != 4:
            raise RuntimeError(
                "Scaler must accept exactly four numeric features."
            )

        if not np.isfinite(self.scaler.mean_).all():
            raise RuntimeError("Scaler contains non-finite means.")

        if (
            not np.isfinite(self.scaler.scale_).all()
            or (self.scaler.scale_ <= 0).any()
        ):
            raise RuntimeError("Scaler contains invalid scale values.")

        self.scaler_hash = calculate_sha256(SCALER_PATH)

        # --------------------------------------------------
        # 5. Load frozen PyTorch model
        # --------------------------------------------------

        torch.set_num_threads(1)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

        checkpoint_features = checkpoint["input_features"]

        if checkpoint_features != self.features:
            raise RuntimeError(
                "Checkpoint feature order does not match manifest."
            )

        self.model = FinancialRiskMLP(
            input_features=len(self.features),
            dropout=float(checkpoint["dropout"]),
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.best_epoch = int(checkpoint["best_epoch"])

        self.best_validation_pr_auc = float(
            checkpoint["best_validation_pr_auc"]
        )

        self.model_version = self.model_hash[:12]

        logger.info(
            "Financial Risk Service initialized successfully. "
            "Model version=%s",
            self.model_version,
        )

    # ------------------------------------------------------
    # Raw transaction -> exact training feature structure
    # ------------------------------------------------------

    def transform_transaction(self, transaction) -> np.ndarray:

        numeric_values = np.array(
            [[
                float(transaction.step),
                float(transaction.amount),
                float(transaction.oldbalanceOrg),
                float(transaction.oldbalanceDest),
            ]],
            dtype=np.float64,
        )

        if not np.isfinite(numeric_values).all():
            raise ValueError("Non-finite numeric input.")

        if (numeric_values[:, 1:] < 0).any():
            raise ValueError("Negative amount or balance.")

        transformed = numeric_values.copy()

        # Same log1p transformation as training.
        transformed[:, 1:] = np.log1p(
            transformed[:, 1:]
        )

        # Use the original training-fitted scaler.
        scaled = self.scaler.transform(
            transformed
        ).astype(np.float32)

        transaction_type = transaction.transaction_type.value

        one_hot = np.array(
            [
                float(transaction_type == category)
                for category in CATEGORIES
            ],
            dtype=np.float32,
        )

        features = np.concatenate(
            [scaled.reshape(-1), one_hot]
        ).astype(np.float32)

        if features.shape != (9,):
            raise RuntimeError("Unexpected feature vector shape.")

        if not np.isfinite(features).all():
            raise ValueError("Invalid transformed feature values.")

        return features

    # ------------------------------------------------------
    # Model inference
    # ------------------------------------------------------


    def predict(self, transaction) -> dict:

        started_at = time.perf_counter()

        features = self.transform_transaction(transaction)

        input_tensor = torch.from_numpy(
            np.array(features, dtype=np.float32, copy=True)
        ).reshape(1, -1)

        with torch.inference_mode():

            logits = self.model(input_tensor)

            probability = torch.sigmoid(logits).item()

        if not np.isfinite(probability):
            raise RuntimeError(
                "Model returned an invalid probability."
            )

        probability = float(
            np.clip(probability, 0.0, 1.0)
        )

        decision = (
            "REVIEW"
            if probability >= self.threshold
            else "NO_REVIEW"
        )

        request_id = str(uuid.uuid4())

        latency_ms = (
            time.perf_counter() - started_at
        ) * 1000

        result = {
            "request_id": request_id,
            "risk_probability": probability,
            "risk_decision": decision,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "model_sha256": self.model_hash,
            "deployment_mode": DEPLOYMENT_MODE,
            "shadow_only": True,
        }

        # Monitoring must never silently change model decisions.
        # If audit writing fails, preserve the scoring response
        # and record the monitoring failure in application logs.
        try:

            prediction_monitor.record_prediction(
                prediction=result,
                features=features,
                transaction_type=transaction.transaction_type.value,
                latency_ms=latency_ms,
            )

        except Exception:

            logger.exception(
                "Prediction monitoring write failed. "
                "request_id=%s",
                request_id,
            )

        logger.info(
            "Scoring completed: request_id=%s decision=%s latency_ms=%.3f",
            request_id,
            decision,
            latency_ms,
        )

        return result

    # ------------------------------------------------------
    # Model information
    # ------------------------------------------------------

    def get_model_info(self) -> dict:

        return {
            "project": "Neural Financial Risk Prediction System",
            "model_name": "PyTorch FinancialRiskMLP",
            "model_version": self.model_version,
            "model_sha256": self.model_hash,
            "scaler_sha256": self.scaler_hash,
            "architecture": [9, 64, 32, 1],
            "input_features": self.features,
            "best_epoch": self.best_epoch,
            "validation_pr_auc": self.best_validation_pr_auc,
            "frozen_threshold": self.threshold,
            "deployment_mode": DEPLOYMENT_MODE,
            "autonomous_decisions_enabled": False,
            "human_review_required": True,
        }


# Load once when the API service starts.
risk_service = FinancialRiskService()
