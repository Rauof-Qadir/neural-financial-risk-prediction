
import json
import logging
import threading

from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = ROOT / "logs" / "monitoring" / "predictions.jsonl"

REFERENCE_PATH = (
    ROOT
    / "reports"
    / "monitoring"
    / "training_reference_profile.json"
)

LOCK = threading.Lock()

NUMERIC_FEATURES = [
    "step",
    "log_amount",
    "log_oldbalanceOrg",
    "log_oldbalanceDest",
]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

PSI_EPSILON = 1e-6

# Initial safeguard: do not calculate PSI on tiny live samples.
# This should be calibrated using real transaction volumes before deployment.
MIN_PSI_SAMPLE_SIZE = 500


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def calculate_psi(reference_counts, current_counts):

    reference = np.asarray(
        reference_counts,
        dtype=np.float64,
    )

    current = np.asarray(
        current_counts,
        dtype=np.float64,
    )

    if reference.shape != current.shape:
        raise ValueError(
            "Reference and current distributions differ in shape."
        )

    if reference.sum() == 0 or current.sum() == 0:
        return None

    reference_pct = reference / reference.sum()
    current_pct = current / current.sum()

    reference_pct = np.clip(
        reference_pct,
        PSI_EPSILON,
        None,
    )

    current_pct = np.clip(
        current_pct,
        PSI_EPSILON,
        None,
    )

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


def psi_status(value):

    if value is None:
        return "INSUFFICIENT_DATA"

    if value < 0.10:
        return "LOW"

    if value <= 0.25:
        return "INVESTIGATE"

    return "ESCALATE"


class PredictionMonitor:

    def __init__(self):

        self.log_path = LOG_PATH
        self.reference_path = REFERENCE_PATH

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def record_prediction(
        self,
        prediction,
        features,
        transaction_type,
        latency_ms,
    ):

        feature_values = {
            name: float(features[index])
            for index, name in enumerate(NUMERIC_FEATURES)
        }

        event = {
            "timestamp": utc_now(),
            "request_id": prediction["request_id"],
            "model_version": prediction["model_version"],
            "model_sha256": prediction.get("model_sha256"),
            "feature_schema_version": "v2",
            "threshold": float(prediction["threshold"]),
            "risk_probability": float(
                prediction["risk_probability"]
            ),
            "risk_decision": prediction["risk_decision"],
            "transaction_type": transaction_type,
            "numeric_features": feature_values,
            "latency_ms": float(latency_ms),
            "deployment_mode": prediction["deployment_mode"],
        }

        line = json.dumps(
            event,
            separators=(",", ":"),
        )

        with LOCK:

            with self.log_path.open(
                "a",
                encoding="utf-8",
            ) as file:

                file.write(line + "\n")

    def read_events(self, limit=100_000):

        if not self.log_path.exists():
            return []

        if limit <= 0:
            return []

        # Keep only the latest events instead of loading the full log.
        latest_events = deque(maxlen=limit)

        with LOCK:

            with self.log_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        latest_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed monitoring log line."
                        )

        return list(latest_events)

    def get_summary(self, limit=100_000):

        events = self.read_events(limit=limit)

        if not events:

            return {
                "status": "NO_PREDICTIONS",
                "prediction_count": 0,
                "review_count": 0,
                "review_rate": None,
                "mean_risk_probability": None,
                "mean_latency_ms": None,
                "p95_latency_ms": None,
                "model_versions": [],
            }

        probabilities = np.array(
            [
                event["risk_probability"]
                for event in events
            ],
            dtype=np.float64,
        )

        latencies = np.array(
            [
                event["latency_ms"]
                for event in events
            ],
            dtype=np.float64,
        )

        review_count = sum(
            event["risk_decision"] == "REVIEW"
            for event in events
        )

        versions = sorted({
            event["model_version"]
            for event in events
        })

        return {
            "status": "OK",
            "prediction_count": len(events),
            "review_count": int(review_count),
            "review_rate": float(
                review_count / len(events)
            ),
            "mean_risk_probability": float(
                probabilities.mean()
            ),
            "mean_latency_ms": float(
                latencies.mean()
            ),
            "p95_latency_ms": float(
                np.percentile(latencies, 95)
            ),
            "model_versions": versions,
            "window": "latest_events",
        }

    def get_drift(self, limit=100_000):

        if not self.reference_path.exists():

            return {
                "status": "REFERENCE_NOT_AVAILABLE",
                "message": (
                    "Run the training reference profile builder first."
                ),
            }

        with self.reference_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            reference = json.load(file)

        events = self.read_events(limit=limit)

        if not events:

            return {
                "status": "NO_PREDICTIONS",
                "prediction_count": 0,
                "numeric_feature_psi": {},
                "transaction_type_psi": None,
            }

        prediction_count = len(events)

        # Do not calculate or report PSI until enough live observations exist.
        if prediction_count < MIN_PSI_SAMPLE_SIZE:

            numeric_results = {}

            for feature in NUMERIC_FEATURES:

                profile = reference["numeric_features"][feature]

                numeric_results[feature] = {
                    "psi": None,
                    "status": "INSUFFICIENT_DATA",
                    "reference_sample_size": profile["sample_size"],
                    "current_sample_size": 0,
                    "minimum_required_sample_size": (
                        MIN_PSI_SAMPLE_SIZE
                    ),
                }

            return {
                "status": "INSUFFICIENT_DATA",
                "prediction_count": prediction_count,
                "minimum_required_sample_size": (
                    MIN_PSI_SAMPLE_SIZE
                ),
                "numeric_feature_psi": numeric_results,
                "transaction_type_psi": {
                    "psi": None,
                    "status": "INSUFFICIENT_DATA",
                    "current_counts": dict(
                        Counter(
                            event["transaction_type"]
                            for event in events
                        )
                    ),
                    "minimum_required_sample_size": (
                        MIN_PSI_SAMPLE_SIZE
                    ),
                },
                "interpretation": (
                    "Not enough prediction events for reliable PSI screening. "
                    "Collect more observations before interpreting drift."
                ),
            }

        numeric_results = {}

        for feature in NUMERIC_FEATURES:

            profile = reference["numeric_features"][feature]

            edges = np.asarray(
                profile["edges"],
                dtype=np.float64,
            )

            reference_counts = profile["counts"]

            current_values = np.array(
                [
                    event["numeric_features"][feature]
                    for event in events
                ],
                dtype=np.float64,
            )

            current_values = current_values[
                np.isfinite(current_values)
            ]

            if len(current_values) < MIN_PSI_SAMPLE_SIZE:

                numeric_results[feature] = {
                    "psi": None,
                    "status": "INSUFFICIENT_DATA",
                    "reference_sample_size": profile["sample_size"],
                    "current_sample_size": int(
                        len(current_values)
                    ),
                    "minimum_required_sample_size": (
                        MIN_PSI_SAMPLE_SIZE
                    ),
                }

                continue

            current_counts, _ = np.histogram(
                current_values,
                bins=edges,
            )

            psi = calculate_psi(
                reference_counts,
                current_counts,
            )

            numeric_results[feature] = {
                "psi": psi,
                "status": psi_status(psi),
                "reference_sample_size": profile["sample_size"],
                "current_sample_size": int(
                    len(current_values)
                ),
                "minimum_required_sample_size": (
                    MIN_PSI_SAMPLE_SIZE
                ),
            }

        reference_type = reference[
            "transaction_type_distribution"
        ]

        current_type_counts = Counter(
            event["transaction_type"]
            for event in events
        )

        if prediction_count < MIN_PSI_SAMPLE_SIZE:

            type_psi = None

        else:

            reference_counts = [
                reference_type[category]
                * reference["transaction_type_rows"]
                for category in TRANSACTION_TYPES
            ]

            current_counts = [
                current_type_counts.get(category, 0)
                for category in TRANSACTION_TYPES
            ]

            type_psi = calculate_psi(
                reference_counts,
                current_counts,
            )

        return {
            "status": "OK",
            "prediction_count": prediction_count,
            "minimum_required_sample_size": (
                MIN_PSI_SAMPLE_SIZE
            ),
            "reference_source": reference["source"],
            "numeric_feature_psi": numeric_results,
            "transaction_type_psi": {
                "psi": type_psi,
                "status": psi_status(type_psi),
                "current_counts": dict(current_type_counts),
                "minimum_required_sample_size": (
                    MIN_PSI_SAMPLE_SIZE
                ),
            },
            "interpretation": (
                "PSI is a screening signal, not proof of concept drift. "
                "Review sample size, data quality and business context."
            ),
        }


prediction_monitor = PredictionMonitor()
