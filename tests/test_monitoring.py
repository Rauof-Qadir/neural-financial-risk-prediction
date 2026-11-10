
import pytest
import json

from src.monitoring.monitor import (
    PredictionMonitor,
    NUMERIC_FEATURES,
    TRANSACTION_TYPES,
    MIN_PSI_SAMPLE_SIZE,
    calculate_psi,
    psi_status,
)




def test_psi_equal_distributions_is_zero():

    result = calculate_psi(
        [100, 200, 300],
        [100, 200, 300],
    )

    assert result == pytest.approx(0.0)


def test_psi_detects_distribution_change():

    result = calculate_psi(
        [100, 100],
        [190, 10],
    )

    assert result > 0.25


def test_psi_rejects_different_shapes():

    with pytest.raises(ValueError):

        calculate_psi(
            [100, 200],
            [100, 200, 300],
        )


def test_psi_empty_distribution_returns_none():

    result = calculate_psi(
        [0, 0],
        [100, 200],
    )

    assert result is None


def test_psi_status_low():

    assert psi_status(0.05) == "LOW"


def test_psi_status_investigate():

    assert psi_status(0.15) == "INVESTIGATE"


def test_psi_status_escalate():

    assert psi_status(0.30) == "ESCALATE"


def test_psi_status_insufficient_data():

    assert psi_status(None) == "INSUFFICIENT_DATA"


def _create_monitor_with_events(tmp_path, event_count):
    monitor = PredictionMonitor()

    monitor.log_path = tmp_path / "predictions.jsonl"
    monitor.reference_path = tmp_path / "reference.json"

    reference = {
        "source": "test_training_reference",
        "numeric_features": {
            feature: {
                "edges": [-float("inf"), 0.0, float("inf")],
                "counts": [1000, 1000],
                "sample_size": 2000,
            }
            for feature in NUMERIC_FEATURES
        },
        "transaction_type_distribution": {
            category: 1.0 / len(TRANSACTION_TYPES)
            for category in TRANSACTION_TYPES
        },
        "transaction_type_rows": 2000,
    }

    monitor.reference_path.write_text(
        json.dumps(reference),
        encoding="utf-8",
    )

    events = []

    for index in range(event_count):
        events.append({
            "request_id": f"test-{index}",
            "transaction_type": "CASH_OUT",
            "numeric_features": {
                feature: 0.25
                for feature in NUMERIC_FEATURES
            },
            "risk_probability": 0.10,
            "risk_decision": "NO_REVIEW",
            "latency_ms": 10.0,
            "model_version": "test-model",
        })

    monitor.log_path.write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )

    return monitor


def test_drift_requires_minimum_sample_size(tmp_path):
    monitor = _create_monitor_with_events(
        tmp_path,
        event_count=9,
    )

    result = monitor.get_drift()

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["prediction_count"] == 9
    assert (
        result["minimum_required_sample_size"]
        == MIN_PSI_SAMPLE_SIZE
    )

    for feature in NUMERIC_FEATURES:
        assert result["numeric_feature_psi"][feature]["psi"] is None
        assert (
            result["numeric_feature_psi"][feature]["status"]
            == "INSUFFICIENT_DATA"
        )

    assert result["transaction_type_psi"]["psi"] is None
    assert (
        result["transaction_type_psi"]["status"]
        == "INSUFFICIENT_DATA"
    )


def test_drift_calculates_psi_at_minimum_sample_size(tmp_path):
    monitor = _create_monitor_with_events(
        tmp_path,
        event_count=MIN_PSI_SAMPLE_SIZE,
    )

    result = monitor.get_drift()

    assert result["status"] == "OK"
    assert result["prediction_count"] == MIN_PSI_SAMPLE_SIZE

    for feature in NUMERIC_FEATURES:
        feature_result = result["numeric_feature_psi"][feature]

        assert feature_result["psi"] is not None
        assert feature_result["current_sample_size"] == (
            MIN_PSI_SAMPLE_SIZE
        )

    assert result["transaction_type_psi"]["psi"] is not None


def test_read_events_returns_only_latest_limit(tmp_path):
    monitor = _create_monitor_with_events(
        tmp_path,
        event_count=10,
    )

    events = monitor.read_events(limit=3)

    assert len(events) == 3
    assert [event["request_id"] for event in events] == [
        "test-7",
        "test-8",
        "test-9",
    ]



