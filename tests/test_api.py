
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import risk_service


client = TestClient(app)


VALID_TRANSACTION = {
    "step": 650,
    "type": "CASH_OUT",
    "amount": 150000.0,
    "oldbalanceOrg": 150000.0,
    "oldbalanceDest": 50000.0,
}


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["deployment_mode"] == "SHADOW_ONLY"


def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_version"] == risk_service.model_version
    assert data["frozen_threshold"] == risk_service.threshold
    assert data["autonomous_decisions_enabled"] is False


def test_single_prediction():

    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["risk_probability"] <= 1

    assert data["risk_decision"] in [
        "REVIEW",
        "NO_REVIEW",
    ]

    assert data["threshold"] == risk_service.threshold
    assert data["shadow_only"] is True
    assert data["deployment_mode"] == "SHADOW_ONLY"
    assert len(data["request_id"]) > 0


def test_prediction_uses_frozen_threshold():

    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    data = response.json()

    expected_decision = (
        "REVIEW"
        if data["risk_probability"] >= risk_service.threshold
        else "NO_REVIEW"
    )

    assert data["risk_decision"] == expected_decision


def test_negative_amount_rejected():

    payload = VALID_TRANSACTION.copy()
    payload["amount"] = -100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_negative_balance_rejected():

    payload = VALID_TRANSACTION.copy()
    payload["oldbalanceOrg"] = -100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_transaction_type_rejected():

    payload = VALID_TRANSACTION.copy()
    payload["type"] = "CRYPTO_TRANSFER"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_unknown_fields_rejected():

    payload = VALID_TRANSACTION.copy()
    payload["isFraud"] = 1

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_batch_prediction():

    response = client.post(
        "/predict/batch",
        json={
            "transactions": [
                VALID_TRANSACTION,
                {
                    **VALID_TRANSACTION,
                    "type": "TRANSFER",
                    "amount": 250000.0,
                },
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2
    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert 0 <= prediction["risk_probability"] <= 1
        assert prediction["shadow_only"] is True


def test_empty_batch_rejected():

    response = client.post(
        "/predict/batch",
        json={"transactions": []},
    )

    assert response.status_code == 422


def test_batch_over_limit_rejected():

    response = client.post(
        "/predict/batch",
        json={
            "transactions": [
                VALID_TRANSACTION
            ] * 501
        },
    )

    assert response.status_code == 422


def test_deployment_policy():

    response = client.get("/deployment-policy")

    assert response.status_code == 200

    data = response.json()

    assert data["autonomous_transaction_actions"] is False
    assert data["automatic_approval"] is False
    assert data["automatic_decline"] is False
    assert data["threshold_retuning_enabled"] is False
