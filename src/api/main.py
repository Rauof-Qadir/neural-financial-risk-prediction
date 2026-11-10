
import logging
from src.monitoring.monitor import prediction_monitor
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.schemas import (
    TransactionInput,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)

from src.api.service import (
    risk_service,
    DEPLOYMENT_MODE,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Neural Financial Risk Prediction API",
    description=(
        "Frozen PyTorch fraud-risk scoring service. "
        "Shadow-only mode: scores are advisory and do not "
        "automatically approve, decline or block transactions."
    ),
    version="1.0.0",
)


@app.get(
    "/",
    tags=["System"],
)
def root():
    return {
        "project": "Neural Financial Risk Prediction System",
        "service": "Financial Risk Scoring API",
        "status": "running",
        "deployment_mode": DEPLOYMENT_MODE,
        "documentation": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "deployment_mode": DEPLOYMENT_MODE,
    }


@app.get(
    "/model-info",
    tags=["Model"],
)
def model_info():

    return risk_service.get_model_info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(transaction: TransactionInput):

    result = risk_service.predict(transaction)

    return result


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(request: BatchPredictionRequest):

    batch_predictions = []

    for transaction in request.transactions:
        result = risk_service.predict(transaction)
        batch_predictions.append(result)

    return {
        "batch_request_id": __import__("uuid").uuid4().hex,
        "count": len(batch_predictions),
        "predictions": batch_predictions,
    }


@app.get(
    "/deployment-policy",
    tags=["Governance"],
)
def deployment_policy():

    return {
        "deployment_mode": DEPLOYMENT_MODE,
        "autonomous_transaction_actions": False,
        "automatic_approval": False,
        "automatic_decline": False,
        "automatic_blocking": False,
        "risk_alert_action": "Human review only",
        "frozen_threshold": risk_service.threshold,
        "threshold_retuning_enabled": False,
        "final_test_access_enabled": False,
    }


@app.get(
    "/monitoring/summary",
    tags=["Monitoring"],
)
def monitoring_summary():

    return prediction_monitor.get_summary()


@app.get(
    "/monitoring/drift",
    tags=["Monitoring"],
)
def monitoring_drift():

    return prediction_monitor.get_drift()
