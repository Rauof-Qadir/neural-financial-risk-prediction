
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TransactionType(str, Enum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class TransactionInput(BaseModel):
    """
    Raw transaction input.

    Account names, customer IDs and fraud labels are deliberately
    excluded from the scoring interface.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    step: int = Field(gt=0, description="Transaction time step")
    transaction_type: TransactionType = Field(alias="type")

    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: List[TransactionInput] = Field(
        min_length=1,
        max_length=500,
    )


class PredictionResponse(BaseModel):
    request_id: str
    risk_probability: float
    risk_decision: str
    threshold: float
    model_version: str
    model_sha256: str
    deployment_mode: str
    shadow_only: bool


class BatchPredictionResponse(BaseModel):
    batch_request_id: str
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    deployment_mode: str
