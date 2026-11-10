
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from torch import nn
from torch.utils.data import IterableDataset, DataLoader

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.neural_network import FinancialRiskMLP


# ---------------- CONFIGURATION ----------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/neural_network")
REPORT_DIR = Path("reports/experiments")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

TARGET = "isFraud"
COLUMNS = FEATURES + [TARGET]

ROW_GROUP_BATCH_SIZE = 100_000
TRAIN_BATCH_SIZE = 4096

EPOCHS = 8
PATIENCE = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
FRAUD_WEIGHT = 50.0
MIN_RECALL = 0.90
RANDOM_SEED = 42


# ---------------- DEVICE ----------------

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nTraining device: {DEVICE}")

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU training.")


# ---------------- STREAMING DATASET ----------------

class ParquetBatchDataset(IterableDataset):
    """
    Streams Parquet row groups without loading
    the complete training dataset into memory.
    """

    def __init__(self, parquet_path, epoch=0):
        super().__init__()
        self.parquet_path = parquet_path
        self.epoch = epoch

    def __iter__(self):
        parquet = pq.ParquetFile(self.parquet_path)

        rng = np.random.default_rng(
            RANDOM_SEED + self.epoch
        )

        row_groups = rng.permutation(
            parquet.num_row_groups
        )

        for group_index in row_groups:
            table = parquet.read_row_group(
                int(group_index),
                columns=COLUMNS,
            )

            df = table.to_pandas()

            X = df[FEATURES].to_numpy(
                dtype=np.float32
            )
            y = df[TARGET].to_numpy(
                dtype=np.float32
            )

            # Shuffle rows within each row group.
            indices = rng.permutation(len(y))
            X = X[indices]
            y = y[indices]

            # Yield mini-batches.
            for start in range(
                0, len(y), TRAIN_BATCH_SIZE
            ):
                end = start + TRAIN_BATCH_SIZE

                x_batch = torch.from_numpy(
                    X[start:end].copy()
                )

                y_batch = torch.from_numpy(
                    y[start:end].copy()
                )

                yield x_batch, y_batch


# ---------------- VALIDATION ----------------

def load_validation():
    path = DATA_DIR / "validation_features_v2.parquet"

    table = pq.read_table(
        path,
        columns=COLUMNS,
    )

    df = table.to_pandas()

    X = df[FEATURES].to_numpy(
        dtype=np.float32
    )

    y = df[TARGET].to_numpy(
        dtype=np.int8
    )

    return X, y


@torch.no_grad()
def predict_probabilities(model, X):
    model.eval()

    probabilities = []

    for start in range(0, len(X), 8192):
        end = start + 8192

        x_batch = torch.from_numpy(
            X[start:end].copy()
        ).to(DEVICE)

        logits = model(x_batch).squeeze(1)

        probs = torch.sigmoid(logits)

        probabilities.extend(
            probs.cpu().numpy()
        )

    return np.asarray(
        probabilities,
        dtype=np.float32,
    )


# ---------------- THRESHOLD ----------------

def choose_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    eligible = np.flatnonzero(
        recall[:-1] >= MIN_RECALL
    )

    if len(eligible) == 0:
        return 0.5

    best_index = eligible[
        np.argmax(precision[eligible])
    ]

    return float(thresholds[best_index])


# ---------------- TRAINING ----------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_rows = 0

    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X).squeeze(1)

        loss = criterion(logits, y)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_rows = len(y)

        total_loss += loss.item() * batch_rows
        total_rows += batch_rows

    return total_loss / total_rows


# ---------------- MAIN ----------------

def main():
    print("\n========== PYTORCH FINANCIAL RISK MODEL ==========")

    train_path = DATA_DIR / "train_features_v2.parquet"

    X_val, y_val = load_validation()

    model = FinancialRiskMLP(
        input_features=len(FEATURES),
        dropout=0.20,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [FRAUD_WEIGHT],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_ap = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = []

    training_start = time.perf_counter()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        dataset = ParquetBatchDataset(
            train_path,
            epoch=epoch,
        )

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
        )

        epoch_start = time.perf_counter()

        train_loss = train_epoch(
            model,
            loader,
            optimizer,
            criterion,
        )

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        probabilities = predict_probabilities(
            model,
            X_val,
        )

        val_ap = average_precision_score(
            y_val,
            probabilities,
        )

        val_roc = roc_auc_score(
            y_val,
            probabilities,
        )

        print(f"Training loss: {train_loss:.6f}")
        print(f"Validation PR-AUC: {val_ap:.6f}")
        print(f"Validation ROC-AUC: {val_roc:.6f}")
        print(f"Epoch time: {epoch_seconds:.2f} sec")

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "validation_pr_auc": float(val_ap),
            "validation_roc_auc": float(val_roc),
            "epoch_seconds": float(epoch_seconds),
        })

        if val_ap > best_ap + 0.0001:
            best_ap = val_ap
            best_epoch = epoch + 1
            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print("New best model checkpoint!")

        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: "
                f"{patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    training_seconds = (
        time.perf_counter() - training_start
    )

    if best_state is None:
        raise RuntimeError("No valid model checkpoint created.")

    model.load_state_dict(best_state)
    model.eval()

    model_path = MODEL_DIR / "financial_risk_mlp.pt"

    torch.save(
        {
            "model_state_dict": best_state,
            "input_features": FEATURES,
            "best_epoch": best_epoch,
            "best_validation_pr_auc": best_ap,
            "dropout": 0.20,
        },
        model_path,
    )

    # Final validation evaluation using best checkpoint.
    probabilities = predict_probabilities(
        model,
        X_val,
    )

    threshold = choose_threshold(
        y_val,
        probabilities,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    report = {
        "model": "PyTorch FinancialRiskMLP",
        "architecture": [9, 64, 32, 1],
        "dropout": 0.20,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "fraud_pos_weight": FRAUD_WEIGHT,
        "batch_size": TRAIN_BATCH_SIZE,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "validation_average_precision": float(
            average_precision_score(
                y_val, probabilities
            )
        ),
        "validation_roc_auc": float(
            roc_auc_score(
                y_val, probabilities
            )
        ),
        "validation_threshold": threshold,
        "validation_precision": float(precision),
        "validation_recall": float(recall),
        "validation_f1": float(f1),
        "validation_alert_rate": float(
            predictions.mean()
        ),
        "validation_confusion_matrix": (
            confusion_matrix(
                y_val, predictions
            ).tolist()
        ),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "probability_median": float(
            np.median(probabilities)
        ),
        "history": history,
    }

    report_path = (
        REPORT_DIR / "neural_network_results.json"
    )

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\n========== FINAL NEURAL NETWORK RESULTS ==========")

    for key, value in report.items():
        if key != "history":
            print(f"{key}: {value}")

    print("\nEpoch history:")
    for item in history:
        print(item)

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")

    print("\n========== NEURAL NETWORK COMPLETE ==========")


if __name__ == "__main__":
    main()
