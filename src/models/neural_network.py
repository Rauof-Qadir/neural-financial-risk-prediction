
import torch
from torch import nn


class FinancialRiskMLP(nn.Module):
    def __init__(self, input_features=9, dropout=0.20):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)
