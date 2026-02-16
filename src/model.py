"""MLP classifier for phishing-email detection (PyTorch).

Architecture
------------
Input (TF-IDF dim)
  -> Linear(hidden1) -> ReLU -> Dropout
  -> Linear(hidden2) -> ReLU -> Dropout
  -> Linear(1)                           # raw logit for BCEWithLogitsLoss
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Two-hidden-layer MLP for binary text classification."""

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),          # single logit output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch,)``."""
        return self.net(x).squeeze(-1)
