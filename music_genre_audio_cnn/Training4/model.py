"""Neural Network architectures and models for GTZAN audio feature classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from xgboost import XGBClassifier


class ResidualBlock(nn.Module):
    """Linear residual block with BatchNorm, Mish, and Dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.act1 = nn.Mish()
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.act2 = nn.Mish()
        self.dropout2 = nn.Dropout(dropout)

        self.skip = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.dropout1(self.act1(self.bn1(self.linear1(x))))
        out = self.dropout2(self.act2(self.bn2(self.linear2(out))))
        return out + residual


class DeepAudioMLP(nn.Module):
    """Deep residual neural network for tabular acoustic features."""

    def __init__(self, in_features: int = 57, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.2),
        )

        self.block1 = ResidualBlock(512, 512, dropout=0.25)
        self.block2 = ResidualBlock(512, 256, dropout=0.25)
        self.block3 = ResidualBlock(256, 128, dropout=0.20)

        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent 128-dimensional embedding for clustering / recommendation."""
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embeddings(x)
        return self.classifier(emb)


def build_xgboost_classifier(seed: int = 42) -> XGBClassifier:
    """Build tuned XGBoost classifier for GTZAN tabular audio features."""
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
