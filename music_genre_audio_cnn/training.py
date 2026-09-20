"""Model-agnostic PyTorch training with validation-selected checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_evaluator: Callable[[nn.Module, torch.device], tuple[float, float]],
    epochs: int,
    model_path: Path,
    checkpoint_metadata: dict[str, object],
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    early_stopping_patience: int = 20,
) -> tuple[nn.Module, nn.Module, torch.device, dict[str, list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_accuracy = -1.0
    best_loss = float("inf")
    remaining_patience = early_stopping_patience

    for epoch in range(epochs):
        model.train()
        total_loss = total_correct = total_samples = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        validation_loss, validation_accuracy = validation_evaluator(model, device)
        history["loss"].append(total_loss / total_samples)
        history["accuracy"].append(total_correct / total_samples)
        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)
        scheduler.step(validation_loss)
        print(f"Epoch {epoch + 1}/{epochs}: loss={history['loss'][-1]:.4f}, val_accuracy={validation_accuracy:.4f}")

        improved = validation_accuracy > best_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_loss - 1e-4
        )
        if improved:
            best_accuracy, best_loss, remaining_patience = validation_accuracy, validation_loss, early_stopping_patience
            torch.save({"model_state_dict": model.state_dict(), **checkpoint_metadata}, model_path)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model_state_dict"])
    return model, criterion, device, history
