"""Reusable data, evaluation, and reporting helpers for audio CNN experiments."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn

SEED = 42
SAMPLE_RATE = 22050
DURATION_SECONDS = 30
CHUNK_SECONDS = 3
CHUNKS_PER_TRACK = DURATION_SECONDS // CHUNK_SECONDS
HOP_LENGTH = 512
TARGET_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / HOP_LENGTH) + 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_feature_into_chunks(feature: np.ndarray) -> np.ndarray:
    """Split a frequency-by-time feature into fixed-length channel-last excerpts."""
    chunks = []
    for chunk_index in range(CHUNKS_PER_TRACK):
        start = chunk_index * TARGET_FRAMES
        chunk = feature[:, start : start + TARGET_FRAMES]
        if chunk.shape[1] < TARGET_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TARGET_FRAMES - chunk.shape[1])))
        chunks.append(chunk[:, :, np.newaxis])
    return np.asarray(chunks, dtype=np.float32)


def collect_labeled_audio(
    audio_root: Path,
    labels: Sequence[str],
    feature_loader: Callable[[str], np.ndarray],
    extensions: set[str] = AUDIO_EXTENSIONS,
) -> list[tuple[str, int]]:
    collected: list[tuple[str, int]] = []
    for label, name in enumerate(labels):
        genre_dir = audio_root / name
        if not genre_dir.is_dir():
            continue
        for path in sorted(genre_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                feature_loader(str(path))
                collected.append((str(path), label))
            except Exception as error:
                print(f"Skipping unreadable audio: {path} ({error})")
    return collected


def split_tracks(
    paths: list[str], labels: list[int], test_size: float = 0.15, validation_size: float = 0.15
) -> tuple[list[str], list[str], list[str], list[int], list[int], list[int]]:
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, random_state=SEED, stratify=labels
    )
    validation_fraction = validation_size / (1.0 - test_size)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_paths, train_labels, test_size=validation_fraction, random_state=SEED, stratify=train_labels
    )
    return train_paths, validation_paths, test_paths, train_labels, validation_labels, test_labels


def channel_last_chunks_to_tensor(chunks: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(chunks).permute(0, 3, 1, 2)


def track_probabilities(
    model: nn.Module,
    paths: Sequence[str],
    feature_loader: Callable[[str], np.ndarray],
    device: torch.device,
    tensor_converter: Callable[[np.ndarray], torch.Tensor] = channel_last_chunks_to_tensor,
) -> torch.Tensor:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for path in paths:
            chunks = tensor_converter(feature_loader(path)).to(device)
            probabilities.append(torch.softmax(model(chunks), dim=1).mean(dim=0).cpu())
    return torch.stack(probabilities)


def evaluate_tracks(
    model: nn.Module,
    paths: Sequence[str],
    labels: Sequence[int],
    feature_loader: Callable[[str], np.ndarray],
    device: torch.device,
    tensor_converter: Callable[[np.ndarray], torch.Tensor] = channel_last_chunks_to_tensor,
) -> tuple[float, float, list[int], list[int]]:
    probabilities = track_probabilities(model, paths, feature_loader, device, tensor_converter)
    label_tensor = torch.tensor(labels)
    loss = -torch.log(probabilities[torch.arange(len(labels)), label_tensor].clamp_min(1e-8)).mean()
    predictions = probabilities.argmax(dim=1)
    accuracy = (predictions == label_tensor).float().mean()
    return loss.item(), accuracy.item(), list(labels), predictions.tolist()


def save_training_curves(history: dict[str, list[float]], path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(history["accuracy"], label="train accuracy")
    axes[0].plot(history["val_accuracy"], label="validation accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].plot(history["loss"], label="train loss")
    axes[1].plot(history["val_loss"], label="validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_confusion_matrix(true_labels: Sequence[int], predicted_labels: Sequence[int], labels: Sequence[str], path: Path) -> None:
    ConfusionMatrixDisplay(confusion_matrix(true_labels, predicted_labels), display_labels=labels).plot(
        xticks_rotation=45, cmap="Blues"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
