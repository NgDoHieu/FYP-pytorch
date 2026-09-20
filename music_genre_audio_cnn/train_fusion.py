"""Train mel, CQT, and chroma CNNs with validation-selected track-level fusion."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from shared import (
    AUDIO_EXTENSIONS,
    CHUNKS_PER_TRACK,
    DURATION_SECONDS,
    SAMPLE_RATE,
    TARGET_FRAMES,
    collect_labeled_audio,
    evaluate_tracks,
    set_seed,
    split_feature_into_chunks,
    split_tracks,
    track_probabilities,
)
from train import GENRES, GenreCNN, load_mel
from training import train_model

CQT_BINS = 96
CHROMA_BINS = 24
FUSION_WEIGHT_STEP = 0.05


def load_cqt(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SECONDS)
    cqt = librosa.cqt(
        audio,
        sr=SAMPLE_RATE,
        hop_length=512,
        fmin=librosa.note_to_hz("C1"),
        n_bins=CQT_BINS,
        bins_per_octave=12,
    )
    cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt = np.clip((cqt + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
    return split_feature_into_chunks(cqt)


def load_chroma(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SECONDS)
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=512,
        fmin=librosa.note_to_hz("C1"),
        n_chroma=CHROMA_BINS,
        n_octaves=7,
        bins_per_octave=48,
    )
    chroma = np.clip(np.nan_to_num(chroma), 0.0, 1.0).astype(np.float32)
    return split_feature_into_chunks(chroma)


class FeatureChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        feature_loader: Callable[[str], np.ndarray],
        training: bool,
    ) -> None:
        self.training = training
        self.samples = [
            (chunk, label)
            for path, label in zip(paths, labels)
            for chunk in feature_loader(path)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature, label = self.samples[index]
        tensor = torch.from_numpy(feature).permute(2, 0, 1)
        if self.training:
            tensor = torch.clamp(tensor + torch.empty(1).uniform_(-0.08, 0.08), 0.0, 1.0)
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * 0.015, 0.0, 1.0)
            if random.random() < 0.5:
                start = random.randrange(TARGET_FRAMES - 20)
                width = random.randrange(5, 20)
                tensor[:, :, start : start + width] = 0.0
            if random.random() < 0.5:
                max_width = max(1, min(16, tensor.shape[1] // 8))
                width = random.randint(1, max_width)
                start = random.randrange(tensor.shape[1] - width + 1)
                tensor[:, start : start + width, :] = 0.0
        return tensor, torch.tensor(label, dtype=torch.long)


def evaluate_feature_tracks(
    model: torch.nn.Module,
    paths: list[str],
    labels: list[int],
    feature_loader: Callable[[str], np.ndarray],
    device: torch.device,
) -> tuple[float, float]:
    return evaluate_tracks(model, paths, labels, feature_loader, device)[:2]


def select_fusion_weights(
    probabilities_by_representation: dict[str, torch.Tensor], labels: list[int], weight_step: float
) -> tuple[dict[str, float], float, float, dict[str, dict[str, float]]]:
    if not 0.0 < weight_step <= 1.0 or not np.isclose(1.0 / weight_step, round(1.0 / weight_step)):
        raise ValueError("weight_step must divide 1 exactly, for example 0.05 or 0.1.")

    label_tensor = torch.tensor(labels)
    units = round(1.0 / weight_step)
    names = ("mel", "cqt", "chroma")
    scores: dict[str, dict[str, float]] = {}
    best_weights: dict[str, float] = {}
    best_accuracy = -1.0
    best_loss = float("inf")
    for mel_units in range(units + 1):
        for cqt_units in range(units - mel_units + 1):
            chroma_units = units - mel_units - cqt_units
            weights = dict(zip(names, (mel_units / units, cqt_units / units, chroma_units / units)))
            fused = sum(weights[name] * probabilities_by_representation[name] for name in names)
            accuracy = (fused.argmax(dim=1) == label_tensor).float().mean().item()
            loss = -torch.log(fused[torch.arange(len(labels)), label_tensor].clamp_min(1e-8)).mean().item()
            key = ", ".join(f"{name}={weights[name]:.2f}" for name in names)
            scores[key] = {"accuracy": accuracy, "loss": loss}
            if accuracy > best_accuracy or (np.isclose(accuracy, best_accuracy) and loss < best_loss):
                best_weights, best_accuracy, best_loss = weights, accuracy, loss
    return best_weights, best_accuracy, best_loss, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mel, CQT, and chroma CNNs with validation-selected late fusion.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-step", type=float, default=FUSION_WEIGHT_STEP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_fusion_chroma",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    audio_root = workspace_root / "GTZAN" / "genres_original"
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    collected = collect_labeled_audio(audio_root, GENRES, load_mel)
    if not collected:
        raise FileNotFoundError(f"No labeled audio files found in {audio_root}.")

    paths = [path for path, _ in collected]
    labels = [label for _, label in collected]
    train_paths, validation_paths, test_paths, train_labels, validation_labels, test_labels = split_tracks(paths, labels)

    def train_representation(name: str, feature_loader: Callable[[str], np.ndarray]):
        set_seed()
        train_loader = DataLoader(
            FeatureChunkDataset(train_paths, train_labels, feature_loader, True),
            batch_size=args.batch_size,
            shuffle=True,
        )
        return train_model(
            GenreCNN(len(GENRES)),
            train_loader,
            lambda model, device: evaluate_feature_tracks(
                model, validation_paths, validation_labels, feature_loader, device
            ),
            args.epochs,
            output_dir / f"gtzan_{name}_cnn.pt",
            {"genres": GENRES, "representation": name},
        )

    feature_loaders = {"mel": load_mel, "cqt": load_cqt, "chroma": load_chroma}
    models: dict[str, torch.nn.Module] = {}
    device: torch.device | None = None
    for name, feature_loader in feature_loaders.items():
        models[name], _, device, _ = train_representation(name, feature_loader)

    if device is None:
        raise RuntimeError("No fusion models were trained.")

    validation_probabilities = {
        name: track_probabilities(models[name], validation_paths, feature_loader, device)
        for name, feature_loader in feature_loaders.items()
    }
    weights, validation_accuracy, validation_loss, validation_scores = select_fusion_weights(
        validation_probabilities, validation_labels, args.weight_step
    )
    test_probabilities_by_representation = {
        name: track_probabilities(models[name], test_paths, feature_loader, device)
        for name, feature_loader in feature_loaders.items()
    }
    test_probabilities = sum(
        weights[name] * test_probabilities_by_representation[name] for name in feature_loaders
    )
    test_labels_tensor = torch.tensor(test_labels)
    predictions = test_probabilities.argmax(dim=1)
    test_loss = -torch.log(test_probabilities[torch.arange(len(test_labels)), test_labels_tensor].clamp_min(1e-8)).mean()
    test_accuracy = (predictions == test_labels_tensor).float().mean()
    results = {
        "dataset": "GTZAN WAV audio",
        "fusion": "Track-level average probabilities with validation-selected three-way late fusion.",
        "representations": {
            "mel": "128-bin mel spectrogram",
            "cqt": f"{CQT_BINS}-bin CQT",
            "chroma": f"{CHROMA_BINS}-bin CQT chroma",
        },
        "fusion_weights": weights,
        "validation_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "validation_weight_scores": validation_scores,
        "test_accuracy": test_accuracy.item(),
        "test_loss": test_loss.item(),
        "classification_report": classification_report(
            test_labels, predictions.tolist(), target_names=GENRES, output_dict=True, zero_division=0
        ),
        "track_split": {"train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)},
        "model_files": {
            name: str(output_dir / f"gtzan_{name}_cnn.pt") for name in feature_loaders
        },
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print("Best validation fusion: " + ", ".join(f"{name}={weight:.2f}" for name, weight in weights.items()))
    print(f"Fusion test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()