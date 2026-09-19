"""Train mel and CQT CNNs, then select a late-fusion weight on validation tracks."""

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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from train import (
    AUDIO_EXTENSIONS,
    CHUNKS_PER_TRACK,
    DURATION_SECONDS,
    GENRES,
    SAMPLE_RATE,
    SEED,
    TARGET_FRAMES,
    load_mel,
    set_seed,
    train_model,
)

CQT_BINS = 84


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
    chunks = []
    for chunk_index in range(CHUNKS_PER_TRACK):
        start = chunk_index * TARGET_FRAMES
        chunk = cqt[:, start : start + TARGET_FRAMES]
        if chunk.shape[1] < TARGET_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TARGET_FRAMES - chunk.shape[1])))
        chunks.append(chunk[:, :, np.newaxis])
    return np.asarray(chunks, dtype=np.float32)


def collect_audio_files(audio_root: Path) -> list[tuple[str, int]]:
    collected: list[tuple[str, int]] = []
    for label, genre in enumerate(GENRES):
        for path in sorted((audio_root / genre).glob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                try:
                    load_mel(str(path))
                    collected.append((str(path), label))
                except Exception as error:
                    print(f"Skipping unreadable audio: {path} ({error})")
    return collected


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
        return tensor, torch.tensor(label, dtype=torch.long)


def track_probabilities(
    model: torch.nn.Module,
    paths: list[str],
    feature_loader: Callable[[str], np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for path in paths:
            chunks = torch.from_numpy(feature_loader(path)).permute(0, 3, 1, 2).to(device)
            probabilities.append(torch.softmax(model(chunks), dim=1).mean(dim=0).cpu())
    return torch.stack(probabilities)


def select_weight(
    mel_probabilities: torch.Tensor, cqt_probabilities: torch.Tensor, labels: list[int]
) -> tuple[float, float, dict[str, float]]:
    label_tensor = torch.tensor(labels)
    validation_scores = {}
    for mel_weight in np.arange(0.0, 1.01, 0.1):
        probabilities = mel_weight * mel_probabilities + (1.0 - mel_weight) * cqt_probabilities
        validation_scores[f"{mel_weight:.1f}"] = (probabilities.argmax(dim=1) == label_tensor).float().mean().item()
    best_weight = max(validation_scores, key=validation_scores.get)
    return float(best_weight), validation_scores[best_weight], validation_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a mel-CNN and CQT-CNN with validation-selected late fusion.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    audio_root = workspace_root / "GTZAN" / "genres_original"
    output_dir = Path(__file__).resolve().parent / "results_fusion"
    output_dir.mkdir(exist_ok=True)
    collected = collect_audio_files(audio_root)
    if not collected:
        raise FileNotFoundError(f"No labeled audio files found in {audio_root}.")

    paths = [path for path, _ in collected]
    labels = [label for _, label in collected]
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_paths, train_labels, test_size=0.15 / 0.85, random_state=SEED, stratify=train_labels
    )

    def train_representation(name: str, feature_loader: Callable[[str], np.ndarray]):
        set_seed()
        train_loader = DataLoader(
            FeatureChunkDataset(train_paths, train_labels, feature_loader, True),
            batch_size=args.batch_size,
            shuffle=True,
        )
        validation_loader = DataLoader(
            FeatureChunkDataset(validation_paths, validation_labels, feature_loader, False),
            batch_size=args.batch_size,
        )
        return train_model(
            train_loader,
            validation_loader,
            len(GENRES),
            args.epochs,
            output_dir / f"gtzan_{name}_cnn.pt",
            {"genres": GENRES, "representation": name},
        )

    mel_model, _, device, _ = train_representation("mel", load_mel)
    cqt_model, _, _, _ = train_representation("cqt", load_cqt)
    mel_validation = track_probabilities(mel_model, validation_paths, load_mel, device)
    cqt_validation = track_probabilities(cqt_model, validation_paths, load_cqt, device)
    mel_weight, validation_accuracy, validation_scores = select_weight(
        mel_validation, cqt_validation, validation_labels
    )
    mel_test = track_probabilities(mel_model, test_paths, load_mel, device)
    cqt_test = track_probabilities(cqt_model, test_paths, load_cqt, device)
    test_probabilities = mel_weight * mel_test + (1.0 - mel_weight) * cqt_test
    test_labels_tensor = torch.tensor(test_labels)
    predictions = test_probabilities.argmax(dim=1)
    test_loss = -torch.log(test_probabilities[torch.arange(len(test_labels)), test_labels_tensor].clamp_min(1e-8)).mean()
    test_accuracy = (predictions == test_labels_tensor).float().mean()
    results = {
        "dataset": "GTZAN WAV audio",
        "fusion": "Track-level average probabilities with validation-selected late fusion.",
        "mel_weight": mel_weight,
        "cqt_weight": 1.0 - mel_weight,
        "validation_accuracy": validation_accuracy,
        "validation_weight_accuracy": validation_scores,
        "test_accuracy": test_accuracy.item(),
        "test_loss": test_loss.item(),
        "classification_report": classification_report(
            test_labels, predictions.tolist(), target_names=GENRES, output_dict=True, zero_division=0
        ),
        "track_split": {"train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)},
        "model_files": {
            "mel": str(output_dir / "gtzan_mel_cnn.pt"),
            "cqt": str(output_dir / "gtzan_cqt_cnn.pt"),
        },
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Best validation fusion: mel={mel_weight:.1f}, CQT={1.0 - mel_weight:.1f}")
    print(f"Fusion test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()