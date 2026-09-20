"""Train a chroma-and-rhythm CNN on GTZAN with track-level evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset

SHARED_MODULE_DIR = Path(__file__).resolve().parents[1] / "music_genre_audio_cnn"
sys.path.insert(0, str(SHARED_MODULE_DIR))

from shared import (
    AUDIO_EXTENSIONS,
    CHUNKS_PER_TRACK,
    DURATION_SECONDS,
    GTZAN_GENRES as GENRES,
    HOP_LENGTH,
    SAMPLE_RATE,
    TARGET_FRAMES,
    collect_labeled_audio,
    evaluate_tracks,
    save_confusion_matrix,
    save_training_curves,
    set_seed,
    split_tracks,
)
from training import train_model

CHROMA_BINS = 24


def pad_or_trim(feature: np.ndarray, frame_count: int) -> np.ndarray:
    if feature.shape[1] >= frame_count:
        return feature[:, :frame_count]
    return np.pad(feature, ((0, 0), (0, frame_count - feature.shape[1])))


def chunk_features(chroma: np.ndarray, rhythm: np.ndarray) -> np.ndarray:
    chunks = []
    for chunk_index in range(CHUNKS_PER_TRACK):
        start = chunk_index * TARGET_FRAMES
        end = start + TARGET_FRAMES
        chunks.append(np.stack((chroma[:, start:end], rhythm[:, start:end]), axis=0))
    return np.asarray(chunks, dtype=np.float32)


def load_chroma_rhythm(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SECONDS)
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz("C1"),
        n_chroma=CHROMA_BINS,
        n_octaves=7,
        bins_per_octave=48,
    )
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        win_length=384,
    )
    rhythm = np.stack([band.mean(axis=0) for band in np.array_split(tempogram, CHROMA_BINS, axis=0)])
    rhythm /= np.maximum(rhythm.max(axis=1, keepdims=True), 1e-8)

    total_frames = CHUNKS_PER_TRACK * TARGET_FRAMES
    chroma = pad_or_trim(np.nan_to_num(chroma), total_frames)
    rhythm = pad_or_trim(np.nan_to_num(rhythm), total_frames)
    return chunk_features(chroma.astype(np.float32), rhythm.astype(np.float32))


class ChromaRhythmDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: list[str], labels: list[int], training: bool) -> None:
        self.training = training
        self.samples = [
            (chunk, label)
            for path, label in zip(paths, labels)
            for chunk in load_chroma_rhythm(path)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature, label = self.samples[index]
        tensor = torch.from_numpy(feature).clone()
        if self.training and random.random() < 0.3:
            width = random.randrange(4, 13)
            start = random.randrange(TARGET_FRAMES - width + 1)
            tensor[:, :, start : start + width] = 0.0
        return tensor, torch.tensor(label, dtype=torch.long)


class ChromaRhythmCNN(nn.Module):
    """Pool over time first so early layers retain chroma-pitch structure."""

    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.15),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, class_count)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(features))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a chroma-and-rhythm CNN on GTZAN.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_chroma_rhythm"
    )
    args = parser.parse_args()

    set_seed()
    workspace_root = Path(__file__).resolve().parents[2]
    audio_root = workspace_root / "GTZAN" / "genres_original"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"GTZAN audio directory not found: {audio_root}")
    args.output_dir.mkdir(exist_ok=True)

    collected = collect_labeled_audio(audio_root, GENRES, load_chroma_rhythm)
    if not collected:
        raise FileNotFoundError("No readable GTZAN audio files were found.")
    paths, labels = zip(*collected)
    train_paths, validation_paths, test_paths, train_labels, validation_labels, test_labels = split_tracks(
        list(paths), list(labels)
    )

    train_loader = DataLoader(
        ChromaRhythmDataset(train_paths, train_labels, training=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model_path = args.output_dir / "gtzan_chroma_rhythm_cnn.pt"
    model, _, device, history = train_model(
        ChromaRhythmCNN(len(GENRES)),
        train_loader,
        lambda current_model, current_device: evaluate_tracks(
            current_model,
            validation_paths,
            validation_labels,
            load_chroma_rhythm,
            current_device,
            torch.from_numpy,
        )[:2],
        args.epochs,
        model_path,
        {"genres": GENRES, "representation": "chroma-rhythm"},
        learning_rate=3e-4,
    )
    test_loss, test_accuracy, _, predictions = evaluate_tracks(
        model, test_paths, test_labels, load_chroma_rhythm, device, torch.from_numpy
    )

    save_training_curves(history, args.output_dir / "training_curves.png")
    save_confusion_matrix(test_labels, predictions, GENRES, args.output_dir / "confusion_matrix.png")

    results = {
        "dataset": "GTZAN WAV audio",
        "representation": "24-bin CQT chroma and 24-band pooled tempogram",
        "evaluation": "Track-level metrics after averaging probabilities from ten three-second excerpts.",
        "augmentation": ["30% light time masking (4-12 frames)"],
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "classification_report": classification_report(
            test_labels, predictions, target_names=GENRES, output_dict=True, zero_division=0
        ),
        "track_split": {"train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)},
        "model_file": str(model_path),
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Chroma-rhythm test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
