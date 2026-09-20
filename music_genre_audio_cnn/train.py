"""Train a CNN on GTZAN WAV audio and mel spectrograms."""

from __future__ import annotations

import json
import random
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shared import (
    AUDIO_EXTENSIONS,
    CHUNKS_PER_TRACK,
    DURATION_SECONDS,
    GTZAN_GENRES as GENRES,
    SAMPLE_RATE,
    SEED,
    TARGET_FRAMES,
    collect_labeled_audio,
    evaluate_tracks,
    save_confusion_matrix,
    save_training_curves,
    set_seed,
    split_feature_into_chunks,
    split_tracks,
)
from training import train_model

N_MELS = 128


def load_mel(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SECONDS)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.clip((mel + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
    
    return split_feature_into_chunks(mel)


class MelChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: list[str], labels: list[int], training: bool) -> None:
        self.training = training
        self.samples = [
            (chunk, label)
            for path, label in zip(paths, labels)
            for chunk in load_mel(path)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        mel, label = self.samples[index]
        mel_tensor = torch.from_numpy(mel).permute(2, 0, 1)
        if self.training:
            mel_tensor = torch.clamp(mel_tensor + torch.empty(1).uniform_(-0.08, 0.08), 0.0, 1.0)
            mel_tensor = torch.clamp(mel_tensor + torch.randn_like(mel_tensor) * 0.015, 0.0, 1.0)
            if random.random() < 0.5:
                start = random.randrange(TARGET_FRAMES - 20)
                width = random.randrange(5, 20)
                mel_tensor[:, :, start : start + width] = 0.0
            if random.random() < 0.5:
                width = random.randrange(5, 20)
                start = random.randrange(N_MELS - width + 1)
                mel_tensor[:, start : start + width, :] = 0.0
        return mel_tensor, torch.tensor(label, dtype=torch.long)


class GenreCNN(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.35), nn.Linear(64, class_count))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(mel))


def main() -> None:
    epochs = 40
    batch_size = 32
    set_seed()
    workspace_root = Path(__file__).resolve().parents[2]
    audio_root = workspace_root / "GTZAN" / "genres_original"
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    collected = collect_labeled_audio(audio_root, GENRES, load_mel)
    if not collected:
        raise FileNotFoundError(f"No labeled audio files found in {audio_root}. Supported extensions: {sorted(AUDIO_EXTENSIONS)}")

    paths = [path for path, _ in collected]
    labels = [label for _, label in collected]

    train_paths, validation_paths, test_paths, train_labels, validation_labels, test_labels = split_tracks(paths, labels)
    train = DataLoader(MelChunkDataset(train_paths, train_labels, True), batch_size=batch_size, shuffle=True)
    model_path = output_dir / "gtzan_audio_cnn.pt"
    model, _, device, history = train_model(
        GenreCNN(len(GENRES)),
        train,
        lambda current_model, current_device: evaluate_tracks(
            current_model, validation_paths, validation_labels, load_mel, current_device
        )[:2],
        epochs,
        model_path,
        {"genres": GENRES},
    )
    test_loss, test_accuracy, true_labels, predicted_labels = evaluate_tracks(
        model, test_paths, test_labels, load_mel, device
    )
    report = classification_report(true_labels, predicted_labels, target_names=GENRES, output_dict=True, zero_division=0)
    save_training_curves(history, output_dir / "training_curves.png")
    save_confusion_matrix(true_labels, predicted_labels, GENRES, output_dir / "confusion_matrix.png")
    results = {
        "dataset": "GTZAN WAV audio",
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "classification_report": report,
        "track_split": {"train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)},
        "genres": GENRES,
        "augmentation": ["brightness", "Gaussian noise", "time masking", "frequency masking"],
        "evaluation": "Track-level metrics after averaging probabilities from ten three-second excerpts.",
        "model_file": str(model_path),
        "training_graph": str(output_dir / "training_curves.png"),
        "confusion_matrix": str(output_dir / "confusion_matrix.png"),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"GTZAN audio test accuracy: {test_accuracy:.4f}")
    print(f"Results saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
