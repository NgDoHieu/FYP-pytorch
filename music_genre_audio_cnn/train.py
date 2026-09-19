"""Train a CNN on GTZAN WAV audio and mel spectrograms."""

from __future__ import annotations

import json
import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
SAMPLE_RATE = 22050
DURATION_SECONDS = 30
CHUNK_SECONDS = 3
CHUNKS_PER_TRACK = int(DURATION_SECONDS / CHUNK_SECONDS)
N_MELS = 128
TARGET_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / 512) + 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_mel(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SECONDS)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.clip((mel + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
    
    chunks = []
    for i in range(CHUNKS_PER_TRACK):
        start = i * TARGET_FRAMES
        end = start + TARGET_FRAMES
        chunk = mel[:, start:end]
        if chunk.shape[1] < TARGET_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TARGET_FRAMES - chunk.shape[1])))
        chunks.append(chunk[:, :, np.newaxis])
    return np.array(chunks)


def is_readable_audio(path: str) -> bool:
    try:
        load_mel(path)
    except Exception as error:
        print(f"Skipping unreadable audio: {path} ({error})")
        return False
    return True


def collect_audio_files(audio_root: Path) -> list[tuple[str, int]]:
    collected: list[tuple[str, int]] = []
    for label, genre in enumerate(GENRES):
        genre_dir = audio_root / genre
        if not genre_dir.exists():
            continue
        for path in sorted(genre_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            path_string = str(path)
            if is_readable_audio(path_string):
                collected.append((path_string, label))
    return collected


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


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = total_correct = total_samples = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    with torch.no_grad():
        for audio, labels in loader:
            audio, labels = audio.to(device), labels.to(device)
            logits = model(audio)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
    return total_loss / total_samples, total_correct / total_samples, true_labels, predicted_labels


def train_model(
    train: DataLoader,
    validation: DataLoader,
    class_count: int,
    epochs: int,
    model_path: Path,
    checkpoint_metadata: dict[str, list[str]],
) -> tuple[nn.Module, nn.Module, torch.device, dict[str, list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenreCNN(class_count).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, min_lr=1e-6)
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_accuracy = -1.0
    remaining_patience = 10
    for epoch in range(epochs):
        model.train()
        total_loss = total_correct = total_samples = 0
        for audio, labels in train:
            audio, labels = audio.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(audio)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
        validation_loss, validation_accuracy, _, _ = evaluate(model, validation, criterion, device)
        history["loss"].append(total_loss / total_samples)
        history["accuracy"].append(total_correct / total_samples)
        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)
        scheduler.step(validation_loss)
        print(f"Epoch {epoch + 1}/{epochs}: loss={history['loss'][-1]:.4f}, val_accuracy={validation_accuracy:.4f}")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            remaining_patience = 10
            torch.save({"model_state_dict": model.state_dict(), **checkpoint_metadata}, model_path)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model_state_dict"])
    return model, criterion, device, history


def main() -> None:
    epochs = 40
    batch_size = 32
    set_seed()
    workspace_root = Path(__file__).resolve().parents[2]
    audio_root = workspace_root / "GTZAN" / "genres_original"
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    collected = collect_audio_files(audio_root)
    if not collected:
        raise FileNotFoundError(f"No labeled audio files found in {audio_root}. Supported extensions: {sorted(AUDIO_EXTENSIONS)}")

    paths = [path for path, _ in collected]
    labels = [label for _, label in collected]

    train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.15, random_state=SEED, stratify=labels)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(train_paths, train_labels, test_size=0.15 / 0.85, random_state=SEED, stratify=train_labels)
    train = DataLoader(MelChunkDataset(train_paths, train_labels, True), batch_size=batch_size, shuffle=True)
    validation = DataLoader(MelChunkDataset(validation_paths, validation_labels, False), batch_size=batch_size)
    test = DataLoader(MelChunkDataset(test_paths, test_labels, False), batch_size=batch_size)
    model_path = output_dir / "gtzan_audio_cnn.pt"
    model, criterion, device, history = train_model(
        train, validation, len(GENRES), epochs, model_path, {"genres": GENRES}
    )
    test_loss, test_accuracy, true_labels, predicted_labels = evaluate(model, test, criterion, device)
    report = classification_report(true_labels, predicted_labels, target_names=GENRES, output_dict=True, zero_division=0)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)
    ConfusionMatrixDisplay(confusion_matrix(true_labels, predicted_labels), display_labels=GENRES).plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    results = {
        "dataset": "GTZAN WAV audio",
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "classification_report": report,
        "track_split": {"train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)},
        "genres": GENRES,
        "augmentation": ["brightness", "Gaussian noise", "time masking"],
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
