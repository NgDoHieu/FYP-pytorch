"""Train a simple mel-spectrogram CNN on an Artist20-style directory dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader

from train import MelChunkDataset, load_mel, set_seed, train_model

SEED = 42
SAMPLE_RATE = 22050
DURATION_SECONDS = 30
CHUNK_SECONDS = 3
CHUNKS_PER_TRACK = DURATION_SECONDS // CHUNK_SECONDS
N_MELS = 128
TARGET_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / 512) + 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def collect_audio_files(audio_root: Path) -> tuple[list[str], list[int], list[str]]:
    artist_dirs = sorted(path for path in audio_root.iterdir() if path.is_dir())
    artists = [path.name for path in artist_dirs]
    paths: list[str] = []
    labels: list[int] = []
    for label, artist_dir in enumerate(artist_dirs):
        for path in sorted(artist_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                paths.append(str(path))
                labels.append(label)
    return paths, labels, artists


def evaluate_tracks(model: nn.Module, paths: list[str], labels: list[int], device: torch.device) -> tuple[float, float, list[int]]:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for path in paths:
            chunks = torch.from_numpy(load_mel(path)).permute(0, 3, 1, 2).to(device)
            probabilities.append(torch.softmax(model(chunks), dim=1).mean(dim=0).cpu())
    probability_tensor = torch.stack(probabilities)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    predictions = probability_tensor.argmax(dim=1)
    loss = -torch.log(probability_tensor[torch.arange(len(labels)), label_tensor].clamp_min(1e-8)).mean()
    accuracy = (predictions == label_tensor).float().mean()
    return loss.item(), accuracy.item(), predictions.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Artist20 audio classifier.") 
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "artist20" / "artist20" / "mp3s-32k",
        help="Directory containing one subdirectory per artist.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Artist20 directory not found: {args.data_dir}. "
            "Expected one folder per artist containing audio files."
        )

    paths, labels, artists = collect_audio_files(args.data_dir)
    if len(artists) < 2 or not paths:
        raise ValueError("Expected audio files in at least two artist subdirectories.")
    if any(labels.count(label) < 7 for label in range(len(artists))):
        raise ValueError("Each artist needs at least seven audio files for train/validation/test splits.")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=SEED, stratify=train_labels
    )
    train = DataLoader(MelChunkDataset(train_paths, train_labels, True), batch_size=args.batch_size, shuffle=True)
    validation = DataLoader(MelChunkDataset(validation_paths, validation_labels, False), batch_size=args.batch_size)

    output_dir = Path(__file__).resolve().parent / "results_artist20"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "artist20_audio_cnn.pt"
    model, _, device, _ = train_model(
        train, validation, len(artists), args.epochs, model_path, {"artists": artists}
    )
    test_loss, test_accuracy, predicted_labels = evaluate_tracks(model, test_paths, test_labels, device)
    results = {
        "dataset": "Artist20",
        "artists": artists,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "evaluation": "Track-level accuracy after averaging predictions from ten three-second excerpts.",
        "augmentation": ["brightness", "Gaussian noise", "time masking"],
        "classification_report": classification_report(
            test_labels, predicted_labels, target_names=artists, output_dict=True, zero_division=0
        ),
        "track_split": {
            "train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)
        },
        "model_file": str(model_path),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with (output_dir / "artist_labels.json").open("w", encoding="utf-8") as file:
        json.dump(artists, file, indent=2)
    print(f"Artist20 test accuracy: {test_accuracy:.4f}")
    print(f"Saved model and results to: {output_dir}")


if __name__ == "__main__":
    main()