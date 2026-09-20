"""Train a simple mel-spectrogram CNN on an Artist20-style directory dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from shared import evaluate_tracks, set_seed, split_tracks
from train import GenreCNN, MelChunkDataset, load_mel
from training import train_model

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

    train_paths, validation_paths, test_paths, train_labels, validation_labels, test_labels = split_tracks(
        paths, labels, test_size=0.2, validation_size=0.16
    )
    train = DataLoader(MelChunkDataset(train_paths, train_labels, True), batch_size=args.batch_size, shuffle=True)

    output_dir = Path(__file__).resolve().parent / "results_artist20"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "artist20_audio_cnn.pt"
    model, _, device, _ = train_model(
        GenreCNN(len(artists)),
        train,
        lambda current_model, current_device: evaluate_tracks(
            current_model, validation_paths, validation_labels, load_mel, current_device
        )[:2],
        args.epochs,
        model_path,
        {"artists": artists},
    )
    test_loss, test_accuracy, _, predicted_labels = evaluate_tracks(
        model, test_paths, test_labels, load_mel, device
    )
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