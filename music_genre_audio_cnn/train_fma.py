"""Train a CNN on FMA Small audio and mel spectrograms."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train import MelChunkDataset, evaluate, load_mel, set_seed, train_model

SEED = 42
SAMPLE_RATE = 22050
DURATION_SECONDS = 30
CHUNK_SECONDS = 3
CHUNKS_PER_TRACK = int(DURATION_SECONDS / CHUNK_SECONDS)
N_MELS = 128
TARGET_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / 512) + 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
# Using the 8 true FMA Small genres
GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]

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


def main() -> None:
    epochs = 40
    batch_size = 32
    set_seed()
    project_root = Path(__file__).resolve().parents[1]
    audio_root = project_root / "raw" / "fma_small_organized" / "genres"
    output_dir = Path(__file__).resolve().parent / "results_fma"
    output_dir.mkdir(exist_ok=True)

    collected = collect_audio_files(audio_root)
    if not collected:
        raise FileNotFoundError(f"No labeled audio files found in {audio_root}. Please run the organize script first.")

    paths = [path for path, _ in collected]
    labels = [label for _, label in collected]

    train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.15, random_state=SEED, stratify=labels)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(train_paths, train_labels, test_size=0.15 / 0.85, random_state=SEED, stratify=train_labels)
    train = DataLoader(MelChunkDataset(train_paths, train_labels, True), batch_size=batch_size, shuffle=True)
    validation = DataLoader(MelChunkDataset(validation_paths, validation_labels, False), batch_size=batch_size)
    test = DataLoader(MelChunkDataset(test_paths, test_labels, False), batch_size=batch_size)
    model_path = output_dir / "fma_audio_cnn.pt"
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
        "dataset": "FMA Small audio",
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "classification_report": report,
        "track_split": {"train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)},
        "genres": GENRES,
        "augmentation": ["brightness", "Gaussian noise", "time masking", "L2 Regularization"],
        "model_file": str(model_path),
        "training_graph": str(output_dir / "training_curves.png"),
        "confusion_matrix": str(output_dir / "confusion_matrix.png"),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"FMA Small audio test accuracy: {test_accuracy:.4f}")
    print(f"Results saved to: {output_dir / 'results.json'}")

if __name__ == "__main__":
    main()
