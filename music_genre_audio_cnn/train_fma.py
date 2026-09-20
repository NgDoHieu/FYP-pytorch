"""Train a CNN on FMA Small audio and mel spectrograms."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from shared import (
    collect_labeled_audio,
    evaluate_tracks,
    save_confusion_matrix,
    save_training_curves,
    set_seed,
    split_tracks,
)
from train import GenreCNN, MelChunkDataset, load_mel
from training import train_model

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

def main() -> None:
    epochs = 40
    batch_size = 32
    set_seed()
    project_root = Path(__file__).resolve().parents[1]
    audio_root = project_root / "raw" / "fma_small_organized" / "genres"
    output_dir = Path(__file__).resolve().parent / "results_fma"
    output_dir.mkdir(exist_ok=True)

    collected = collect_labeled_audio(audio_root, GENRES, load_mel)
    if not collected:
        raise FileNotFoundError(f"No labeled audio files found in {audio_root}. Please run the organize script first.")

    paths = [path for path, _ in collected]
    labels = [label for _, label in collected]

    train_paths, validation_paths, test_paths, train_labels, validation_labels, test_labels = split_tracks(paths, labels)
    train = DataLoader(MelChunkDataset(train_paths, train_labels, True), batch_size=batch_size, shuffle=True)
    model_path = output_dir / "fma_audio_cnn.pt"
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
