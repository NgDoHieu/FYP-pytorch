"""Evaluate the fixed mel, CQT, and chroma late-fusion ensemble on GTZAN."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report

from shared import (
    GTZAN_GENRES as GENRES,
    collect_labeled_audio,
    save_confusion_matrix,
    set_seed,
    split_tracks,
    track_probabilities,
)
from train import GenreCNN, load_mel
from train_fusion import load_chroma, load_cqt

FUSION_WEIGHTS = {"mel": 0.60, "cqt": 0.30, "chroma": 0.10}


def load_model(model_path: Path, device: torch.device) -> GenreCNN:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = GenreCNN(len(GENRES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train fusion branches, then evaluate the fixed late-fusion ensemble."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed()
    project_dir = Path(__file__).resolve().parent
    workspace_root = project_dir.parents[1]
    audio_root = workspace_root / "GTZAN" / "genres_original"
    checkpoint_dir = project_dir / "results_fusion_chroma"
    output_dir = project_dir / "final_fusion_result"
    output_dir.mkdir(exist_ok=True)

    print("Training mel, CQT, and chroma fusion branches...", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(project_dir / "train_fusion.py"),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(checkpoint_dir),
        ],
        check=True,
    )
    print("Evaluating fixed fusion weights...", flush=True)

    feature_loaders = {"mel": load_mel, "cqt": load_cqt, "chroma": load_chroma}
    collected = collect_labeled_audio(audio_root, GENRES, load_mel)
    if not collected:
        raise FileNotFoundError(f"No labeled audio files found in {audio_root}.")
    paths, labels = zip(*collected)
    _, _, test_paths, _, _, test_labels = split_tracks(list(paths), list(labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        name: load_model(checkpoint_dir / f"gtzan_{name}_cnn.pt", device)
        for name in feature_loaders
    }
    probabilities = {
        name: track_probabilities(models[name], test_paths, loader, device)
        for name, loader in feature_loaders.items()
    }
    fused_probabilities = sum(FUSION_WEIGHTS[name] * probabilities[name] for name in feature_loaders)
    label_tensor = torch.tensor(test_labels)
    predictions = fused_probabilities.argmax(dim=1)
    test_loss = -torch.log(
        fused_probabilities[torch.arange(len(test_labels)), label_tensor].clamp_min(1e-8)
    ).mean()
    test_accuracy = (predictions == label_tensor).float().mean()

    confusion_matrix_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(test_labels, predictions.tolist(), GENRES, confusion_matrix_path)
    results = {
        "dataset": "GTZAN WAV audio",
        "fusion": "Fixed track-level late fusion of saved mel, CQT, and chroma CNNs.",
        "fusion_weights": FUSION_WEIGHTS,
        "evaluation": "Held-out track metrics after averaging probabilities from ten three-second excerpts.",
        "test_accuracy": test_accuracy.item(),
        "test_loss": test_loss.item(),
        "classification_report": classification_report(
            test_labels, predictions.tolist(), target_names=GENRES, output_dict=True, zero_division=0
        ),
        "track_split": {"train": 699, "validation": 150, "test": len(test_paths)},
        "model_files": {name: str(checkpoint_dir / f"gtzan_{name}_cnn.pt") for name in feature_loaders},
        "confusion_matrix": str(confusion_matrix_path),
        "training_curves": str(checkpoint_dir / "fusion_training_curves.png"),
    }
    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Fixed fusion test accuracy: {test_accuracy:.4f}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()