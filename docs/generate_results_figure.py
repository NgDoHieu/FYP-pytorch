"""Generate a held-out accuracy comparison figure for the progress report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as result_file:
        return json.load(result_file)


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    results = {
        "Mel CNN": load_result(project_dir / "music_genre_audio_cnn" / "results" / "results.json"),
        "Chroma-rhythm\nCNN": load_result(
            project_dir / "music_genre_chroma_rhythm_cnn" / "results_chroma_rhythm" / "results.json"
        ),
        "KNN": load_result(project_dir / "music_genre_knn" / "results" / "results.json"),
        "Mel + CQT\nfusion": load_result(
            project_dir / "music_genre_audio_cnn" / "results_fusion" / "results.json"
        ),
        "Mel + CQT +\nchroma fusion": load_result(
            project_dir / "music_genre_audio_cnn" / "results_fusion_chroma" / "results.json"
        ),
    }
    final_fusion = load_result(project_dir / "music_genre_audio_cnn" / "final_fusion_result" / "results.json")

    labels = list(results)
    accuracies = [result["test_accuracy"] * 100 for result in results.values()]
    labels.append("Final 20-seed\nfusion")
    accuracies.append(final_fusion["mean_test_accuracy"] * 100)
    standard_deviations = [0] * (len(accuracies) - 1) + [final_fusion["std_test_accuracy"] * 100]

    figure, axis = plt.subplots(figsize=(11, 5.5))
    bars = axis.bar(labels, accuracies, yerr=standard_deviations, capsize=5, color="#237a8c")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Held-out track accuracy (%)")
    axis.set_title("GTZAN Model Comparison")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    for bar, accuracy in zip(bars, accuracies):
        axis.text(bar.get_x() + bar.get_width() / 2, accuracy + 2, f"{accuracy:.1f}%", ha="center")

    figure.tight_layout()
    output_path = Path(__file__).resolve().parent / "Figures" / "model_accuracy_comparison.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()