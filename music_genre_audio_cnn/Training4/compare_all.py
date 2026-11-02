"""Run both Option 3 (segment split) and Option 1 (track split) and generate side-by-side comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    train_script = current_dir / "train_high_acc.py"

    seg_res_path = current_dir / "results_segment" / "results.json"
    track_res_path = current_dir / "results_track" / "results.json"

    if not seg_res_path.exists():
        print("================================================================")
        print("STEP 1: Running Option 3 (Segment-Level Split -> 92-95% Target)")
        print("================================================================")
        subprocess.run([sys.executable, str(train_script), "--split", "segment", "--epochs", "80"], check=True)

    if not track_res_path.exists():
        print("\n================================================================")
        print("STEP 2: Running Option 1 (Track-Level Split -> Strict Song Isolation)")
        print("================================================================")
        subprocess.run([sys.executable, str(train_script), "--split", "track", "--epochs", "80"], check=True)

    # Load results
    seg_res_path = current_dir / "results_segment" / "results.json"
    track_res_path = current_dir / "results_track" / "results.json"

    with seg_res_path.open("r", encoding="utf-8") as f:
        seg_res = json.load(f)
    with track_res_path.open("r", encoding="utf-8") as f:
        track_res = json.load(f)

    # Plot comparison bar chart
    labels = ["DeepAudioMLP", "XGBoost", "Ensemble", "Track-Voted (Song)"]
    seg_scores = [
        seg_res["results"]["deep_audio_mlp_accuracy"] * 100,
        seg_res["results"]["xgboost_accuracy"] * 100,
        seg_res["results"]["ensemble_accuracy"] * 100,
        np.nan,
    ]
    track_scores = [
        track_res["results"]["deep_audio_mlp_accuracy"] * 100,
        track_res["results"]["xgboost_accuracy"] * 100,
        track_res["results"]["ensemble_accuracy"] * 100,
        track_res["results"]["track_voted_accuracy"] * 100,
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, seg_scores, width, label="Option 3: Segment-Level Split", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, track_scores, width, label="Option 1: Strict Track-Level Split", color="#2ca02c")

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("GTZAN Genre Classification: Segment-Level vs. Track-Level Rigor", fontsize=13, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    for rect in rects1:
        height = rect.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    for rect in rects2:
        height = rect.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    fig.tight_layout()
    comparison_plot = current_dir / "accuracy_comparison.png"
    fig.savefig(comparison_plot, dpi=180)
    plt.close(fig)

    print("\n================================================================")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Comparison plot saved to: {comparison_plot}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
