"""Explore GTZAN feature-space structure at the track level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, trustworthiness
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from dataset import GENRES, extract_track_id


def load_track_features(csv_path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Average excerpt-level features so every song contributes one sample."""
    excerpts = pd.read_csv(csv_path)
    feature_names = [
        column for column in excerpts.columns if column not in {"filename", "length", "label"}
    ]
    excerpts["track_id"] = excerpts["filename"].map(extract_track_id)

    inconsistent_labels = excerpts.groupby("track_id")["label"].nunique()
    if (inconsistent_labels != 1).any():
        raise ValueError("Found tracks whose excerpts have inconsistent genre labels.")

    tracks = (
        excerpts.groupby(["track_id", "label"], as_index=False, sort=True)[feature_names]
        .mean()
        .sort_values("track_id")
        .reset_index(drop=True)
    )
    if tracks.empty:
        raise ValueError(f"No track features found in {csv_path}.")

    labels = LabelEncoder().fit(GENRES).transform(tracks["label"])
    return tracks, feature_names, labels


def save_embedding_plot(
    embedding: np.ndarray,
    track_labels: np.ndarray,
    cluster_labels: np.ndarray,
    title: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, colors, legend_title in (
        (axes[0], track_labels, "Genre"),
        (axes[1], cluster_labels, "Cluster"),
    ):
        names = (
            GENRES
            if legend_title == "Genre"
            else [f"Cluster {value}" for value in sorted(np.unique(cluster_labels))]
        )
        scatter = axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            cmap=plt.get_cmap("tab10", len(names)),
            vmin=-0.5,
            vmax=len(names) - 0.5,
            s=24,
            alpha=0.8,
            linewidths=0,
        )
        axis.set_xlabel("Component 1")
        axis.set_ylabel("Component 2")
        axis.set_title(f"Colored by {legend_title.lower()}")
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=scatter.cmap(index),
                markersize=7,
                label=name,
            )
            for index, name in enumerate(names)
        ]
        axis.legend(handles, names, title=legend_title, loc="best", fontsize="small")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def score_clusters(
    name: str, features: np.ndarray, truth: np.ndarray, predicted: np.ndarray
) -> dict[str, float | str]:
    return {
        "method": name,
        "silhouette": float(silhouette_score(features, predicted)),
        "davies_bouldin": float(davies_bouldin_score(features, predicted)),
        "adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(truth, predicted)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run track-level PCA, t-SNE, and unsupervised clustering on GTZAN features."
    )
    workspace_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--csv",
        type=Path,
        default=workspace_root / "GTZAN" / "features_3_sec.csv",
        help="Path to GTZAN features_3_sec.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "clustering_results",
        help="Directory for clustering plots and result files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(f"GTZAN feature CSV not found: {args.csv}")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be positive.")

    args.output.mkdir(parents=True, exist_ok=True)
    tracks, feature_names, truth = load_track_features(args.csv)
    features = tracks[feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Feature CSV contains missing or non-finite numeric values.")
    if args.perplexity >= len(tracks):
        raise ValueError(
            f"t-SNE perplexity ({args.perplexity}) must be smaller than track count ({len(tracks)})."
        )

    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=args.seed)
    pca_2d = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=len(GENRES), n_init=20, random_state=args.seed)
    kmeans_labels = kmeans.fit_predict(scaled)
    gmm = GaussianMixture(
        n_components=len(GENRES), covariance_type="diag", n_init=3, random_state=args.seed
    )
    gmm_labels = gmm.fit_predict(scaled)

    scores = [
        score_clusters("KMeans on standardized track features", scaled, truth, kmeans_labels),
        score_clusters("Diagonal GMM on standardized track features", scaled, truth, gmm_labels),
    ]

    tsne_2d = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=args.seed,
    ).fit_transform(scaled)
    tsne_trustworthiness = float(trustworthiness(scaled, tsne_2d, n_neighbors=10))

    save_embedding_plot(
        pca_2d,
        truth,
        kmeans_labels,
        "GTZAN track-level feature PCA",
        args.output / "pca_tracks.png",
    )
    save_embedding_plot(
        tsne_2d,
        truth,
        kmeans_labels,
        "GTZAN track-level feature t-SNE",
        args.output / "tsne_tracks.png",
    )

    assignments = tracks[["track_id", "label"]].copy()
    assignments["kmeans_cluster"] = kmeans_labels
    assignments["gmm_cluster"] = gmm_labels
    assignments["pca_1"] = pca_2d[:, 0]
    assignments["pca_2"] = pca_2d[:, 1]
    assignments["tsne_1"] = tsne_2d[:, 0]
    assignments["tsne_2"] = tsne_2d[:, 1]
    assignments.to_csv(args.output / "track_assignments.csv", index=False)

    results = {
        "dataset": "GTZAN features_3_sec.csv, averaged to one row per track",
        "track_count": int(len(tracks)),
        "feature_count": len(feature_names),
        "seed": args.seed,
        "pca_2d_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_2d_total_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "tsne": {
            "perplexity": args.perplexity,
            "trustworthiness_10_neighbors": tsne_trustworthiness,
        },
        "cluster_evaluation_space": "57 standardized mean track features, not 2D plots",
        "cluster_metrics": scores,
        "metric_notes": {
            "silhouette": "Higher is better; measures separation without using genre labels.",
            "davies_bouldin": "Lower is better; measures within-cluster spread relative to separation.",
            "adjusted_rand_index": "Higher is better; compares cluster assignments with genre labels.",
            "normalized_mutual_information": "Higher is better; compares cluster assignments with genre labels.",
        },
        "plots": ["pca_tracks.png", "tsne_tracks.png"],
        "assignments_csv": "track_assignments.csv",
    }
    with (args.output / "clustering_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Analyzed {len(tracks)} tracks using {len(feature_names)} standardized features.")
    print(f"PCA 2D explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"t-SNE trustworthiness (10 neighbors): {tsne_trustworthiness:.3f}")
    for result in scores:
        print(
            f"{result['method']}: silhouette={result['silhouette']:.3f}, "
            f"Davies-Bouldin={result['davies_bouldin']:.3f}, "
            f"ARI={result['adjusted_rand_index']:.3f}, "
            f"NMI={result['normalized_mutual_information']:.3f}"
        )
    print(f"Saved plots, metrics, and assignments to {args.output}")


if __name__ == "__main__":
    main()
