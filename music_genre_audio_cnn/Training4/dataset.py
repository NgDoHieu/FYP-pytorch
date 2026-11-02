"""Data loading and preprocessing utilities for GTZAN tabular audio features."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def extract_track_id(filename: str) -> str:
    """Extract parent track ID from 3-sec excerpt filename (e.g., 'blues.00000.0.wav' -> 'blues.00000')."""
    parts = str(filename).split(".")
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}"
    return str(filename)


def load_gtzan_data(
    csv_path: Path | str,
    split_mode: Literal["segment", "track"] = "segment",
    test_size: float = 0.15,
    val_size: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Load GTZAN features_3_sec.csv and create stratified splits.

    split_mode:
        - 'segment': (Option 3) Independent random split of 3-sec excerpts.
                     Yields 92-95% accuracy as commonly reported in Kaggle / Lab benchmarks.
        - 'track':   (Option 1) Strictly grouped by song. All 10 excerpts of a song stay
                     in the same split, preventing audio leakage.
    """
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in ("filename", "length", "label")]
    df["track_id"] = df["filename"].apply(extract_track_id)

    le = LabelEncoder()
    le.fit(GENRES)
    df["label_idx"] = le.transform(df["label"])

    scaler = StandardScaler()

    if split_mode == "segment":
        # Standard train / val / test split on segments
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["label_idx"]
        )
        val_fraction = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_fraction, random_state=seed, stratify=train_val_df["label_idx"]
        )
    elif split_mode == "track":
        # Group by unique track_id to ensure complete song isolation
        unique_tracks = df[["track_id", "label_idx"]].drop_duplicates().reset_index(drop=True)
        train_val_tracks, test_tracks = train_test_split(
            unique_tracks, test_size=test_size, random_state=seed, stratify=unique_tracks["label_idx"]
        )
        val_fraction = val_size / (1.0 - test_size)
        train_tracks, val_tracks = train_test_split(
            train_val_tracks, test_size=val_fraction, random_state=seed, stratify=train_val_tracks["label_idx"]
        )

        train_df = df[df["track_id"].isin(train_tracks["track_id"])].copy()
        val_df = df[df["track_id"].isin(val_tracks["track_id"])].copy()
        test_df = df[df["track_id"].isin(test_tracks["track_id"])].copy()
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    y_train = train_df["label_idx"].values
    y_val = val_df["label_idx"].values
    y_test = test_df["label_idx"].values

    return {
        "split_mode": split_mode,
        "feature_names": feature_cols,
        "genres": GENRES,
        "label_encoder": le,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }
