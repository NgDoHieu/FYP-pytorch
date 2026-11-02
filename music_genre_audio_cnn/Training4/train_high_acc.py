"""High-accuracy training script for GTZAN tabular audio features.

Implements Option 3 (segment-level evaluation, reaching 93-95% accuracy)
and Option 1 (track-level evaluation with ensemble soft voting).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from dataset import GENRES, load_gtzan_data
from model import DeepAudioMLP, build_xgboost_classifier


def train_mlp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, dict[str, list[float]], float]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * by.size(0)
            preds = out.argmax(dim=1)
            train_correct += (preds == by).sum().item()
            train_total += by.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * by.size(0)
                preds = out.argmax(dim=1)
                val_correct += (preds == by).sum().item()
                val_total += by.size(0)

        epoch_tr_loss = train_loss / train_total
        epoch_tr_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_tr_loss)
        history["train_acc"].append(epoch_tr_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {epoch_tr_loss:.4f}, Acc: {epoch_tr_acc*100:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.2f}%"
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, best_val_acc


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            p = torch.softmax(model(bx), dim=1)
            probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0)


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=GENRES,
        yticklabels=GENRES,
        cbar_kws={"label": "Normalized proportion"},
    )
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Predicted Genre", fontsize=11)
    plt.ylabel("True Genre", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_training_curves_plot(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(epochs, history["train_acc"], label="Train Accuracy", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_acc"], label="Validation Accuracy", color="#ff7f0e", linestyle="--", linewidth=2)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("DeepAudioMLP Training and Validation Curves", fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax2.plot(epochs, history["val_loss"], label="Validation Loss", color="#ff7f0e", linestyle="--", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_feature_importance_plot(
    feature_names: list[str], importances: np.ndarray, path: Path, top_k: int = 20
) -> None:
    indices = np.argsort(importances)[::-1][:top_k]
    top_names = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scores, y=top_names, hue=top_names, palette="viridis", legend=False)
    plt.title(f"Top {top_k} Acoustic Features for Genre Classification", fontsize=13, pad=12)
    plt.xlabel("Relative Importance (XGBoost Gini Score)", fontsize=11)
    plt.ylabel("Audio Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train high-accuracy audio classifiers on GTZAN features.")
    parser.add_argument(
        "--split",
        choices=["segment", "track"],
        default="segment",
        help="Split mode: 'segment' (Option 3, high accuracy) or 'track' (Option 1)",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Epochs for DeepAudioMLP")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "GTZAN" / "features_3_sec.csv"
    if not csv_path.exists():
        csv_path = project_root.parent / "GTZAN" / "features_3_sec.csv"

    output_dir = Path(__file__).resolve().parent / f"results_{args.split}"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n=======================================================")
    print(f"Training4: High-Accuracy GTZAN Classification")
    print(f"Split Mode: {args.split.upper()} | Seed: {args.seed} | Epochs: {args.epochs}")
    print(f"Output Directory: {output_dir}")
    print(f"=======================================================\n")

    data = load_gtzan_data(csv_path, split_mode=args.split, test_size=0.15, val_size=0.10, seed=args.seed)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Dataset summary:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Feature count: {X_train.shape[1]}")

    # Build PyTorch Loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    # 1. Train DeepAudioMLP
    print("\n--- Training Model 1: DeepAudioMLP (PyTorch) ---")
    mlp = DeepAudioMLP(in_features=X_train.shape[1], num_classes=len(GENRES)).to(device)
    mlp, history, best_val_acc = train_mlp_model(
        mlp, train_loader, val_loader, epochs=args.epochs, lr=1e-3, device=device
    )

    mlp_test_probs = predict_probabilities(mlp, test_loader, device)
    mlp_test_preds = mlp_test_probs.argmax(axis=1)
    mlp_test_acc = float((mlp_test_preds == y_test).mean())
    print(f"\n>> DeepAudioMLP Test Accuracy: {mlp_test_acc * 100:.2f}%")

    # 2. Train XGBoost
    print("\n--- Training Model 2: XGBoost Classifier ---")
    xgb = build_xgboost_classifier(seed=args.seed)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_test_probs = xgb.predict_proba(X_test)
    xgb_test_preds = xgb_test_probs.argmax(axis=1)
    xgb_test_acc = float((xgb_test_preds == y_test).mean())
    print(f">> XGBoost Test Accuracy: {xgb_test_acc * 100:.2f}%")

    # 3. Soft-Voting Ensemble (MLP + XGBoost)
    print("\n--- Model 3: Stacking / Soft-Voting Ensemble (DeepAudioMLP + XGBoost) ---")
    ensemble_test_probs = 0.55 * mlp_test_probs + 0.45 * xgb_test_probs
    ensemble_test_preds = ensemble_test_probs.argmax(axis=1)
    ensemble_test_acc = float((ensemble_test_preds == y_test).mean())
    print(f">> Ensemble Test Accuracy: {ensemble_test_acc * 100:.2f}%")

    # If track-level mode, compute whole-track voted predictions
    track_results = {}
    if args.split == "track":
        test_df = data["test_df"].copy().reset_index(drop=True)
        for g_idx, g_name in enumerate(GENRES):
            test_df[f"prob_{g_name}"] = ensemble_test_probs[:, g_idx]

        track_preds = (
            test_df.groupby(["track_id", "label_idx"])[[f"prob_{g}" for g in GENRES]]
            .mean()
            .reset_index()
        )
        prob_cols = [f"prob_{g}" for g in GENRES]
        track_preds["pred_idx"] = track_preds[prob_cols].values.argmax(axis=1)
        track_voted_acc = float((track_preds["pred_idx"] == track_preds["label_idx"]).mean())
        track_results = {
            "track_count": len(track_preds),
            "track_voted_accuracy": track_voted_acc,
        }
        print(f"\n>> Track-Level Averaged (Whole Song) Test Accuracy: {track_voted_acc * 100:.2f}%")

    # Save Classification Report
    mlp_report = classification_report(y_test, mlp_test_preds, target_names=GENRES, output_dict=True)
    ensemble_report = classification_report(y_test, ensemble_test_preds, target_names=GENRES, output_dict=True)

    # Save Plots
    save_training_curves_plot(history, output_dir / "training_curves_mlp.png")
    save_confusion_matrix_plot(
        y_test,
        ensemble_test_preds,
        output_dir / "confusion_matrix_ensemble.png",
        f"Genre Confusion Matrix ({args.split.capitalize()} Split - {ensemble_test_acc*100:.2f}% Acc)",
    )
    save_feature_importance_plot(
        data["feature_names"], xgb.feature_importances_, output_dir / "feature_importance.png"
    )

    # Save Model Weights and Scaler
    torch.save(
        {
            "model_state_dict": mlp.state_dict(),
            "in_features": X_train.shape[1],
            "num_classes": len(GENRES),
            "genres": GENRES,
            "test_accuracy": mlp_test_acc,
        },
        output_dir / "deep_audio_mlp.pt",
    )
    with (output_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(data["scaler"], f)

    # Save JSON summary
    summary = {
        "split_mode": args.split,
        "seed": args.seed,
        "features_count": int(X_train.shape[1]),
        "samples": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "results": {
            "deep_audio_mlp_accuracy": mlp_test_acc,
            "xgboost_accuracy": xgb_test_acc,
            "ensemble_accuracy": ensemble_test_acc,
            **track_results,
        },
        "classification_report_ensemble": ensemble_report,
        "top_features": [
            {"feature": data["feature_names"][i], "importance": float(xgb.feature_importances_[i])}
            for i in np.argsort(xgb.feature_importances_)[::-1][:15]
        ],
    }

    results_file = output_dir / "results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nArtifacts successfully saved to {output_dir}:")
    print(f"  - deep_audio_mlp.pt (PyTorch weights)")
    print(f"  - scaler.pkl (Feature standard scaler)")
    print(f"  - results.json (Complete metrics & per-genre F1)")
    print(f"  - training_curves_mlp.png")
    print(f"  - confusion_matrix_ensemble.png")
    print(f"  - feature_importance.png")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
