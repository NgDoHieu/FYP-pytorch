# Music Genre Audio Intelligence & CNN Architectures

This directory contains the audio classification models and experiments for the FYP project. The **primary active pipeline moving forward is [Training4/](Training4/)**, which implements high-accuracy tabular acoustic ensembling and data leakage analysis on GTZAN.

---

## 1. Active Focus: Training4 (High-Accuracy Stacking Ensemble)

Located in `music_genre_audio_cnn/Training4/`, this module implements:
- **PyTorch `DeepAudioMLP`** (Residual MLP with cosine annealing and label smoothing).
- **`XGBoost` Gradient Boosted Decision Tree Classifier**.
- **Stacking Meta-Ensemble** combining both representations.
- **Leakage Analysis:** Compares **Option 3** (Segment-level split: **92.93% test accuracy**) with **Option 1** (Strict track isolation: **82.00% track-voted accuracy**), providing an empirical solution to Bob Sturm's (2014) GTZAN segment leakage phenomenon.
- **Unsupervised Acoustic Latent Space:** PCA, t-SNE, K-Means, and Gaussian Mixture Models (GMM) on 57-dimensional song-averaged feature vectors.

### Running Training4
```powershell
# Segment-level evaluation (92.9% accuracy)
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\train_high_acc.py --split segment --epochs 80

# Strict track-level evaluation (82.0% track-voted accuracy)
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\train_high_acc.py --split track --epochs 80

# Benchmark comparison plot
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\compare_all.py

# Unsupervised clustering & latent embeddings
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\cluster_analysis.py
```

See [Training4/README.md](Training4/README.md) for full documentation, per-class metrics, confusion matrices, and feature importance analyses.

---

## 2. Directory Layout

```text
music_genre_audio_cnn/
├── Training4/               # ACTIVE PIPELINE: DeepAudioMLP, XGBoost, Stacking Ensemble & Clustering
│   ├── dataset.py           # Stratified data loader for segment and track splits
│   ├── model.py             # PyTorch DeepAudioMLP and XGBoost definitions
│   ├── train_high_acc.py    # Main training and evaluation engine
│   ├── compare_all.py       # Comparative benchmark chart generator
│   ├── cluster_analysis.py  # PCA, t-SNE, K-Means, and GMM analysis
│   ├── results_segment/     # Option 3 artifacts (92.93% accuracy)
│   ├── results_track/       # Option 1 artifacts (82.00% track-voted accuracy)
│   └── clustering_results/  # Unsupervised projections and metrics
├── requirements.txt         # Core dependencies for Training4
├── shared.py                # Reusable track ID, evaluation, and plotting helpers
├── Training4_FMA/           # Exploratory tabular experiments on FMA Small
└── (Historical CNN files)   # Legacy spectrogram CNN/CRNN scripts (train.py, final_fusion_result.py)
```

---

## 3. Dependencies

Dependencies are listed in [requirements.txt](requirements.txt). The primary pipeline uses `torch`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn`. Install into the workspace `.conda` environment with:

```powershell
.\.conda\python.exe -m pip install -r .\FYP-main\music_genre_audio_cnn\requirements.txt
```

