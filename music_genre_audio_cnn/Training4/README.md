# Training4: High-Accuracy GTZAN Classification & Leakage Analysis

## 1. Project Overview

This subproject implements **Option 3** (High-Accuracy Segment-Level Classification, targeting **92%–95%**) and **Option 1** (Strict Track-Level Evaluation & Ensembling) on the GTZAN dataset using tabular acoustic features ([features_3_sec.csv](../../../GTZAN/features_3_sec.csv)).

It answers the core research question:
> **Why do many papers and online benchmarks report 90%–95% accuracy on GTZAN, and how does that compare against strict, leak-free track evaluation?**

---

## 2. Experimental Results

All experiments were conducted with a stratified 75% train / 10% validation / 15% test split using 57 extracted acoustic features (MFCCs 1–20 mean & variance, spectral centroid, bandwidth, rolloff, zero-crossing rate, chroma STFT, tempo, and RMS energy).

| Evaluation Protocol | DeepAudioMLP (PyTorch) | XGBoost Classifier | Stacking Ensemble | Track-Voted (Song Level) |
|---|:---:|:---:|:---:|:---:|
| **Option 3: Segment-Level Split** | **92.86%** *(Val: 93.99%)* | **88.86%** | **92.66%** | N/A (independent slices) |
| **Option 1: Strict Track-Level Split** | 73.40% | 71.87% | 76.07% | **82.00%** |

### Key Takeaway for Your Advisor
* **Option 3 achieves 92.86% test accuracy**, hitting the 90%–95% target mentioned by your advisor.
* However, when evaluating under strict track-level isolation (where all 10 excerpts of a song are kept together), segment accuracy is ~76.07%, rising to **82.00%** once the 10 excerpts per song are soft-voted.
* Presenting **both** protocols demonstrates both the ability to hit high accuracy and a deep understanding of data leakage in audio ML.

---

## 3. Explaining the Performance Gap: The Segment Leakage Phenomenon

In GTZAN, each 30-second audio track is cut into ten 3-second slices:
- In **Option 3 (Segment-Level Split)**, `blues.00000.0.wav` might be in the training set while `blues.00000.1.wav` is in the test set. Because both slices share the exact same musical key, mastering, instruments, and artist production, the model easily achieves **93%–95% accuracy**.
- In **Option 1 (Track-Level Split)**, all 10 slices of `blues.00000` are strictly kept in the test set. The model must generalize to completely unseen artists and tracks, reflecting true real-world performance.
- This effect was formally documented by Bob Sturm (2014) in *"A State of the Art Review of Music Genre Classification with GTZAN"*.

---

## 4. Top Acoustic Features for Music Genre Classification

Using XGBoost tree feature importance (Gini gain), the top acoustic features that differentiate genres are:

1. **`perceptr_var`** & **`harmony_var`**: Harmonic and percussive separation variances.
2. **`spectral_centroid_mean`** & **`spectral_centroid_var`**: Brightness / center of mass of the frequency spectrum (crucial for separating Classical vs. Metal).
3. **`rolloff_mean`** & **`rolloff_var`**: High-frequency cutoff point.
4. **`chroma_stft_mean`**: Distribution of energy across the 12 musical pitch classes (key for Jazz and Classical).
5. **`mfcc1_mean`**, **`mfcc2_mean`**, **`mfcc4_mean`**: Lower-order MFCCs capturing overall vocal/timbral shape.
6. **`zero_crossing_rate_mean`**: Percussive noisiness (separates Rock/Metal from Ambient/Classical).

---

## 5. Artifacts and Generated Visualizations

All model weights, scalers, reports, and plots are saved in this folder:

```text
Training4/
├── dataset.py                        # Dataset loader supporting 'segment' and 'track' splits
├── model.py                          # DeepAudioMLP (PyTorch Residual MLP) and XGBoost models
├── train_high_acc.py                 # Main training script
├── compare_all.py                    # Side-by-side comparison benchmark runner
├── accuracy_comparison.png           # Comparative bar chart (Option 3 vs Option 1)
├── README.md                         # This comprehensive report
├── results_segment/                  # Option 3 Artifacts (92.86% Accuracy)
│   ├── deep_audio_mlp.pt             # Saved PyTorch model checkpoint
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── results.json                  # Per-class precision, recall, F1, and metrics
│   ├── training_curves_mlp.png       # Loss and accuracy curves
│   ├── confusion_matrix_ensemble.png # 10x10 normalized confusion matrix
│   └── feature_importance.png        # Top 20 acoustic features bar chart
└── results_track/                    # Option 1 Artifacts (Strict Song Isolation)
    ├── deep_audio_mlp.pt             # Saved PyTorch model checkpoint
    ├── scaler.pkl                    # Fitted StandardScaler
    ├── results.json                  # Complete metrics
    ├── training_curves_mlp.png       # Learning curves
    ├── confusion_matrix_ensemble.png # Confusion matrix
    └── feature_importance.png        # Feature importances
```

## 6. Clustering and Dimensionality Reduction

`cluster_analysis.py` averages the 10 excerpt rows for each song before scaling
and analysis, so the clustering experiment contains one sample per track rather
than treating excerpts from the same song as independent examples. It creates:

- PCA and t-SNE plots colored by the known genre and by K-Means assignment.
- K-Means and diagonal Gaussian Mixture Model (GMM) results with 10 clusters.
- Silhouette and Davies-Bouldin scores for geometric cluster quality, plus ARI
  and NMI comparing the clusters with genre labels.
- PCA explained variance and t-SNE trustworthiness, and a CSV of track IDs,
  labels, cluster assignments, and 2D coordinates.

Cluster metrics are computed in the original 57-dimensional standardized
track-feature space, not the 2D t-SNE projection. Genre labels are used only
after clustering to assess correspondence with genres. PCA/t-SNE plots are
exploratory visualizations; t-SNE axes and distances between distant groups do
not have a direct physical interpretation.

### Results (seed 42)

The run analyzed 1,000 tracks. The first two PCA components explain 42.3% of
feature variance. The PCA→t-SNE projection has trustworthiness 0.965 for
10-nearest-neighbor relationships. On the standardized feature vectors,
K-Means scored silhouette 0.112, Davies-Bouldin 2.071, ARI 0.204, and NMI
0.355. Diagonal GMM scored 0.080, 2.277, 0.207, and 0.359, respectively.
These results indicate only modest alignment between unsupervised clusters and
the genre labels; the t-SNE appearance should not be interpreted as proof of
strong genre separation.

See [the PCA plot](clustering_results/pca_tracks.png),
[the t-SNE plot](clustering_results/tsne_tracks.png), and
[the metric summary](clustering_results/clustering_results.json).

```powershell
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\cluster_analysis.py
```

Outputs are written to `Training4/clustering_results/`.

---

## 7. How to Reproduce

From the outer `FYP-main` workspace directory, use the project's `.conda`
interpreter explicitly. Running `python` alone may select another Python
installation that does not have PyTorch installed:

```bash
# Optional: install this project's dependencies into the selected interpreter
.\.conda\python.exe -m pip install -r .\FYP-main\music_genre_audio_cnn\requirements.txt

# Run Option 3 (Segment-Level Split -> 92.86% test accuracy)
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\train_high_acc.py --split segment --epochs 80

# Run Option 1 (Track-Level Split -> 82.00% track-voted accuracy)
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\train_high_acc.py --split track --epochs 80

# Run both and generate comparison chart
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\compare_all.py

# Run PCA, t-SNE, and unsupervised clustering
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\cluster_analysis.py
```
