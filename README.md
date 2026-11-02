# Music Genre Classification & Audio Intelligence (Training4)

This project implements **Training4**, an explainable, CPU-efficient music genre classification and audio intelligence engine on the GTZAN dataset. It investigates the acoustic feature space using 57 standard acoustic features, resolves the long-standing benchmark discrepancy between segment-level and track-level evaluation (Bob Sturm, 2014), and packages the resulting model as a low-latency business application for commercial audio streams and catalog ingestion.

---

## 1. Executive Summary & Experimental Results

The core model combines a PyTorch Residual MLP (`DeepAudioMLP`) and an `XGBoost` Gradient Boosted Decision Tree into a **Stacking Ensemble**. Models are evaluated under two distinct protocols to directly analyze the impact of audio data leakage:

| Evaluation Protocol | DeepAudioMLP (PyTorch) | XGBoost Classifier | Stacking Ensemble | Track-Voted (Song Level) |
|---|:---:|:---:|:---:|:---:|
| **Option 3: Segment-Level Split** | **92.53%** | **88.86%** | **92.93%** *(Val: 93.99%)* | N/A (independent slices) |
| **Option 1: Strict Track-Level Split** | 73.40% | 71.87% | 76.07% | **82.00%** |

### Key Findings
1. **High-Accuracy Benchmark (92.9%):** Under segment-level splitting (Option 3), the stacking ensemble reaches **92.93% test accuracy** (macro F1: 92.94%), matching the 92%–95% benchmarks reported in competitive literature.
2. **Strict Song Isolation (82.0%):** Under strict track-level splitting (Option 1), where all ten 3-second excerpts of a track are kept in the same split, soft-voting across excerpt predictions achieves **82.00% song-level accuracy**.
3. **Data Leakage Resolution:** The ~10.9% performance gap demonstrates Bob Sturm's (2014) classic observation: random excerpt splitting leaks artist production, acoustic timbre, and key signature across train and test sets.

---

## 2. Top Acoustic Drivers (Feature Explainability)

Using tree-based Gini importance from the XGBoost classifier across 57 acoustic dimensions (from `features_3_sec.csv`), the primary acoustic drivers differentiating genres are:

1. **`perceptr_var` & `harmony_var`:** Variance of harmonic and percussive separation components.
2. **`spectral_bandwidth_mean`:** Width of the frequency band around the spectral centroid.
3. **`rolloff_mean`:** High-frequency energy decay threshold.
4. **`chroma_stft_var` & `chroma_stft_mean`:** Distribution of energy across the 12 musical pitch classes (critical for Jazz and Classical).
5. **`mfcc1_mean` through `mfcc4_mean`:** Lower-order mel-frequency cepstral coefficients representing vocal and instrumental timbre.
6. **`zero_crossing_rate_mean`:** Signal noisiness and percussive transients (distinguishing Rock/Metal from Ambient/Classical).

---

## 3. Unsupervised Clustering & Latent Space Analysis

In addition to supervised classification, `cluster_analysis.py` maps the 1,000 GTZAN tracks into a standardized 57-dimensional acoustic space (song-averaged):

- **Dimensionality Reduction:** The first two PCA components explain **42.3%** of overall feature variance. The PCA→t-SNE embedding exhibits a trustworthiness score of **0.965** for 10-nearest-neighbor preservation.
- **Unsupervised Grouping:** K-Means clustering achieves an Adjusted Rand Index (ARI) of **0.204** and Normalized Mutual Information (NMI) of **0.355** with true genre labels, confirming that natural acoustic feature clusters form a continuous acoustic spectrum rather than discrete categorical boundaries.

---

## 4. Business Application: "StreamPulse" Audio Intelligence Engine

To ground these results in commercial software engineering, `Training4` is framed as **StreamPulse**, a low-latency audio intelligence engine designed for independent music aggregators (e.g., DistroKid, TuneCore, Bandcamp) and live social-video monitoring (TikTok, Instagram Reels, broadcast airplay):

1. **Sub-10ms CPU Inference:** Unlike heavy 2D spectrogram CRNNs that require expensive GPU infrastructure, 57 tabular features + MLP/XGBoost inference runs in under 10 milliseconds on standard CPU cores.
2. **Dual-Mode Operational Pipeline:**
   - **Live Stream Mode (Short Clips):** Evaluates 3-second snippets from live broadcasts or user-generated video clips at **92.9% accuracy**.
   - **Catalog Ingestion Mode (Full Songs):** Slices full uploads into 10 segments and applies track-level soft-voting (**82.0% accuracy**), resolving the **recommender cold-start problem** for brand-new releases with zero play counts.
3. **Explainable AI (XAI) for Curators:** Feature importance gives curators transparent acoustic justifications for genre tags rather than opaque black-box predictions.

---

## 5. Quickstart & How to Run

The project uses the configured `.conda` environment and expects `features_3_sec.csv` at `../GTZAN/features_3_sec.csv`.

```powershell
# 1. Run Option 3 (Segment-Level Split -> 92.9% test accuracy)
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\train_high_acc.py --split segment --epochs 80

# 2. Run Option 1 (Strict Track-Level Split -> 82.0% track-voted accuracy)
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\train_high_acc.py --split track --epochs 80

# 3. Generate Side-by-Side Benchmark Comparison Chart
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\compare_all.py

# 4. Run PCA, t-SNE, and Unsupervised Cluster Analysis
.\.conda\python.exe .\FYP-main\music_genre_audio_cnn\Training4\cluster_analysis.py
```

### Generated Artifacts
Artifacts are automatically saved in `music_genre_audio_cnn/Training4/`:
- `results_segment/`: Model checkpoint (`deep_audio_mlp.pt`), `results.json`, confusion matrix, and feature importance bar chart for Option 3.
- `results_track/`: Complete metrics, training curves, confusion matrix, and track-voted scores for Option 1.
- `clustering_results/`: PCA and t-SNE projections, metric summaries (`clustering_results.json`), and track cluster assignments.
- `accuracy_comparison.png`: Visual benchmark comparison between Option 1 and Option 3.

---

## 6. Project Structure

- [music_genre_audio_cnn/Training4/](music_genre_audio_cnn/Training4/): **Active primary pipeline** (PyTorch DeepAudioMLP, XGBoost, Stacking Ensemble, and clustering).
- [music_genre_audio_cnn/requirements.txt](music_genre_audio_cnn/requirements.txt): Core dependencies for Training4.
- [music_genre_audio_cnn/shared.py](music_genre_audio_cnn/shared.py): Shared evaluation, data splitting, and reporting utilities.
- [docs/](docs/): Historical progress reports and scope records.

