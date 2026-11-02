# Core Project Scope Freeze and Finalization Record

## Decision

The core project is **10-class music genre classification on GTZAN using audio-feature fusion**. The research question is:

> How does late fusion of mel-spectrogram, CQT, and chroma CRNN predictions compare with the project's audio-classification baselines, and how stable is its performance across seeded splits?

The core is frozen to the existing GTZAN classification experiments and their evaluation. Personalized music recommendation is a separate research problem and is not part of this project's core deliverable.

## Frozen scope

### In scope

- The ten GTZAN labels: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.
- A mel-spectrogram CNN baseline, chroma-rhythm CNN, KNN baseline, and mel/CQT/chroma late-fusion experiments.
- Track-level train/validation/test separation; excerpts from a track must stay within its track's split.
- The final fixed-weight ensemble: independent mel, CQT, and chroma CRNN branches, averaging each branch's ten excerpt probabilities per track and fusing the resulting track probabilities with weights **0.60 / 0.30 / 0.10**.
- Reporting accuracy, macro F1, per-class results, confusion matrices, and variation across seeded runs.

### Out of scope for the core

- Last.fm, LFM-1b, or Million Song Dataset listening histories.
- Artist/title matching between Last.fm and FMA.
- Recommendation-profile methods M1-M5, collaborative filtering, ranking metrics, or recommender deployment.
- Treating FMA genre mapping or Artist20 artist classification as evidence for the GTZAN core result.

Those items can be considered only as a separately approved extension after the core evaluation and report are finalized.

## Implemented pipeline

1. Load up to 30 seconds of mono audio at 22,050 Hz.
2. Generate mel, 96-bin CQT, and 24-bin CQT-chroma features.
3. Divide each track into ten three-second excerpts.
4. Train one CRNN per feature representation; training uses augmentation, validation-based checkpoint selection, and early stopping.
5. Average excerpt probabilities to get one probability vector per track.
6. Combine the three track probability vectors with the fixed weights above, then select the highest-probability genre.

The 20-run evaluation uses a stratified **699/150/150 track** train/validation/test split per seed. Each seed changes the split as well as model initialization. The reported accuracy is the mean and population standard deviation across the 20 run-level test accuracies. The pooled classification report combines 20 test predictions per genre per run (300 per genre, 3,000 predictions total); these are repeated evaluations over a small dataset, not 3,000 unique tracks.

## Results found in the repository

| Experiment | Held-out accuracy | Macro F1 | Evidence and interpretation |
| --- | ---: | ---: | --- |
| Mel CNN | 75.4% | 74.8% | Archived [results](../music_genre_audio_cnn/results/results.json). **Do not use as a confirmed track-level comparator yet:** its report has support 150 per genre while the recorded test split has 150 tracks total (15 per genre). That count mismatch needs to be reconciled by regenerating the report with the current track-level evaluator. |
| Chroma-rhythm CNN | 66.7% | 65.7% | [Results](../music_genre_chroma_rhythm_cnn/results_chroma_rhythm/results.json), 150 test tracks, 15 per genre. |
| KNN | 64.0% | 63.4% | [Results](../music_genre_knn/results/results.json), 150 test tracks, 15 per genre. |
| Mel/CQT fusion search | 80.7% | 80.2% | [Results](../music_genre_audio_cnn/results_fusion/results.json). Validation selected mel 1.00 and CQT 0.00, so this run did not show a useful CQT contribution. |
| Validation-selected mel/CQT/chroma fusion | 82.7% | 82.6% | [Results](../music_genre_audio_cnn/results_fusion_chroma/results.json), one run with weights mel 0.50, CQT 0.20, chroma 0.30. |
| Fixed mel/CQT/chroma fusion | **83.3% mean ± 3.2%** | **83.2% pooled** | [20-run results](../music_genre_audio_cnn/final_fusion_result/results.json), fixed weights mel 0.60, CQT 0.30, chroma 0.10. This is the best currently measured core result. |

Do not describe the final 83.3% as statistically significant or as proof of generalization beyond GTZAN. The standard deviation reflects variation across the recorded runs; it is not a confidence interval. GTZAN is small and has known dataset-quality limitations.

## Dataset and experiment boundaries

GTZAN audio is present outside the repository at `..\GTZAN\genres_original`; the project scripts use that location. The FMA-to-GTZAN utility and FMA/Artist20 training scripts are available, but the organized FMA Small dataset and corresponding FMA/Artist20 result files were not found in the project at review time. Artist20 archives are present but are not extracted into a training directory. They are therefore not part of the evidence for this locked core.

## Finalization checklist

1. **Reconcile the mel baseline.** Re-run or regenerate its evaluation report so that the classification report has one prediction per test track (15 per genre for the recorded 150-track split). If that cannot be done, retain the 75.4% value only as a historical result and exclude it from direct track-level comparisons.
2. **Freeze the comparison table.** Use one consistent evaluation unit and explicitly identify each single-split result versus the 20-run mean. Do not tune on test results.
3. **Preserve the experiment record.** Keep the current seeds, per-seed result JSON files, checkpoints, and aggregate plots alongside the report. The runner accepts `--seeds-from music_genre_audio_cnn/final_fusion_result/results.json` to replay the original seed assignments and writes each run into a new timestamped output directory by default, preventing accidental replacement of the saved results.
4. **Complete the written analysis.** Explain the fixed fusion choice, the run-to-run spread, per-class weaknesses (especially rock), GTZAN limitations, and why external dataset results are excluded.
5. **Freeze scope for submission.** Do not add recommendation datasets or ranking features to the core results. If an extension is later approved, evaluate it as a separate study with its own data split and metrics.

The existing [first progress report](first_progress_report.md) is historical progress documentation. This record clarifies the final scope and adds the evidence caveats that must be resolved before claiming a fully reconciled final comparison.
