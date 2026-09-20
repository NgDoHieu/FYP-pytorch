# Chroma and Rhythm Genre CNN

This standalone GTZAN experiment uses a two-channel input for every three-second excerpt:

- 24-bin CQT chroma for harmonic and pitch-class information.
- A 24-band pooled tempogram for rhythmic periodicity.

The CNN preserves the chroma axis in its first pooling stage and applies only light time masking during training. Track-level validation and test metrics average probabilities across ten excerpts before scoring a complete track.

## Run

Create or select a Python environment with the packages in `requirements.txt`, then run:

```powershell
python train_chroma_rhythm.py
```

Useful options:

```powershell
python train_chroma_rhythm.py --epochs 100 --batch-size 32
```

Outputs are written to `results_chroma_rhythm/`. The saved checkpoint can be evaluated as a separate representation in a later late-fusion comparison with mel and CQT.
