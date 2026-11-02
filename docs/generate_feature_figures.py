"""Generate representative mel, CQT, chroma, and tempogram figures for the report."""

import os
from pathlib import Path
import sys

if sys.platform == "win32":
    dll_directory = Path(sys.executable).resolve().parent / "Library" / "bin"
    if dll_directory.is_dir():
        os.add_dll_directory(str(dll_directory))
        os.environ["PATH"] = f"{dll_directory}{os.pathsep}{os.environ['PATH']}"

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 22_050
HOP_LENGTH = 512
MEL_BINS = 128
CQT_BINS = 96
CHROMA_BINS = 24


def save_figure(data: np.ndarray, title: str, y_axis: str | None, colorbar_label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    image = librosa.display.specshow(
        data,
        x_axis="time",
        y_axis=y_axis,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        ax=axis,
    )
    axis.set_title(title)
    if y_axis is None:
        axis.set_ylabel("Pooled tempo band")
    figure.colorbar(image, ax=axis, format="%+2.0f dB" if colorbar_label == "dB" else None, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_audio(path: Path) -> np.ndarray:
    source_rate, audio = wavfile.read(path)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(wavfile.read(path, mmap=True)[1].dtype, np.integer):
        audio /= np.iinfo(wavfile.read(path, mmap=True)[1].dtype).max
    if source_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, source_rate).astype(np.float32)
    return audio[: SAMPLE_RATE * 30]


def main() -> None:
    document_dir = Path(__file__).resolve().parent
    audio_path = document_dir.parents[1] / "GTZAN" / "genres_original" / "blues" / "blues.00000.wav"
    output_dir = document_dir / "Figures"
    output_dir.mkdir(exist_ok=True)

    audio = load_audio(audio_path)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2_048,
        hop_length=HOP_LENGTH,
        n_mels=MEL_BINS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    save_figure(
        mel_db,
        "128-bin Mel Spectrogram",
        "mel",
        "dB",
        output_dir / "mel_spectrogram.png",
    )

    cqt = librosa.cqt(
        audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz("C1"),
        n_bins=CQT_BINS,
        bins_per_octave=12,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    save_figure(
        cqt_db,
        "96-bin Constant-Q Transform (CQT) Spectrogram",
        "cqt_note",
        "dB",
        output_dir / "cqt_spectrogram.png",
    )

    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz("C1"),
        n_chroma=CHROMA_BINS,
        n_octaves=7,
        bins_per_octave=48,
    )
    save_figure(
        chroma,
        "24-bin CQT Chroma",
        "chroma",
        "Normalized magnitude",
        output_dir / "cqt_chroma.png",
    )

    onset_envelope = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        win_length=384,
    )
    rhythm = np.stack([band.mean(axis=0) for band in np.array_split(tempogram, CHROMA_BINS, axis=0)])
    rhythm /= np.maximum(rhythm.max(axis=1, keepdims=True), 1e-8)
    save_figure(
        rhythm,
        "24-band Pooled Tempogram",
        None,
        "Normalized rhythmic periodicity",
        output_dir / "pooled_tempogram.png",
    )


if __name__ == "__main__":
    main()