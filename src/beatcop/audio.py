"""Audio loading: any libsndfile format -> mono float32 at the model's sample rate."""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 22050


def to_model_input(signal: np.ndarray, sr: int) -> np.ndarray:
    """Downmix to mono (channel mean) and resample to 22.05 kHz with soxr, as Beat This! does."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    elif signal.ndim != 1:
        raise ValueError(f"expected 1D or 2D signal, got shape {signal.shape}")
    if sr != SAMPLE_RATE:
        signal = soxr.resample(signal, in_rate=sr, out_rate=SAMPLE_RATE)
    return signal.astype(np.float32)


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (samples, sample_rate) exactly as stored in the file, shape (n,) or (n, channels)."""
    signal, sr = sf.read(str(path), dtype="float64", always_2d=False)
    return signal, sr
