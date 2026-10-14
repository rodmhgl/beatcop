"""NumPy port of Beat This!'s LogMelSpect (torchaudio MelSpectrogram + log1p(1000 * x)).

Parameters: 22.05 kHz, n_fft 1024, hop 441 (50 fps), 128 Slaney-scale mel bands from 30 Hz to
11 kHz, magnitude (power=1), frame-length normalisation, centred frames with reflect padding.
"""

from functools import lru_cache

import numpy as np

from beatcop.audio import SAMPLE_RATE

N_FFT = 1024
HOP = 441
FPS = SAMPLE_RATE / HOP  # 50.0
N_MELS = 128
F_MIN = 30.0
F_MAX = 11000.0
LOG_MULTIPLIER = 1000.0

_BLOCK_FRAMES = 4096  # bounds peak memory on long files


def _hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        f >= min_log_hz,
        min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep,
        f / f_sp,
    )


def _mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """(n_fft // 2 + 1, n_mels) triangular filters, no area normalisation (torchaudio norm=None)."""
    all_freqs = np.linspace(0, SAMPLE_RATE // 2, N_FFT // 2 + 1)
    m_pts = np.linspace(_hz_to_mel_slaney(F_MIN), _hz_to_mel_slaney(F_MAX), N_MELS + 2)
    f_pts = _mel_to_hz_slaney(m_pts)
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    n = np.arange(N_FFT)
    return (0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32)  # periodic Hann


def log_mel(signal: np.ndarray) -> np.ndarray:
    """Mono 22.05 kHz float32 signal -> (frames, 128) log-mel spectrogram at 50 fps."""
    padded = np.pad(signal.astype(np.float32), N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:n_frames]
    window = _window()
    norm = np.float32(np.sqrt(N_FFT))  # torch.stft(normalized=True), i.e. "frame_length" mode
    fb = mel_filterbank()
    out = np.empty((n_frames, N_MELS), dtype=np.float32)
    for start in range(0, n_frames, _BLOCK_FRAMES):
        block = frames[start : start + _BLOCK_FRAMES] * window
        mag = np.abs(np.fft.rfft(block, axis=1)).astype(np.float32) / norm
        out[start : start + len(block)] = mag @ fb
    return np.log1p(LOG_MULTIPLIER * out)
