"""Frame logits -> beat/downbeat times. Port of Beat This!'s "minimal" postprocessor."""

import numpy as np

from beatcop.spectrogram import FPS

PEAK_RADIUS = 3  # max-pool window 7 frames = +/-70 ms at 50 fps


def _peaks(logits: np.ndarray) -> np.ndarray:
    """Frames that are the maximum within +/-70 ms and above probability 0.5 (logit > 0)."""
    padded = np.pad(logits, PEAK_RADIUS, constant_values=-np.inf)
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * PEAK_RADIUS + 1).max(axis=1)
    return np.flatnonzero((logits == local_max) & (logits > 0))


def deduplicate(frames: np.ndarray, width: int = 1) -> np.ndarray:
    """Merge runs of peaks no more than `width` frames apart into their running mean."""
    result = []
    for f in frames.tolist():
        if result and f - p <= width:
            c += 1
            p += (f - p) / c
            result[-1] = p
        else:
            p, c = float(f), 1
            result.append(p)
    return np.array(result, dtype=np.float64)


def frames_to_beats(
    beat_logits: np.ndarray, downbeat_logits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (beat_times, downbeat_times) in seconds; every downbeat is snapped onto a beat."""
    beats = deduplicate(_peaks(beat_logits)) / FPS
    downbeats = deduplicate(_peaks(downbeat_logits)) / FPS
    if len(beats):
        downbeats = beats[np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1)]
    return beats, np.unique(downbeats)
