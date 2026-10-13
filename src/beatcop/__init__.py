"""beatcop: beat, downbeat and tempo-map detection with the Beat This! model on ONNX Runtime."""

from pathlib import Path

import numpy as np

from beatcop import audio, postprocess, spectrogram
from beatcop.model import BeatModel

__all__ = ["BeatModel", "detect", "detect_file"]


def detect(
    signal: np.ndarray, sr: int, model: BeatModel | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Raw audio -> (beat_times, downbeat_times) in seconds."""
    model = model or BeatModel()
    spect = spectrogram.log_mel(audio.to_model_input(signal, sr))
    return postprocess.frames_to_beats(*model(spect))


def detect_file(path: str | Path, model: BeatModel | None = None) -> tuple[np.ndarray, np.ndarray]:
    return detect(*audio.load(path), model=model)
