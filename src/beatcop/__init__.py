"""beatcop: beat, downbeat and tempo-map detection with the Beat This! model on ONNX Runtime."""

from pathlib import Path

import numpy as np

from beatcop import audio, postprocess, spectrogram
from beatcop.model import BeatModel

__all__ = ["BeatModel", "detect", "detect_file", "detect_frames"]


def detect_frames(
    signal: np.ndarray, sr: int, model: BeatModel | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Raw audio -> frame-wise (beat_logits, downbeat_logits) at 50 fps."""
    model = model or BeatModel()
    return model(spectrogram.log_mel(audio.to_model_input(signal, sr)))


def detect(
    signal: np.ndarray, sr: int, model: BeatModel | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Raw audio -> (beat_times, downbeat_times) in seconds."""
    return postprocess.frames_to_beats(*detect_frames(signal, sr, model))


def detect_file(path: str | Path, model: BeatModel | None = None) -> tuple[np.ndarray, np.ndarray]:
    return detect(*audio.load(path), model=model)
