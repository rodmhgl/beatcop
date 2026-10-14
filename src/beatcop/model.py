"""Chunked ONNX inference, mirroring beat_this.inference.split_predict_aggregate.

The model sees 30 s windows (1500 frames). Each chunk's outer 6 frames are discarded because the
model was not trained on window edges, so consecutive chunks overlap by that border and earlier
chunks win on overlap ("keep_first").
"""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

CHUNK_FRAMES = 1500
BORDER_FRAMES = 6
_UNSET = -1000.0  # logit for "probability zero"

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "beatthis_final0.onnx"


def model_path(explicit: str | Path | None = None) -> Path:
    path = Path(explicit or os.environ.get("BEATCOP_MODEL") or DEFAULT_MODEL)
    if not path.is_file():
        raise FileNotFoundError(
            f"model not found at {path}; run scripts/export_onnx.py or set BEATCOP_MODEL"
        )
    return path


def _split(n_frames: int) -> np.ndarray:
    step = CHUNK_FRAMES - 2 * BORDER_FRAMES
    starts = np.arange(-BORDER_FRAMES, n_frames - BORDER_FRAMES, step)
    if n_frames > step:
        # shift the last chunk left so it ends at the end of the piece instead of being short
        starts[-1] = n_frames - (CHUNK_FRAMES - BORDER_FRAMES)
    return starts


class BeatModel:
    def __init__(self, path: str | Path | None = None, threads: int | None = None):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path(path)), opts, providers=["CPUExecutionProvider"]
        )
        # A fixed-length export (e.g. one with input [1, 1500, 128]) needs chunks padded to size.
        dim = self.session.get_inputs()[0].shape[1]
        self.fixed_frames = dim if isinstance(dim, int) else None

    def _run(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(chunk)
        if self.fixed_frames and n < self.fixed_frames:
            chunk = np.pad(chunk, ((0, self.fixed_frames - n), (0, 0)))
        beat, downbeat = self.session.run(None, {"spect": chunk[None].astype(np.float32)})
        return beat[0, :n], downbeat[0, :n]

    def __call__(self, spect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(frames, 128) log-mel -> (beat_logits, downbeat_logits), each shape (frames,)."""
        n = len(spect)
        beat = np.full(n, _UNSET, dtype=np.float32)
        downbeat = np.full(n, _UNSET, dtype=np.float32)
        # reverse order so earlier chunks overwrite later ones on overlap
        for start in _split(n)[::-1]:
            lo, hi = max(start, 0), min(start + CHUNK_FRAMES, n)
            chunk = np.pad(
                spect[lo:hi],
                ((max(0, -start), max(0, min(BORDER_FRAMES, start + CHUNK_FRAMES - n))), (0, 0)),
            )
            b, d = self._run(chunk)
            b, d = b[BORDER_FRAMES:-BORDER_FRAMES], d[BORDER_FRAMES:-BORDER_FRAMES]
            dst = slice(start + BORDER_FRAMES, start + BORDER_FRAMES + len(b))
            beat[dst], downbeat[dst] = b, d
        return beat, downbeat
