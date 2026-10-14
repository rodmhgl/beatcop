"""Doctor mode: re-track beats from the model's frame-wise beat logits with a dynamic Bayesian
network, for tracks the steadiness test rejects.

Parameters follow madmom's DBNBeatTrackingProcessor defaults: activations resampled to 200 fps
and clipped to [1e-6, 0.999999], observation lambda 16 (non-beat states score
log((1 - p) / 15)), and tempo transitions scored -100 * |new_interval / old_interval - 1|. The
tempo range is centred on the estimated tempo, up to 1.5x either way.
"""

import numpy as np

from beatcop.spectrogram import FPS

DBN_FPS = 200
OBS_LAMBDA = 16
TRANSITION_LAMBDA = 100
TEMPO_SPREAD = 1.5


def _activation(beat_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Beat probability resampled from model frames to DBN frames. Returns (times, probs)."""
    p = 1.0 / (1.0 + np.exp(-beat_logits.astype(np.float64)))
    t_model = np.arange(len(p)) / FPS
    times = np.arange(0.0, t_model[-1], 1.0 / DBN_FPS)
    return times, np.clip(np.interp(times, t_model, p), 1e-6, 0.999999)


def track(beat_logits: np.ndarray, bpm: float, span: tuple[float, float] | None = None) -> np.ndarray:
    """Most likely beat sequence (seconds) for the activation, with tempo near `bpm`.
    `span` limits the result to beats within 0.1 s of the original first..last beat."""
    times, p = _activation(beat_logits)
    beat_obs = np.log(p)
    other_obs = np.log((1.0 - p) / (OBS_LAMBDA - 1))

    frames_per_beat = 60.0 * DBN_FPS / bpm
    intervals = np.arange(
        max(2, int(np.floor(frames_per_beat / TEMPO_SPREAD))),
        int(np.ceil(frames_per_beat * TEMPO_SPREAD)) + 1,
    )
    # transition from interval i (row) to interval j (column) at a beat, log-normalised per row
    trans = -TRANSITION_LAMBDA * np.abs(intervals[None, :] / intervals[:, None] - 1.0)
    trans -= np.logaddexp.reduce(trans, axis=1, keepdims=True)

    # flat state space: for each interval L, positions 0..L-1 (0 = beat)
    offsets = np.concatenate([[0], np.cumsum(intervals)])
    n_states = int(offsets[-1])
    beat_idx = offsets[:-1]
    last_idx = offsets[1:] - 1
    is_beat = np.zeros(n_states, dtype=bool)
    is_beat[beat_idx] = True
    advance_from = np.arange(n_states) - 1  # within an interval, position k comes from k - 1

    delta = np.where(is_beat, beat_obs[0], other_obs[0]) - np.log(n_states)
    back = np.empty((len(times), len(intervals)), dtype=np.int32)
    for t in range(1, len(times)):
        new = np.empty(n_states)
        new[~is_beat] = delta[advance_from[~is_beat]]
        cand = delta[last_idx][:, None] + trans  # previous interval end -> new interval start
        back[t] = cand.argmax(axis=0)
        new[beat_idx] = cand.max(axis=0)
        new += np.where(is_beat, beat_obs[t], other_obs[t])
        delta = new

    # backtrack: positions count down to 0, where the stored choice gives the previous interval
    state = int(delta.argmax())
    interval = int(np.searchsorted(offsets, state, side="right") - 1)
    beats = []
    for t in range(len(times) - 1, -1, -1):
        pos = state - offsets[interval]
        if pos == 0:
            beats.append(times[t])
            if t > 0:
                interval = int(back[t, interval])
                state = int(last_idx[interval])
        else:
            state -= 1
    beats = np.array(beats[::-1])
    if span is not None:
        beats = beats[(beats >= span[0] - 0.1) & (beats <= span[1] + 0.1)]
    return beats
