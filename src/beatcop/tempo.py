"""Turn detected beats into a tempo map: a locked single tempo or a per-beat following map.

Detection modes:
  single - force one locked tempo (least-squares grid through the beats)
  multi  - follow every detected beat (one tempo per beat interval)
  auto   - single when the beats sit on a steady grid, multi otherwise
"""

from dataclasses import dataclass, field

import numpy as np

STEADY_RMS_S = 0.020  # beats within ~1 frame RMS of a straight grid count as "steady"
MAX_METER = 12


@dataclass
class TempoMap:
    beats: np.ndarray
    downbeats: np.ndarray
    mode: str  # "single" or "multi"
    # (beat_index, beats_per_bar) at each metre change; beat_index refers to `beats`
    meters: list[tuple[int, int]] = field(default_factory=list)

    @property
    def bpm(self) -> float:
        if len(self.beats) < 3:
            return float("nan") if len(self.beats) < 2 else 60.0 / float(np.diff(self.beats)[0])
        return 60.0 / fit_grid(self.beats)[1]

    @property
    def beat_bpms(self) -> np.ndarray:
        return 60.0 / np.diff(self.beats)


@dataclass(frozen=True)
class Bar:
    first_beat: int  # index into TempoMap.beats
    beats: int  # beats (quarter notes) in the bar
    start: float  # seconds
    end: float  # seconds; the last bar is extrapolated past the final detected beat


def bars(tm: TempoMap) -> list[Bar]:
    """The bars of a tempo map in audio time: a pickup bar for any beats before the first bar
    line (whole bars of them stay whole bars), then bars of the current metre."""
    beats = tm.beats
    n = len(beats)
    last_ibi = float(beats[-1] - beats[-2])

    def at(i: int) -> float:
        return float(beats[i]) if i < n else float(beats[-1]) + (i - n + 1) * last_ibi

    starts: list[tuple[int, int]] = []
    first_bar, first_meter = tm.meters[0]
    pickup = first_bar % first_meter
    if pickup:
        starts.append((0, pickup))
    i = pickup
    changes = dict(tm.meters)
    meter = first_meter
    while i < n:
        meter = changes.get(i, meter)
        # a metre change mid-bar (from --meter or odd downbeats) cuts the bar short there
        nxt = next((c for c in changes if i < c < i + meter), i + meter)
        starts.append((i, nxt - i))
        i = nxt
    return [Bar(i, k, at(i), at(i + k)) for i, k in starts]


def beat_indices(beats: np.ndarray) -> np.ndarray:
    """Integer grid position of each beat, so a dropped beat leaves a gap instead of shifting the rest."""
    ibi = np.diff(beats)
    steps = np.maximum(1, np.rint(ibi / np.median(ibi))).astype(int)
    return np.concatenate([[0], np.cumsum(steps)])


def fit_grid(beats: np.ndarray) -> tuple[float, float, float]:
    """Least-squares straight grid. Returns (t0, period, rms_residual_seconds)."""
    idx = beat_indices(beats)
    period, t0 = np.polyfit(idx, beats, 1)
    rms = float(np.sqrt(np.mean((beats - (t0 + period * idx)) ** 2)))
    return float(t0), float(period), rms


def _snap(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if not len(times) or not len(grid):
        return np.array([], dtype=np.float64)
    return np.unique(grid[np.abs(grid[None, :] - times[:, None]).argmin(axis=1)])


def rescale(beats: np.ndarray, downbeats: np.ndarray, how: str) -> np.ndarray:
    """Octave fix: "half" keeps every other beat (phase-locked to the first downbeat), "double"
    inserts midpoints."""
    if how == "double":
        mids = (beats[:-1] + beats[1:]) / 2
        return np.sort(np.concatenate([beats, mids]))
    if how == "half":
        phase = int(np.searchsorted(beats, downbeats[0])) % 2 if len(downbeats) else 0
        return beats[phase::2]
    raise ValueError(how)


def meters_from_downbeats(beats: np.ndarray, downbeats: np.ndarray) -> list[tuple[int, int]]:
    """Metre changes as (beat_index, beats_per_bar).

    Downbeats less than 2 beats after the previous one are treated as spurious. The most common
    bar length is the song's metre; a gap that is a whole number of those bars (missed downbeats)
    keeps it, anything else is written as an odd bar so later bar lines stay on the downbeats.
    """
    raw = [p for p in np.searchsorted(beats, downbeats).tolist() if p < len(beats)]
    if len(raw) < 2:
        return [(raw[0] if raw else 0, 4)]
    gaps = np.diff(raw)
    regular = gaps[(gaps >= 2) & (gaps <= MAX_METER)]
    meter = int(np.bincount(regular).argmax()) if len(regular) else 4

    pos: list[int] = []
    for i, p in enumerate(raw):
        if pos and p - pos[-1] < 2:
            # two bar lines a beat apart: keep the one a whole number of bars before the next
            nxt = raw[i + 1] if i + 1 < len(raw) else None
            if nxt is not None and (nxt - p) % meter == 0 and (nxt - pos[-1]) % meter != 0:
                pos[-1] = p
            continue
        pos.append(p)
    if len(pos) < 2:
        return [(pos[0], meter)]
    counts = np.diff(pos)
    changes: list[tuple[int, int]] = []
    for p, count in zip(pos[:-1], counts.tolist()):
        want = meter if count % meter == 0 else count
        if not changes or want != changes[-1][1]:
            changes.append((p, want))
    return changes


def build(
    beats: np.ndarray,
    downbeats: np.ndarray,
    mode: str = "auto",
    scale: str | None = None,
    meter: int | None = None,
) -> TempoMap:
    if scale:
        beats = rescale(beats, downbeats, scale)
        downbeats = _snap(downbeats, beats)

    if len(beats) >= 3 and mode in ("auto", "single"):
        t0, period, rms = fit_grid(beats)
        if mode == "single" or rms <= STEADY_RMS_S:
            idx = beat_indices(beats)
            beats = t0 + period * np.arange(idx[0], idx[-1] + 1)
            beats = beats[beats >= 0]
            downbeats = _snap(downbeats, beats)
            mode = "single"
    if mode != "single":
        mode = "multi"

    if meter:
        first = int(np.searchsorted(beats, downbeats[0])) if len(downbeats) else 0
        meters = [(first, meter)]
        downbeats = beats[first::meter]
    else:
        meters = meters_from_downbeats(beats, downbeats)
    return TempoMap(beats=beats, downbeats=downbeats, mode=mode, meters=meters)
