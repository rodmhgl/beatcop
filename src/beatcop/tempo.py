"""Turn detected beats into a tempo map: a locked single tempo or a per-beat following map.

Detection modes (see analyse_grid for the steadiness test):
  single - force one locked tempo (the grid most beats land on)
  multi  - follow every detected beat (one tempo per beat interval)
  auto   - single when the beats sit on a steady grid, multi otherwise
  doctor - like auto, but an unsteady track is re-tracked with a DBN (beatcop.dbn) first
"""

from dataclasses import dataclass, field

import numpy as np

MAX_METER = 12

# Steadiness test constants.
MIN_BEATS = 8
TEMPO_TOL = 0.06  # an interval is "consistent" within 6% of the tempo...
MAX_SKIP = 3  # ...even if it spans up to 3 beats (missed beats)
ON_GRID_S = 0.025  # a beat within 25 ms of the fitted grid is on it
WINDOW = 12  # local steadiness is judged over 12-beat windows...
WINDOW_ON_GRID = 0.75  # ...needing 75% of their beats on grid
STEADY_PCT = 85.0  # lock when 85% of windows are steady
MIN_RUN = 16  # second chance: steady runs of at least 16 beats
RELATED_RATIOS = (2 / 3, 3 / 4, 4 / 3, 3 / 2)  # tempo changes that still allow one grid


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


def _phase(beats: np.ndarray, period: float) -> float:
    """Grid offset in seconds from the circular mean of the beats' positions in the period."""
    ang = 2 * np.pi * beats / period
    return float(np.arctan2(np.sin(ang).sum(), np.cos(ang).sum()) / (2 * np.pi) * period)


def _on_grid(beats: np.ndarray, period: float) -> np.ndarray:
    phase = _phase(beats, period)
    return np.abs(beats - (np.rint((beats - phase) / period) * period + phase)) < ON_GRID_S


def refine_bpm(beats: np.ndarray, bpm: float) -> float:
    """Two-stage grid search for the tempo the most beats land on: 0.9-1.1x in 0.05 BPM steps,
    then +/-0.1 BPM in 0.004 BPM steps. Ties on the on-grid fraction go to the smallest mean
    residual."""

    def score(b: float) -> tuple[float, float]:
        period = 60.0 / b
        phase = _phase(beats, period)
        res = np.abs(beats - (np.rint((beats - phase) / period) * period + phase))
        return float(np.mean(res < ON_GRID_S)), -float(np.mean(np.minimum(res, ON_GRID_S)))

    coarse = np.arange(bpm * 0.9, bpm * 1.1, 0.05)
    best = max(coarse, key=lambda b: score(b)[0])
    fine = np.arange(best - 0.1, best + 0.1, 0.004)
    return float(max(fine, key=score))


def round_bpm(bpm: float) -> float:
    """Snap to a whole BPM when within 0.25 of it, else keep two decimals."""
    whole = round(bpm)
    return float(whole) if abs(bpm - whole) < 0.25 else round(bpm, 2)


@dataclass
class GridStats:
    stable: bool = False
    bpm: float = float("nan")  # refined tempo of the steady part
    steady_pct: float = 0.0  # % of 12-beat windows with >= 75% of beats on grid
    first: int = 0  # index of the first beat after any intro
    second_chance: bool = False  # locked because steady runs bracket off-grid breaks


def _steady_windows(on_grid: np.ndarray) -> np.ndarray:
    n = len(on_grid)
    counts = np.array([on_grid[max(0, i - WINDOW // 2): i + WINDOW // 2].sum() for i in range(n)])
    return counts / WINDOW >= WINDOW_ON_GRID


def _second_chance(beats: np.ndarray, steady: np.ndarray, bpm: float) -> bool:
    """Lock anyway if long steady runs bracket breaks that are off-grid rather than in a new tempo.

    A steady stretch inside a break is one whose intervals vary by under 3%."""
    runs, i, n = [], 0, len(steady)
    while i < n:
        if steady[i]:
            j = i
            while j < n and steady[j]:
                j += 1
            if j - i >= MIN_RUN:
                runs.append((i, j))
            i = j
        else:
            i += 1
    span = beats[-1] - beats[0]
    if not runs or span <= 0:
        return False
    covered = sum(beats[j - 1] - beats[i] for i, j in runs)
    if (covered / span < 0.4 or beats[runs[0][0]] - beats[0] > 0.25 * span
            or beats[-1] - beats[runs[-1][1] - 1] > 0.25 * span):
        return False
    period = 60.0 / bpm
    ibi = np.diff(beats)
    gap = ~steady[:-1]
    if gap.any() and np.mean(np.abs(ibi[gap] - period) / period <= TEMPO_TOL) > 0.35:
        return False
    for (_, a), (b, _) in zip(runs[:-1], runs[1:]):
        seg = ibi[a:b]
        for k in range(0, max(0, len(seg) - MIN_RUN) + 1):
            win = seg[k : k + MIN_RUN]
            if len(win) < MIN_RUN or np.std(win) / np.mean(win) >= 0.03:
                continue
            ratio = period / np.mean(win)
            if 0.6 <= ratio <= 1.67 and not any(abs(ratio - r) / r <= 0.03 for r in RELATED_RATIOS):
                return False
    return True


def analyse_grid(beats: np.ndarray) -> GridStats:
    """Is the track steady enough to lock to one tempo? See the README for the rules."""
    if len(beats) < MIN_BEATS:
        return GridStats()
    ibi = np.diff(beats)
    bpms = 60.0 / np.maximum(ibi, 0.001)
    med = np.median(bpms)
    plausible = ibi[(bpms > 0.75 * med) & (bpms < 1.5 * med)]
    period = float(np.median(plausible if len(plausible) else ibi))
    tempo = 60.0 / period

    k = np.maximum(1, np.rint(ibi / period))
    consistent = (np.abs(60.0 / (ibi / k) - tempo) / tempo <= TEMPO_TOL) & (k <= MAX_SKIP)
    first = int(np.argmax(consistent)) if consistent.any() else 0

    tail = beats[first:]
    stats = GridStats(bpm=tempo, first=first)
    if len(tail) < WINDOW:
        return stats
    stats.bpm = refine_bpm(tail, tempo)
    steady = _steady_windows(_on_grid(tail, 60.0 / stats.bpm))
    stats.steady_pct = 100.0 * float(steady.mean())
    stats.stable = stats.steady_pct >= STEADY_PCT
    if not stats.stable and _second_chance(tail, steady, stats.bpm):
        stats.stable = stats.second_chance = True
    return stats


def lock_grid(beats: np.ndarray, bpm: float) -> np.ndarray:
    """A constant-tempo grid at `bpm`, phase-fitted to `beats`, spanning the beats."""
    period = 60.0 / bpm
    phase = _phase(beats, period)
    k0 = int(np.rint((beats[0] - phase) / period))
    k1 = int(np.rint((beats[-1] - phase) / period))
    grid = phase + period * np.arange(k0, k1 + 1)
    return grid[grid >= 0]


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
    beat_logits: np.ndarray | None = None,
) -> TempoMap:
    """Turn detected beats into a TempoMap. `beat_logits` (the model's frame-wise beat output)
    is only needed for mode="doctor"."""
    if scale:
        beats = rescale(beats, downbeats, scale)
        downbeats = _snap(downbeats, beats)

    stats = analyse_grid(beats)
    locked = mode == "single" or (mode in ("auto", "doctor") and stats.stable)
    if mode == "doctor" and not locked and beat_logits is not None and len(beats) >= MIN_BEATS:
        from beatcop.dbn import track

        beats = track(beat_logits, stats.bpm, span=(beats[0], beats[-1]))
        downbeats = _snap(downbeats, beats)
    if locked and len(beats) >= 3:
        bpm = stats.bpm if np.isfinite(stats.bpm) else 60.0 / fit_grid(beats)[1]
        # beats before the first consistent interval are an intro: keep them as detected
        beats = np.concatenate([beats[: stats.first], lock_grid(beats[stats.first:], round_bpm(bpm))])
        downbeats = _snap(downbeats, beats)
    mode = "single" if locked else "multi"

    if meter:
        first = int(np.searchsorted(beats, downbeats[0])) if len(downbeats) else 0
        meters = [(first, meter)]
        downbeats = beats[first::meter]
    else:
        meters = meters_from_downbeats(beats, downbeats)
    return TempoMap(beats=beats, downbeats=downbeats, mode=mode, meters=meters)
