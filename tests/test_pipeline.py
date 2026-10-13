import mido
import numpy as np
import pytest

from beatcop import midi, postprocess, spectrogram, tempo
from beatcop.model import DEFAULT_MODEL, _split, CHUNK_FRAMES, BORDER_FRAMES


def test_peaks_threshold_window_and_dedupe():
    logits = np.full(200, -5.0, dtype=np.float32)
    logits[[50, 51]] = 3.0  # adjacent equal peaks -> merged to 50.5
    logits[100], logits[102] = 2.0, 1.0  # 102 is within +/-3 frames of a bigger peak -> dropped
    logits[150] = -0.1  # below probability 0.5 -> dropped
    db = np.full(200, -5.0, dtype=np.float32)
    db[99] = 1.0  # snaps onto the nearest beat
    beats, downbeats = postprocess.frames_to_beats(logits, db)
    np.testing.assert_allclose(beats, [50.5 / 50, 100 / 50])
    np.testing.assert_allclose(downbeats, [2.0])


def test_split_covers_every_frame_once_after_borders():
    for n in (10, 1488, 1489, 5000):
        covered = np.zeros(n, dtype=int)
        for s in _split(n):
            covered[max(0, s + BORDER_FRAMES) : min(n, s + CHUNK_FRAMES - BORDER_FRAMES)] += 1
        assert covered.min() >= 1


def test_spectrogram_shape_and_rate():
    x = np.zeros(22050 * 3, dtype=np.float32)
    assert spectrogram.log_mel(x).shape == (151, 128)
    assert spectrogram.FPS == 50


def _grid(bpm, n, start=0.5, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return start + np.arange(n) * 60 / bpm + rng.normal(0, jitter, n)


def test_auto_locks_steady_and_follows_rubato():
    steady = _grid(120, 64, jitter=0.005)
    tm = tempo.build(steady, steady[::4], mode="auto")
    assert tm.mode == "single" and tm.bpm == pytest.approx(120, abs=0.2)

    ibis = 60 / np.linspace(90, 130, 63)
    rubato = np.concatenate([[0.5], 0.5 + np.cumsum(ibis)])
    assert tempo.build(rubato, rubato[::4], mode="auto").mode == "multi"


def test_dropped_beat_does_not_shift_grid():
    beats = np.delete(_grid(100, 40), 17)
    t0, period, rms = tempo.fit_grid(beats)
    assert period == pytest.approx(0.6) and rms < 1e-9


def test_metre_ignores_spurious_and_missed_downbeats():
    beats = _grid(120, 40)
    down = np.concatenate([[beats[1]], beats[2::3]])  # spurious one beat before a 3/4 bar line
    down = np.delete(down, 4)  # a missed downbeat = two whole bars
    assert tempo.meters_from_downbeats(beats, down) == [(2, 3)]


def test_rescale_half_and_double():
    beats = _grid(120, 16)
    assert len(tempo.rescale(beats, beats[::4], "double")) == 31
    assert np.allclose(tempo.rescale(beats, beats[1::4], "half"), beats[1::2])


def _midi_beat_times(path):
    mid = mido.MidiFile(path)
    tempo_us, t, times, tick = 500000, 0.0, [0.0], 0
    for msg in mid.tracks[0]:
        for _ in range(msg.time // mid.ticks_per_beat):
            t += tempo_us / 1e6
            times.append(t)
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
    return np.array(times)


def test_midi_round_trip_places_every_beat(tmp_path):
    ibis = 60 / np.linspace(95, 115, 31)
    beats = np.concatenate([[0.6], 0.6 + np.cumsum(ibis)])
    tm = tempo.build(beats, beats[2::4], mode="multi")
    path = midi.to_midi(tm, tmp_path / "t.mid")
    times = _midi_beat_times(path)
    n_lead = int(round(beats[0] / ibis[0]))
    np.testing.assert_allclose(times[n_lead : n_lead + len(beats)], beats, atol=1e-5)
    sigs = [(m.numerator, m.denominator) for m in mido.MidiFile(path).tracks[0] if m.type == "time_signature"]
    assert sigs[0] == ((n_lead + 2) % 4, 4) and sigs[1] == (4, 4)


@pytest.mark.skipif(not DEFAULT_MODEL.is_file(), reason="run scripts/export_onnx.py first")
def test_model_finds_click_track():
    import beatcop

    sr, bpm = 44100, 120
    y = np.zeros(sr * 20)
    for t in np.arange(0.5, 19.5, 60 / bpm):
        i = int(t * sr)
        y[i : i + 2000] += np.sin(2 * np.pi * 1000 * np.arange(2000) / sr) * np.exp(-np.arange(2000) / 300)
    beats, _ = beatcop.detect(y, sr)
    assert 60 / np.median(np.diff(beats)) == pytest.approx(bpm, rel=0.05)


def test_steadiness_locks_jittered_grid_and_rounds_bpm():
    beats = _grid(120.1, 96, jitter=0.008)
    stats = tempo.analyse_grid(beats)
    assert stats.stable and stats.steady_pct >= tempo.STEADY_PCT
    assert stats.bpm == pytest.approx(120.1, abs=0.1)
    assert tempo.round_bpm(stats.bpm) == 120.0
    assert tempo.round_bpm(127.6) == 127.6


def test_free_intro_is_kept_and_the_rest_locked():
    # free intro: no interval within 6% of the song's beat (0.513 s) or a multiple of it
    intro = np.cumsum([0.5, 0.71, 0.37, 0.83, 0.66, 0.41, 0.77, 0.62, 0.35, 0.88])
    song = intro[-1] + 0.7 + np.arange(80) * 60 / 117
    beats = np.concatenate([intro, song])
    stats = tempo.analyse_grid(beats)
    assert stats.stable and stats.first >= 9
    tm = tempo.build(beats, song[::4])
    assert tm.mode == "single"
    np.testing.assert_allclose(tm.beats[: stats.first], beats[: stats.first])


def test_steadiness_rejects_drifting_tempo():
    ibis = 60 / np.linspace(96, 124, 120)
    beats = 0.5 + np.concatenate([[0], np.cumsum(ibis)])
    assert not tempo.analyse_grid(beats).stable


def test_second_chance_locks_through_an_off_grid_break():
    a = _grid(120, 48)
    # free-time break: no interval within 6% of the 0.5 s beat
    brk = a[-1] + np.cumsum([0.7, 0.35, 0.8, 0.62, 0.4, 0.9, 0.33, 0.75, 0.65, 0.38, 0.85, 0.6])
    c = a[0] + 0.5 * np.arange(int(np.ceil((brk[-1] + 0.3 - a[0]) / 0.5)), 200)[:48]  # back on grid
    stats = tempo.analyse_grid(np.concatenate([a, brk, c]))
    assert stats.stable and stats.second_chance


def test_dbn_tracks_activation_with_tempo_drift():
    from beatcop import dbn

    ibis = 60 / np.linspace(100, 116, 40)
    truth = 0.5 + np.concatenate([[0], np.cumsum(ibis)])
    frames = np.arange(int((truth[-1] + 1) * 50))
    logits = np.full(len(frames), -6.0)
    for t in truth:
        logits[int(round(t * 50))] = 3.0
    logits[int(round(truth[10] * 50)) + 12] = 1.0  # a spurious off-beat peak
    beats = dbn.track(logits, 108.0, span=(truth[0], truth[-1]))
    assert len(beats) == len(truth)
    assert np.abs(beats - truth).max() < 0.02
