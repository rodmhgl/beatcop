"""Write a TempoMap as a Standard MIDI File tempo map (one quarter note per detected beat).

Drop it into a DAW at bar 1 / 0:00 and the bar lines follow the recording. The stretch from 0:00
to the first beat becomes whole lead-in beats at roughly the song's tempo, and a pickup bar is
added when needed so the first detected downbeat lands on a bar line.
"""

from pathlib import Path

import mido
import numpy as np

from beatcop.tempo import TempoMap

PPQ = 480
MAX_BPM = 960  # DAWs reject faster tempos


def _quarter_lengths(tm: TempoMap) -> tuple[np.ndarray, int]:
    """Seconds per quarter note from 0:00 through the last beat, and the number of lead-in beats."""
    beats = tm.beats
    ibis = np.diff(beats)
    n_lead = int(round(beats[0] / ibis[0]))
    if n_lead == 0 and beats[0] >= 60.0 / MAX_BPM:
        n_lead = 1
    if n_lead == 0:
        # first beat is too close to 0:00 for a lead-in tempo: pin it to 0:00 and let the first
        # quarter absorb the offset; every later beat stays exact
        ibis = ibis.copy()
        ibis[0] = beats[1]
        return ibis, 0
    return np.concatenate([[beats[0] / n_lead] * n_lead, ibis]), n_lead


def to_midi(tm: TempoMap, path: str | Path, name: str = "beatcop tempo map") -> Path:
    if len(tm.beats) < 2:
        raise ValueError("need at least two beats for a tempo map")
    quarters, n_lead = _quarter_lengths(tm)

    events: list[tuple[int, mido.MetaMessage]] = []
    last_tempo = None
    for i, sec in enumerate(quarters):
        tempo = int(round(sec * 1e6))
        if tempo != last_tempo:
            events.append((i * PPQ, mido.MetaMessage("set_tempo", tempo=tempo)))
            last_tempo = tempo

    # Metre: the pickup (lead-in + beats before the first bar line) gets its own short bar.
    first_bar, first_meter = tm.meters[0]
    pickup = (n_lead + first_bar) % first_meter
    if pickup:
        events.append((0, mido.MetaMessage("time_signature", numerator=pickup, denominator=4)))
    for beat_idx, meter in tm.meters:
        tick = (n_lead + beat_idx) * PPQ if beat_idx != first_bar else pickup * PPQ
        events.append((tick, mido.MetaMessage("time_signature", numerator=meter, denominator=4)))

    events.sort(key=lambda e: (e[0], e[1].type != "time_signature"))
    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
    now = 0
    for tick, msg in events:
        track.append(msg.copy(time=tick - now))
        now = tick
    end_tick = len(quarters) * PPQ
    track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - now)))

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    mid.tracks.append(track)
    path = Path(path)
    mid.save(str(path))
    return path
