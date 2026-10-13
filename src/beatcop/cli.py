"""beatcop CLI: detect beats/downbeats in an audio file and write a .beats TSV and MIDI tempo map."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from beatcop import detect_file, midi, tempo
from beatcop.model import BeatModel


def _write_beats_tsv(tm: tempo.TempoMap, path: Path) -> None:
    """Beat This!-compatible TSV: time<TAB>position-in-bar (1 = downbeat)."""
    down = np.searchsorted(tm.beats, tm.downbeats)
    pos = np.zeros(len(tm.beats), dtype=int)
    down_set = set(down.tolist())
    counter = None
    for i in range(len(tm.beats)):
        if i in down_set:
            counter = 1
        elif counter is not None:
            counter += 1
        pos[i] = counter or 0
    # beats before the first downbeat count backwards into a pickup bar
    first_meter = tm.meters[0][1]
    first = int(down[0]) if len(down) else 0
    for i in range(first):
        pos[i] = (first_meter - (first - i) % first_meter) % first_meter + 1
    path.write_text("".join(f"{t:.3f}\t{p}\n" for t, p in zip(tm.beats, pos)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="beatcop", description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("-o", "--outdir", type=Path, help="default: beside the audio file")
    p.add_argument("--mode", choices=["auto", "single", "multi"], default="auto",
                   help="auto: locked tempo if steady, else follow beats; single: force one "
                        "tempo; multi: per-beat tempo map")
    p.add_argument("--scale", choices=["half", "double"], help="rescale tempo an octave")
    p.add_argument("--meter", type=int, help="force N/4 and regroup bars every N beats")
    p.add_argument("--no-midi", action="store_true", help="skip the _tempo.mid tempo map")
    p.add_argument("--json", action="store_true", help="print results as JSON")
    p.add_argument("--model", type=Path, help="ONNX model (default: models/ or $BEATCOP_MODEL)")
    args = p.parse_args(argv)

    beats, downbeats = detect_file(args.audio, BeatModel(args.model))
    if len(beats) < 2:
        print("no beats detected", file=sys.stderr)
        return 1
    tm = tempo.build(beats, downbeats, mode=args.mode, scale=args.scale, meter=args.meter)

    outdir = args.outdir or args.audio.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.audio.stem
    beats_path = outdir / f"{stem}.beats"
    _write_beats_tsv(tm, beats_path)
    midi_path = None if args.no_midi else midi.to_midi(tm, outdir / f"{stem}_tempo.mid")

    if args.json:
        print(json.dumps({
            "mode": tm.mode,
            "bpm": round(tm.bpm, 2),
            "meters": tm.meters,
            "beats": np.round(tm.beats, 3).tolist(),
            "downbeats": np.round(tm.downbeats, 3).tolist(),
            "beats_file": str(beats_path),
            "midi_file": str(midi_path) if midi_path else None,
        }))
    else:
        meters = ", ".join(f"{m}/4 from beat {i}" for i, m in tm.meters)
        print(f"{args.audio.name}: {len(tm.beats)} beats, {len(tm.downbeats)} downbeats, "
              f"{tm.bpm:.2f} BPM ({tm.mode}), metre {meters}")
        print(f"wrote {beats_path}" + (f" and {midi_path}" if midi_path else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
