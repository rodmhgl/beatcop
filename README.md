# BeatCop

Beat, downbeat and tempo-map detection from audio. Runs the [Beat This!](https://github.com/CPJKU/beat_this) model (CPJKU, MIT) on ONNX Runtime with a NumPy front end, so there's no PyTorch at runtime. BeatCop will export files compatible with Guitar Pro 8 and Transcribe!.

## Setup

Needs Python 3.10+ and [uv](https://docs.astral.sh/uv/). The model isn't checked in. Export it once (this step needs torch):

```bash
uv sync
uv run --with torch --with "beat_this @ git+https://github.com/CPJKU/beat_this" \
    --with onnxscript python scripts/export_onnx.py   # -> models/beatthis_final0.onnx
```

To keep the model somewhere else, set `BEATCOP_MODEL=/path/to/model.onnx` or pass `--model`.

## Usage

```bash
uv run beatcop song.wav                      # writes song.beats, song_tempo.mid, song_tempo.gp, song.xsc
uv run beatcop song.wav --mode single        # force one locked tempo
uv run beatcop song.wav --scale half --meter 3 -o out/ --json
```

| Option | Effect |
| --- | --- |
| `--mode auto` | Locked tempo if the track is steady (see below), else per-beat (default) |
| `--mode single` | One tempo: the grid the most beats land on |
| `--mode multi` | One tempo per beat interval |
| `--mode doctor` | Like auto, but unsteady tracks are re-tracked with a DBN and then followed per beat (complex or polyrhythmic music) |
| `--scale half\|double` | Fix octave errors: `half` keeps every other beat (in phase with the first downbeat), `double` adds midpoints |
| `--meter N` | Force N/4, regroup bars every N beats from the first downbeat |
| `--no-midi` | Skip the MIDI file |
| `--no-gp` | Skip the Guitar Pro file |
| `--no-xsc` | Skip the Transcribe! file |
| `-o, --outdir DIR` | Write outputs here (default: beside the audio) |
| `--json` | Print mode, BPM, metres, beat and downbeat times and output paths as JSON |
| `--model PATH` | ONNX model (default: `models/` or `$BEATCOP_MODEL`) |

Outputs:

- `song.beats` — TSV of `time<TAB>position-in-bar` (1 = downbeat); same format as Beat This!
- `song_tempo.mid` — tempo map with one quarter note per beat. Import it at 0:00 / bar 1. A pickup bar puts the first detected downbeat on a bar line.
- `song_tempo.gp` — Guitar Pro 8 file: an empty score with one bar per detected bar, the audio embedded as its audio track and synced bar by bar. Add tracks and write on top. Embeds a full copy of the audio.
- `song.xsc` — Transcribe! document: a measure marker on every bar line (with its beat count) and a beat marker on every other beat. Keep it beside the audio. Under WSL, a `/mnt/<drive>/` audio path is written as the Windows path Transcribe! will see.

Python API:

```python
import beatcop
from beatcop import tempo

beats, downbeats = beatcop.detect_file("song.wav")      # seconds
beatcop.detect(signal, sr)                               # same, from an array
beat_logits, downbeat_logits = beatcop.detect_frames(signal, sr)  # raw 50 fps logits

tm = tempo.build(beats, downbeats, mode="doctor", beat_logits=beat_logits)
tm.mode, tm.bpm, tm.meters, tempo.bars(tm)
```

Pass a `beatcop.BeatModel(path)` as `model=` to reuse one ONNX session across files.

## Pipeline

1. Mono, soxr resample to 22.05 kHz.
2. Log-mel spectrogram: 128 Slaney bands from 30 Hz to 11 kHz, 50 fps, `log1p(1000·x)` (`spectrogram.py`).
3. Model on 1500-frame (30 s) chunks. The 6 frames at each chunk edge are discarded and the earlier chunk wins where chunks overlap (`model.py`).
4. Peaks: local max within ±70 ms and logit > 0; adjacent peaks merged; downbeats snapped to beats (`postprocess.py`).
5. Tempo map, metre and bars (`tempo.py`); MIDI, Guitar Pro and Transcribe! export (`midi.py`, `gp.py`, `xsc.py`).

Steps 1–4 reproduce upstream `beat_this` with its default `minimal` post-processing. On the test audio the beat and downbeat times came out identical, and the spectrogram matches torchaudio to within 2.4e-5.

### Steadiness test (auto / doctor)

Implemented in `tempo.analyse_grid`:

- Tempo = median beat interval among intervals within 0.75–1.5× the median BPM. Beats before the first interval within 6% of it (allowing up to 3 skipped beats) are an intro, kept as detected.
- Tempo refined by grid search (0.9–1.1×, 0.05 BPM steps, then ±0.1 in 0.004 steps), phase from the circular mean. A beat within 25 ms of the grid is on it.
- **Steady** when ≥ 85% of 12-beat windows have ≥ 75% of beats on grid.
- Second chance: steady runs of ≥ 16 beats covering ≥ 40% of the song, starting in the first 25% and ending in the last 25%, with off-tempo breaks (≤ 35% of break intervals at tempo) and no steady section at an unrelated tempo (2/3, 3/4, 4/3 and 3/2 are allowed).
- Fewer than 8 beats, or fewer than 12 after the intro, is never steady.
- Ties in the fine search go to the smallest mean residual. A steady section inside a break means 16 intervals varying < 3%; it must sit at 0.6–1.67× the tempo to count against locking.
- Locked tempo snaps to a whole BPM when within 0.25, and the grid is phase-fitted to the beats after the intro.

Doctor's DBN (`dbn.py`): 200 fps, observation λ 16, transition λ 100 (madmom's defaults), tempo range ±1.5× around the estimate. It runs on the model's beat logits, keeps beats within 0.1 s of the originally detected span, and downbeats are snapped onto the new beats.

## Caveats

- If the first beat is less than 62.5 ms from 0:00, the MIDI pins it to 0:00 (DAWs cap tempo around 960 BPM). Every later beat stays exact.
- Mistakes the model makes, such as briefly locking onto eighth notes, appear as odd bars in the metre. Override them with `--meter`, or try `--mode doctor`.
- Pieces under 30 s run through the model unpadded, as upstream does.

## Tests

```bash
uv run pytest
```

The model test is skipped until `models/beatthis_final0.onnx` exists.

## Third-party

`src/beatcop/assets/blank.gp`, the empty Guitar Pro 8 score the `.gp` export starts from, is from [guitar-pro-youtube-sync](https://github.com/teaqu/guitar-pro-youtube-sync) (MIT, see `blank.gp.LICENSE` beside it).
