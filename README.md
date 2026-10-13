# BeatCop

Beat, downbeat and tempo-map detection from audio. Runs the [Beat This!](https://github.com/CPJKU/beat_this) model (CPJKU, MIT) on ONNX Runtime with a NumPy front end, so there's no PyTorch at runtime.

## Setup

The model isn't checked in. Export it once (this step needs torch):

```bash
uv sync
uv run --with torch --with "beat_this @ git+https://github.com/CPJKU/beat_this" \
    --with onnxscript python scripts/export_onnx.py   # -> models/beatthis_final0.onnx
```

To keep the model somewhere else, set `BEATCOP_MODEL=/path/to/model.onnx` or pass `--model`.

## Usage

```bash
uv run beatcop song.wav                      # writes song.beats + song_tempo.mid beside the audio
uv run beatcop song.wav --mode single        # force one locked tempo
uv run beatcop song.wav --scale half --meter 3 -o out/ --json
```

| Option | Effect |
| --- | --- |
| `--mode auto` | Locked tempo if the beats fit a straight grid (RMS ≤ 20 ms), else per-beat (default) |
| `--mode single` | One tempo, least-squares grid through the beats |
| `--mode multi` | One tempo per beat interval |
| `--scale half\|double` | Fix octave errors |
| `--meter N` | Force N/4, regroup bars every N beats from the first downbeat |
| `--no-midi` | Skip the MIDI file |

Outputs:

- `song.beats` — TSV of `time<TAB>position-in-bar` (1 = downbeat); same format as Beat This!
- `song_tempo.mid` — tempo map with one quarter note per beat. Import it at 0:00 / bar 1. A pickup bar puts the first detected downbeat on a bar line.

Python API:

```python
import beatcop
beats, downbeats = beatcop.detect_file("song.wav")  # seconds
```

## Pipeline

1. Mono, soxr resample to 22.05 kHz.
2. Log-mel spectrogram: 128 Slaney bands from 30 Hz to 11 kHz, 50 fps, `log1p(1000·x)` (`spectrogram.py`).
3. Model on 1500-frame (30 s) chunks. The 6 frames at each chunk edge are discarded and the earlier chunk wins where chunks overlap (`model.py`).
4. Peaks: local max within ±70 ms and logit > 0; adjacent peaks merged; downbeats snapped to beats (`postprocess.py`).
5. Tempo map and metre (`tempo.py`), MIDI export (`midi.py`).

Steps 1–4 reproduce upstream `beat_this` with its default `minimal` post-processing. On the test audio the beat and downbeat times came out identical, and the spectrogram matches torchaudio to within 2.4e-5.

## Caveats

- If the first beat is less than 62.5 ms from 0:00, the MIDI pins it to 0:00 (DAWs cap tempo around 960 BPM). Every later beat stays exact.
- Mistakes the model makes, such as briefly locking onto eighth notes, appear as odd bars in the metre. Override them with `--meter`.

## Tests

```bash
uv run pytest
```

The model test is skipped until `models/beatthis_final0.onnx` exists.
