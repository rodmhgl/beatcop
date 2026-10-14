import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from beatcop import gp, tempo, xsc


def _rubato(n=33, start=0.6, first_down=2):
    ibis = 60 / np.linspace(95, 115, n - 1)
    beats = np.concatenate([[start], start + np.cumsum(ibis)])
    return tempo.build(beats, beats[first_down::4], mode="multi")


def test_bars_start_with_a_pickup_then_follow_the_downbeats():
    tm = _rubato()  # downbeats on beats 2, 6, 10, ...
    bars = tempo.bars(tm)
    assert [b.beats for b in bars[:3]] == [2, 4, 4]
    assert bars[0].start == tm.beats[0] and bars[1].start == tm.beats[2]
    # every bar ends where the next starts; the last runs one beat past the final detected beat
    assert all(a.end == b.start for a, b in zip(bars, bars[1:]))
    assert bars[-1].end > tm.beats[-1]


def test_bars_follow_a_metre_change():
    beats = 0.5 + np.arange(30) * 0.5
    tm = tempo.TempoMap(beats=beats, downbeats=beats[[0, 4, 8, 11, 14]], mode="single",
                        meters=[(0, 4), (8, 3)])
    assert [b.beats for b in tempo.bars(tm)][:5] == [4, 4, 3, 3, 3]


def _markers(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = lines.index("SectionStart,Markers")
    count = int(lines[i + 1].split(",")[1])
    rows = [l.split(",") for l in lines[i + 2 : lines.index("SectionEnd,Markers")]]
    assert len(rows) == count
    return rows


def _secs(stamp):
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_xsc_marks_every_bar_and_beat(tmp_path):
    tm = _rubato()
    audio = tmp_path / "song.mp3"
    path = xsc.to_xsc(tm, audio, tmp_path / "song.xsc")
    text = path.read_bytes()
    assert text.startswith(b"XSC Transcribe.Document") and b"\r\n" in text
    assert "SoundFileName,song.mp3," in path.read_text()

    rows = _markers(path)
    kinds = [r[0] for r in rows]
    assert kinds[0] == "S" and set(kinds[1:]) == {"M", "B"}
    np.testing.assert_allclose([_secs(r[5]) for r in rows], tm.beats, atol=1e-6)
    bars = [r for r in rows if r[0] in "SM"]
    assert [r[3] for r in bars[:3]] == ["A1", "A2", "A3"]
    # one beat between consecutive markers, else Transcribe! subdivides each beat
    assert {int(r[4]) for r in rows} == {1}
    first_bars = [i for i, r in enumerate(rows) if r[0] in "SM"][:3]
    assert np.diff(first_bars).tolist() == [2, 4]


def test_xsc_points_at_a_windows_path_under_wsl(tmp_path):
    line = xsc.sound_file_line("/mnt/c/Users/me/Music/a b.mp3")
    assert line == r"SoundFileName,a b.mp3,Win,C:\\Users\\me\\Music\\a b.mp3"


def _gpif(path):
    with zipfile.ZipFile(path) as z:
        return z, ET.fromstring(z.read("Content/score.gpif")), z.namelist()


def test_gp_has_a_bar_per_bar_synced_to_the_audio(tmp_path):
    tm = _rubato()
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"ID3 fake mp3")
    path = gp.to_gp(tm, audio, tmp_path / "song_tempo.gp")
    bars = tempo.bars(tm)

    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("Content/score.gpif"))
        assets = root.findall("Assets/Asset")
        assert len(assets) == 1
        assert z.read(assets[0].findtext("EmbeddedFilePath")) == b"ID3 fake mp3"

    master_bars = root.findall("MasterBars/MasterBar")
    assert len(master_bars) == len(bars)
    assert [mb.findtext("Time") for mb in master_bars[:3]] == ["2/4", "4/4", "4/4"]
    assert [mb.findtext("Bars") for mb in master_bars[:3]] == ["0", "1", "2"]
    assert len(root.findall("Bars/Bar")) == len(bars)

    pad = int(root.findtext("BackingTrack/FramePadding"))
    assert pad == -round(bars[0].start * gp.FRAME_RATE)
    syncs = [a.find("Value") for a in root.findall("MasterTrack/Automations/Automation")
             if a.findtext("Type") == "SyncPoint"]
    assert [int(s.findtext("BarIndex")) for s in syncs] == list(range(len(bars)))
    frames = np.array([int(s.findtext("FrameOffset")) for s in syncs])
    np.testing.assert_allclose((frames - pad) / gp.FRAME_RATE, [b.start for b in bars], atol=1e-4)
    tempos = np.array([float(s.findtext("ModifiedTempo")) for s in syncs])
    np.testing.assert_allclose(tempos, [60 * b.beats / (b.end - b.start) for b in bars], rtol=1e-4)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_gp_last_bar_runs_to_the_end_of_the_audio(tmp_path, mode):
    beats = 1.0 + np.arange(40) * 0.5
    tm = tempo.build(beats, beats[::4], mode=mode)
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")
    last = tempo.bars(tm)[-1]
    duration = last.end + 6.0
    with zipfile.ZipFile(gp.to_gp(tm, audio, tmp_path / "t.gp", duration=duration)) as z:
        root = ET.fromstring(z.read("Content/score.gpif"))
    syncs = [a.find("Value") for a in root.findall("MasterTrack/Automations/Automation")
             if a.findtext("Type") == "SyncPoint"]
    final = syncs[-1]
    assert int(final.findtext("BarIndex")) == len(tempo.bars(tm)) - 1
    pad = int(root.findtext("BackingTrack/FramePadding"))
    start = (int(final.findtext("FrameOffset")) - pad) / gp.FRAME_RATE
    expected = 60 * last.beats / (duration - last.start)
    assert float(final.findtext("ModifiedTempo")) == pytest.approx(expected, rel=1e-4)
    assert start == pytest.approx(last.start, abs=1e-4)


def test_gp_single_tempo_needs_one_sync_point(tmp_path):
    beats = 1.0 + np.arange(40) * 0.5
    tm = tempo.build(beats, beats[::4], mode="single")
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")
    with zipfile.ZipFile(gp.to_gp(tm, audio, tmp_path / "t.gp")) as z:
        gpif = z.read("Content/score.gpif").decode()
    assert gpif.count("<Type>SyncPoint</Type>") == 1
    assert re.search(r"<Type>Tempo</Type>.*?<Value>120 2</Value>", gpif, re.S)
