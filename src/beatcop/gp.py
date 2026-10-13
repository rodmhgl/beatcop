"""Write a TempoMap as a Guitar Pro 8 file (.gp): an empty score with one bar per detected bar and
the audio embedded as the audio track, synced bar by bar.

GP8 plays the audio track from its sync points, not from the score's tempo: each SyncPoint ties a
bar to an audio frame (counted at 44.1 kHz whatever the file's rate) and gives the bar's tempo in
the audio (ModifiedTempo). BackingTrack/FramePadding is minus the frame at which bar 1 starts.
The empty score comes from assets/blank.gp (MIT, from teaqu/guitar-pro-youtube-sync).
"""

import hashlib
import json
import re
import uuid
import zipfile
from importlib import resources
from pathlib import Path

from beatcop.tempo import TempoMap, bars

FRAME_RATE = 44100


def _automation(kind: str, bar: int, value: str) -> str:
    return (f"<Automation>\n<Type>{kind}</Type>\n<Linear>false</Linear>\n<Bar>{bar}</Bar>\n"
            f"<Position>0</Position>\n<Visible>true</Visible>\n<Value>{value}</Value>\n</Automation>")


def _score(gpif: str, tm: TempoMap, asset_id: str, asset_path: str, audio_name: str) -> str:
    bs = bars(tm)
    score_bpm = round(tm.bpm)
    pad = -round(bs[0].start * FRAME_RATE)

    autos = [_automation("Tempo", 0, f"{score_bpm} 2")]
    for i, b in enumerate(bs if tm.mode == "multi" else bs[:1]):
        bpm = 60 * b.beats / (b.end - b.start)
        frame = round(b.start * FRAME_RATE) + pad
        autos.append(_automation("SyncPoint", i, (
            f"\n<BarIndex>{i}</BarIndex>\n<BarOccurrence>0</BarOccurrence>\n"
            f"<ModifiedTempo>{bpm:.5f}</ModifiedTempo>\n<OriginalTempo>{score_bpm}</OriginalTempo>\n"
            f"<FrameOffset>{frame}</FrameOffset>\n")))

    key = re.search(r"<Key>.*?</Key>", gpif, re.S).group(0)
    master_bars = "".join(
        f"<MasterBar>\n{key}\n<Time>{b.beats}/4</Time>\n<Bars>{i}</Bars>\n</MasterBar>\n"
        for i, b in enumerate(bs))
    bar_list = "".join(
        f'<Bar id="{i}">\n<Clef>G2</Clef>\n<Voices>-1 -1 -1 -1</Voices>\n</Bar>\n' for i in range(len(bs)))
    backing = (
        "<BackingTrack>\n<IconId>21</IconId>\n<Color>0 0 0</Color>\n"
        "<Name><![CDATA[Audio Track]]></Name>\n<ShortName><![CDATA[a.track]]></ShortName>\n"
        "<PlaybackState>Default</PlaybackState>\n<ChannelStrip>\n<Parameters>0.500000 0.500000 "
        "0.500000 0.500000 0.500000 0.500000 0.500000 0.500000 0.500000 0.000000 0.500000 0.500000 "
        "0.800000 0.500000 0.500000 0.500000</Parameters>\n</ChannelStrip>\n<Enabled>true</Enabled>\n"
        "<Source>Local</Source>\n<AssetId>0</AssetId>\n<YouTubeVideoUrl></YouTubeVideoUrl>\n"
        "<Filter>6</Filter>\n<FramesPerPixel>300</FramesPerPixel>\n"
        f"<FramePadding>{pad}</FramePadding>\n<Semitones>0</Semitones>\n<Cents>0</Cents>\n"
        "</BackingTrack>\n")
    assets = (
        f'<Assets>\n<Asset id="0">\n<OriginalFilePath><![CDATA[{audio_name}]]></OriginalFilePath>\n'
        f"<OriginalFileSha1><![CDATA[{asset_id}]]></OriginalFileSha1>\n"
        f"<EmbeddedFilePath><![CDATA[{asset_path}]]></EmbeddedFilePath>\n</Asset>\n</Assets>\n")

    def sub(pattern: str, repl: str, text: str) -> str:
        out, n = re.subn(pattern, lambda _: repl, text, count=1, flags=re.S)
        if n != 1:
            raise ValueError(f"template has no {pattern}")
        return out

    gpif = sub(r"<Automations>.*?</Automations>", "<Automations>\n" + "\n".join(autos) + "\n</Automations>", gpif)
    gpif = sub(r"(?<=<!-- order is important here -->\n)<MasterBar>.*</MasterBar>\n", master_bars, gpif)
    gpif = sub(r"<Bars>\n<Bar id=.*?</Bars>", "<Bars>\n" + bar_list + "</Bars>", gpif)
    gpif = gpif.replace("</MasterTrack>\n", "</MasterTrack>\n" + backing, 1)
    return gpif.replace("</GPIF>", assets + "</GPIF>")


def to_gp(tm: TempoMap, audio: str | Path, path: str | Path) -> Path:
    audio = Path(audio)
    data = audio.read_bytes()
    asset_id = str(uuid.UUID(hashlib.sha1(data).hexdigest()[:32]))
    asset_path = f"Content/Assets/{asset_id}{audio.suffix.lower()}"
    path = Path(path)
    template = resources.files("beatcop").joinpath("assets/blank.gp")
    with zipfile.ZipFile(template.open("rb")) as src, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            content = src.read(item.filename)
            if item.filename == "Content/score.gpif":
                content = _score(content.decode("utf-8"), tm, asset_id, asset_path, audio.name).encode("utf-8")
            elif item.filename == "meta.json":
                content = json.dumps({"hasAudio": True, "version": "1.0.0"}, indent=4).encode()
            elif item.filename == "Content/Preferences.json":
                prefs = json.loads(content)
                prefs.setdefault("view", {})["backingTrackVisible"] = True
                content = json.dumps(prefs, separators=(",", ":")).encode()
            out.writestr(item, content)
        out.mkdir("Content/Assets/")
        out.writestr(asset_path, data, compress_type=zipfile.ZIP_STORED)
    return path
