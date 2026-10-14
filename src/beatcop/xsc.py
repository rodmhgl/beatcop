"""Write a TempoMap as a Transcribe! document (.xsc) with a measure marker on every bar line and a
beat marker on every other beat.

Open the .xsc in Transcribe! with the audio beside it. Marker lines are
`type,position,auto-named,label,beats-to-next,h:mm:ss.ffffff,`: type S(ection)/M(easure)/B(eat),
position -1 (use the time), labels auto-named like Transcribe!'s own (A1, A2, ...), and
beats-to-next 1 on every marker since each marker is one beat from the next (a bar's beat count
there would make Transcribe! subdivide each beat again); the metre follows from the beat markers.
"""

import re
from pathlib import Path, PurePosixPath

from beatcop.tempo import TempoMap, bars

VERSION_LINES = ["XSC Transcribe.Document Version 6089.00", "Transcribe!,Windows,9,42,7,S,0"]


def _stamp(t: float) -> str:
    us = round(t * 1e6)
    s, us = divmod(us, 1_000_000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02}:{s:02}.{us:06}"


def sound_file_line(audio: str | Path) -> str:
    """Transcribe! finds the audio by name beside the .xsc, or by the full path (backslashes doubled
    on Windows). A WSL /mnt/<drive>/ path is written as the Windows path Transcribe! will see."""
    p = PurePosixPath(str(audio))
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(p))
    if m:
        win = f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\")
        return f"SoundFileName,{p.name},Win,{win.replace(chr(92), chr(92) * 2)}"
    return f"SoundFileName,{p.name},Mac,{p}"


def to_xsc(tm: TempoMap, audio: str | Path, path: str | Path) -> Path:
    markers = []
    bar_starts = {b.first_beat for b in bars(tm)}
    bar_no = 0
    for i, t in enumerate(tm.beats):
        if i in bar_starts:
            bar_no += 1
            kind = "S" if bar_no == 1 else "M"
            markers.append(f"{kind},-1,1,A{bar_no},1,{_stamp(t)},")
        else:
            markers.append(f"B,-1,1,,1,{_stamp(t)},")

    lines = [
        *VERSION_LINES,
        "",
        "SectionStart,Main",
        sound_file_line(Path(audio).resolve()),
        "MarkerNumberContinuously,0",
        "SectionEnd,Main",
        "",
        "SectionStart,Markers",
        f"Howmany,{len(markers)}",
        *markers,
        "SectionEnd,Markers",
        "",
    ]
    path = Path(path)
    path.write_bytes("\r\n".join(lines).encode("utf-8"))
    return path
