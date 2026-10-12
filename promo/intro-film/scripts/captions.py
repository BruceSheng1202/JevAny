"""Write SRT, VTT and burn-in ASS captions from the narration timing.

Usage: python scripts/captions.py VO_DIR/timing.json OUT_DIR
Cues hold at most two lines of 42 characters and split long narration lines
at sentence or clause boundaries, timed in proportion to their length.
"""

import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_LINE = 42


def wrapped(text: str) -> list[str]:
    lines = textwrap.wrap(text, MAX_LINE)
    if len(lines) > 2 and ", " in text:
        # Prefer one clause per line when both clauses fit a slightly longer line.
        head, tail = text.split(", ", 1)
        if len(head) + 1 <= MAX_LINE + 4 and len(tail) <= MAX_LINE + 4:
            return [head + ",", tail]
    return lines


def chunks(text: str) -> list[str]:
    """Split narration into cues that wrap to at most two lines."""
    out = []
    for sentence in re.split(r"(?<=[.?!])\s+", text):
        if len(wrapped(sentence)) <= 2:
            out.append(sentence)
            continue
        cur = ""
        for clause in re.split(r"(?<=,)\s+", sentence):
            trial = f"{cur} {clause}".strip()
            if cur and len(wrapped(trial)) > 2:
                out.append(cur)
                cur = clause
            else:
                cur = trial
        out.append(cur)
    final = []
    for cue in out:
        lines = wrapped(cue)
        final.extend(" ".join(lines[i:i + 2]) for i in range(0, len(lines), 2))
    return final


def stamp(t: float, sep: str) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def ass_time(t: float) -> str:
    cs = round(t * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def main() -> None:
    timing = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    overrides = json.loads((ROOT / "audio" / "narration.json").read_text()).get("captions", {})
    cues = []
    for line in timing:
        parts = chunks(overrides.get(line["id"], line["text"]))
        total = sum(len(p) for p in parts)
        t = line["start"]
        for p in parts:
            dur = line["duration"] * len(p) / total
            cues.append((t, t + dur, wrapped(p)))
            t += dur
    # Hold each cue until shortly before the next one, at most 0.6 s longer.
    held = []
    for i, (a, b, lines) in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else 90.0
        held.append((a, min(nxt - 0.05, b + 0.6), lines))
    srt = "".join(f"{i}\n{stamp(a, ',')} --> {stamp(b, ',')}\n" + "\n".join(lines) + "\n\n" for i, (a, b, lines) in enumerate(held, 1))
    vtt = "WEBVTT\n\n" + "".join(f"{stamp(a, '.')} --> {stamp(b, '.')}\n" + "\n".join(lines) + "\n\n" for a, b, lines in held)
    (out / "jevany-intro.en.srt").write_text(srt)
    (out / "jevany-intro.en.vtt").write_text(vtt)
    assert all(1 <= len(lines) <= 2 for _, _, lines in held), "cue wraps beyond two lines"
    for layout, (w, h, size, margin) in {"landscape": (1920, 1080, 40, 36), "feed": (1080, 1350, 38, 40)}.items():
        side = 70 if layout == "feed" else 100
        head = (
            "[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
            "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Cap,DM Sans,%d,&H00F1F3ED,&H00F1F3ED,&H00000000,&H66080C0E,0,0,0,0,100,100,0,0,3,10,0,2,%d,%d,%d,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        ) % (w, h, size, side, side, margin)
        events = "".join(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{chr(92)}N".replace(f"{chr(92)}N", "", 1) + f"{chr(92)}N".join(lines) + "\n" for a, b, lines in held)
        (out / f"captions-{layout}.ass").write_text(head + events)
    print(f"{len(held)} cues")


if __name__ == "__main__":
    main()
