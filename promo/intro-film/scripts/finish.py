"""Master the audio and mux the delivery MP4s.

Usage:
  python scripts/finish.py audio MIX.wav OUT.m4a --ffmpeg BIN
  python scripts/finish.py mux VIDEO.mp4 AUDIO.m4a OUT.mp4 --ffmpeg BIN [--trim SECONDS]
  python scripts/finish.py burn IN.mp4 CAPTIONS.ass FONTS_DIR OUT.mp4 --ffmpeg BIN

Audio targets -14 LUFS integrated with true peak at or below -1 dBTP.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from render import X264

TARGET_LUFS = -14.0


def loudness(ffmpeg: str, path: str, prefilter: str = "") -> dict:
    chain = f"{prefilter + ',' if prefilter else ''}ebur128=peak=true"
    err = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", path, "-af", chain, "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = err[err.rindex("Summary:"):]
    return {
        "integrated_lufs": float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1)),
        "lra_lu": float(re.search(r"LRA:\s+([\d.]+) LU", summary).group(1)),
        "true_peak_dbtp": float(re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary).group(1)),
    }


def master(args) -> None:
    before = loudness(args.ffmpeg, args.inputs[0])
    gain = TARGET_LUFS - before["integrated_lufs"]
    for _ in range(3):
        chain = f"volume={gain:.2f}dB,alimiter=limit=0.79:attack=4:release=60:level=false"
        after = loudness(args.ffmpeg, args.inputs[0], chain)
        miss = TARGET_LUFS - after["integrated_lufs"]
        if abs(miss) < 0.2:
            break
        gain += miss
    subprocess.run([args.ffmpeg, "-v", "error", "-y", "-i", args.inputs[0], "-af", chain, "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", args.inputs[1]], check=True)
    final = loudness(args.ffmpeg, args.inputs[1])
    report = {"source": before, "gain_db": round(gain, 2), "chain": chain, "encoded": final}
    Path(args.inputs[1]).with_suffix(".loudness.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def mux(args) -> None:
    video, audio, out = args.inputs
    trim = ["-t", str(args.trim)] if args.trim else []
    subprocess.run([args.ffmpeg, "-v", "error", "-y", "-i", video, "-i", audio, *trim, "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                    "-metadata", "title=JevAny: open infrastructure for System 1 decision models", "-metadata:s:a:0", "language=eng",
                    "-movflags", "+faststart", out], check=True)
    print(f"wrote {out}")


def burn(args) -> None:
    src, ass, fonts, out = args.inputs
    esc = lambda p: str(Path(p).resolve()).replace(":", r"\:")
    subprocess.run([args.ffmpeg, "-v", "error", "-y", "-i", src, "-vf", f"ass={esc(ass)}:fontsdir={esc(fonts)}", *X264, "-c:a", "copy",
                    "-movflags", "+faststart", out], check=True)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audio", "mux", "burn"])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--trim", type=float)
    args = ap.parse_args()
    {"audio": master, "mux": mux, "burn": burn}[args.mode](args)


if __name__ == "__main__":
    main()
