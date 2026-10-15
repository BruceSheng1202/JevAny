"""Check delivered MP4s and export review frames.

Usage: python scripts/qa.py OUT_DIR --ffmpeg BIN --ffprobe BIN VIDEO.mp4 [VIDEO.mp4 ...]
Writes qa.json with stream facts, full-decode results, loudness and
black/frozen-frame scans, plus a contact sheet per video.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

# One frame per scene plus the transitions between them.
REVIEW_TIMES = [0.0, 2.0, 4.6, 5.6, 7.0, 8.6, 9.9, 11.5, 15.95, 19.0, 23.0, 24.5, 27.5, 31.5, 36.5, 39.0, 42.0, 47.0, 52.5, 56.0, 62.0,
                64.5, 67.0, 71.0, 72.0, 74.5, 76.5, 80.0, 85.0, 89.96]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def check(args, video: Path, out: Path) -> dict:
    probe = json.loads(run([args.ffprobe, "-v", "error", "-show_streams", "-show_format", "-count_frames", "-of", "json", str(video)]).stdout)
    streams = []
    for s in probe["streams"]:
        keep = {k: s.get(k) for k in ("codec_type", "codec_name", "profile", "width", "height", "pix_fmt", "r_frame_rate", "avg_frame_rate",
                                      "nb_read_frames", "duration", "sample_rate", "channels", "bit_rate", "color_space", "color_primaries")}
        streams.append({k: v for k, v in keep.items() if v is not None})
    head = video.open("rb").read(4_000_000)
    moov_first = -1 < head.find(b"moov") < head.find(b"mdat")
    decode = run([args.ffmpeg, "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"])
    blacks = run([args.ffmpeg, "-hide_banner", "-i", str(video), "-vf", "blackdetect=d=0.25:pix_th=0.06", "-an", "-f", "null", "-"]).stderr
    freezes = run([args.ffmpeg, "-hide_banner", "-i", str(video), "-vf", "freezedetect=n=0.0008:d=1.5", "-an", "-f", "null", "-"]).stderr
    loud = run([args.ffmpeg, "-hide_banner", "-nostats", "-i", str(video), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"]).stderr
    summary = loud[loud.rindex("Summary:"):] if "Summary:" in loud else ""
    # Edge levels: first and last 0.25 s should fade cleanly.
    edges = {}
    for name, ss in (("first_0.25s", 0), ("last_0.25s", 89.75)):
        vol = run([args.ffmpeg, "-hide_banner", "-ss", str(ss), "-t", "0.25", "-i", str(video), "-vn", "-af", "volumedetect", "-f", "null", "-"]).stderr
        m = re.search(r"max_volume: (-?[\d.]+|-inf) dB", vol)
        edges[name] = m.group(1) if m else None
    # Decoded sample peak below 0 dBFS means no clipped samples.
    clip = run([args.ffmpeg, "-hide_banner", "-i", str(video), "-vn", "-af", "astats=measure_overall=all:measure_perchannel=none", "-f", "null", "-"]).stderr
    frames_dir = out / f"{video.stem}-frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for t in REVIEW_TIMES:
        run([args.ffmpeg, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(frames_dir / f"t{t:06.2f}.png")])
    sheet = out / f"{video.stem}-contact-sheet.jpg"
    width = 480 if "4x5" not in video.stem else 300
    run([args.ffmpeg, "-v", "error", "-y", "-pattern_type", "glob", "-i", str(frames_dir / "*.png"), "-vf", f"scale={width}:-1,tile=6x5:padding=6:color=0x202020",
         "-frames:v", "1", "-q:v", "3", str(sheet)])
    return {
        "file": video.name,
        "bytes": video.stat().st_size,
        "format": {k: probe["format"].get(k) for k in ("format_name", "duration", "bit_rate")},
        "streams": streams,
        "faststart_moov_before_mdat": moov_first,
        "full_decode_errors": decode.stderr.strip().splitlines()[:20],
        "full_decode_exit": decode.returncode,
        "black_segments": re.findall(r"black_start:\S+ black_end:\S+ black_duration:\S+", blacks),
        "freeze_segments": re.findall(r"freeze_start: \S+|freeze_end: \S+", freezes),
        "loudness": {
            "integrated_lufs": re.search(r"I:\s+(-?[\d.]+)", summary).group(1) if summary else None,
            "lra_lu": re.search(r"LRA:\s+([\d.]+)", summary).group(1) if summary else None,
            "true_peak_dbtp": re.search(r"Peak:\s+(-?[\d.]+)", summary).group(1) if summary else None,
        },
        "audio_edge_max_db": edges,
        "decoded_sample_peak_dbfs": float(re.findall(r"Peak level dB: (-?[\d.]+)", clip)[-1]),
        "contact_sheet": sheet.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--ffprobe", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = [check(args, Path(v), out) for v in args.videos]
    (out / "qa.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
