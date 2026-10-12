"""Extract cropped JPEG frames from the repository's case recordings.

Usage: python scripts/extract_clips.py FFMPEG OUT_DIR
The recordings place the replay or interface viewport at x=360..1260,
y=128..574 of a 1280x720 frame. Frames stay at the source 20 fps; the film
blends neighbouring frames when it plays them slower.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MEDIA = REPO / "site/assets/media"
VIEWPORT = "crop=900:446:360:128"
CLIPS = {
    "crosswalk": ("crosswalk.mp4", f"{VIEWPORT},scale=1800:892:flags=lanczos,unsharp=5:5:0.6"),
    "peg_insertion": ("peg_insertion.mp4", VIEWPORT),
    "drone": ("drone.mp4", VIEWPORT),
    "lab": ("lab.mp4", VIEWPORT),
    "inbox": ("inbox.mp4", VIEWPORT),
    "sql": ("sql.mp4", VIEWPORT),
    "driving": ("driving.mp4", VIEWPORT),
}


def main() -> None:
    ffmpeg, out = sys.argv[1], Path(sys.argv[2])
    manifest = {}
    for name, (source, vf) in CLIPS.items():
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        for old in target.glob("*.jpg"):
            old.unlink()
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(MEDIA / source), "-vf", vf, "-q:v", "2", str(target / "%04d.jpg")], check=True)
        manifest[name] = {"source": f"site/assets/media/{source}", "fps": 20, "frames": len(list(target.glob("*.jpg")))}
    (out / "clips.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
