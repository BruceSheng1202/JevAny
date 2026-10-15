"""Extract cropped JPEG frames from the repository's case recordings.

Usage: python scripts/extract_clips.py FFMPEG OUT_DIR
Case recordings place the replay or interface viewport at x=360..1260,
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
# Playground recordings (1120x900) place the scene at x=16..816, y=76..616.
PLAYGROUND = "crop=800:540:16:76"
WALL = ",scale=450:-2:flags=lanczos"
CASES = ["peg_insertion", "block_stacking", "drawer_opening", "conveyor_sorting", "assembly", "control_panel", "obstacle_push",
         "frontend", "research", "sql", "ci", "inbox", "calendar", "spreadsheet", "drone", "lab", "driving", "warehouse_rover"]
# Wall tiles are half-size crops; "arm-hd" is the full-size playground replay for the agent-loop scene.
CLIPS = {name: (f"{name}.mp4", VIEWPORT + WALL) for name in CASES}
CLIPS.update({f"pg-{name}": (f"docs/playground-{name}.mp4", PLAYGROUND + WALL) for name in ("arm", "doom", "crafter")})
CLIPS["arm-hd"] = ("docs/playground-arm.mp4", PLAYGROUND)


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
