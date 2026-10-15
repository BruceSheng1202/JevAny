"""Synthesize narration lines with Kokoro-82M (Apache-2.0) on CPU.

Usage: python scripts/tts.py OUT_DIR
Writes one WAV per line plus timing.json with measured durations.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parents[1]
# Misaki inline phoneme overrides keep project names consistent.
PRONUNCIATION = {
    # IPA written as escapes to keep the source ASCII.
    "SimpleJev": "[SimpleJev](/s\u02c8\u026amp\u0259l d\u0292\u02c8\u025bv/)",
    "JevAny": "[JevAny](/d\u0292\u02c8\u025bv \u02c8\u025bni/)",
    "Jev": "[Jev](/d\u0292\u02c8\u025bv/)",
    "WebShop": "web shop",
    "LLM": "L L M",
}


def speakable(text: str) -> str:
    for word, spoken in PRONUNCIATION.items():
        text = re.sub(rf"\b{word}\b", spoken, text)
    return text


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "audio" / "narration.json").read_text())
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    timing = []
    for i, line in enumerate(spec["lines"]):
        chunks = [r.audio.numpy() for r in pipeline(speakable(line["text"]), voice=spec["voice"], speed=spec["speed"])]
        audio = np.concatenate(chunks).astype(np.float32)
        # Trim synthesizer edge silence so start times mark the first phoneme.
        loud = np.flatnonzero(np.abs(audio) > 0.01)
        audio = audio[max(loud[0] - 240, 0): loud[-1] + 2400]
        sf.write(out / f"{line['id']}.wav", audio, spec["sample_rate"])
        duration = len(audio) / spec["sample_rate"]
        nxt = spec["lines"][i + 1]["start"] if i + 1 < len(spec["lines"]) else 90.0
        timing.append({**line, "duration": round(duration, 3), "end": round(line["start"] + duration, 3), "gap_to_next": round(nxt - line["start"] - duration, 3)})
        print(f"{line['id']:10s} {line['start']:6.2f} +{duration:5.2f} gap {timing[-1]['gap_to_next']:+.2f}")
    (out / "timing.json").write_text(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
