"""Mix narration over the score with voice-keyed ducking.

Usage: python scripts/mix.py VO_DIR MUSIC.wav OUT.wav [--no-voice]
The narration WAVs come from scripts/tts.py; start times from
audio/narration.json. Final loudness normalization happens in finish.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
SR = 48000
N = SR * 90


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def smooth(env, attack, release):
    out = np.empty_like(env)
    a, r = np.exp(-1 / (attack * SR)), np.exp(-1 / (release * SR))
    level = 0.0
    for i, v in enumerate(env):
        coef = a if v > level else r
        level = coef * level + (1 - coef) * v
        out[i] = level
    return out


def main() -> None:
    vo_dir, music_path, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    with_voice = "--no-voice" not in sys.argv
    spec = json.loads((ROOT / "audio" / "narration.json").read_text())
    music, sr = sf.read(music_path, dtype="float64")
    assert sr == SR and len(music) == N
    voice = np.zeros(N)
    for line in spec["lines"]:
        clip, vsr = sf.read(vo_dir / f"{line['id']}.wav", dtype="float64")
        clip = resample_poly(clip, SR, vsr)
        i = int(line["start"] * SR)
        voice[i:i + len(clip)] += clip[: N - i]
    active = np.abs(voice) > 0.02
    voice_db = rms_db(voice[active])
    # Bed sits 9 dB under the voice and ducks a further 7 dB while it speaks.
    music *= 10 ** ((voice_db - 9 - rms_db(music)) / 20)
    gate = np.zeros(N)
    # Envelope at 100 Hz resolution keeps the follower fast.
    hop = SR // 100
    blocks = np.sqrt(np.mean(voice[: N // hop * hop].reshape(-1, hop) ** 2, axis=1))
    key = (blocks > 0.01).astype(float)
    key = smooth(np.repeat(key, hop), 0.06, 0.35)
    gate[: len(key)] = key
    duck = 10 ** (-7 * gate / 20)
    mix = music * duck[:, None]
    if with_voice:
        mix += voice[:, None] * 0.98
    sf.write(out, mix.astype(np.float32), SR, subtype="FLOAT")
    print(f"voice active rms {voice_db:.1f} dBFS, peak {np.max(np.abs(mix)):.3f}")


if __name__ == "__main__":
    main()
