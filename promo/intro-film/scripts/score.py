"""Synthesize the original electronic score, cue-aligned to the film.

Usage: python scripts/score.py OUT.wav
Deterministic (fixed seeds); no samples or third-party recordings. 120 BPM in
A minor. Cue times match the chapter boundaries in src/film.js.
"""

import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000
DUR = 90.0
N = int(SR * DUR)
BEAT = 0.5
rng = np.random.default_rng(7)


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def place(buf: np.ndarray, sig: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += gain * sig[: j - i]


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x)


def env_ar(n, attack, release):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(attack, 1e-4)) * np.minimum(1, (n / SR - t) / max(release, 1e-4))
    return np.clip(e, 0, 1)


def pad_voice(freq, dur, bright=0.35):
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for detune in (-0.07, 0.0, 0.06):
        f = freq * 2 ** (detune / 12)
        vib = 0.0015 * np.sin(2 * np.pi * 0.23 * t + detune * 40)
        for h, amp in ((1, 1.0), (2, 0.5 * bright), (3, 0.28 * bright), (4, 0.12 * bright)):
            out += amp * np.sin(2 * np.pi * f * h * t * (1 + vib) + rng.uniform(0, 6.28))
    return out / 6


def pluck(freq, dur=0.6, decay=5.5):
    t = np.arange(int(dur * SR)) / SR
    tone = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t) + 0.12 * np.sin(2 * np.pi * 3 * freq * t)
    return tone * np.exp(-decay * t) * np.minimum(1, t / 0.004)


def kick():
    t = np.arange(int(0.45 * SR)) / SR
    f = 46 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) * np.exp(-t * 7.5) + 0.25 * np.exp(-t * 400) * rng.standard_normal(len(t))) * 0.9


def hat(length=0.05):
    n = int(length * SR)
    return hp(rng.standard_normal(n), 7000, 4) * np.exp(-np.arange(n) / SR * 70) * 0.35


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = hp(rng.standard_normal(n), 1500, 2) * (t / dur) ** 2.2
    sweep = np.sin(2 * np.pi * np.cumsum(220 + 660 * (t / dur) ** 2) / SR) * (t / dur) ** 3 * 0.25
    return (noise * 0.18 + sweep) * np.minimum(1, (dur - t) / 0.03)


def impact():
    t = np.arange(int(2.6 * SR)) / SR
    sub = np.sin(2 * np.pi * (38 + 30 * np.exp(-t * 6)) * t) * np.exp(-t * 1.6)
    burst = lp(rng.standard_normal(len(t)), 2500, 2) * np.exp(-t * 9) * 0.5
    return sub + burst


def reverb_ir(seconds=2.6):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    left = rng.standard_normal(n) * np.exp(-t * 3.0)
    right = rng.standard_normal(n) * np.exp(-t * 3.0)
    return lp(left, 6000) / np.sqrt(n), lp(right, 6000) / np.sqrt(n)


# A minor: Am, F, C, G; voicings as MIDI notes.
CHORDS = {"Am": [57, 60, 64, 69], "F": [53, 57, 60, 65], "C": [55, 60, 64, 67], "G": [55, 59, 62, 67], "Am9": [57, 60, 64, 71]}
ROOT = {"Am": 45, "F": 41, "C": 48, "G": 43, "Am9": 45}
PROG = ["Am", "F", "C", "G"]


def main() -> None:
    out = sys.argv[1]
    pads, plucks, drums, bass, fx = (np.zeros(N) for _ in range(5))

    # Hook: the wall reveal lands on an impact; the groove starts at once.
    place(fx, impact(), 0.0, 0.7)
    place(fx, riser(2.0), 8.0, 0.8)
    place(fx, impact(), 10.0, 0.85)

    # Chord progression, one chord per 2 s bar from 0 s. Section starts match the
    # chapter boundaries in src/film.js.
    sections = [  # (start, end, kick pattern, hats, arp, bass, brightness)
        (0.0, 10.0, "four", True, True, True, 0.5),
        (10.0, 17.0, "half", False, True, False, 0.35),
        (17.0, 27.0, "half", True, True, True, 0.4),
        (27.0, 36.0, "four", True, True, True, 0.45),
        (36.0, 49.0, "four", True, True, True, 0.55),
        (49.0, 63.0, "four", True, True, True, 0.5),
        (63.0, 75.0, "four", True, True, True, 0.6),
        (75.0, 80.5, "none", False, True, False, 0.35),
    ]
    for start, end, kicks, hats, arp, has_bass, bright in sections:
        t = start
        while t < end - 1e-6:
            bar = int(round(t / 2.0))
            name = PROG[bar % 4]
            length = min(2.0, end - t)
            chord = sum(pad_voice(hz(m), length + 0.6, bright) for m in CHORDS[name]) * env_ar(int((length + 0.6) * SR), 0.25, 0.6)
            place(pads, chord, t, 0.32)
            for b in range(int(round(length / BEAT))):
                bt = t + b * BEAT
                if kicks == "four" or (kicks == "half" and b % 2 == 0):
                    place(drums, kick(), bt, 0.55)
                if hats:
                    place(drums, hat(), bt + BEAT / 2, 0.5)
                if has_bass:
                    for s in (0, 1):
                        note = pluck(hz(ROOT[name]), 0.24, 9) * 1.4
                        place(bass, note, bt + s * BEAT / 2, 0.55 if s == 0 else 0.35)
            if arp:
                notes = CHORDS[name] + [CHORDS[name][1] + 12]
                for s in range(int(round(length / (BEAT / 2)))):
                    place(plucks, pluck(hz(notes[(s * 3 + bar) % len(notes)] + 12), 0.5, 7.5), t + s * BEAT / 2, 0.11)
            t += 2.0
    place(fx, riser(1.8), 78.7, 0.6)

    # Finale: impact, wide chords, long resolve to A minor.
    place(fx, impact(), 80.5, 0.9)
    for i, (name, at, length) in enumerate((("Am9", 80.5, 2.0), ("F", 82.5, 2.0), ("C", 84.5, 2.0), ("Am", 86.5, 3.4))):
        chord = sum(pad_voice(hz(m), length + 0.4, 0.45) for m in CHORDS[name]) * env_ar(int((length + 0.4) * SR), 0.3, 0.8 if i < 3 else 3.0)
        place(pads, chord, at, 0.36)
        for s in range(int(length / (BEAT / 2))):
            notes = CHORDS[name] + [CHORDS[name][2] + 12]
            place(plucks, pluck(hz(notes[(s * 2 + i) % len(notes)] + 12), 0.5, 7.0), at + s * BEAT / 2, 0.09 * (1 - s / (length / (BEAT / 2))) if i == 3 else 0.09)
    for i in range(16):
        place(drums, kick(), 80.5 + i * BEAT, 0.42 if i % 2 == 0 and i < 14 else 0.0)

    ir_l, ir_r = reverb_ir()
    wet_src = pads * 0.6 + plucks + fx * 0.5
    dry = lp(pads, 3500) + plucks * 0.8 + drums + lp(bass, 600) + fx
    mix_l = dry + fftconvolve(wet_src, ir_l)[:N] * 0.9
    mix_r = dry + fftconvolve(wet_src, ir_r)[:N] * 0.9
    stereo = np.stack([mix_l, mix_r], axis=1)
    stereo = hp(stereo.T, 28).T
    fade = np.ones(N)
    tail = int(0.4 * SR)
    fade[-tail:] = np.linspace(1, 0, tail) ** 2
    stereo *= fade[:, None]
    stereo = np.tanh(stereo / np.max(np.abs(stereo)) * 1.2) * 0.7
    sf.write(out, stereo.astype(np.float32), SR, subtype="FLOAT")
    print(f"wrote {out} peak {np.max(np.abs(stereo)):.3f}")


if __name__ == "__main__":
    main()
