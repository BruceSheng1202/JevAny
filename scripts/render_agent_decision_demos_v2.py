#!/usr/bin/env python3
"""Render the paired README animations for Jev inside LLM agent loops.

Both lanes of a pair share one clock and one playback rate, so the moment each
lane stops matches the recorded wall-clock ratio. Only the totals are measured;
the per-step artwork is schematic. A finished lane freezes until the loop ends.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


WIDTH, HEIGHT = 1120, 560
MARGIN = 36
CARD_W, CARD_H = 512, 392
CARD_Y = 78
PAD = 28
SCENE_Y = 150
SCENE_W = CARD_W - 2 * PAD

PLAY_MS = 2400
FRAME_MS = 40
END_HOLD_MS = 600

BG = "#070b14"
CARD = "#111a2d"
SURFACE = "#18233a"
SUNKEN = "#0b1221"
LINE = "#24314c"
TEXT = "#f4f7ff"
MUTED = "#9aa7c1"
DIM = "#66748f"
TEAL = "#2dd4bf"
BLUE = "#60a5fa"

FONT_DIRS = ("/usr/share/fonts/dejavu", "/usr/share/fonts/truetype/dejavu")

TITLES = {
    "webshop": ("WebShop", "buy the matching blazer"),
    "frozen_lake": ("FrozenLake", "reach the goal"),
    "sqlite": ("Terminal-Bench", "recover the truncated SQLite rows"),
}
OUTCOMES = {
    "webshop": "purchased, reward 1",
    "frozen_lake": "goal reached, reward 1",
    "sqlite": "10 rows recovered, reward 1",
}


@lru_cache
def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = f"{'DejaVuSansMono' if mono else 'DejaVuSans'}{'-Bold' if bold else ''}.ttf"
    for directory in FONT_DIRS:
        candidate = Path(directory, name)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError(f"{name} is not in any of {FONT_DIRS}")


def blend(base: str, other: str, weight: float) -> tuple[int, int, int]:
    first, second = ImageColor.getrgb(base), ImageColor.getrgb(other)
    return tuple(round(a + (b - a) * weight) for a, b in zip(first, second))


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, size: int = 26,
         color=TEXT, bold: bool = False, mono: bool = False, anchor: str = "la") -> None:
    draw.text(xy, value, font=font(size, bold, mono), fill=color, anchor=anchor)


def width_of(draw: ImageDraw.ImageDraw, value: str, size: int, bold: bool = False) -> float:
    return draw.textlength(value, font=font(size, bold))


def run(draw: ImageDraw.ImageDraw, x: float, y: float, parts, size: int = 25) -> float:
    """Draw coloured segments on one baseline and return the end of the line."""
    for value, color, bold in parts:
        draw.text((x, y), value, font=font(size, bold), fill=color)
        x += width_of(draw, value, size, bold)
    return x


def thousands(value: int) -> str:
    if value < 1000:
        return str(value)
    return f"{value / 1000:.1f}k" if value < 100_000 else f"{value / 1000:.0f}k"


def tick_mark(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, color) -> None:
    draw.line([(cx - size, cy + size * 0.1), (cx - size * 0.3, cy + size * 0.8),
               (cx + size, cy - size * 0.8)], fill=color,
              width=max(2, round(size / 3.6)), joint="curve")


def arrow(draw: ImageDraw.ImageDraw, cx: float, cy: float, direction: str,
          color, size: int = 13) -> None:
    dx, dy = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, -1), "Down": (0, 1)}[direction]
    tip = (cx + dx * size, cy + dy * size)
    base = (cx - dx * size * 0.7, cy - dy * size * 0.7)
    wing = (-dy * size * 0.9, dx * size * 0.9)
    draw.polygon([tip, (base[0] + wing[0], base[1] + wing[1]),
                  (base[0] - wing[0], base[1] - wing[1])], fill=color)


@dataclass(frozen=True)
class Step:
    actor: str
    title: str
    state: int


def steps_for(key: str, trace: dict, assisted: bool) -> list[Step]:
    if key == "webshop":
        if assisted:
            actors = ["LLM", "LLM", "Jev", "Jev", "LLM"]
            titles = [item["action"] for item in trace["jev_harness"]["path"]]
            states = [0, 1, 2, 3, 3]
        else:
            titles = trace["baseline"]["path"]
            actors = ["LLM"] * len(titles)
            states = [0, 1, 2, 3, 3, 3, 3]
        return [Step(a, t, s) for a, t, s in zip(actors, titles, states)]
    if key == "frozen_lake":
        moves = [Step("Jev" if assisted else "LLM", f"move {item['action'].lower()}", index + 1)
                 for index, item in enumerate(trace["steps"])]
        return [Step("LLM", "plan the route", 0)] + moves if assisted else moves
    if assisted:
        return [Step("LLM", "propose 3 probes", 0), Step("Jev", "inspect the raw page", 1),
                Step("LLM", "parse cell pointers", 2), Step("LLM", "write recover.json", 3)]
    return [Step("LLM", "read the schema", 0), Step("LLM", "dump the raw page", 1),
            Step("LLM", "parse cell pointers", 2), Step("LLM", "write recover.json", 3)]


def webshop_scene(draw: ImageDraw.ImageDraw, x: int, y: int, state: int, color: str) -> None:
    if state == 0:
        draw.rounded_rectangle((x, y + 10, x + SCENE_W, y + 74), 16, fill=SURFACE)
        draw.ellipse((x + 24, y + 30, x + 50, y + 56), outline=MUTED, width=3)
        draw.line((x + 48, y + 54, x + 58, y + 64), fill=MUTED, width=3)
        text(draw, (x + 74, y + 28), "green blazer, small", 27)
        text(draw, (x + 4, y + 94), "long sleeve, under $80", 23, MUTED)
        for row in range(2):
            top = y + 132 + row * 58
            draw.rounded_rectangle((x, top, x + SCENE_W, top + 48), 12, fill=SURFACE)
            draw.rounded_rectangle((x + 12, top + 8, x + 52, top + 40), 8, fill=LINE)
            draw.rounded_rectangle((x + 68, top + 13, x + 300, top + 21), 4, fill=LINE)
            draw.rounded_rectangle((x + 68, top + 29, x + 212, top + 35), 3,
                                   fill=blend(LINE, CARD, 0.5))
        return

    draw.rounded_rectangle((x, y + 6, x + 186, y + 218), 16, fill=SURFACE)
    chosen = state >= 2
    cloth = "#2f6f5a" if chosen else "#414d68"
    mid, top, hem = x + 93, y + 36, y + 196
    draw.polygon([(mid - 28, top), (mid + 28, top), (mid + 46, top + 14), (mid + 60, top + 56),
                  (mid + 46, top + 64), (mid + 40, top + 42), (mid + 40, hem), (mid - 40, hem),
                  (mid - 40, top + 42), (mid - 46, top + 64), (mid - 60, top + 56),
                  (mid - 46, top + 14)], fill=cloth)
    draw.polygon([(mid - 26, top), (mid + 26, top), (mid, top + 54)],
                 fill=blend(cloth, SUNKEN, 0.45))
    draw.line((mid, top + 54, mid, hem - 6), fill=blend(cloth, TEXT, 0.35), width=3)

    right = x + 206
    text(draw, (right, y + 4), "Women's blazer", 27, bold=True)
    text(draw, (right, y + 40), "long sleeve, under $80", 22, MUTED)
    text(draw, (right, y + 78), "COLOR", 20, DIM, bold=True)
    for index, swatch in enumerate(("#245c49", "#55613a", "#b3a07c")):
        cx, cy = right + 22 + index * 64, y + 124
        draw.ellipse((cx - 20, cy - 20, cx + 20, cy + 20), fill=swatch)
        if chosen and index == 0:
            draw.ellipse((cx - 27, cy - 27, cx + 27, cy + 27), outline=color, width=3)
    if chosen:
        text(draw, (right, y + 154), "z-dark green", 22, color, bold=True)
        tick_mark(draw, right + width_of(draw, "z-dark green", 22, True) + 18, y + 166, 9, color)
    text(draw, (right, y + 190), "SIZE", 20, DIM, bold=True)
    for index, label in enumerate(("XS", "S", "M")):
        left = right + index * 80
        picked = state >= 3 and index == 1
        draw.rounded_rectangle((left, y + 220, left + 68, y + 264), 12,
                               fill=blend(color, CARD, 0.82) if picked else SURFACE,
                               outline=color if picked else None, width=2)
        text(draw, (left + 34, y + 242), label, 24, color if picked else MUTED, picked, anchor="mm")


def lake_scene(draw: ImageDraw.ImageDraw, x: int, y: int, trace: dict, step: Step,
               fraction: float, finished: bool, color: str, assisted: bool) -> None:
    cell, gap = 52, 6
    places = [tuple(item["player"]) for item in trace["steps"]] + [(0, 3)]
    state = step.state
    for row in range(4):
        for col in range(4):
            left, top = x + col * (cell + gap), y + 10 + row * (cell + gap)
            goal, hole = (row, col) == (0, 3), (row, col) == (3, 3)
            draw.rounded_rectangle((left, top, left + cell, top + cell), 11,
                                   fill="#12343a" if goal else SUNKEN if hole else "#1b2742")
            if goal:
                draw.line((left + 20, top + 14, left + 20, top + 40), fill=TEAL, width=3)
                draw.polygon([(left + 22, top + 14), (left + 41, top + 21),
                              (left + 22, top + 28)], fill=TEAL)
            elif hole:
                draw.ellipse((left + 13, top + 13, left + 39, top + 39),
                             outline="#3a4663", width=3)
    if assisted and state:
        for row, col in places[1:]:
            draw.ellipse((x + col * 58 + 22, y + row * 58 + 32,
                          x + col * 58 + 30, y + row * 58 + 40), fill=blend(TEAL, CARD, 0.55))
    spot = places[state]
    if state and not finished:
        start, target = places[state - 1], places[state]
        shift = min(1.0, max(0.0, (fraction - 0.4) / 0.4))
        spot = tuple(a + (b - a) * shift for a, b in zip(start, target))
    cx, cy = x + spot[1] * 58 + 26, y + spot[0] * 58 + 36
    draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), fill=blend(color, CARD, 0.8))
    draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=color)

    text(draw, (x + 344, y + 14), "4 directions", 22, DIM, anchor="ma")
    taken = trace["steps"][state - 1]["action"] if state else None
    for direction, (dx, dy) in (("Up", (0, -54)), ("Down", (0, 54)),
                                ("Left", (-74, 0)), ("Right", (74, 0))):
        cx, cy = x + 344 + dx, y + 148 + dy
        picked = direction == taken
        draw.rounded_rectangle((cx - 32, cy - 25, cx + 32, cy + 25), 12,
                               fill=blend(color, CARD, 0.8) if picked else SURFACE,
                               outline=color if picked else None, width=2)
        arrow(draw, cx, cy, direction, color if picked else DIM)


def terminal_scene(draw: ImageDraw.ImageDraw, x: int, y: int, options: list[dict], state: int,
                   assisted: bool, finished: bool, color: str) -> None:
    draw.rounded_rectangle((x, y + 6, x + SCENE_W, y + 236), 14, fill=SUNKEN)
    for index in range(3):
        cx = x + 28 + index * 20
        draw.ellipse((cx - 5, y + 25, cx + 5, y + 35), fill=blend(LINE, TEXT, 0.15))
    text(draw, (x + 102, y + 18), "trunc.db", 21, DIM, mono=True)
    draw.line((x + 1, y + 50, x + SCENE_W - 1, y + 50), fill=LINE)

    if assisted and state == 0:
        for index, option in enumerate(options):
            top = y + 70 + index * 54
            label = option["display_command"].split(";")[0].split("|")[0].strip()
            if index == 0:
                draw.rounded_rectangle((x + 14, top - 8, x + SCENE_W - 14, top + 38), 10,
                                       fill=blend(color, SUNKEN, 0.86))
            draw.ellipse((x + 28, top + 6, x + 46, top + 24), fill=color if index == 0 else None,
                         outline=color if index == 0 else DIM, width=2)
            text(draw, (x + 62, top + 3), label, 21, TEXT if index == 0 else DIM, mono=True)
        return

    first = "$ od -A x -t x1z trunc.db" if assisted else "$ sqlite3 trunc.db '.schema'"
    lines = [(first, TEXT), ("$ od -A x -t x1z trunc.db", TEXT),
             ("cell pointers -> 10 records", TEXT), ("recover.json: 10 rows", TEXT)]
    if assisted:
        lines[1] = ("0d 00 00 00 0a 0f 49 ...", MUTED)
    shown = lines[:state + 1]
    if finished:
        shown = shown[-3:] + [("10 rows, verifier passed", TEAL)]
    shown = shown[-4:]
    for index, (line, shade) in enumerate(shown):
        last = index == len(shown) - 1
        text(draw, (x + 28, y + 72 + index * 40), line, 22, shade if last else DIM, mono=True)


def pair_for(key: str, trace: dict) -> tuple[dict, dict]:
    pair = trace["pair"] if key == "frozen_lake" else trace
    baseline, assisted = pair["baseline"], pair["jev_harness"]
    for item in (baseline, assisted):
        if not 0 < item["wall_time_seconds"] < float("inf"):
            raise ValueError(f"{key}: wall_time_seconds must be finite and positive")
    return baseline, assisted


def finish_ms(item: dict, longest: float) -> int:
    # GIF timestamps have 10 ms precision; include the actual finish boundary.
    return round(PLAY_MS * item["wall_time_seconds"] / longest / 10) * 10


def render_frame(key: str, trace: dict, tick: int) -> Image.Image:
    baseline, assisted = pair_for(key, trace)
    longest = max(item["wall_time_seconds"] for item in (baseline, assisted))
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    name, subject = TITLES[key]
    text(draw, (MARGIN, 22), name, 33, bold=True)
    text(draw, (MARGIN + width_of(draw, name, 33, True) + 22, 29), subject, 26, MUTED)

    for lane, item in enumerate((baseline, assisted)):
        card = MARGIN + lane * (CARD_W + 24)
        inner = card + PAD
        edge = card + CARD_W - PAD
        color = TEAL if lane else BLUE
        end = finish_ms(item, longest)
        finished = tick >= end
        elapsed = item["wall_time_seconds"] if finished else tick / PLAY_MS * longest
        steps = steps_for(key, trace, bool(lane))
        progress = min(tick / end, 1.0)
        position = min(progress * len(steps), len(steps) - 0.000001)
        step, fraction = steps[int(position)], position % 1

        draw.rounded_rectangle((card, CARD_Y, card + CARD_W, CARD_Y + CARD_H), 20,
                               fill=CARD, outline=LINE)
        draw.ellipse((inner, CARD_Y + 31, inner + 16, CARD_Y + 47), fill=color)
        text(draw, (inner + 28, CARD_Y + 21), "LLM + Jev" if lane else "LLM only", 28, color, True)
        clock = f"{elapsed:.2f} s" if key == "webshop" else f"{elapsed:.1f} s"
        text(draw, (edge, CARD_Y + 16), clock, 36, TEXT, True, anchor="ra")
        if finished:
            tick_mark(draw, edge - width_of(draw, clock, 36, True) - 24, CARD_Y + 38, 11, color)

        if key == "webshop":
            webshop_scene(draw, inner, SCENE_Y, step.state, color)
        elif key == "frozen_lake":
            lake_scene(draw, inner, SCENE_Y, trace, step, fraction, finished, color, bool(lane))
        else:
            terminal_scene(draw, inner, SCENE_Y, trace["options"], step.state,
                           bool(lane), finished, color)

        if finished:
            tick_mark(draw, inner + 11, CARD_Y + 344, 11, color)
            text(draw, (inner + 34, CARD_Y + 330), OUTCOMES[key], 26)
        else:
            run(draw, inner, CARD_Y + 330,
                [(step.actor, color if step.actor == "Jev" else BLUE, True),
                 ("  ", DIM, False), (step.title, TEXT, False)], 26)
        track = CARD_Y + 372
        draw.rounded_rectangle((inner, track, inner + SCENE_W, track + 8), 4, fill=LINE)
        filled = round(SCENE_W * min(tick / PLAY_MS, item["wall_time_seconds"] / longest))
        if filled:
            draw.rounded_rectangle((inner, track, inner + filled, track + 8), 4, fill=color)

    run(draw, MARGIN, 500, [
        (f"{baseline['steps']} -> ", MUTED, False), (str(assisted["steps"]), TEAL, True),
        (f" {baseline['step_label']}     ", DIM, False),
        (f"{baseline['llm_calls']} -> ", MUTED, False), (str(assisted["llm_calls"]), TEAL, True),
        (" LLM calls     ", DIM, False),
        (f"{thousands(baseline['tokens'])} -> ", MUTED, False),
        (thousands(assisted["tokens"]), TEAL, True), (" tokens", DIM, False),
    ], 24)
    rate = longest * 1000 / PLAY_MS
    speed = f"{rate:.0f}x" if rate >= 10 else f"{rate:.1f}x"
    text(draw, (WIDTH - MARGIN, 502), f"{speed} speed, measured totals", 22, DIM, anchor="ra")
    return image


def save(path: str, key: str, trace: dict) -> None:
    pair = pair_for(key, trace)
    longest = max(item["wall_time_seconds"] for item in pair)
    finishes = {finish_ms(item, longest) for item in pair}
    # Keep finish boundaries without introducing sub-20 ms frame delays.
    ticks = sorted(finishes | {
        tick for tick in range(0, PLAY_MS + 1, FRAME_MS)
        if all(abs(tick - end) >= 20 for end in finishes)
    })
    durations = [b - a for a, b in zip(ticks, ticks[1:])] + [END_HOLD_MS]
    # A shared palette prevents flicker and lets GIF store just changed pixels.
    samples = [render_frame(key, trace, tick).resize((280, 140)) for tick in ticks[::8]]
    sheet = Image.new("RGB", (280, 140 * len(samples)))
    for index, sample in enumerate(samples):
        sheet.paste(sample, (0, 140 * index))
    palette = sheet.quantize(colors=128)
    frames = [
        render_frame(key, trace, tick).quantize(palette=palette, dither=Image.Dither.NONE)
        for tick in ticks
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    print(f"{output}: {len(frames)} frames, {sum(durations) / 1000:.1f}s, "
          f"Jev completes at {finish_ms(pair[1], longest) / 1000:.2f}s, "
          f"{output.stat().st_size / 1024:.0f} KiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="docs/demos/jev-agent-harness-traces.json")
    parser.add_argument("--terminal-out", default="docs/demos/jev-decision-terminal-v2.gif")
    parser.add_argument("--frozen-lake-out", default="docs/demos/jev-decision-frozen-lake-v2.gif")
    parser.add_argument("--webshop-out", default="docs/demos/jev-decision-webshop-v2.gif")
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    for key, output in (("webshop", args.webshop_out), ("frozen_lake", args.frozen_lake_out),
                        ("sqlite", args.terminal_out)):
        save(output, key, trace[key])


if __name__ == "__main__":
    main()
