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

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont


WIDTH, HEIGHT = 1120, 552
MARGIN = 36
CARD_W, CARD_H = 512, 420
CARD_Y, HEADER_H = 70, 56
PAD = 28
SCENE_Y, SCENE_H = 142, 278
SCENE_W = CARD_W - 2 * PAD

PLAY_MS = 4000
FRAME_MS = 80
END_HOLD_MS = 1000

PAGE = "#f3f5fb"
CARD = "#ffffff"
HAIRLINE = "#e2e8f4"
SURFACE = "#eef2f9"
SUNKEN = "#e3e9f4"
INK = "#1f2a44"
MUTED = "#5f6d86"
DIM = "#95a2b8"
SLATE = "#6e7d99"
JEV = "#6b4fbb"
ICE = "#d6e1f2"
ICE_GOAL = "#d6ecdf"
ICE_HOLE = "#aab6cc"
FLAG = "#2f7d66"
TERM_BG = "#1d2438"
TERM_INK = "#e9edf8"
TERM_DIM = "#8d99b2"

FONT_DIR = Path(__file__).resolve().parents[1] / "site/assets/fonts"

TITLES = {
    "webshop": ("WebShop", "buy the matching blazer"),
    "frozen_lake": ("FrozenLake", "reach the goal"),
    "sqlite": ("Terminal-Bench", "recover the truncated SQLite rows"),
}
OUTCOMES = {"webshop": "purchased", "frozen_lake": "goal reached",
            "sqlite": "10 rows recovered"}


@lru_cache
def font(size: int, bold: bool = False, display: bool = False) -> ImageFont.FreeTypeFont:
    name = "ibm-plex-sans-condensed-medium.woff2" if display else "dm-sans-latin.woff2"
    face = ImageFont.truetype(str(FONT_DIR / name), size)
    if not display:
        face.set_variation_by_axes([min(40, max(9, size)), 500 if bold else 400])
    return face


def blend(base: str, other: str, weight: float) -> tuple[int, int, int]:
    first, second = ImageColor.getrgb(base), ImageColor.getrgb(other)
    return tuple(round(a + (b - a) * weight) for a, b in zip(first, second))


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, size: int = 24,
         color=INK, bold: bool = False, tabular: bool = False, anchor: str = "la",
         display: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold, display), fill=color, anchor=anchor,
              features=["tnum"] if tabular else None)


def width_of(draw: ImageDraw.ImageDraw, value: str, size: int, bold: bool = False,
             tabular: bool = False, display: bool = False) -> float:
    return draw.textlength(value, font=font(size, bold, display),
                           features=["tnum"] if tabular else None)


def run(draw: ImageDraw.ImageDraw, x: float, y: float, parts, size: int = 24) -> float:
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
              width=max(2, round(size / 4)), joint="curve")


def arrow(draw: ImageDraw.ImageDraw, cx: float, cy: float, direction: str,
          color, size: int = 12) -> None:
    dx, dy = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, -1), "Down": (0, 1)}[direction]
    tip = (cx + dx * size, cy + dy * size)
    base = (cx - dx * size * 0.7, cy - dy * size * 0.7)
    wing = (-dy * size * 0.85, dx * size * 0.85)
    draw.polygon([tip, (base[0] + wing[0], base[1] + wing[1]),
                  (base[0] - wing[0], base[1] - wing[1])], fill=color)


def panel(draw: ImageDraw.ImageDraw, box, radius: int = 14, fill=SURFACE, outline=None) -> None:
    draw.rounded_rectangle(box, radius, fill=fill, outline=outline)


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
        return [Step("LLM", "propose 3 probes", 0), Step("Jev", "pick the raw-page dump", 1),
                Step("LLM", "parse cell pointers", 2), Step("LLM", "write recover.json", 3)]
    return [Step("LLM", "read the schema", 0), Step("LLM", "dump the raw page", 1),
            Step("LLM", "parse cell pointers", 2), Step("LLM", "write recover.json", 3)]


def webshop_scene(draw: ImageDraw.ImageDraw, x: int, y: int, state: int, color: str) -> None:
    if state == 0:
        panel(draw, (x, y + 16, x + SCENE_W, y + 80), 16, SURFACE)
        draw.ellipse((x + 26, y + 36, x + 52, y + 62), outline=MUTED, width=3)
        draw.line((x + 50, y + 60, x + 60, y + 70), fill=MUTED, width=3)
        text(draw, (x + 76, y + 35), "green blazer, small", 26)
        text(draw, (x + 4, y + 100), "long sleeve, under $80", 22, MUTED)
        for row in range(2):
            top = y + 142 + row * 62
            panel(draw, (x, top, x + SCENE_W, top + 50), 12, SURFACE)
            panel(draw, (x + 12, top + 9, x + 54, top + 41), 8, SUNKEN)
            panel(draw, (x + 70, top + 14, x + 306, top + 22), 4, SUNKEN)
            panel(draw, (x + 70, top + 31, x + 218, top + 37), 3, blend(SUNKEN, SURFACE, 0.5))
        return

    panel(draw, (x, y + 4, x + 196, y + 248), 16, SURFACE)
    chosen = state >= 2
    cloth = "#3f7a63" if chosen else "#9aa7bd"
    shade = blend(cloth, INK, 0.22)
    mid, top, hem = x + 98, y + 36, y + 224
    # A long-sleeve blazer: shoulders, sleeves down to the hem, then the lapels.
    draw.polygon([(mid - 26, top), (mid + 26, top), (mid + 52, top + 14), (mid + 44, top + 54),
                  (mid + 44, hem), (mid - 44, hem), (mid - 44, top + 54), (mid - 52, top + 14)],
                 fill=cloth)
    for side in (-1, 1):
        draw.polygon([(mid + side * 44, top + 14), (mid + side * 62, top + 30),
                      (mid + side * 58, top + 150), (mid + side * 38, top + 150),
                      (mid + side * 42, top + 40)], fill=shade)
        draw.polygon([(mid + side * 24, top + 2), (mid + side * 6, top + 96),
                      (mid, top + 60)], fill=shade)
    draw.line((mid, top + 96, mid, hem - 8), fill=blend(cloth, "#ffffff", 0.4), width=3)
    for button in (top + 112, top + 146):
        draw.ellipse((mid - 5, button - 5, mid + 5, button + 5),
                     fill=blend(cloth, "#ffffff", 0.55))

    right = x + 216
    text(draw, (right, y + 4), "Women's blazer", 26)
    text(draw, (right, y + 40), "long sleeve, under $80", 21, MUTED)
    text(draw, (right, y + 70), "COLOR", 19, DIM)
    for index, swatch in enumerate(("#2f6b55", "#6f7a4c", "#c0ad88")):
        cx, cy = right + 22 + index * 64, y + 124
        draw.ellipse((cx - 20, cy - 20, cx + 20, cy + 20), fill=swatch)
        if chosen and index == 0:
            draw.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), outline=color, width=3)
    if chosen:
        text(draw, (right, y + 152), "z-dark green", 22, color)
        tick_mark(draw, right + width_of(draw, "z-dark green", 22) + 20, y + 164, 9, color)
    text(draw, (right, y + 190), "SIZE", 19, DIM)
    for index, label in enumerate(("XS", "S", "M")):
        left = right + index * 80
        picked = state >= 3 and index == 1
        panel(draw, (left, y + 216, left + 68, y + 260), 12,
              blend(color, CARD, 0.88) if picked else SURFACE, color if picked else None)
        text(draw, (left + 34, y + 238), label, 23, color if picked else MUTED, anchor="mm")


def flag(draw: ImageDraw.ImageDraw, left: int, top: int) -> None:
    draw.line((left + 20, top + 12, left + 20, top + 44), fill=FLAG, width=3)
    draw.polygon([(left + 22, top + 12), (left + 44, top + 21), (left + 22, top + 30)], fill=FLAG)


def lake_scene(draw: ImageDraw.ImageDraw, x: int, y: int, trace: dict, step: Step,
               fraction: float, finished: bool, color: str, assisted: bool) -> None:
    cell, pitch = 56, 61
    places = [tuple(item["player"]) for item in trace["steps"]] + [(0, 3)]
    state = step.state
    goal_at = (0, 3)
    for row in range(4):
        for col in range(4):
            left, top = x + col * pitch, y + 12 + row * pitch
            goal, hole = (row, col) == goal_at, (row, col) == (3, 3)
            panel(draw, (left, top, left + cell, top + cell), 12,
                  ICE_GOAL if goal else ICE_HOLE if hole else ICE)
            if goal:
                flag(draw, left, top)
            elif hole:
                draw.ellipse((left + 13, top + 13, left + 43, top + 43), fill="#6f7c95")
    if assisted and state:
        for row, col in places[1:-1]:
            cx, cy = x + col * pitch + 28, y + 12 + row * pitch + 28
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=blend(color, ICE, 0.4))
    spot = places[state]
    if state and not finished:
        start, target = places[state - 1], places[state]
        shift = min(1.0, max(0.0, (fraction - 0.35) / 0.45))
        spot = tuple(a + (b - a) * shift for a, b in zip(start, target))
    cx, cy = x + spot[1] * pitch + 28, y + 12 + spot[0] * pitch + 28
    draw.ellipse((cx - 23, cy - 23, cx + 23, cy + 23), fill=blend(color, CARD, 0.84))
    draw.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=color)
    # Keep the goal visible once the agent stands on it.
    flag(draw, x + goal_at[1] * pitch, y + 12 + goal_at[0] * pitch)

    center = x + 353
    text(draw, (center, y + 14), "4 candidate moves", 21, DIM, anchor="ma")
    taken = trace["steps"][state - 1]["action"] if state else None
    for direction, (dx, dy) in (("Up", (0, -52)), ("Down", (0, 52)),
                                ("Left", (-64, 0)), ("Right", (64, 0))):
        bx, by = center + dx, y + 152 + dy
        picked = direction == taken
        panel(draw, (bx - 31, by - 23, bx + 31, by + 23), 12,
              blend(color, CARD, 0.86) if picked else SURFACE, color if picked else None)
        arrow(draw, bx, by, direction, color if picked else DIM)


def terminal_scene(draw: ImageDraw.ImageDraw, x: int, y: int, options: list[dict], state: int,
                   assisted: bool, finished: bool, color: str) -> None:
    panel(draw, (x, y + 4, x + SCENE_W, y + 254), 14, TERM_BG)
    for index in range(3):
        cx = x + 30 + index * 20
        draw.ellipse((cx - 5, y + 25, cx + 5, y + 35), fill="#4a5572")
    text(draw, (x + 104, y + 18), "trunc.db", 20, TERM_DIM, tabular=True)
    draw.line((x + 1, y + 50, x + SCENE_W - 1, y + 50), fill="#2b3350")

    if assisted and state <= 1:
        picked = state == 1
        for index, option in enumerate(options):
            top = y + 78 + index * 56
            label = option["display_command"].split(";")[0].split("|")[0].strip()
            chosen = picked and index == 0
            if chosen:
                panel(draw, (x + 16, top - 10, x + SCENE_W - 16, top + 38), 10,
                      blend(color, TERM_BG, 0.7))
            mark = blend(color, "#ffffff", 0.4)
            draw.ellipse((x + 30, top + 4, x + 48, top + 22), fill=mark if chosen else None,
                         outline=mark if chosen else TERM_DIM, width=2)
            text(draw, (x + 64, top + 1), label, 21,
                 TERM_INK if chosen else TERM_DIM, tabular=True)
        return

    if assisted:
        lines = [("$ od -A x -t x1z trunc.db", TERM_INK), ("0d 00 00 00 0a 0f 49 ...", TERM_DIM),
                 ("cell pointers -> 10 records", TERM_INK), ("recover.json: 10 rows", TERM_INK)]
        shown = lines[:state + 1]
    else:
        lines = [("$ sqlite3 trunc.db '.schema'", TERM_INK), ("$ od -A x -t x1z trunc.db", TERM_INK),
                 ("cell pointers -> 10 records", TERM_INK), ("recover.json: 10 rows", TERM_INK)]
        shown = lines[:state + 1]
    if finished:
        shown = shown[-3:] + [("verifier passed", "#64d8b6")]
    shown = shown[-4:]
    for index, (line, shade) in enumerate(shown):
        last = index == len(shown) - 1
        text(draw, (x + 30, y + 74 + index * 42), line, 21,
             shade if last else TERM_DIM, tabular=True)
    if not finished and len(shown) < 4:
        top = y + 74 + len(shown) * 42
        draw.rectangle((x + 30, top + 4, x + 42, top + 24), fill=blend(TERM_DIM, TERM_BG, 0.35))


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


def card_frame(draw: ImageDraw.ImageDraw, card: int, tint: str) -> None:
    box = (card, CARD_Y, card + CARD_W, CARD_Y + CARD_H)
    draw.rounded_rectangle(box, 20, fill=CARD, outline=HAIRLINE)
    draw.rounded_rectangle((card + 1, CARD_Y + 1, card + CARD_W - 1, CARD_Y + HEADER_H),
                           20, fill=tint)
    draw.rectangle((card + 1, CARD_Y + HEADER_H - 20, card + CARD_W - 1, CARD_Y + HEADER_H),
                   fill=tint)
    draw.line((card + 1, CARD_Y + HEADER_H, card + CARD_W - 1, CARD_Y + HEADER_H), fill=HAIRLINE)


def render_frame(key: str, trace: dict, tick: int) -> Image.Image:
    baseline, assisted = pair_for(key, trace)
    longest = max(item["wall_time_seconds"] for item in (baseline, assisted))
    image = Image.new("RGB", (WIDTH, HEIGHT), PAGE)
    draw = ImageDraw.Draw(image)
    name, subject = TITLES[key]
    text(draw, (MARGIN, 14), name, 34, INK, display=True)
    text(draw, (MARGIN + width_of(draw, name, 34, display=True) + 20, 24), subject, 24, MUTED)

    for lane, item in enumerate((baseline, assisted)):
        card = MARGIN + lane * (CARD_W + 24)
        inner = card + PAD
        edge = card + CARD_W - PAD
        color = JEV if lane else SLATE
        end = finish_ms(item, longest)
        finished = tick >= end
        elapsed = item["wall_time_seconds"] if finished else tick / PLAY_MS * longest
        steps = steps_for(key, trace, bool(lane))
        progress = min(tick / end, 1.0)
        position = min(progress * len(steps), len(steps) - 0.000001)
        step, fraction = steps[int(position)], position % 1

        card_frame(draw, card, blend(color, CARD, 0.93))
        text(draw, (inner, CARD_Y + 14), "LLM + Jev" if lane else "LLM only", 26, color, True)
        clock = f"{elapsed:.2f} s" if key == "webshop" else f"{elapsed:.1f} s"
        text(draw, (edge, CARD_Y + 12), clock, 32, INK, True, tabular=True, anchor="ra")
        if finished:
            tick_mark(draw, edge - width_of(draw, clock, 32, True, tabular=True) - 24,
                      CARD_Y + 32, 11, color)

        if key == "webshop":
            webshop_scene(draw, inner, SCENE_Y, step.state, color)
        elif key == "frozen_lake":
            lake_scene(draw, inner, SCENE_Y, trace, step, fraction, finished, color, bool(lane))
        else:
            terminal_scene(draw, inner, SCENE_Y, trace["options"], step.state,
                           bool(lane), finished, color)

        if finished:
            tick_mark(draw, inner + 11, CARD_Y + 376, 11, color)
            text(draw, (inner + 34, CARD_Y + 362), OUTCOMES[key], 25, MUTED)
        else:
            run(draw, inner, CARD_Y + 362,
                [(step.actor, color if step.actor == "Jev" else SLATE, True),
                 ("  ", DIM, False), (step.title, INK, False)], 25)
        track = CARD_Y + 400
        draw.rounded_rectangle((inner, track, inner + SCENE_W, track + 7), 4, fill=SUNKEN)
        # A finished lane freezes on its measured share, which rounding in
        # finish_ms can place just before the current tick.
        share = item["wall_time_seconds"] / longest if finished else tick / PLAY_MS
        filled = round(SCENE_W * min(share, item["wall_time_seconds"] / longest))
        if filled:
            draw.rounded_rectangle((inner, track, inner + filled, track + 7), 4, fill=color)

    run(draw, MARGIN, 512, [
        (f"{baseline['steps']} -> ", MUTED, False), (str(assisted["steps"]), JEV, True),
        (f" {baseline['step_label']}     ", DIM, False),
        (f"{baseline['llm_calls']} -> ", MUTED, False), (str(assisted["llm_calls"]), JEV, True),
        (" LLM calls     ", DIM, False),
        (f"{thousands(baseline['tokens'])} -> ", MUTED, False),
        (thousands(assisted["tokens"]), JEV, True), (" tokens", DIM, False),
    ], 23)
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
    samples = [render_frame(key, trace, tick).resize((280, 138)) for tick in ticks[::8]]
    sheet = Image.new("RGB", (280, 138 * len(samples)))
    for index, sample in enumerate(samples):
        sheet.paste(sample, (0, 138 * index))
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


def save_overview(path: str, sources: list[str]) -> None:
    """Join the three approved loops without changing their pixels or timing."""
    frames = []
    durations = []
    for source in sources:
        with Image.open(source) as clip:
            if clip.size != (WIDTH, HEIGHT):
                raise ValueError(f"{source}: expected a {WIDTH}×{HEIGHT} replay")
            for index in range(clip.n_frames):
                clip.seek(index)
                original = clip.convert("RGB")
                frame = original.quantize(colors=256, dither=Image.Dither.NONE)
                if ImageChops.difference(original, frame.convert("RGB")).getbbox():
                    raise ValueError(f"{source}: frame {index} cannot be preserved exactly")
                frames.append(frame)
                durations.append(clip.info["duration"])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    print(f"{output}: {len(frames)} frames, {sum(durations) / 1000:.1f}s, "
          f"{output.stat().st_size / 1024:.0f} KiB")


def save_site_media(sources: list[str]) -> None:
    """Export only this presentation's videos and posters at native resolution."""
    try:
        import av
    except ImportError as error:
        raise SystemExit("Install av to export site media: python -m pip install av") from error

    directory = Path(__file__).resolve().parents[1] / "site/assets/media/docs"
    directory.mkdir(parents=True, exist_ok=True)
    for source in sources:
        name = Path(source).stem
        timeline = []
        duration = 0
        with Image.open(source) as image:
            for index in range(image.n_frames):
                image.seek(index)
                timeline.append((duration, image.convert("RGB")))
                duration += image.info["duration"]
        for suffix, codec, options in (
            ("mp4", "libx264", {"crf": "26", "preset": "fast"}),
            ("webm", "libvpx-vp9", {"crf": "36", "b:v": "0", "cpu-used": "3", "row-mt": "1"}),
        ):
            container_options = {"movflags": "+faststart"} if suffix == "mp4" else {}
            with av.open(str(directory / f"{name}.{suffix}"), "w", options=container_options) as output:
                stream = output.add_stream(codec, rate=25)
                stream.width, stream.height = timeline[0][1].size
                stream.pix_fmt = "yuv420p"
                stream.thread_count = 2
                stream.options = options
                cursor = 0
                for tick in range(0, duration, 40):
                    while cursor + 1 < len(timeline) and timeline[cursor + 1][0] <= tick:
                        cursor += 1
                    for packet in stream.encode(av.VideoFrame.from_image(timeline[cursor][1])):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
        timeline[min(12, len(timeline) - 1)][1].save(
            directory / f"{name}.webp", quality=88, method=6,
        )
        print(f"Prepared site media: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="docs/demos/jev-agent-harness-traces.json")
    parser.add_argument("--terminal-out", "--sqlite-out", default="docs/demos/jev-decision-terminal-v2.gif")
    parser.add_argument("--frozen-lake-out", default="docs/demos/jev-decision-frozen-lake-v2.gif")
    parser.add_argument("--webshop-out", default="docs/demos/jev-decision-webshop-v2.gif")
    parser.add_argument("--overview-out", "--out", default="docs/demos/jev-agent-harness.gif")
    parser.add_argument("--site-media", action="store_true",
                        help="Also export MP4, WebM, and WebP for these four GIFs")
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    for key, output in (("webshop", args.webshop_out), ("frozen_lake", args.frozen_lake_out),
                        ("sqlite", args.terminal_out)):
        save(output, key, trace[key])
    clips = [args.webshop_out, args.frozen_lake_out, args.terminal_out]
    save_overview(args.overview_out, clips)
    if args.site_media:
        save_site_media([args.overview_out, *clips])


if __name__ == "__main__":
    main()
