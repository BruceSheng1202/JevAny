"""Render the film with headless Chromium and stream frames into ffmpeg.

Stills:  python scripts/render.py stills --layout landscape --clips DIR --out DIR --times 0 4.5 9
Video:   python scripts/render.py video  --layout landscape --clips DIR --out FILE.mp4 --ffmpeg BIN
         [--workers 4] [--start 0] [--end 2700]

Video mode splits the frame range across workers. Each worker pipes PNG
screenshots into its own x264 encoder; segments share encoder settings and
are joined with the concat demuxer without re-encoding. Audio is muxed
separately by scripts/finish.py.
"""

import argparse
import asyncio
import subprocess
import time
from pathlib import Path

from playwright.async_api import async_playwright

FILM = Path(__file__).resolve().parents[1]
SIZES = {"landscape": (1920, 1080), "feed": (1080, 1350)}
X264 = ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2",
        "-x264-params", "keyint=60:min-keyint=30:scenecut=0", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


async def open_page(pw, layout: str, clips: Path):
    width, height = SIZES[layout]
    browser = await pw.chromium.launch(args=["--force-color-profile=srgb", "--disable-lcd-text", "--font-render-hinting=none", "--allow-file-access-from-files"])
    page = await browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    await page.goto(f"{(FILM / 'film.html').as_uri()}?layout={layout}&clips={clips.resolve().as_uri()}")
    await page.evaluate("FILM.ready")
    return browser, page, errors


async def shoot(page, frame: int) -> bytes:
    await page.evaluate(f"FILM.seek({frame})")
    return await page.screenshot(type="png", animations="disabled", caret="hide")


async def stills(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser, page, errors = await open_page(pw, args.layout, Path(args.clips))
        for t in args.times:
            frame = min(round(float(t) * 30), 2699)
            (out / f"{args.layout}-{frame:04d}.png").write_bytes(await shoot(page, frame))
        await browser.close()
    if errors:
        raise SystemExit("page errors: " + "; ".join(errors[:5]))


async def worker(pw, args, index: int, start: int, end: int, segment: Path) -> None:
    browser, page, errors = await open_page(pw, args.layout, Path(args.clips))
    enc = subprocess.Popen([args.ffmpeg, "-v", "error", "-y", "-f", "image2pipe", "-framerate", "30", "-c:v", "png", "-i", "-", *X264, "-r", "30", str(segment)],
                           stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for frame in range(start, end):
            enc.stdin.write(await shoot(page, frame))
            if (frame - start) % 150 == 0:
                print(f"[w{index}] frame {frame} ({frame - start}/{end - start}) {time.time() - t0:.0f}s", flush=True)
    except BaseException:
        # Stop this worker's encoder and browser before reporting the failure.
        enc.kill()
        enc.wait()
        await browser.close()
        raise
    enc.stdin.close()
    await browser.close()
    if enc.wait() != 0:
        raise SystemExit(f"encoder for segment {index} failed")
    if errors:
        raise SystemExit(f"page errors in segment {index}: " + "; ".join(errors[:5]))


async def video(args) -> None:
    out = Path(args.out)
    seg_dir = out.with_suffix(".segments")
    seg_dir.mkdir(parents=True, exist_ok=True)
    total = args.end - args.start
    bounds = [args.start + round(total * i / args.workers) for i in range(args.workers + 1)]
    segments = [seg_dir / f"seg{i:02d}.mp4" for i in range(args.workers)]
    async with async_playwright() as pw:
        # A TaskGroup cancels sibling workers as soon as one fails.
        async with asyncio.TaskGroup() as group:
            for i in range(args.workers):
                group.create_task(worker(pw, args, i, bounds[i], bounds[i + 1], segments[i]))
    listing = seg_dir / "concat.txt"
    listing.write_text("".join(f"file '{s.resolve()}'\n" for s in segments))
    subprocess.run([args.ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)], check=True)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stills", "video"])
    ap.add_argument("--layout", choices=list(SIZES), default="landscape")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", nargs="*", default=[])
    ap.add_argument("--ffmpeg")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=2700)
    args = ap.parse_args()
    asyncio.run(stills(args) if args.mode == "stills" else video(args))


if __name__ == "__main__":
    main()
