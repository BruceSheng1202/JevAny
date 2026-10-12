# JevAny intro film

Source for the 90-second JevAny overview video: a 1920x1080 master and a
1080x1350 (4:5) feed version, both at 30 fps with English narration, an
original synthesized score and English captions. A moving wall of the
project's demo recordings opens the film under "Build your own decision
models". Chapters follow for checkpoints, data, training, evaluation and
deployment. The first cut (crosswalk opening) is in Git history at 817d2b5.

Every frame is a pure function of its frame number. `film.html` loads
`src/film.js`, which rebuilds the stage on `FILM.seek(frame)`. Headless
Chromium captures each frame and pipes it into ffmpeg. The motion rules follow
the Remotion agent skills (frame-driven animation only, no CSS transitions,
generous safe areas), but the film does not depend on the Remotion package.

| Path | Contents |
|---|---|
| `film.html`, `src/film.css`, `src/film.js` | Scenes, layout for both aspect ratios, motion |
| `src/data.js` | Facts, records and demo provenance shown on screen, generated from `docs/demos/`, `results/`, `recipes/` and `examples/` |
| `audio/narration.json` | Narration lines, start times, voice and caption overrides |
| `scripts/` | Data and clip extraction, TTS, score, mix, captions, render and finishing |

Fonts come from `site/assets/fonts` (DM Sans and IBM Plex Sans Condensed, OFL)
and the system URW Gothic used by the README title. Demo footage comes from
`site/assets/media` (case library) and `site/assets/media/docs` (playground
replays); on-screen labels state that these are archived replays or interface
recordings.

## Render

Requires Python 3.12, ffmpeg with libx264 and libass, and Chromium for
Playwright 1.55. Set `WORK` to a scratch directory outside the repository and
`FFMPEG` to the ffmpeg binary.

```bash
python3.12 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
"$WORK/venv/bin/pip" install -r promo/intro-film/requirements.txt
"$WORK/venv/bin/python" -m playwright install chromium
PY="$WORK/venv/bin/python"; F=promo/intro-film/scripts

$PY $F/extract_data.py                                   # refresh src/data.js from repository evidence
$PY $F/extract_clips.py "$FFMPEG" "$WORK/clips"         # cropped JPEG frames from site/assets/media
HF_HOME="$WORK/hf" $PY $F/tts.py "$WORK/vo"            # Kokoro-82M narration + timing.json
$PY $F/score.py "$WORK/music.wav"
$PY $F/mix.py "$WORK/vo" "$WORK/music.wav" "$WORK/mix.wav"
$PY $F/finish.py audio "$WORK/mix.wav" "$WORK/audio.m4a" --ffmpeg "$FFMPEG"
$PY $F/captions.py "$WORK/vo/timing.json" "$WORK/captions"
$PY $F/prepare_fonts.py "$WORK/fonts"

for L in landscape feed; do
  $PY $F/render.py video --layout $L --clips "$WORK/clips" --out "$WORK/video-$L.mp4" --ffmpeg "$FFMPEG" --workers 6
  $PY $F/finish.py mux "$WORK/video-$L.mp4" "$WORK/audio.m4a" "$WORK/jevany-intro-$L.mp4" --ffmpeg "$FFMPEG"
  $PY $F/finish.py burn "$WORK/jevany-intro-$L.mp4" "$WORK/captions/captions-$L.ass" "$WORK/fonts" "$WORK/jevany-intro-$L-captioned.mp4" --ffmpeg "$FFMPEG"
done
```

Preview single frames with
`render.py stills --layout feed --clips "$WORK/clips" --out "$WORK/stills" --times 4.5 67`.
The commands prepare the caption font and produce both clean masters and
versions with captions burned in.

Playwright's bundled Node needs glibc 2.28 or newer. On older hosts, point
`playwright/driver/node` in the virtual environment at a compatible Node
binary; Chromium itself runs unchanged.

## Editing

Scene timing lives in `SCENES` at the end of `src/film.js`. Narration start
times live in `audio/narration.json`. After changing either, regenerate the
narration, mix and captions, and keep cue times aligned with the scenes. On-screen
numbers must come from `src/data.js` or carry an "illustrative" label.
