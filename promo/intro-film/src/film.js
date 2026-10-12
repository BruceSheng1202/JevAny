/* JevAny overview film.
 * Every visual is a pure function of the frame number: FILM.seek(frame)
 * rebuilds the stage, so frames render identically in any order and any
 * worker. Layout "landscape" is 1920x1080; layout "feed" is 1080x1350.
 */
(() => {
  const FPS = 30;
  const FRAMES = 2700;
  const q = new URLSearchParams(location.search);
  const FEED = q.get("layout") === "feed";
  const W = FEED ? 1080 : 1920;
  const H = FEED ? 1350 : 1080;
  const CLIPS = q.get("clips") || "clips";
  const D = window.FILM_DATA;
  // Frame counts from scripts/extract_clips.py (clips.json).
  const CLIP_FRAMES = {"peg_insertion": 48, "block_stacking": 48, "drawer_opening": 48, "conveyor_sorting": 96, "assembly": 192, "control_panel": 192, "obstacle_push": 96, "frontend": 64, "research": 96, "sql": 32, "ci": 48, "inbox": 128, "calendar": 64, "spreadsheet": 64, "drone": 128, "lab": 128, "driving": 128, "warehouse_rover": 192, "pg-arm": 73, "pg-doom": 42, "pg-crafter": 37, "arm-hd": 73};
  const MID = "&middot;";
  const ARROW = "&rarr;";

  const stage = document.getElementById("stage");
  stage.style.width = `${W}px`;
  stage.style.height = `${H}px`;

  // ---------- timing helpers ----------
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, p) => a + (b - a) * p;
  function bezier(x1, y1, x2, y2) {
    const f = (a, b, t) => 3 * a * t * (1 - t) * (1 - t) + 3 * b * t * t * (1 - t) + t * t * t;
    return (x) => {
      if (x <= 0) return 0;
      if (x >= 1) return 1;
      let lo = 0, hi = 1, t = x;
      for (let i = 0; i < 24; i++) {
        t = (lo + hi) / 2;
        if (f(x1, x2, t) < x) lo = t; else hi = t;
      }
      return f(y1, y2, t);
    };
  }
  const OUT = bezier(0.16, 1, 0.3, 1);
  const INOUT = bezier(0.65, 0, 0.35, 1);
  const BACK = (p) => 1 + 2.2 * Math.pow(p - 1, 3) + 1.2 * Math.pow(p - 1, 2);
  const k = (t, a, b, e = OUT) => e(clamp((t - a) / (b - a)));
  // Reveal: fade, rise and unblur from time a over d seconds.
  const rv = (t, a, d = 0.7, dy = 28, blur = 10) => {
    const p = k(t, a, a + d);
    return `opacity:${p.toFixed(4)};transform:translateY(${((1 - p) * dy).toFixed(2)}px);filter:blur(${((1 - p) * blur).toFixed(2)}px)`;
  };
  // Reveal then exit at time b.
  const rvx = (t, a, b, d = 0.7, dy = 28) => {
    const p = k(t, a, a + d) * (1 - k(t, b, b + 0.45, INOUT));
    const lift = (1 - k(t, a, a + d)) * dy - k(t, b, b + 0.45, INOUT) * 18;
    return `opacity:${p.toFixed(4)};transform:translateY(${lift.toFixed(2)}px);filter:blur(${((1 - p) * 10).toFixed(2)}px)`;
  };
  const pct = (p) => `${(Math.round(p * 1000) / 10).toFixed(1)}%`;
  const fmt = (n) => n.toLocaleString("en-US");
  const px = (n) => `${Math.round(n * 100) / 100}px`;
  const box = (b) => `left:${px(b.x)};top:${px(b.y)};width:${px(b.w)};height:${px(b.h)}`;

  function rng(seed) {
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let r = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- shared pieces ----------
  const GRAIN = [0, 1, 2, 3].map((i) => {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d");
    const img = g.createImageData(256, 256);
    const r = rng(101 + i);
    for (let j = 0; j < img.data.length; j += 4) {
      const v = Math.floor(r() * 255);
      img.data[j] = img.data[j + 1] = img.data[j + 2] = v;
      img.data[j + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return c.toDataURL("image/png");
  });

  function background(t, gridAlpha) {
    const off = (t * 22) % 90;
    const breathe = 0.85 + 0.15 * Math.sin(t * 0.6);
    return `
      <div class="layer" style="background:
        radial-gradient(${W * 0.75}px ${H * 0.6}px at 50% 112%, rgba(45,212,191,${(0.13 * breathe).toFixed(3)}), transparent 70%),
        radial-gradient(${W * 0.55}px ${H * 0.5}px at 8% -12%, rgba(139,92,246,.08), transparent 70%),
        radial-gradient(${W * 0.5}px ${H * 0.45}px at 100% 0%, rgba(168,240,200,.05), transparent 70%)"></div>
      <div class="abs" style="left:-60%;width:220%;top:${px(H * 0.52)};height:${px(H * 0.95)};
        transform:perspective(760px) rotateX(72deg);transform-origin:50% 0;
        background-image:linear-gradient(rgba(168,240,200,.11) 1px, transparent 1px),
          linear-gradient(90deg, rgba(168,240,200,.11) 1px, transparent 1px);
        background-size:90px 90px;background-position:0 ${px(off)};
        -webkit-mask-image:linear-gradient(to bottom, transparent 0%, #000 25%, #000 55%, transparent 100%);
        opacity:${gridAlpha.toFixed(3)}"></div>`;
  }

  function grain(frame) {
    return `<div class="layer" style="background-image:url(${GRAIN[frame % 4]});background-size:256px 256px;
      mix-blend-mode:overlay;opacity:.07;pointer-events:none"></div>
      <div class="layer" style="background:radial-gradient(ellipse at 50% 45%, transparent 55%, rgba(0,0,0,.55) 100%)"></div>`;
  }

  function logo(size, t0, t, glow = 1) {
    const tile = k(t, t0, t0 + 0.5);
    const a = k(t, t0 + 0.15, t0 + 0.85, INOUT);
    const b = k(t, t0 + 0.35, t0 + 1.15, INOUT);
    const dots = (d) => BACK(k(t, t0 + d, t0 + d + 0.45, (x) => x));
    return `<div style="position:relative;width:${size}px;height:${size}px;flex:none">
      <div class="abs" style="inset:-60%;background:radial-gradient(circle, rgba(45,212,191,${(0.28 * tile * glow).toFixed(3)}), transparent 62%)"></div>
      <svg viewBox="0 0 512 512" width="${size}" height="${size}" style="position:relative;overflow:visible">
        <rect width="512" height="512" rx="96" fill="#101629" stroke="rgba(168,240,200,.28)" stroke-width="3" opacity="${tile.toFixed(3)}"/>
        <path d="M144 128 256 240V128" fill="none" stroke="#8b5cf6" stroke-width="28" stroke-linecap="round" stroke-linejoin="round"
          pathLength="1" stroke-dasharray="1" stroke-dashoffset="${(1 - a).toFixed(4)}"/>
        <circle cx="144" cy="128" r="${(28 * dots(0.2)).toFixed(2)}" fill="#8b5cf6"/>
        <circle cx="256" cy="128" r="${(28 * dots(0.3)).toFixed(2)}" fill="#8b5cf6"/>
        <path d="M128 320a64 64 0 0 0 128 0v-80l112-112" fill="none" stroke="#2dd4bf" stroke-width="44" stroke-linecap="round" stroke-linejoin="round"
          pathLength="1" stroke-dasharray="1" stroke-dashoffset="${(1 - b).toFixed(4)}"/>
        <circle cx="368" cy="128" r="${(36 * dots(0.9)).toFixed(2)}" fill="#2dd4bf"/>
      </svg></div>`;
  }

  const wordmark = (size) =>
    `<div style="font-family:'URW Gothic',sans-serif;font-weight:600;font-size:${size}px;letter-spacing:-.01em;line-height:1;white-space:nowrap">` +
    `<span style="color:#8b5cf6">Jev</span><span style="color:#2dd4bf">Any</span></div>`;

  function kicker(t, a, text, x, y, b = 1e9) {
    return `<div class="abs kicker" style="left:${px(x)};top:${px(y)};font-size:${FEED ? 24 : 26}px;${rvx(t, a, b, 0.6, 14)}">${text}</div>`;
  }

  function headline(t, a, html, x, y, size, width, b = 1e9) {
    return `<div class="abs head" style="left:${px(x)};top:${px(y)};width:${px(width)};font-size:${size}px;${rvx(t, a, b)}">${html}</div>`;
  }

  function foot(t, a, text, x, y, w) {
    return `<div class="abs foot" style="left:${px(x)};top:${px(y)};width:${px(w)};${rv(t, a, 0.8, 8, 4)}">${text}</div>`;
  }

  // A light that sweeps across the frame at chapter breaks.
  function sweep(t, t0) {
    const p = clamp((t - t0) / 0.7);
    if (p <= 0 || p >= 1) return "";
    const x = lerp(-0.2, 1.2, INOUT(p)) * W;
    const a = Math.sin(Math.PI * p);
    return `<div class="abs" style="left:${px(x - 360)};top:0;width:720px;height:${H}px;
      background:linear-gradient(90deg, transparent, rgba(168,240,200,${(0.10 * a).toFixed(3)}), transparent);mix-blend-mode:screen"></div>
      <div class="abs" style="left:${px(x - 1)};top:0;width:2px;height:${H}px;background:rgba(168,240,200,${(0.55 * a).toFixed(3)});
      box-shadow:0 0 24px rgba(168,240,200,${(0.7 * a).toFixed(3)})"></div>`;
  }

  // Clip canvases are filled after the DOM update; see drawClips().
  function clip(name, seconds, b, extra = "", loop = false, zoom = 1) {
    return `<div class="abs" style="${box(b)};overflow:hidden;border-radius:${FEED ? 18 : 20}px;${extra}">
      <canvas data-clip="${name}" data-t="${seconds.toFixed(4)}" data-loop="${loop ? 1 : 0}" width="${Math.round(b.w)}" height="${Math.round(b.h)}"
        style="width:100%;height:100%;transform:scale(${zoom.toFixed(4)})"></canvas></div>`;
  }

  // ---------- demo wall ----------
  // A tilted plane of scrolling columns, one tile per archived replay or
  // interface recording. `level` sets how strongly the footage shows.
  const WALL = D.wall;
  const WALL_GEO = FEED
    ? { cols: 5, tw: 300, th: 168, gap: 16, pw: 1700, ph: 1900 }
    : { cols: 7, tw: 340, th: 190, gap: 18, pw: 2560, ph: 1500 };
  function wall(t, level, zoom) {
    const g = WALL_GEO;
    const step = g.th + g.gap;
    const rows = Math.ceil(g.ph / step) + 2;
    const x0 = (g.pw - (g.cols * (g.tw + g.gap) - g.gap)) / 2;
    const mid = Math.floor(g.cols / 2);
    // The middle column holds the first wall clip at the plane centre at t = 0.
    const k0 = Math.floor(rows / 2);
    let tiles = "";
    for (let c = 0; c < g.cols; c++) {
      const speed = (c % 2 ? -1 : 1) * (26 + 7 * ((c * 3) % 4));
      const shift = c === mid ? 0 : (c * 97) % step;
      for (let r = -1; r < rows; r++) {
        const raw = r * step + shift + speed * t;
        const y = ((raw % (rows * step)) + rows * step) % (rows * step) - step + (g.ph - rows * step) / 2 + step / 2;
        const slot = r - k0;
        const idx = (((c - mid) * 5 + slot) % WALL.length + WALL.length) % WALL.length;
        const tile = WALL[idx];
        const label = level > 0.35 ? `<div class="abs" style="left:10px;bottom:8px;font-size:${FEED ? 15 : 15}px;font-weight:600;color:rgba(241,243,237,.82);
          text-shadow:0 1px 6px rgba(0,0,0,.9)">${tile.title}</div>` : "";
        tiles += `<div class="abs" style="left:${px(x0 + c * (g.tw + g.gap))};top:${px(y - g.th / 2)};width:${g.tw}px;height:${g.th}px;
          border-radius:12px;overflow:hidden;box-shadow:0 18px 40px rgba(0,0,0,.5);outline:1px solid rgba(210,239,229,.10)">
          <canvas data-clip="${tile.clip}" data-t="${(t * 0.9 + idx * 0.37).toFixed(4)}" data-loop="1" width="${g.tw}" height="${g.th}" style="width:100%;height:100%"></canvas>${label}</div>`;
      }
    }
    return `<div class="layer" style="perspective:2200px;overflow:hidden">
      <div class="abs" style="left:${px((W - g.pw) / 2)};top:${px((H - g.ph) / 2)};width:${g.pw}px;height:${g.ph}px;
        transform:rotateX(${FEED ? 14 : 18}deg) rotateZ(-7deg) scale(${zoom.toFixed(4)});transform-origin:50% 50%;
        opacity:${(0.18 + 0.82 * level).toFixed(3)};filter:brightness(${(0.42 + 0.5 * level).toFixed(3)}) saturate(${(0.55 + 0.4 * level).toFixed(3)})">${tiles}</div></div>
      <div class="layer" style="background:radial-gradient(ellipse ${FEED ? "85% 55%" : "62% 58%"} at 50% 48%, rgba(6,10,12,${(0.55 + 0.3 * (1 - level)).toFixed(3)}), rgba(6,10,12,${(0.25 + 0.55 * (1 - level)).toFixed(3)}))"></div>`;
  }

  // Wall strength over the film: bright under the hook and close, dim behind chapters.
  function wallLevel(t) {
    if (t < 9.6) return 1;
    if (t < 17) return lerp(1, 0.42, k(t, 9.6, 10.6, INOUT));
    if (t < 80.2) return lerp(0.42, 0.14, k(t, 16.6, 17.6, INOUT));
    return lerp(0.14, 0.75, k(t, 80.2, 81.6, INOUT));
  }

  // ---------- shared chapter pieces ----------
  const STATIONS = ["Checkpoints", "Data", "Train", "Evaluate", "Deploy"];
  const CHAPTER_TIMES = [17, 27, 36, 49, 63, 75.3];
  function stationRail(t) {
    if (t < 17.1 || t > 75.6) return "";
    const cur = CHAPTER_TIMES.findIndex((c, i) => t >= c && t < CHAPTER_TIMES[i + 1]);
    const fade = k(t, 17.1, 17.7) * (1 - k(t, 75.0, 75.6));
    const items = STATIONS.map((s, i) => {
      const on = i === cur, done = i < cur;
      return `<span style="display:inline-flex;align-items:center;gap:10px;color:${on ? "var(--mint)" : done ? "rgba(168,240,200,.55)" : "rgba(169,180,182,.5)"}">
        <span class="mono" style="font-size:${FEED ? 17 : 19}px">0${i + 1}</span>${s}</span>`;
    }).join(`<span style="color:rgba(169,180,182,.35)">&mdash;</span>`);
    return `<div class="abs" style="left:${FEED ? 60 : 80}px;top:${FEED ? 76 : 62}px;display:flex;gap:${FEED ? 12 : 18}px;align-items:center;
      font-size:${FEED ? 19 : 22}px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;opacity:${fade.toFixed(3)}">${items}</div>`;
  }

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // Terminal line typed over [a, a + d]; shows a prompt and a caret while typing.
  function typed(t, a, d, text) {
    const n = Math.round(text.length * clamp((t - a) / d));
    const caret = t >= a && t < a + d + 0.4 ? `<span style="background:var(--mint);opacity:.85">&nbsp;</span>` : "";
    return t < a ? "" : `<span class="mint">$</span> ${esc(text.slice(0, n))}${caret}`;
  }
  function terminal(b, lines, t, a, size) {
    const body = lines.map((l) => `<div style="white-space:pre;min-height:${size * 1.5}px">${typed(t, l[0], l[1], l[2])}</div>`).join("");
    return `<div class="abs card mono" style="${box(b)};padding:${FEED ? "20px 26px" : "26px 34px"};font-size:${size}px;line-height:1.5;
      background:rgba(5,9,11,.9);${rv(t, a, 0.6, 20, 6)}">
      <div style="display:flex;gap:9px;margin-bottom:${FEED ? 10 : 14}px">${["#f87171", "#fbbf24", "#34d399"].map((c) => `<span class="dot" style="width:12px;height:12px;background:${c};opacity:.55"></span>`).join("")}</div>${body}</div>`;
  }

  function nodeCard(b, t, a, title, sub, accent = "var(--mint)", lit = 0, extra = "") {
    return `<div class="abs card" style="${box(b)};padding:${FEED ? "20px 26px" : "26px 30px"};border-color:color-mix(in srgb, ${accent} ${Math.round(22 + 50 * lit)}%, transparent);
      box-shadow:0 30px 80px rgba(0,0,0,.45), 0 0 ${(50 * lit).toFixed(1)}px color-mix(in srgb, ${accent} 40%, transparent);${rv(t, a, 0.7, 24, 8)}">
      <div style="font-size:${FEED ? 34 : 38}px;font-weight:600;line-height:1.1">${title}</div>
      ${sub ? `<div style="margin-top:10px;font-size:${FEED ? 24 : 27}px;color:var(--muted);line-height:1.3">${sub}</div>` : ""}${extra}</div>`;
  }

  // Animated connector between two points with a travelling pulse.
  function link(x1, y1, x2, y2, t, a, color = "#a8f0c8", curve = 0) {
    const p = k(t, a, a + 0.5, INOUT);
    if (p <= 0) return "";
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - curve;
    const d = `M${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
    let s = `<path d="${d}" fill="none" stroke="${color}" stroke-width="3" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${(1 - p).toFixed(4)}" opacity=".75"/>`;
    if (p >= 1) {
      const c = ((t - a - 0.5) * 0.8) % 1;
      const pt = pointOn(d, c);
      s += `<circle cx="${pt.x.toFixed(1)}" cy="${pt.y.toFixed(1)}" r="7" fill="${color}" style="filter:drop-shadow(0 0 10px ${color})"/>`;
    }
    return s;
  }
  const svgLayer = (inner) => `<svg class="abs" width="${W}" height="${H}" style="left:0;top:0;overflow:visible">${inner}</svg>`;

  // Point at fraction p along an SVG path, measured with a hidden element.
  const probe = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  probe.setAttribute("style", "position:absolute;width:0;height:0;visibility:hidden");
  const probePath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  probe.appendChild(probePath);
  document.body.appendChild(probe);
  function pointOn(d, p) {
    probePath.setAttribute("d", d);
    return probePath.getPointAtLength(probePath.getTotalLength() * clamp(p));
  }

  const HX = FEED ? 60 : 80;
  const HY = FEED ? 120 : 108;
  const HS = FEED ? 72 : 84;
  const HW = FEED ? 960 : 1760;

  // ---------- 0-10 s: hook ----------
  function hook(t) {
    let html = `<div class="abs" style="left:${HX}px;top:${FEED ? 70 : 56}px;display:flex;align-items:center;gap:18px;${rv(t, -0.2, 0.6, 10, 4)}">
      ${logo(FEED ? 60 : 56, -0.6, t, 0.6)}${wordmark(FEED ? 50 : 46)}</div>`;
    const y1 = FEED ? 380 : 300;
    const size = FEED ? 124 : 150;
    html += `<div class="abs head" style="left:0;width:${W}px;top:${y1}px;text-align:center;font-size:${size}px;text-shadow:0 8px 40px rgba(0,0,0,.6)">
      <div style="${rv(t, 0.3, 0.8, 40, 14)}">Build your own</div>
      <div class="mint" style="${rv(t, 0.75, 0.8, 40, 14)};text-shadow:0 0 50px rgba(168,240,200,.35)">decision models.</div></div>`;
    html += `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 690 : 650}px;text-align:center;font-size:${FEED ? 40 : 46}px;font-weight:500;color:var(--ink);
      line-height:1.3;${rv(t, 3.35, 0.7, 16, 6)}">Open checkpoints, and the infrastructure for</div>`;
    const chips = [["Research", 6.2], ["Training", 6.85], ["Deployment", 7.45]].map(([c, at]) => {
      const lit = k(t, at, at + 0.3);
      return `<span class="pill" style="padding:${FEED ? "14px 30px" : "16px 36px"};font-size:${FEED ? 38 : 44}px;font-weight:600;
        color:color-mix(in srgb, var(--mint) ${Math.round(lit * 100)}%, var(--muted));border-color:rgba(168,240,200,${(0.15 + 0.6 * lit).toFixed(3)});
        background:rgba(168,240,200,${(0.03 + 0.07 * lit).toFixed(3)});box-shadow:0 0 ${(36 * lit).toFixed(1)}px rgba(168,240,200,${(0.3 * lit).toFixed(3)});
        ${rv(t, 3.9 + (at - 6.2) * 0.25, 0.6, 16, 6)}">${c}</span>`;
    }).join("");
    html += `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 820 : 750}px;display:flex;justify-content:center;gap:${FEED ? 16 : 24}px">${chips}</div>`;
    html += foot(t, 1.2, "Background: archived replays and original interface recordings from the JevAny case library and playground.",
      HX, FEED ? 1120 : 880, HW);
    return html;
  }

  // ---------- 10-17 s: one framework ----------
  function framework(t) {
    let html = kicker(t, 10.1, "One open repository", HX, FEED ? 80 : 66);
    html += headline(t, 10.3, `From checkpoint <span class="mint">to deployment.</span>`, HX, HY, HS, HW);
    const subs = ["5 released", "your JSONL", "SFT recipes", "held-out data", "HTTP and Python"];
    const times = [10.9, 11.3, 11.7, 12.1, 12.5];
    let svg = "";
    STATIONS.forEach((s, i) => {
      const lit = Math.max(0, 1 - Math.abs(t - times[i] - 0.3) / 0.6);
      const [cx, cy] = FEED ? [270, 330 + i * 150] : [240 + i * 360, 470];
      if (i > 0) {
        const [px0, py0] = FEED ? [270, 330 + (i - 1) * 150] : [240 + (i - 1) * 360, 470];
        svg += link(FEED ? px0 : px0 + 30, FEED ? py0 + 30 : py0, FEED ? cx : cx - 30, FEED ? cy - 30 : cy, t, times[i] - 0.3);
      }
      html += `<div class="abs" style="left:${cx - 26}px;top:${cy - 26}px;width:52px;height:52px;border-radius:50%;border:2px solid var(--mint);
        background:radial-gradient(circle, rgba(168,240,200,${(0.25 + 0.5 * lit).toFixed(3)}), rgba(6,10,12,.9) 70%);
        box-shadow:0 0 ${(20 + 40 * lit).toFixed(1)}px rgba(168,240,200,${(0.35 + 0.4 * lit).toFixed(3)});${rv(t, times[i], 0.5, 12, 4)}"></div>`;
      html += FEED
        ? `<div class="abs" style="left:330px;top:${cy - 36}px;${rv(t, times[i], 0.5, 12, 4)}"><div style="font-size:40px;font-weight:600">${s}</div>
            <div style="font-size:26px;color:var(--muted);margin-top:2px">${subs[i]}</div></div>`
        : `<div class="abs" style="left:${cx - 170}px;width:340px;top:${cy + 50}px;text-align:center;${rv(t, times[i], 0.5, 12, 4)}">
            <div style="font-size:40px;font-weight:600">${s}</div><div style="font-size:27px;color:var(--muted);margin-top:6px">${subs[i]}</div></div>`;
    });
    // Research loop: return path from Evaluate back to Data, drawn apart from the forward rail.
    const rp = k(t, 13.0, 13.8, INOUT);
    const loopPath = FEED ? `M 244 772 C 150 735, 150 525, 244 488` : `M ${240 + 3 * 360} 430 C ${240 + 3 * 360} 300, ${240 + 360} 300, ${240 + 360} 430`;
    if (rp > 0) {
      svg += `<path d="${loopPath}" fill="none" stroke="#a78bfa" stroke-width="3" stroke-dasharray="10 9" opacity="${(0.85 * rp).toFixed(3)}"
        marker-end="url(#rl)"/>`;
    }
    svg = `<defs><marker id="rl" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="#a78bfa"/></marker></defs>` + svg;
    html += svgLayer(svg);
    html += `<div class="abs" style="${FEED ? "left:-12px;top:612px;width:220px;text-align:center;rotate:-90deg" : `left:${240 + 360 + 40}px;width:${2 * 360 - 80}px;top:296px;text-align:center`};font-size:${FEED ? 28 : 30}px;
      color:var(--violet);font-weight:600;${rv(t, 13.4, 0.5, 8, 3)}">research loop</div>`;
    const chips = [`${D.supported.count} supported open models`, `${D.supported.families.length} model families`, "Apache-2.0"].map((c, i) =>
      `<span class="pill" style="padding:12px 26px;font-size:${FEED ? 28 : 32}px;color:var(--ink);${rv(t, 12.4 + 0.25 * i, 0.5, 12, 4)}">${c}</span>`).join("");
    html += `<div class="abs" style="left:${HX}px;top:${FEED ? 1060 : 720}px;width:${HW}px;display:flex;flex-wrap:wrap;gap:16px;justify-content:${FEED ? "flex-start" : "center"}">${chips}</div>`;
    return html;
  }

  // ---------- 17-27 s: checkpoints ----------
  function checkpoints(t) {
    let html = headline(t, 17.2, `Start from an <span class="mint">open checkpoint.</span>`, HX, HY, HS, HW);
    D.checkpoints.forEach((c, i) => {
      const a = 17.8 + 0.18 * i;
      const b = FEED ? { x: 60, y: 260 + i * 104, w: 960, h: 90 } : { x: 80 + i * 360, y: 280, w: 330, h: 230 };
      const readout = c.readout === "pointer" ? "Pointer" : "Direct-token";
      const [fam, ...rest] = c.name.split(" ");
      html += `<div class="abs card" style="${box(b)};padding:${FEED ? "0 30px" : "26px 28px"};${FEED ? "display:flex;align-items:center;justify-content:space-between" : ""};${rv(t, a, 0.6, 24, 6)}">
        <div><div class="mono" style="font-size:${FEED ? 18 : 19}px;color:var(--subtle)">SimpleJev/JevAny-</div>
        <div style="font-size:${FEED ? 34 : 36}px;font-weight:600;margin-top:4px">${fam}</div>${rest.length && !FEED ? `<div style="font-size:28px;color:var(--ink)">${rest.join(" ")}</div>` : ""}</div>
        <span class="pill" style="${FEED ? "" : "position:absolute;left:28px;bottom:24px;"}padding:8px 18px;font-size:${FEED ? 22 : 23}px;color:${c.readout === "pointer" ? "var(--mint)" : "var(--violet)"}">${readout}</span></div>`;
    });
    const tb = FEED ? { x: 60, y: 810, w: 960, h: 250 } : { x: 80, y: 570, w: 1760, h: 230 };
    html += terminal(tb, [
      [21.0, 1.6, "jevany serve --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA"],
      [23.0, 1.3, "jevany train --config recipes/finetune.toml"],
    ], t, 20.6, FEED ? 23 : 30);
    html += foot(t, 19.0, `LoRA adapters on open base models, published on Hugging Face. Base-model licences apply.`, HX, FEED ? 1080 : 836, HW);
    return html;
  }

  // ---------- 27-36 s: data ----------
  function jsonHtml(obj, indent = 0, labelGlow = 0) {
    const pad = (n) => "&nbsp;".repeat(n * 2);
    if (typeof obj === "string") return `<span style="color:#e6e9e3">"${esc(obj)}"</span>`;
    if (typeof obj === "boolean") return `<span class="violet">${obj}</span>`;
    const keys = Object.keys(obj);
    return "{<br>" + keys.map((key, i) => {
      const v = obj[key];
      const isLabel = key === "label";
      const keyHtml = `<span style="color:${isLabel ? "var(--mint)" : "#8fb8c4"};${isLabel ? `text-shadow:0 0 ${(18 * labelGlow).toFixed(1)}px rgba(168,240,200,.9)` : ""}">"${key}"</span>`;
      const val = v === null ? "{...}" : jsonHtml(v, indent + 1, labelGlow);
      return `${pad(indent + 1)}${keyHtml}: ${isLabel ? `<span style="color:var(--mint);font-weight:700">${typeof v === "string" ? `"${v}"` : v}</span>` : val}${i < keys.length - 1 ? "," : ""}`;
    }).join("<br>") + `<br>${pad(indent)}}`;
  }
  function data(t) {
    let html = headline(t, 27.2, `Bring your <span class="mint">own decisions.</span>`, HX, HY, HS, HW);
    const rec = JSON.parse(D.record);
    // Long policy text and criteria descriptions are elided on screen.
    const shown = { state: rec.state, questions: {} };
    for (const [qk, qv] of Object.entries(rec.questions)) {
      shown.questions[qk] = { type: qv.type, criteria: qv.criteria ? null : undefined, label: qv.label };
      if (!qv.criteria) delete shown.questions[qk].criteria;
    }
    const b = FEED ? { x: 60, y: 260, w: 960, h: 560 } : { x: 80, y: 270, w: 1080, h: 560 };
    const reveal = k(t, 27.7, 30.2, (x) => x);
    const glow = Math.max(0, 1 - Math.abs(t - 31.0) / 0.6);
    html += `<div class="abs card mono" style="${box(b)};padding:${FEED ? "24px 28px" : "30px 36px"};font-size:${FEED ? 23 : 26}px;line-height:1.55;
      background:rgba(5,9,11,.9);overflow:hidden;${rv(t, 27.5, 0.6, 20, 6)}">
      <div style="color:var(--subtle);font-size:${FEED ? 18 : 20}px;margin-bottom:12px">examples/train.jsonl ${MID} line 1</div>
      <div style="-webkit-mask-image:linear-gradient(to bottom, #000 ${(reveal * 100).toFixed(1)}%, transparent ${(reveal * 100 + 6).toFixed(1)}%);white-space:normal;word-break:break-word">${jsonHtml(shown, 0, glow)}</div></div>`;
    const col = FEED ? { x: 60, y: 850 } : { x: 1220, y: 300 };
    const group = (y, a, title, items, color) => `<div class="abs" style="left:${col.x}px;top:${y}px;${rv(t, a, 0.6, 16, 6)}">
      <div class="kicker" style="font-size:${FEED ? 20 : 22}px;color:var(--subtle)">${title}</div>
      <div style="display:flex;gap:12px;margin-top:14px;flex-wrap:wrap">${items.map((x) => `<span class="pill mono" style="padding:10px 22px;font-size:${FEED ? 26 : 30}px;color:${color}">${x}</span>`).join("")}</div></div>`;
    html += group(col.y, 31.0, "Questions with labels", ["choice", "noul", "score"], "var(--mint)");
    html += group(col.y + (FEED ? 120 : 170), 32.6, "State", ["text", "image", "video"], "var(--violet)");
    if (!FEED) html += group(col.y + 340, 34.2, "Check before training", ["jevany data validate"], "var(--ink)");
    html += foot(t, 28.5, "Real starter record; criteria descriptions elided. Image and video state need a multimodal backbone.", HX, FEED ? 1110 : 860, HW);
    return html;
  }

  // ---------- 36-49 s: train ----------
  function train(t) {
    let html = headline(t, 36.2, `Train <span class="mint">your own checkpoint.</span>`, HX, HY, HS, HW);
    const flow = FEED
      ? [{ x: 60, y: 260, w: 300, h: 150 }, { x: 390, y: 260, w: 300, h: 150 }, { x: 720, y: 260, w: 300, h: 150 }]
      : [{ x: 80, y: 260, w: 520, h: 160 }, { x: 700, y: 260, w: 520, h: 160 }, { x: 1320, y: 260, w: 520, h: 160 }];
    const items = [["recipes/sft.toml", "base model, data, output", 36.6], ["jevany train", "SFT with your labels", 37.4], ["runs/my-jev", "your checkpoint", 38.2]];
    let svg = "";
    items.forEach((it, i) => {
      const lit = i === 2 ? Math.max(0, 1 - Math.abs(t - 38.8) / 0.8) : 0;
      html += nodeCard(flow[i], t, it[2], `<span class="mono" style="font-size:${FEED ? 24 : 34}px">${it[0]}</span>`, FEED ? "" : it[1], "var(--mint)", lit);
      if (i) svg += link(flow[i - 1].x + flow[i - 1].w + 6, flow[i].y + flow[i].h / 2, flow[i].x - 6, flow[i].y + flow[i].h / 2, t, it[2] - 0.2);
    });
    html += svgLayer(svg);
    // One GPU or many: the same recipe through the standard launcher.
    const g = FEED ? { x: 60, y: 450, w: 960, h: 270 } : { x: 80, y: 470, w: 860, h: 340 };
    const many = k(t, 41.0, 41.6);
    const cells = Array.from({ length: 8 }, (_, i) => {
      const on = i === 0 ? k(t, 40.0, 40.4) : k(t, 41.0 + 0.08 * i, 41.3 + 0.08 * i);
      const pulse = 0.5 + 0.5 * Math.sin((t - 41) * 5 + i);
      return `<div style="height:${FEED ? 44 : 56}px;border-radius:10px;border:1px solid rgba(168,240,200,${(0.15 + 0.5 * on).toFixed(3)});
        background:rgba(168,240,200,${(on * (0.18 + 0.12 * pulse)).toFixed(3)});display:flex;align-items:center;justify-content:center;
        font-size:${FEED ? 18 : 20}px;color:rgba(241,243,237,${(0.3 + 0.6 * on).toFixed(3)})" class="mono">GPU ${i}</div>`;
    }).join("");
    html += `<div class="abs card" style="${box(g)};padding:${FEED ? "22px 28px" : "28px 34px"};${rv(t, 39.8, 0.6, 20, 6)}">
      <div style="font-size:${FEED ? 32 : 36}px;font-weight:600">One GPU or many</div>
      <div class="mono" style="margin-top:12px;font-size:${FEED ? 20 : 23}px;color:var(--muted);line-height:1.6">
        <div><span class="mint">$</span> jevany train --config recipes/sft.toml</div>
        <div style="opacity:${(0.25 + 0.75 * many).toFixed(3)}"><span class="mint">$</span> bash infra/train.sh --config recipes/sft.toml</div></div>
      <div style="display:grid;grid-template-columns:repeat(8, 1fr);gap:8px;margin-top:${FEED ? 14 : 22}px">${cells}</div></div>`;
    // Research choices: readout heads, then RLCR as a separate experimental continuation.
    const r = FEED ? { x: 60, y: 750, w: 960, h: 130 } : { x: 980, y: 470, w: 860, h: 150 };
    const choose = k(t, 43.6, 44.4);
    const heads = ["Pointer", "Direct-token"].map((h, i) => `<span class="pill" style="padding:10px 24px;font-size:${FEED ? 28 : 30}px;font-weight:600;
      color:${i ? "var(--violet)" : "var(--mint)"};${rv(t, 43.8 + 0.2 * i, 0.5, 10, 4)}">${h}</span>`).join(`<span class="muted" style="font-size:28px">or</span>`);
    html += `<div class="abs card" style="${box(r)};padding:${FEED ? "20px 28px" : "26px 34px"};opacity:${choose.toFixed(3)}">
      <div class="kicker" style="font-size:${FEED ? 19 : 21}px;color:var(--subtle)">Choose a readout head</div>
      <div style="display:flex;align-items:center;gap:16px;margin-top:14px">${heads}</div></div>`;
    const x = FEED ? { x: 60, y: 900, w: 960, h: 150 } : { x: 980, y: 650, w: 860, h: 160 };
    html += `<div class="abs card" style="${box(x)};padding:${FEED ? "20px 28px" : "26px 34px"};border-color:rgba(167,139,250,.45);${rv(t, 45.4, 0.6, 18, 6)}">
      <div style="display:flex;align-items:center;gap:14px"><span style="font-size:${FEED ? 32 : 34}px;font-weight:600">RLCR</span>
        <span class="pill" style="padding:6px 16px;font-size:${FEED ? 20 : 21}px;color:var(--amber);border-color:rgba(245,185,113,.5)">experimental</span></div>
      <div style="margin-top:10px;font-size:${FEED ? 24 : 26}px;color:var(--muted)">Continues from an SFT checkpoint; rewards correctness and calibration.</div></div>`;
    html += foot(t, 37.0, "Commands from the README and docs/TRAINING.md. The starter recipe uses Qwen3.5-0.8B; swap in your data and backbone.", HX, FEED ? 1080 : 860, HW);
    return html;
  }

  // ---------- 49-63 s: evaluate ----------
  function evaluate(t) {
    let html = headline(t, 49.2, `Evaluate, <span class="mint">then compare.</span>`, HX, HY, HS, HW);
    const tb = FEED ? { x: 60, y: 250, w: 960, h: 130 } : { x: 80, y: 260, w: 1760, h: 120 };
    html += terminal(tb, [[49.9, 1.5, FEED ? "jevany eval --run runs/my-jev --data dev.jsonl" : "jevany eval --run runs/research --data data/research/development.jsonl"]], t, 49.6, FEED ? 22 : 28);
    // Accuracy and calibration side by side, released checkpoints on Transfer.
    const c = FEED ? { x: 60, y: 405, w: 960, h: 480 } : { x: 80, y: 405, w: 1120, h: 430 };
    const g = k(t, 51.6, 52.8);
    const rows = D.checkpoints.map((m, i) => {
      const w = m.transfer;
      return `<div style="display:grid;grid-template-columns:${FEED ? "280px 1fr 90px 90px" : "320px 1fr 110px 110px"};align-items:center;gap:16px;height:${FEED ? 58 : 50}px;${rv(t, 51.2 + 0.12 * i, 0.5, 10, 4)}">
        <span style="font-size:${FEED ? 24 : 27}px;font-weight:600;white-space:nowrap">${m.name.replace(" Direct-Token", " Direct")}</span>
        <div style="position:relative;height:14px;border-radius:7px;background:rgba(255,255,255,.07)"><div style="position:absolute;left:0;top:0;bottom:0;border-radius:7px;
          width:${(clamp(w) * 100 * g).toFixed(2)}%;background:linear-gradient(90deg, var(--teal), var(--mint));box-shadow:0 0 14px rgba(168,240,200,.4)"></div></div>
        <span class="mono" style="font-size:${FEED ? 22 : 25}px;text-align:right">${pct(m.transfer)}</span>
        <span class="mono violet" style="font-size:${FEED ? 22 : 25}px;text-align:right">${m.ece.toFixed(3)}</span></div>`;
    }).join("");
    html += `<div class="abs card" style="${box(c)};padding:${FEED ? "22px 28px" : "28px 34px"};${rv(t, 50.8, 0.6, 20, 6)}">
      <div style="font-size:${FEED ? 26 : 28}px;font-weight:600;margin-bottom:${FEED ? 12 : 16}px">Published Transfer diagnostics <span class="muted" style="font-weight:500;font-size:.85em">${MID} informed development</span></div>
      <div style="display:grid;grid-template-columns:${FEED ? "280px 1fr 90px 90px" : "320px 1fr 110px 110px"};gap:16px;font-size:${FEED ? 18 : 20}px;font-weight:600;letter-spacing:.1em;
        text-transform:uppercase;color:var(--subtle);margin-bottom:10px"><span>Released checkpoint</span><span>Transfer accuracy, 0 to 100%</span><span style="text-align:right">Acc.</span><span style="text-align:right">ECE</span></div>
      ${rows}</div>`;
    // Research tools: exact resume and paradigm comparison.
    const side = FEED ? { x: 60, y: 905, w: 960 } : { x: 1240, y: 410, w: 600 };
    html += `<div class="abs card" style="left:${side.x}px;top:${side.y}px;width:${side.w}px;height:${FEED ? 170 : 200}px;padding:${FEED ? "20px 28px" : "26px 30px"};${rv(t, 55.4, 0.6, 18, 6)}">
      <div style="font-size:${FEED ? 30 : 34}px;font-weight:600">Resume exactly</div>
      <div style="margin-top:8px;font-size:${FEED ? 23 : 25}px;color:var(--muted);line-height:1.35">Recipe, optimizer, schedule, data order and random state are saved.</div></div>`;
    if (!FEED) {
      html += `<div class="abs card" style="left:${side.x}px;top:${side.y + 220}px;width:${side.w}px;height:200px;padding:26px 30px;${rv(t, 57.4, 0.6, 18, 6)}">
        <div style="font-size:34px;font-weight:600">Compare paradigms</div>
        <div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap">${["Pointer", "Direct-token", "RLCR"].map((p, i) => `<span class="pill" style="padding:8px 18px;font-size:24px;
          color:${["var(--mint)", "var(--violet)", "var(--amber)"][i]};${rv(t, 57.8 + 0.15 * i, 0.4, 8, 3)}">${p}</span>`).join("")}</div></div>`;
    }
    html += foot(t, 52.0, `Published diagnostics on Transfer (${fmt(D.protocols.transfer)} decisions, results/model-family-v2.json). The suite informed development; not a held-out test. Lower ECE is better calibrated.`,
      HX, FEED ? 1095 : 860, HW);
    return html;
  }

  // ---------- 63-75 s: deploy ----------
  function deploy(t) {
    let html = headline(t, 63.2, `Serve it <span class="mint">behind one API.</span>`, HX, HY, HS, HW);
    const N = FEED
      ? { ck: { x: 60, y: 260, w: 960, h: 120 }, sv: { x: 60, y: 420, w: 960, h: 150 }, http: { x: 60, y: 620, w: 465, h: 150 }, py: { x: 555, y: 620, w: 465, h: 150 }, app: { x: 60, y: 820, w: 960, h: 250 } }
      : { ck: { x: 80, y: 300, w: 420, h: 170 }, sv: { x: 580, y: 300, w: 700, h: 170 }, http: { x: 1360, y: 250, w: 480, h: 130 }, py: { x: 1360, y: 400, w: 480, h: 130 }, app: { x: 580, y: 560, w: 1260, h: 260 } };
    html += nodeCard(N.ck, t, 63.6, `<span class="mono" style="font-size:${FEED ? 30 : 32}px">runs/my-jev</span>`, "the checkpoint you trained", "var(--mint)", Math.max(0, 1 - Math.abs(t - 64.2) / 0.6));
    html += terminal(N.sv, [[64.4, 1.5, "jevany serve --checkpoint runs/my-jev"]], t, 64.2, FEED ? 24 : 27);
    html += nodeCard(N.http, t, 67.2, "HTTP", `<span class="mono" style="font-size:${FEED ? 21 : 23}px">POST /v1/systemone</span>`, "var(--mint)");
    html += nodeCard(N.py, t, 67.7, "Python", `<span class="mono" style="font-size:${FEED ? 21 : 23}px">JevClient().system_one(...)</span>`, "var(--violet)");
    let svg = "";
    if (FEED) {
      svg += link(540, N.ck.y + N.ck.h, 540, N.sv.y, t, 64.2);
      svg += link(300, N.sv.y + N.sv.h, 300, N.http.y, t, 67.0);
      svg += link(780, N.sv.y + N.sv.h, 780, N.py.y, t, 67.5);
      svg += link(300, N.http.y + N.http.h, 300, N.app.y, t, 69.4);
      svg += link(780, N.py.y + N.py.h, 780, N.app.y, t, 69.4);
    } else {
      svg += link(N.ck.x + N.ck.w, 385, N.sv.x, 385, t, 64.2);
      svg += link(N.sv.x + N.sv.w, 360, N.http.x, 315, t, 67.0);
      svg += link(N.sv.x + N.sv.w, 410, N.py.x, 465, t, 67.5);
      svg += link(N.http.x + 240, N.py.y + N.py.h, 1210, N.app.y, t, 69.4, "#a8f0c8", -30);
    }
    html += svgLayer(svg);
    // Applications: three live recordings from the demo library.
    const apps = ["frontend", "inbox", "pg-arm"];
    const titles = ["Browser agent", "Email triage", "Robot control"];
    const a = N.app;
    const tw = (a.w - 60 - 2 * 20) / 3, th = a.h - 96;
    let tiles = "";
    apps.forEach((c, i) => {
      tiles += `<div class="abs" style="left:${30 + i * (tw + 20)}px;top:70px;width:${tw}px;height:${th}px;${rv(t, 69.8 + 0.2 * i, 0.6, 16, 6)}">
        ${clip(c, Math.max(0, t - 69.8), { x: 0, y: 0, w: tw, h: th }, "outline:1px solid rgba(210,239,229,.14)", true)}
        <div class="abs" style="left:12px;bottom:10px;font-size:${FEED ? 20 : 22}px;font-weight:600;text-shadow:0 1px 8px #000">${titles[i]}</div></div>`;
    });
    html += `<div class="abs card" style="${box(a)};${rv(t, 69.5, 0.6, 20, 6)}">
      <div class="abs" style="left:30px;top:20px;font-size:${FEED ? 30 : 32}px;font-weight:600">Your applications <span class="muted" style="font-size:.8em;font-weight:500">get a choice and its probabilities</span></div>${tiles}</div>`;
    html += foot(t, 66.0, "Application tiles are archived replays and interface recordings from the JevAny demos.", HX, FEED ? 1100 : 860, HW);
    return html;
  }

  // ---------- 75-80.5 s: agent loops ----------
  function loops(t) {
    let html = kicker(t, 75.4, "Inside agent loops", HX, FEED ? 80 : 66);
    html += headline(t, 75.6, `The LLM plans. <span class="mint">Jev picks.</span>`, HX, HY, HS, HW);
    const v = FEED ? { x: 60, y: 260, w: 960, h: 520 } : { x: 80, y: 260, w: 900, h: 600 };
    html += `<div class="abs" style="${box(v)};${rv(t, 75.5, 0.6, 20, 6)}">${clip("arm-hd", Math.max(0, t - 75.5) * 0.9, { x: 0, y: 0, w: v.w, h: v.h }, "outline:1px solid rgba(210,239,229,.18);box-shadow:0 40px 100px rgba(0,0,0,.6)", true)}
      <div class="abs pill" style="left:22px;top:20px;padding:10px 20px;font-size:${FEED ? 22 : 24}px;font-weight:600"><span class="dot" style="background:var(--mint)"></span>Playground replay ${MID} recorded model run</div></div>`;
    const s = FEED ? { x: 60, y: 810, w: 960 } : { x: 1040, y: 260, w: 800 };
    const row = (y, a, dot, name, sub) => `<div class="abs card" style="left:${s.x}px;top:${y}px;width:${s.w}px;height:${FEED ? 100 : 150}px;padding:${FEED ? "18px 28px" : "28px 32px"};${rv(t, a, 0.6, 18, 6)}">
      <div style="display:flex;align-items:center;gap:14px;font-size:${FEED ? 30 : 36}px;font-weight:600"><span class="dot" style="background:${dot};box-shadow:0 0 12px ${dot}"></span>${name}</div>
      ${FEED ? "" : `<div style="margin-top:10px;font-size:27px;color:var(--muted)">${sub}</div>`}</div>`;
    html += row(s.y, 76.0, "var(--llm)", "LLM", "plans, recovers and finishes");
    html += row(s.y + (FEED ? 115 : 170), 77.2, "var(--mint)", "Jev", "takes repeated choices among valid options");
    const st = { x: s.x, y: s.y + (FEED ? 230 : 340), w: s.w, h: FEED ? 0 : 200 };
    if (!FEED) {
      html += `<div class="abs card" style="left:${st.x}px;top:${st.y}px;width:${st.w}px;height:${st.h}px;padding:28px 32px;border-color:rgba(168,240,200,.3);${rv(t, 78.2, 0.6, 18, 6)}">
        <div style="font-size:27px;color:var(--muted)">One paired WebShop run, same reward</div>
        <div style="margin-top:8px;font-size:58px;font-weight:600">LLM calls <span style="color:var(--llm)">${D.webshop.llm_calls[0]}</span> <span class="muted" style="font-size:.7em">${ARROW}</span> <span class="mint">${D.webshop.llm_calls[1]}</span></div></div>`;
    }
    html += foot(t, 76.4, FEED ? "Gains vary by task; see the paired agent-harness evaluations." : "Gains vary by task; see the paired agent-harness evaluations. The robot replay is a PyBullet simulation, not a physical deployment.",
      HX, FEED ? 1100 : 880, HW);
    return html;
  }

  // ---------- 80.5-90 s: close ----------
  function close(t) {
    let html = `<div class="abs" style="left:0;top:${FEED ? 210 : 150}px;width:${W}px;display:flex;flex-direction:${FEED ? "column" : "row"};align-items:center;justify-content:center;gap:${FEED ? 30 : 36}px">
      ${logo(FEED ? 160 : 130, 80.6, t, 1.3)}<div style="${rv(t, 81.0, 0.8, 24, 12)}">${wordmark(FEED ? 112 : 118)}</div></div>`;
    html += `<div class="abs head" style="left:0;width:${W}px;top:${FEED ? 560 : 350}px;text-align:center;font-size:${FEED ? 96 : 104}px;text-shadow:0 8px 40px rgba(0,0,0,.6);${rv(t, 81.6, 0.8)}">
      Build your own${FEED ? "<br>" : " "}<span class="mint">decision models.</span></div>`;
    const glow = 0.5 + 0.5 * Math.sin((t - 83) * 2.2);
    html += `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 830 : 540}px;display:flex;justify-content:center;${rv(t, 83.4, 0.7, 20, 6)}">
      <div class="pill" style="padding:${FEED ? "22px 40px" : "24px 48px"};font-size:${FEED ? 46 : 54}px;font-weight:600;border:2px solid rgba(168,240,200,.7);
        background:rgba(6,10,12,.75);box-shadow:0 0 ${(30 + 30 * glow).toFixed(1)}px rgba(168,240,200,.35)"><span>github.com/<span class="mint">SimpleJev/JevAny</span></span></div></div>`;
    html += `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 974 : 700}px;text-align:center;font-size:${FEED ? 36 : 40}px;${rv(t, 84.2, 0.7, 14, 5)}">simplejev.github.io/JevAny</div>`;
    html += `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 1038 : 766}px;text-align:center;font-size:${FEED ? 28 : 30}px;color:var(--muted);letter-spacing:.06em;${rv(t, 84.8, 0.7, 10, 4)}">
      Open checkpoints ${MID} Research ${MID} Training ${MID} Deployment ${MID} Apache-2.0</div>`;
    return html;
  }

  // Scene windows overlap slightly for crossfades: [scene, start, end, fadeIn, fadeOut].
  const SCENES = [
    [hook, 0, 10.2, 0, 0.6],
    [framework, 9.9, 17.2, 0.4, 0.45],
    [checkpoints, 16.9, 27.2, 0.4, 0.45],
    [data, 26.9, 36.2, 0.4, 0.45],
    [train, 35.9, 49.2, 0.4, 0.45],
    [evaluate, 48.9, 63.2, 0.4, 0.45],
    [deploy, 62.9, 75.3, 0.4, 0.45],
    [loops, 75.0, 80.7, 0.4, 0.45],
    [close, 80.4, 90.5, 0.45, 0.01],
  ];
  const SWEEPS = [9.75, 16.7, 75.0, 80.2];

  function compose(frame) {
    const t = frame / FPS;
    let html = "";
    for (const [fn, a, b, fi, fo] of SCENES) {
      if (t < a || t >= b) continue;
      const pin = fi > 0 ? k(t, a, a + fi) : 1, pout = 1 - k(t, b - fo, b, INOUT);
      const o = Math.min(pin, pout);
      const scale = (1 + 0.025 * (1 - pin)) * (1 - 0.02 * (1 - pout));
      html += `<div class="layer" style="opacity:${o.toFixed(4)};transform:scale(${scale.toFixed(5)});filter:blur(${((1 - pout) * 10).toFixed(2)}px)">${fn(t)}</div>`;
    }
    // Hook: start tight on the first wall tile, then pull back to reveal the wall.
    const zoom = lerp(2.7, FEED ? 1.08 : 1.0, k(t, 0, 2.0, bezier(0.7, 0, 0.2, 1))) * (1 + 0.04 * clamp((t - 2) / 88));
    return background(t, 0.12) + wall(t, wallLevel(t), zoom) + html + stationRail(t) + SWEEPS.map((s) => sweep(t, s)).join("") + grain(frame);
  }
  // ---------- clip frame cache ----------
  const cache = new Map();
  function frameImage(name, i) {
    const key = `${name}/${i}`;
    if (cache.has(key)) {
      const v = cache.get(key);
      cache.delete(key);
      cache.set(key, v);
      return v;
    }
    const img = new Image();
    img.src = `${CLIPS}/${name}/${String(i + 1).padStart(4, "0")}.jpg`;
    const p = img.decode().then(() => img);
    cache.set(key, p);
    while (cache.size > 360) cache.delete(cache.keys().next().value);
    return p;
  }

  async function drawClips() {
    const jobs = [...stage.querySelectorAll("canvas[data-clip]")].map(async (c) => {
      const name = c.dataset.clip;
      const n = CLIP_FRAMES[name];
      let f = parseFloat(c.dataset.t) * 20;
      f = c.dataset.loop === "1" ? ((f % n) + n) % n : clamp(f, 0, n - 1);
      const i0 = Math.floor(f), i1 = (i0 + 1) % n;
      const frac = c.dataset.loop === "1" || i0 < n - 1 ? f - i0 : 0;
      const [a, b] = await Promise.all([frameImage(name, i0), frameImage(name, i1)]);
      const g = c.getContext("2d");
      const s = Math.max(c.width / a.width, c.height / a.height);
      const w = a.width * s, h = a.height * s, x = (c.width - w) / 2, y = (c.height - h) / 2;
      g.globalAlpha = 1;
      g.drawImage(a, x, y, w, h);
      if (frac > 0.001) {
        g.globalAlpha = frac;
        g.drawImage(b, x, y, w, h);
      }
    });
    await Promise.all(jobs);
  }

  const ready = (async () => {
    await document.fonts.load("500 40px 'Plex Condensed'");
    await document.fonts.load("500 40px 'DM Sans'");
    await document.fonts.load("600 40px 'URW Gothic'");
    await document.fonts.ready;
  })();

  window.FILM = {
    FPS, FRAMES, W, H, ready,
    async seek(frame) {
      await ready;
      stage.innerHTML = compose(frame);
      await drawClips();
      return frame;
    },
  };
})();
