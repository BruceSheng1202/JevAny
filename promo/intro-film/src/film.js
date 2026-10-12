/* JevAny intro film.
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
  const CLIP_FRAMES = { crosswalk: 128, peg_insertion: 48, drone: 128, lab: 128, inbox: 128, sql: 32, driving: 128 };
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

  // Probability row: label, value, bar. `hot` (0..1) lights the chosen option.
  function probRow(label, value, hot, x, y, w, size, appear, numAlpha = 1, flare = 0) {
    const glow = hot.toFixed(3);
    const labelColor = hot > 0.01 ? `color-mix(in srgb, var(--mint) ${Math.round(hot * 100)}%, var(--ink))` : "var(--ink)";
    return `<div class="abs" style="left:${px(x)};top:${px(y)};width:${px(w)};height:${size * 2}px;${appear}">
      <div class="abs" style="left:0;top:0;font-size:${size}px;font-weight:500;color:${labelColor}">${label}</div>
      <div class="abs" style="right:0;top:0;font-size:${size}px;font-weight:600;color:${labelColor};opacity:${numAlpha.toFixed(3)}">${pct(value)}</div>
      <div class="track" style="left:0;right:0;top:${size * 1.45}px">
        <div class="fill" style="width:${(value * 100).toFixed(3)}%;
          background:linear-gradient(90deg, color-mix(in srgb, var(--mint) ${Math.round(hot * 100)}%, #5f7378), color-mix(in srgb, var(--teal) ${Math.round(hot * 100)}%, #6f8489));
          box-shadow:0 0 ${(22 * hot + 50 * flare).toFixed(1)}px rgba(168,240,200,${Math.min(1, 0.75 * hot + flare).toFixed(3)})"></div>
      </div>
      <div class="abs" style="left:-30px;right:-30px;top:${size * 1.45 - 30}px;height:70px;border-radius:35px;
        background:radial-gradient(closest-side, rgba(168,240,200,${(0.35 * flare).toFixed(3)}), transparent);pointer-events:none"></div>
      <div class="abs" style="left:-22px;top:${size * 0.2}px;width:6px;height:${size * 0.75}px;border-radius:3px;background:var(--mint);
        opacity:${glow};box-shadow:0 0 16px rgba(168,240,200,.8)"></div>
    </div>`;
  }

  const CHAPTERS = [
    [10, "01", "The idea"],
    [24.5, "02", "One API"],
    [39, "03", "Range"],
    [56, "04", "Agent loops"],
    [76.5, "05", "Get started"],
  ];
  function chapterRail(t) {
    if (t < 10.2 || t > 76.2) return "";
    const cur = CHAPTERS.filter((c) => t >= c[0]).pop();
    const fade = k(t, 10.2, 10.9) * (1 - k(t, 75.6, 76.2));
    const items = CHAPTERS.slice(0, 4).map((c) => {
      const on = c === cur;
      return `<span style="display:inline-flex;align-items:center;gap:10px;color:${on ? "var(--mint)" : "rgba(169,180,182,.55)"}">
        <span style="width:${on ? 26 : 10}px;height:3px;border-radius:2px;background:currentColor"></span>${on ? `${c[1]} ${MID} ${c[2]}` : ""}</span>`;
    }).join("");
    return `<div class="abs" style="right:${FEED ? 60 : 80}px;top:${FEED ? 52 : 46}px;display:flex;gap:14px;font-size:20px;font-weight:600;
      letter-spacing:.12em;text-transform:uppercase;opacity:${fade.toFixed(3)}">${items}</div>`;
  }

  // ---------- scene 1: hook (0-10 s) ----------
  function hook(t) {
    const steps = D.crosswalk.steps;
    const s1 = steps[0], s3 = steps[2];
    const flip = k(t, 5.45, 5.95, INOUT);
    const reorder = k(t, 5.95, 6.75, INOUT);
    const grow = k(t, 3.7, 4.7);
    const regrow = k(t, 5.95, 6.9);
    const cb = FEED ? { x: 60, y: 290, w: 960, h: 476 } : { x: 80, y: 228, w: 1160, h: 575 };
    const ct = t < 5.6 ? lerp(0.2, 2.6, clamp(t / 5.6)) : Math.min(3.8 + (t - 5.6), 6.35);
    const zoom = t < 5.6 ? lerp(1.07, 1.0, OUT(clamp(t / 5.6))) : lerp(1.0, 1.06, clamp((t - 5.6) / 4.4));
    const reveal = 0.5 + 0.5 * k(t, 0, 0.8);
    let html = "";

    // Footage with a soft frame and depth.
    html += `<div class="abs" style="${box({ x: cb.x - 2, y: cb.y - 2, w: cb.w + 4, h: cb.h + 4 })};border-radius:22px;border:1px solid var(--line);
      box-shadow:0 40px 120px rgba(0,0,0,.6), 0 0 80px rgba(45,212,191,${(0.10 * reveal).toFixed(3)});opacity:${reveal.toFixed(3)}"></div>`;
    html += clip("crosswalk", ct, cb, `opacity:${reveal.toFixed(3)};filter:brightness(${(0.6 + 0.4 * reveal).toFixed(3)})`, false, zoom);
    // Cut flash at the state change, masked by the sweep.
    const flash = Math.max(0, 1 - Math.abs(t - 5.6) / 0.18);
    if (flash > 0) html += `<div class="abs" style="${box(cb)};border-radius:20px;background:rgba(168,240,200,${(0.22 * flash).toFixed(3)})"></div>`;

    // State pill inside the footage.
    const pill = (txt, color, a) => `<div class="abs pill" style="left:${px(cb.x + 26)};top:${px(cb.y + 24)};padding:12px 22px;font-size:${FEED ? 28 : 30}px;
      font-weight:600;opacity:${a.toFixed(3)};backdrop-filter:blur(6px)"><span class="dot" style="background:${color};box-shadow:0 0 14px ${color}"></span>${txt}</div>`;
    html += pill("Pedestrian in the roadway", "var(--amber)", k(t, 0.2, 0.8) * (1 - flip));
    html += pill("Roadway clear", "var(--mint)", flip);

    // Kicker and headline.
    const hx = FEED ? 60 : 80;
    const stepText = (n) => `JevAny decision record ${MID} Step ${n} of 3`;
    html += kicker(t, -0.2, stepText(1), hx, FEED ? 92 : 70, 5.35);
    html += kicker(t, 5.75, stepText(3), hx, FEED ? 92 : 70);
    html += headline(t, -0.3, "A pedestrian is still in the road.", hx, FEED ? 134 : 108, FEED ? 76 : 88, FEED ? 960 : 1400, 1.85);
    html += headline(t, 2.1, "What should the car do?", hx, FEED ? 134 : 108, FEED ? 76 : 88, FEED ? 960 : 1400, 7.55);
    html += headline(t, 7.85, `New state. <span class="mint">New choice.</span>`, hx, FEED ? 134 : 108, FEED ? 76 : 88, FEED ? 960 : 1400, 9.35);

    // Options with probabilities from the archived record.
    const panel = FEED ? { x: 82, y: 812, w: 916, row: 74, size: 32 } : { x: 1316, y: 290, w: 524, row: 124, size: 36 };
    html += `<div class="abs" style="left:${px(panel.x)};top:${px(panel.y - (FEED ? 0 : 62))};font-size:22px;font-weight:600;letter-spacing:.14em;
      color:var(--subtle);text-transform:uppercase;${rv(t, 2.1, 0.6, 10, 4)};${FEED ? "display:none" : ""}">4 of 7 options ${MID} probability</div>`;
    const rank = (step) => step.shown.slice().sort((a, b) => b.probability - a.probability).map((o) => o.action);
    const r1 = rank(s1), r3 = rank(s3);
    s1.shown.forEach((o) => {
      const p1 = o.probability;
      const p3 = s3.shown.find((x) => x.action === o.action).probability;
      const value = lerp(p1 * grow, p3, regrow);
      const pos = lerp(r1.indexOf(o.action), r3.indexOf(o.action), reorder);
      const hot = (o.action === s1.action ? k(t, 4.35, 4.75) * (1 - flip) : 0) + (o.action === s3.action ? k(t, 6.7, 7.1) : 0);
      const appear = rv(t, 2.2 + 0.11 * r1.indexOf(o.action), 0.55, 22, 6);
      const flare = o.action === s1.action ? Math.max(0, 1 - Math.abs(t - 4.5) / 0.35) * (1 - flip) : o.action === s3.action ? Math.max(0, 1 - Math.abs(t - 6.9) / 0.35) : 0;
      html += probRow(o.label, value, hot, panel.x, panel.y + pos * panel.row, panel.w, panel.size, appear, k(t, 3.65, 3.95), flare);
    });

    html += foot(t, 2.0, "Archived replay from an earlier compatible JevAny checkpoint. The 3D view illustrates the recorded run; 4 of 7 options shown.",
      hx, FEED ? 1122 : 836, FEED ? 960 : 1160);
    return html;
  }

  // ---------- scene 2a: title (10-16 s) ----------
  function title(t) {
    const L = FEED ? 230 : 190;
    const lockup = `<div class="abs" style="left:0;top:${FEED ? 250 : 270}px;width:${W}px;display:flex;flex-direction:${FEED ? "column" : "row"};
      align-items:center;justify-content:center;gap:${FEED ? 46 : 52}px">
      ${logo(L, 10.0, t)}
      <div style="${rv(t, 10.55, 0.9, 30, 14)}">${wordmark(FEED ? 150 : 168)}</div></div>`;
    const sub = `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 760 : 560}px;text-align:center;font-size:${FEED ? 46 : 52}px;font-weight:500;
      line-height:1.25;${rv(t, 11.0, 0.8)}">Open infrastructure for${FEED ? "<br>" : " "}<span class="mint">System 1 decision models</span></div>`;
    const chips = ["Data", "Adaptation", "Evaluation", "Deployment"].map((c, i) =>
      `<span class="pill" style="padding:14px 28px;font-size:${FEED ? 30 : 32}px;font-weight:500;color:var(--muted);${rv(t, 12.4 + 0.14 * i, 0.6, 16, 6)}">${c}</span>`).join("");
    const row = `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 930 : 680}px;display:flex;justify-content:center;flex-wrap:wrap;gap:18px;
      padding:0 60px">${chips}</div>`;
    return lockup + sub + row;
  }

  // ---------- scene 2b: why (16-24.5 s) ----------
  const MENU = ["x-small", "small", "medium"];
  function why(t) {
    const hx = FEED ? 60 : 80;
    let html = kicker(t, 16.1, "Why it matters", hx, FEED ? 92 : 70);
    html += headline(t, 16.3, "Agents write paragraphs to pick from a short list.", hx, FEED ? 134 : 108, FEED ? 62 : 70, FEED ? 960 : 1760, 20.25);
    html += headline(t, 20.55, `A decision model reads the state <span class="mint">and picks.</span>`, hx, FEED ? 134 : 108, FEED ? 62 : 70, FEED ? 960 : 1760);
    const A = FEED ? { x: 60, y: 330, w: 960, h: 390 } : { x: 80, y: 268, w: 860, h: 600 };
    const B = FEED ? { x: 60, y: 744, w: 960, h: 362 } : { x: 980, y: 268, w: 860, h: 600 };
    const head = (b, name, sub, color, a) => `<div class="abs" style="left:${px(b.x + 36)};top:${px(b.y + 30)};${rv(t, a, 0.6, 12, 4)}">
      <div style="display:flex;align-items:center;gap:14px;font-size:${FEED ? 32 : 36}px;font-weight:600"><span class="dot" style="background:${color};box-shadow:0 0 12px ${color}"></span>${name}</div>
      <div style="margin-top:8px;font-size:${FEED ? 24 : 26}px;color:var(--muted)">${sub}</div></div>`;
    html += `<div class="abs card" style="${box(A)};${rv(t, 16.5, 0.7, 30, 8)}"></div><div class="abs card" style="${box(B)};${rv(t, 16.75, 0.7, 30, 8)}"></div>`;
    html += head(A, "LLM agent", "Free-text reasoning, then an answer to parse", "var(--llm)", 16.8);
    html += head(B, "Decision model", "Scores each candidate option", "var(--mint)", 17.0);

    // Streaming reasoning: abstract text lines that keep coming.
    const r = rng(7);
    const lines = Array.from({ length: 60 }, () => 0.35 + 0.6 * r());
    const lh = FEED ? 30 : 36;
    const area = { x: A.x + 36, y: A.y + (FEED ? 130 : 150), w: A.w - 72, h: FEED ? 140 : 300 };
    const typed = Math.max(0, (t - 17.2) * 5.2);
    let rowsHtml = "";
    for (let i = 0; i < Math.min(lines.length, Math.ceil(typed)); i++) {
      const wfrac = i === Math.floor(typed) ? lines[i] * (typed % 1) : lines[i];
      rowsHtml += `<div class="abs" style="left:0;top:${i * lh}px;width:${(wfrac * 100).toFixed(2)}%;height:${lh * 0.42}px;border-radius:6px;
        background:rgba(143,163,196,${i % 7 === 3 ? 0.42 : 0.24})"></div>`;
    }
    const scroll = Math.max(0, typed * lh - area.h + lh);
    html += `<div class="abs" style="${box(area)};overflow:hidden;-webkit-mask-image:linear-gradient(to bottom, transparent, #000 22%, #000 88%, transparent)">
      <div class="abs" style="left:0;top:${px(-scroll)};width:100%">${rowsHtml}</div></div>`;
    const chipsY = A.y + A.h - (FEED ? 84 : 110);
    const llmPick = k(t, 22.7, 23.1);
    html += `<div class="abs" style="left:${px(A.x + 36)};top:${px(chipsY)};display:flex;gap:14px;${rv(t, 17.4, 0.6, 10, 4)}">` +
      MENU.map((m) => `<span class="pill mono" style="padding:10px 22px;font-size:${FEED ? 24 : 28}px;
        ${m === "small" ? `border-color:rgba(143,163,196,${(0.2 + 0.8 * llmPick).toFixed(3)});color:color-mix(in srgb, var(--llm) ${Math.round(llmPick * 100)}%, var(--muted))` : "color:var(--muted)"}">${m}</span>`).join("") + `</div>`;

    // Decision model: the same menu, scored in one step.
    const illus = { "x-small": 0.09, small: 0.86, medium: 0.05 };
    const g = k(t, 20.9, 21.7);
    const rowX = B.x + 58, rowW = B.w - 116;
    MENU.forEach((m, i) => {
      const hot = m === "small" ? k(t, 21.6, 22.0) : 0;
      const y = B.y + (FEED ? 120 : 190) + i * (FEED ? 62 : 110);
      const label = `<span class="mono">${m}</span>`;
      html += `<div class="abs" style="left:${px(rowX)};top:${px(y)};width:${px(rowW)};${rv(t, 17.3 + 0.1 * i, 0.6, 12, 4)}">
        <div style="font-size:${FEED ? 28 : 32}px;color:${hot ? `color-mix(in srgb, var(--mint) ${Math.round(hot * 100)}%, var(--ink))` : "var(--ink)"}">${label}</div>
        <div class="track" style="left:0;right:0;top:${FEED ? 42 : 52}px"><div class="fill" style="width:${(illus[m] * g * 100).toFixed(2)}%;
          background:${hot ? "linear-gradient(90deg, var(--mint), var(--teal))" : "#5f7378"};box-shadow:0 0 ${(20 * hot).toFixed(1)}px rgba(168,240,200,.7)"></div></div></div>`;
    });
    html += `<div class="abs pill" style="left:${px(rowX)};top:${px(B.y + B.h - (FEED ? 60 : 96))};padding:10px 22px;font-size:${FEED ? 24 : 28}px;
      color:var(--mint);${rv(t, 21.9, 0.5, 10, 4)}"><span class="dot" style="background:var(--mint)"></span>small ${MID} done</div>`;
    html += foot(t, 17.5, "Illustration. Menu from the WebShop example later in this video; bar lengths are not measured values.", hx, FEED ? 1120 : 880, FEED ? 960 : 1600);
    return html;
  }

  // ---------- scene 3: API (24.5-39 s) ----------
  function api(t) {
    const hx = FEED ? 60 : 80;
    let html = kicker(t, 24.6, `One API ${MID} POST /v1/systemone`, hx, FEED ? 92 : 70);
    html += headline(t, 24.8, `State and options in.${FEED ? "<br>" : " "}<span class="mint">A choice out.</span>`, hx, FEED ? 134 : 108, FEED ? 70 : 84, FEED ? 960 : 1760);
    const L = FEED
      ? { x: 60, w: 960, s: { y: 320, h: 124 }, q: { y: 460, h: 112 }, o: { y: 588, h: 150 }, core: [540, 806, 58], out: { x: 60, y: 880, w: 960, h: 228 } }
      : { x: 80, w: 660, s: { y: 268, h: 160 }, q: { y: 448, h: 140 }, o: { y: 608, h: 210 }, core: [1010, 543, 118], out: { x: 1250, y: 300, w: 590, h: 488 } };
    const label = (txt) => `<div class="mono" style="font-size:${FEED ? 20 : 22}px;color:var(--mint);letter-spacing:.06em">${txt}</div>`;
    const card = (b, a, inner) => `<div class="abs card" style="left:${px(L.x)};top:${px(b.y)};width:${px(L.w)};height:${px(b.h)};padding:${FEED ? "20px 28px" : "26px 34px"};${rv(t, a, 0.7, 26, 8)}">${inner}</div>`;
    const media = k(t, 35.3, 35.9);
    html += card(L.s, 25.45, `${label('"state"')}<div style="margin-top:${FEED ? 10 : 16}px;font-size:${FEED ? 36 : 40}px;font-weight:500">&ldquo;I was charged twice.&rdquo;</div>
      <div class="abs" style="right:${FEED ? 24 : 30}px;top:${FEED ? 22 : 26}px;display:flex;gap:10px;opacity:${media.toFixed(3)}">` +
      ["text", "image", "video"].map((m, i) => `<span class="pill mono" style="padding:6px 16px;font-size:${FEED ? 20 : 22}px;color:${i ? "var(--violet)" : "var(--muted)"};
        ${rv(t, 35.3 + 0.15 * i, 0.5, 8, 3)}">${m}</span>`).join("") + `</div>`);
    html += card(L.q, 26.3, `${label('"instructions"')}<div style="margin-top:${FEED ? 10 : 14}px;font-size:${FEED ? 32 : 36}px">Which team should handle this?</div>`);
    const opt = (key, desc) => `<div style="display:flex;gap:22px;align-items:baseline;margin-top:${FEED ? 8 : 14}px;font-size:${FEED ? 30 : 32}px">
      <span class="mono mint" style="min-width:${FEED ? 150 : 160}px">${key}</span><span class="muted">${desc}</span></div>`;
    html += card(L.o, 27.55, `${label('"criteria"')}${opt("billing", "Payment problems")}${opt("shipping", "Delivery problems")}`);

    // Beams into the model and out to the answer.
    const [cx, cy, cr] = L.core;
    const beam = (x1, y1, x2, y2, a, d = 0.8, color = "rgba(168,240,200,.75)") => {
      const p = k(t, a, a + d, INOUT);
      const path = FEED ? `M${x1} ${y1} C ${x1} ${(y1 + y2) / 2}, ${x2} ${(y1 + y2) / 2}, ${x2} ${y2}` : `M${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`;
      return `<path d="${path}" fill="none" stroke="${color}" stroke-width="3" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${(1 - p).toFixed(4)}"
        style="filter:drop-shadow(0 0 8px rgba(168,240,200,.8))"/>`;
    };
    let svg = "";
    if (FEED) {
      svg += beam(540, 738, cx, cy - cr, 28.2, 0.7);
      svg += beam(cx, cy + cr, 540, L.out.y, 29.45, 0.4);
    } else {
      const ex = L.x + L.w;
      svg += beam(ex, L.s.y + L.s.h / 2, cx - cr, cy - 40, 28.0);
      svg += beam(ex, L.q.y + L.q.h / 2, cx - cr, cy, 28.12);
      svg += beam(ex, L.o.y + L.o.h / 2, cx - cr, cy + 40, 28.24);
      svg += beam(cx + cr, cy, L.out.x, cy, 29.45, 0.4);
    }
    html += `<svg class="abs" width="${W}" height="${H}" style="left:0;top:0;overflow:visible">${svg}</svg>`;
    const pulse = Math.max(0, 1 - Math.abs(t - 29.3) / 0.5);
    const coreIn = k(t, 27.9, 28.6);
    html += `<div class="abs" style="left:${px(cx - cr)};top:${px(cy - cr)};width:${cr * 2}px;height:${cr * 2}px;border-radius:50%;
      border:2px solid rgba(168,240,200,${(0.35 + 0.5 * pulse).toFixed(3)});background:radial-gradient(circle, rgba(16,22,41,.95) 55%, rgba(45,212,191,${(0.18 + 0.35 * pulse).toFixed(3)}));
      box-shadow:0 0 ${(40 + 80 * pulse).toFixed(1)}px rgba(45,212,191,${(0.25 + 0.4 * pulse).toFixed(3)});opacity:${coreIn.toFixed(3)};
      transform:scale(${(0.85 + 0.15 * coreIn + 0.05 * pulse).toFixed(4)});display:flex;align-items:center;justify-content:center">
      <svg viewBox="0 0 512 512" width="${cr * 1.05}" height="${cr * 1.05}"><path d="M144 128 256 240V128" fill="none" stroke="#8b5cf6" stroke-width="30" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="144" cy="128" r="28" fill="#8b5cf6"/><circle cx="256" cy="128" r="28" fill="#8b5cf6"/>
      <path d="M128 320a64 64 0 0 0 128 0v-80l112-112" fill="none" stroke="#2dd4bf" stroke-width="44" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="368" cy="128" r="36" fill="#2dd4bf"/></svg></div>`;
    if (!FEED) html += `<div class="abs" style="left:${px(cx - 160)};width:320px;top:${px(cy + cr + 26)};text-align:center;font-size:26px;color:var(--muted);${rv(t, 28.4, 0.6, 8, 3)}">Jev model</div>`;

    // Answer card.
    const o = L.out;
    const g = k(t, 30.0, 30.9);
    const bars = [["billing", 0.93, 1], ["shipping", 0.07, 0]].map(([key, v, hot], i) => `<div style="position:relative;height:${FEED ? 40 : 70}px;margin-top:${i ? 6 : 0}px">
      <span class="mono" style="font-size:${FEED ? 26 : 30}px;color:${hot ? "var(--mint)" : "var(--muted)"}">${key}</span>
      <div class="track" style="left:${FEED ? 170 : 0}px;right:0;top:${FEED ? 14 : 48}px"><div class="fill" style="width:${(v * g * 100).toFixed(2)}%;
        background:${hot ? "linear-gradient(90deg, var(--mint), var(--teal))" : "#5f7378"};box-shadow:0 0 ${hot ? 18 : 0}px rgba(168,240,200,.6)"></div></div></div>`).join("");
    html += `<div class="abs card" style="${box(o)};padding:${FEED ? "22px 30px" : "34px 38px"};border-color:rgba(168,240,200,.32);${rv(t, 29.6, 0.7, 26, 8)}">
      ${label('answers.department')}
      <div style="display:flex;align-items:baseline;gap:20px;margin:${FEED ? "2px 0 6px" : "18px 0 22px"}">
        <span class="muted" style="font-size:${FEED ? 26 : 28}px">choice</span>
        <span class="mono" style="font-size:${FEED ? 46 : 64}px;color:var(--mint);text-shadow:0 0 28px rgba(168,240,200,.45)">billing</span></div>
      <div style="${rv(t, 29.95, 0.5, 8, 3)}">${bars}</div>
      ${FEED ? "" : `<div class="mono" style="margin-top:26px;font-size:24px;line-height:1.6;color:var(--subtle);${rv(t, 31.4, 0.6, 8, 3)}">"probabilities": {...}<br>"confidence": ...</div>`}
    </div>`;
    html += foot(t, 30.4, "Request shape from docs/API.md. Answer values illustrative. Image and video input depend on the backbone.", hx, FEED ? 1120 : 862, FEED ? 960 : 1600);
    return html;
  }

  // ---------- scene 4: range (39-47 s) ----------
  const DOMAIN = { peg_insertion: "Robot arm", drone: "Drone delivery", lab: "Lab automation", inbox: "Email triage", sql: "SQL repair", driving: "Driving" };
  function range(t) {
    const hx = FEED ? 60 : 80;
    let html = kicker(t, 39.1, "Same API, different decisions", hx, FEED ? 92 : 70);
    html += headline(t, 39.3, `Tickets, tools, <span class="mint">robots.</span>`, hx, FEED ? 134 : 108, FEED ? 76 : 84, FEED ? 960 : 1700);
    const cols = FEED ? 2 : 3;
    const tw = FEED ? 465 : 560, th = FEED ? 230 : 284, gap = 30;
    const x0 = FEED ? 60 : 90, y0 = FEED ? 290 : 248;
    const push = lerp(1.0, 1.035, clamp((t - 39) / 8));
    let tiles = "";
    D.montage.forEach((m, i) => {
      const c = i % cols, r = Math.floor(i / cols);
      const b = { x: x0 + c * (tw + gap), y: y0 + r * (th + gap), w: tw, h: th };
      const a = 39.45 + 0.13 * i;
      const p = k(t, a, a + 0.8);
      tiles += `<div class="abs" style="${box(b)};opacity:${p.toFixed(3)};transform:translateY(${((1 - p) * 40).toFixed(1)}px) scale(${(0.94 + 0.06 * p).toFixed(4)});">
        <div class="abs" style="inset:-2px;border-radius:22px;border:1px solid var(--line);box-shadow:0 26px 70px rgba(0,0,0,.55)"></div>
        ${clip(m.case, Math.max(0, t - a), { x: 0, y: 0, w: tw, h: th }, "", true, 1.0)}
        <div class="abs" style="left:0;right:0;top:0;height:${th * 0.55}px;border-radius:20px 20px 0 0;background:linear-gradient(to bottom, rgba(5,9,11,.88), transparent)"></div>
        <div class="abs" style="left:22px;top:18px;right:22px">
          <div class="kicker" style="font-size:${FEED ? 18 : 19}px">${DOMAIN[m.case]}</div>
          <div style="margin-top:6px;font-size:${FEED ? 26 : 30}px;font-weight:600;line-height:1.15">${m.title}</div></div>
      </div>`;
    });
    html += `<div class="abs" style="left:0;top:0;width:${W}px;height:${H}px;transform:scale(${push.toFixed(4)});transform-origin:50% 50%">${tiles}</div>`;
    html += foot(t, 40.2, "Archived replays from an earlier compatible JevAny checkpoint. 3D views illustrate recorded runs; inbox and SQL are original interface recordings.",
      hx, FEED ? 1068 : 880, FEED ? 960 : 1700);
    return html;
  }

  // ---------- scene 4b: train (47-56 s) ----------
  function train(t) {
    const hx = FEED ? 60 : 80;
    let html = kicker(t, 47.1, "Train your own", hx, FEED ? 92 : 70);
    html += headline(t, 47.3, `Fit a Jev to <span class="mint">your decisions.</span>`, hx, FEED ? 134 : 108, FEED ? 76 : 84, FEED ? 960 : 1700);
    const nodes = [
      ["01", "Open LLM or VLM", "or a released Jev checkpoint", 47.7],
      ["02", "Your decision data", "states, options, labels", 49.9],
      ["03", "Fine-tune", "adapt the model to your data", 51.0],
      ["04", "Evaluate, then serve", "behind the same API", 53.3],
    ];
    const nb = (i) => FEED ? { x: 110, y: 296 + i * 206, w: 860, h: 156 } : { x: 80 + i * 450, y: 360, w: 390, h: 230 };
    let svg = "";
    nodes.forEach((n, i) => {
      if (i === 0) return;
      const a = nb(i - 1), b = nb(i);
      const p = k(t, n[3] - 0.35, n[3] + 0.15, INOUT);
      const [x1, y1, x2, y2] = FEED ? [a.x + 80, a.y + a.h, b.x + 80, b.y] : [a.x + a.w, a.y + a.h / 2, b.x, b.y + b.h / 2];
      svg += `<line x1="${x1}" y1="${y1}" x2="${lerp(x1, x2, p)}" y2="${lerp(y1, y2, p)}" stroke="rgba(168,240,200,.6)" stroke-width="3"/>`;
      // Data pulse travelling along the finished link.
      const cyc = ((t - n[3]) * 0.9) % 1;
      if (t > n[3] + 0.2) svg += `<circle cx="${lerp(x1, x2, cyc)}" cy="${lerp(y1, y2, cyc)}" r="7" fill="#a8f0c8" style="filter:drop-shadow(0 0 10px #a8f0c8)"/>`;
    });
    html += `<svg class="abs" width="${W}" height="${H}" style="left:0;top:0">${svg}</svg>`;
    nodes.forEach((n, i) => {
      const b = nb(i);
      const lit = Math.max(0, 1 - Math.abs(t - n[3] - 0.4) / 0.9);
      html += `<div class="abs card" style="${box(b)};padding:${FEED ? "26px 34px 0 160px" : "30px 32px"};border-color:rgba(168,240,200,${(0.14 + 0.4 * lit).toFixed(3)});
        box-shadow:0 30px 80px rgba(0,0,0,.45), 0 0 ${(60 * lit).toFixed(1)}px rgba(45,212,191,${(0.3 * lit).toFixed(3)});${rv(t, n[3], 0.7, 26, 8)}">
        <div class="mono mint" style="font-size:${FEED ? 40 : 26}px;${FEED ? "position:absolute;left:44px;top:46px" : ""}">${n[0]}</div>
        <div style="margin-top:${FEED ? 0 : 22}px;font-size:${FEED ? 40 : 40}px;font-weight:600;line-height:1.1">${n[1]}</div>
        <div style="margin-top:10px;font-size:${FEED ? 28 : 28}px;color:var(--muted)">${n[2]}</div></div>`;
    });
    if (!FEED) {
      html += `<div class="abs" style="left:80px;top:690px;width:1760px;font-size:34px;color:var(--muted);${rv(t, 54.0, 0.7, 12, 4)}">
        Data preparation, model adaptation and evaluation live in one open repository.</div>`;
    }
    html += foot(t, 48.2, "Supported model families, recipes and hardware notes are in the repository docs.", hx, FEED ? 1120 : 870, FEED ? 960 : 1700);
    return html;
  }

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

  // ---------- scene 5a: agent loop (56-64.5 s) ----------
  function loop(t) {
    const hx = FEED ? 60 : 80;
    let html = kicker(t, 56.1, "Jev inside LLM agent loops", hx, FEED ? 92 : 70);
    html += headline(t, 56.3, `The LLM plans. <span class="mint">Jev picks.</span>`, hx, FEED ? 134 : 108, FEED ? 76 : 84, FEED ? 960 : 1700);
    const N = FEED
      ? { llm: { x: 120, y: 300, w: 840, h: 160 }, jev: { x: 120, y: 620, w: 840, h: 160 }, env: { x: 120, y: 940, w: 840, h: 150 } }
      : { llm: { x: 120, y: 330, w: 520, h: 190 }, jev: { x: 1280, y: 330, w: 520, h: 190 }, env: { x: 700, y: 650, w: 520, h: 170 } };
    const node = (b, name, sub, color, a, glowAt = -9) => {
      const lit = Math.max(0, 1 - Math.abs(t - glowAt) / 0.5);
      return `<div class="abs card" style="${box(b)};padding:${FEED ? "26px 36px" : "32px 38px"};border-color:color-mix(in srgb, ${color} ${Math.round(30 + 50 * lit)}%, transparent);
        box-shadow:0 30px 80px rgba(0,0,0,.45), 0 0 ${(50 * lit).toFixed(1)}px color-mix(in srgb, ${color} 45%, transparent);${rv(t, a, 0.7, 24, 8)}">
        <div style="display:flex;align-items:center;gap:16px;font-size:${FEED ? 40 : 44}px;font-weight:600"><span class="dot" style="width:18px;height:18px;background:${color};box-shadow:0 0 14px ${color}"></span>${name}</div>
        <div style="margin-top:12px;font-size:${FEED ? 28 : 30}px;color:var(--muted)">${sub}</div></div>`;
    };
    // Repeated Jev <-> environment cycles.
    const cycles = [61.4, 62.15, 62.9];
    const envGlow = cycles.map((c) => c + 0.35).find((c) => Math.abs(t - c) < 0.5) || -9;
    const jevGlow = cycles.find((c) => Math.abs(t - c) < 0.5) || 60.6;
    html += node(N.llm, "LLM", "Plans, recovers, finishes", "var(--llm)", 56.7, 63.6);
    html += node(N.env, "Environment", "Applies actions, returns state", "var(--muted)", 57.8, envGlow);
    html += node(N.jev, "Jev", "Picks among 2&ndash;4 valid options", "var(--mint)", 60.4, jevGlow);

    const L = N.llm, J = N.jev, E = N.env;
    let svg = `<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 10 5 0 10z" fill="#a8f0c8"/></marker><marker id="al" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 10 5 0 10z" fill="#8fa3c4"/></marker></defs>`;
    const draw = (d, a, dur, color, marker, dash = "") => {
      const p = k(t, a, a + dur, INOUT);
      if (p <= 0) return "";
      // The arrowhead appears only once the stroke reaches it.
      const head = dash || p > 0.97 ? `marker-end="url(#${marker})"` : "";
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="3" ${dash ? `stroke-dasharray="${dash}"` : `pathLength="1" stroke-dasharray="1" stroke-dashoffset="${(1 - p).toFixed(4)}"`}
        ${dash ? `opacity="${p.toFixed(3)}"` : ""} ${head}/>`;
    };
    const labelAt = (x, y, txt, a, color = "var(--muted)", anchor = "middle") => `<div class="abs" style="left:${px(anchor === "middle" ? x - 300 : x)};top:${px(y)};width:600px;
      text-align:${anchor === "middle" ? "center" : "left"};font-size:${FEED ? 24 : 26}px;color:${color};${rv(t, a, 0.5, 8, 3)}">${txt}</div>`;
    let labels = "";
    let paths;
    if (FEED) {
      const xa = 330, xb = 750;
      paths = [
        [`M${xa} ${L.y + L.h} L${xa} ${J.y - 4}`, 60.5, "#a8f0c8", "ah"],
        [`M${xa} ${J.y + J.h} L${xa} ${E.y - 4}`, 61.2, "#a8f0c8", "ah"],
        [`M${xb} ${E.y} L${xb} ${J.y + J.h + 4}`, 61.6, "#a8f0c8", "ah"],
      ];
      paths.forEach((p) => { svg += draw(p[0], p[1], 0.5, p[2], p[3]); });
      svg += draw(`M${xb} ${J.y} L${xb} ${L.y + L.h + 4}`, 63.4, 0.4, "#8fa3c4", "al", "10 9");
      labels += labelAt(xa + 24, L.y + L.h + 50, "candidates", 60.7, "var(--mint)", "left");
      labels += labelAt(xa + 24, J.y + J.h + 50, "action", 61.3, "var(--mint)", "left");
      labels += labelAt(xb + 24, J.y + J.h + 80, "new state", 61.7, "var(--muted)", "left");
      labels += labelAt(xb + 24, L.y + L.h + 50, "hand back", 63.5, "var(--llm)", "left");
    } else {
      paths = [
        [`M${L.x + L.w} ${L.y + 70} L${J.x - 6} ${J.y + 70}`, 60.5, "#a8f0c8", "ah"],
        [`M${J.x + 150} ${J.y + J.h} C ${J.x + 150} ${E.y + 40}, ${E.x + E.w + 120} ${E.y + 60}, ${E.x + E.w + 6} ${E.y + 70}`, 61.2, "#a8f0c8", "ah"],
        [`M${E.x + E.w} ${E.y + 130} C ${E.x + E.w + 230} ${E.y + 130}, ${J.x + 300} ${E.y - 40}, ${J.x + 300} ${J.y + J.h + 6}`, 61.6, "#a8f0c8", "ah"],
      ];
      paths.forEach((p) => { svg += draw(p[0], p[1], 0.5, p[2], p[3]); });
      svg += draw(`M${J.x} ${J.y + 140} L${L.x + L.w + 6} ${L.y + 140}`, 63.4, 0.4, "#8fa3c4", "al", "12 10");
      labels += labelAt((L.x + L.w + J.x) / 2, L.y + 22, "plan + candidate actions", 60.7, "var(--mint)");
      labels += labelAt((L.x + L.w + J.x) / 2, L.y + 152, "hand back on novelty, recovery or completion", 63.5, "var(--llm)");
      labels += labelAt(E.x + E.w - 6, J.y + J.h + 62, "action", 61.3, "var(--mint)", "left");
      labels += labelAt(J.x + 330, E.y + 52, "new state", 61.7, "var(--muted)", "left");
    }
    // Packets for the repeated choices.
    cycles.forEach((c) => {
      const p1 = clamp((t - c) / 0.35), p2 = clamp((t - c - 0.35) / 0.35);
      const packet = (d, p, fill) => {
        const pt = pointOn(d, p);
        return `<circle cx="${pt.x.toFixed(2)}" cy="${pt.y.toFixed(2)}" r="9" fill="${fill}" style="filter:drop-shadow(0 0 10px ${fill})"/>`;
      };
      if (p1 > 0 && p1 < 1) svg += packet(paths[1][0], p1, "#a8f0c8");
      if (p2 > 0 && p2 < 1) svg += packet(paths[2][0], p2, "#cfd8da");
    });
    html += `<svg class="abs" width="${W}" height="${H}" style="left:0;top:0;overflow:visible">${svg}</svg>${labels}`;
    html += foot(t, 58.4, "Golden rule from the README: delegate the selection bottleneck, not the task.", hx, FEED ? 1120 : 870, FEED ? 960 : 1700);
    return html;
  }

  // ---------- scene 5b: WebShop race (64.5-72 s) ----------
  function shop(t) {
    const S = D.webshop;
    const hx = FEED ? 60 : 80;
    let html = kicker(t, 64.6, `One paired WebShop run ${MID} same reward`, hx, FEED ? 92 : 70);
    html += headline(t, 64.8, "Buy the matching blazer", hx, FEED ? 134 : 108, FEED ? 72 : 84, FEED ? 960 : 1700);
    html += `<div class="abs" style="left:${hx}px;top:${FEED ? 222 : 212}px;width:${FEED ? 960 : 1700}px;font-size:${FEED ? 26 : 30}px;color:var(--muted);${rv(t, 65.0, 0.6, 10, 4)}">
      Goal: ${S.goal.replace("Women's", "women's")}</div>`;
    const start = 65.4, base = 6.0;
    const jevDur = base * S.jev.wall_time_seconds / S.baseline.wall_time_seconds;
    const speed = S.baseline.wall_time_seconds / base;
    const lanes = [
      { name: "LLM only", run: S.baseline, color: "var(--llm)", dur: base, ctrl: Array(S.baseline.steps).fill("llm") },
      { name: "LLM + Jev", run: S.jev, color: "var(--mint)", dur: jevDur, ctrl: S.jev.path.map((p) => p.controller) },
    ];
    lanes.forEach((ln, li) => {
      const b = FEED ? { x: 60, y: 300 + li * 196, w: 960, h: 176 } : { x: 80, y: 292 + li * 178, w: 1760, h: 156 };
      const el = clamp((t - start) / ln.dur);
      const done = el >= 1;
      const clock = Math.min(t - start, ln.dur) * speed;
      html += `<div class="abs card" style="${box(b)};${rv(t, 65.1 + 0.15 * li, 0.6, 20, 6)};border-color:${done ? "rgba(168,240,200,.35)" : "var(--line)"}"></div>`;
      const lx = b.x + 34, ly = b.y + (FEED ? 20 : 30);
      html += `<div class="abs" style="left:${lx}px;top:${ly}px;${rv(t, 65.2 + 0.15 * li, 0.6, 10, 4)}">
        <div style="display:flex;align-items:center;gap:12px;font-size:${FEED ? 32 : 36}px;font-weight:600"><span class="dot" style="background:${ln.color}"></span>${ln.name}</div>
        <div class="mono" style="margin-top:${FEED ? 0 : 10}px;font-size:${FEED ? 24 : 28}px;color:${done ? "var(--mint)" : "var(--muted)"};${FEED ? "position:absolute;left:330px;top:6px;white-space:nowrap" : ""}">
          ${t < start ? "0.00" : clock.toFixed(2)} s${done && FEED ? ` ${MID} purchased, reward ${ln.run.reward}` : ""}</div>
        ${done && !FEED ? `<div style="margin-top:6px;font-size:24px;color:var(--mint)">Purchased ${MID} reward ${ln.run.reward}</div>` : ""}</div>`;
      const n = ln.ctrl.length;
      const bw = FEED ? 90 : 136, bg = FEED ? 10 : 14, bx0 = FEED ? b.x + 34 : b.x + 410, by = FEED ? b.y + 78 : b.y + 34;
      for (let i = 0; i < n; i++) {
        const on = clamp(el * n - i);
        const isJev = ln.ctrl[i] === "jev";
        const col = isJev ? "168,240,200" : "143,163,196";
        const fresh = on > 0 && on < 1 ? 1 : Math.max(0, 1 - (el * n - i - 1) * 1.5);
        html += `<div class="abs" style="left:${bx0 + i * (bw + bg)}px;top:${by}px;width:${bw}px;height:${FEED ? 76 : 88}px;border-radius:14px;
          border:1px solid rgba(${col},${(0.25 + 0.5 * on).toFixed(3)});background:rgba(${col},${(0.06 + (isJev ? 0.62 : 0.32) * on).toFixed(3)});
          box-shadow:0 0 ${(26 * fresh * on).toFixed(1)}px rgba(${col},.55);display:flex;align-items:center;justify-content:center;
          font-size:${FEED ? 21 : 24}px;font-weight:600;color:${isJev && on > 0.5 ? "#06231a" : "rgba(241,243,237,.8)"};${rv(t, 65.3 + 0.02 * i, 0.4, 10, 3)}">
          ${isJev ? `${FEED ? "" : "Jev: "}${S.jev.path[i].action.includes("small") ? "size" : "color"}` : "LLM"}</div>`;
      }
    });
    const stats = [
      ["LLM calls", S.baseline.llm_calls, S.jev.llm_calls],
      ["Tokens", S.baseline.tokens, S.jev.tokens],
      ["Actions", S.baseline.steps, S.jev.steps],
    ];
    stats.forEach((s, i) => {
      const b = FEED ? { x: 60, y: 708 + i * 128, w: 960, h: 112 } : { x: 80 + i * 600, y: 668, w: 560, h: 170 };
      html += `<div class="abs card" style="${box(b)};padding:${FEED ? "0 34px" : "26px 34px"};${FEED ? "display:flex;align-items:center;justify-content:space-between" : ""};${rv(t, 69.7 + 0.18 * i, 0.6, 20, 6)}">
        <div style="font-size:${FEED ? 28 : 28}px;color:var(--muted)">${s[0]}</div>
        <div style="margin-top:${FEED ? 0 : 10}px;font-size:${FEED ? 50 : 58}px;font-weight:600;white-space:nowrap">
          <span style="color:var(--llm)">${fmt(s[1])}</span> <span class="muted" style="font-size:.7em">${ARROW}</span> <span class="mint">${fmt(s[2])}</span></div></div>`;
    });
    html += foot(t, 66.0, `Steps illustrated. Playback ${speed.toFixed(1)}x faster than measured time (${S.baseline.wall_time_seconds} s vs ${S.jev.wall_time_seconds} s), ratio preserved. Source: docs/demos/jev-agent-harness-traces.json.`,
      hx, FEED ? 1104 : 862, FEED ? 960 : 1760);
    return html;
  }

  // ---------- scene 5c: results vary (72-76.5 s) ----------
  const VARY = [
    ["FrozenLake", "10 pairs", -64.4],
    ["WebShop", "3 pairs", -21.4],
    ["Terminal-Bench", "6 pairs", -9.0],
    ["WebArena", "6 pairs", 5.6],
  ];
  function vary(t) {
    const hx = FEED ? 60 : 80;
    let html = kicker(t, 72.1, "Change in LLM calls, LLM + Jev vs. LLM only", hx, FEED ? 92 : 70);
    html += headline(t, 72.3, `Gains vary by task.${FEED ? "<br>" : " "}<span class="mint">Measure first.</span>`, hx, FEED ? 134 : 108, FEED ? 76 : 84, FEED ? 960 : 1760);
    const axis = FEED ? 700 : 1300, scale = FEED ? 6 : 8.2;
    VARY.forEach((v, i) => {
      const y = FEED ? 410 + i * 168 : 300 + i * 132;
      const a = 72.6 + 0.16 * i;
      const g = k(t, a + 0.2, a + 1.0);
      const w = Math.abs(v[2]) * scale * g;
      const neg = v[2] < 0;
      const col = neg ? "var(--mint)" : "var(--amber)";
      html += `<div class="abs" style="left:${hx}px;top:${FEED ? y - 66 : y}px;${rv(t, a, 0.5, 12, 4)}">
        <span style="font-size:${FEED ? 34 : 40}px;font-weight:600">${v[0]}</span>
        <span style="margin-left:14px;font-size:${FEED ? 24 : 26}px;color:var(--subtle)">${v[1]}</span></div>`;
      html += `<div class="abs" style="left:${px(neg ? axis - w : axis)};top:${FEED ? y + 4 : y + 8}px;width:${px(w)};height:${FEED ? 36 : 40}px;border-radius:8px;
        background:${col};opacity:.88;box-shadow:0 0 22px color-mix(in srgb, ${col} 50%, transparent)"></div>`;
      html += `<div class="abs mono" style="left:${FEED ? axis + 60 : 1460}px;top:${FEED ? y : y + 2}px;font-size:${FEED ? 34 : 40}px;color:${col};${rv(t, a + 0.6, 0.5, 6, 3)}">
        ${v[2] > 0 ? "+" : "&minus;"}${Math.abs(v[2]).toFixed(1)}%</div>`;
    });
    const ay0 = FEED ? 330 : 280, ay1 = FEED ? 1080 : 820;
    html += `<div class="abs" style="left:${axis - 1}px;top:${ay0}px;width:2px;height:${ay1 - ay0}px;background:rgba(241,243,237,.35);${rv(t, 72.5, 0.5, 0, 0)}"></div>`;
    html += foot(t, 73.0, "Paired evaluations from the JevAny README and agent harness report. FrozenLake uses GPT-5.6-sol; success rates differ by task.", hx, FEED ? 1110 : 870, FEED ? 960 : 1760);
    return html;
  }

  // ---------- scene 6: call to action (76.5-90 s) ----------
  function cta(t) {
    let html = "";
    // Faint drifting replays behind the lockup.
    const ghosts = ["drone", "peg_insertion", "inbox"];
    ghosts.forEach((g, i) => {
      const w = FEED ? 620 : 760, h = w * 446 / 900;
      const x = (FEED ? [-180, 640, -120] : [-120, 1280, 600])[i] + (t - 76.5) * (i % 2 ? -9 : 9);
      const y = (FEED ? [120, 560, 1060] : [80, 120, 760])[i];
      html += `<div style="opacity:${(0.13 * k(t, 76.8, 78)).toFixed(3)};filter:blur(1px) saturate(.6)">${clip(g, t - 76.5, { x, y, w, h }, "", true, 1)}</div>`;
    });
    html += `<div class="layer" style="background:radial-gradient(ellipse at 50% 45%, rgba(6,10,12,.55), rgba(6,10,12,.92) 70%)"></div>`;
    const L = FEED ? 170 : 150;
    html += `<div class="abs" style="left:0;top:${FEED ? 200 : 150}px;width:${W}px;display:flex;flex-direction:${FEED ? "column" : "row"};align-items:center;justify-content:center;gap:${FEED ? 34 : 40}px">
      ${logo(L, 76.6, t, 1.3)}<div style="${rv(t, 77.0, 0.8, 24, 12)}">${wordmark(FEED ? 120 : 132)}</div></div>`;
    html += `<div class="abs head" style="left:0;width:${W}px;top:${FEED ? 620 : 380}px;text-align:center;font-size:${FEED ? 96 : 104}px;${rv(t, 77.6, 0.8)}">
      Any model.${FEED ? "<br>" : " "}<span class="mint">Your next move.</span></div>`;
    const glow = 0.5 + 0.5 * Math.sin((t - 79) * 2.2);
    html += `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 878 : 560}px;display:flex;justify-content:center;${rv(t, 79.6, 0.7, 20, 6)}">
      <div class="pill" style="padding:${FEED ? "22px 40px" : "24px 48px"};font-size:${FEED ? 46 : 54}px;font-weight:600;border:2px solid rgba(168,240,200,.7);
        background:rgba(168,240,200,.07);box-shadow:0 0 ${(30 + 30 * glow).toFixed(1)}px rgba(168,240,200,.35)"><span>github.com/<span class="mint">SimpleJev/JevAny</span></span></div></div>`;
    html += `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 1022 : 718}px;text-align:center;font-size:${FEED ? 36 : 40}px;color:var(--ink);${rv(t, 80.4, 0.7, 14, 5)}">
      simplejev.github.io/JevAny</div>`;
    html += `<div class="abs" style="left:0;width:${W}px;top:${FEED ? 1086 : 784}px;text-align:center;font-size:${FEED ? 28 : 30}px;color:var(--muted);letter-spacing:.06em;${rv(t, 81.0, 0.7, 10, 4)}">
      Open source ${MID} Apache-2.0 ${MID} Python 3.12+</div>`;
    return html;
  }

  // Scene windows overlap slightly for crossfades: [start, end, fadeIn, fadeOut, grid].
  const SCENES = [
    [hook, 0, 10.1, 0, 0.65, 0.35],
    [title, 9.85, 16.2, 0.4, 0.5, 0.9],
    [why, 15.95, 24.75, 0.4, 0.5, 0.35],
    [api, 24.45, 39.25, 0.4, 0.55, 0.45],
    [range, 38.95, 47.25, 0.35, 0.5, 0.25],
    [train, 46.95, 56.25, 0.35, 0.5, 0.6],
    [loop, 55.95, 64.75, 0.4, 0.5, 0.55],
    [shop, 64.45, 72.2, 0.35, 0.45, 0.35],
    [vary, 71.95, 76.75, 0.35, 0.5, 0.45],
    [cta, 76.45, 90.5, 0.45, 0.01, 0.9],
  ];
  const SWEEPS = [9.75, 24.2, 38.7, 55.7, 76.2];

  function compose(frame) {
    const t = frame / FPS;
    let grid = 0;
    let html = "";
    for (const [fn, a, b, fi, fo, g] of SCENES) {
      if (t < a || t >= b) continue;
      const pin = fi > 0 ? k(t, a, a + fi) : 1, pout = 1 - k(t, b - fo, b, INOUT);
      const o = Math.min(pin, pout);
      grid = Math.max(grid, g * o);
      const scale = (1 + 0.025 * (1 - pin)) * (1 - 0.02 * (1 - pout));
      html += `<div class="layer" style="opacity:${o.toFixed(4)};transform:scale(${scale.toFixed(5)});filter:blur(${((1 - pout) * 10).toFixed(2)}px)">${fn(t)}</div>`;
    }
    const intro = k(t, 0, 0.8);
    return background(t, grid * intro) + html + chapterRail(t) + SWEEPS.map((s) => sweep(t, s)).join("") + grain(frame);
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
    while (cache.size > 90) cache.delete(cache.keys().next().value);
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
