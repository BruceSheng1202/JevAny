#!/usr/bin/env python3
"""Render the README comparison from the unrounded release results.

Requires matplotlib and cairosvg. Run from the repository root with:
    python scripts/plot_evaluation_overview.py
"""

import copy
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/model-family-v2.json"
OUT = ROOT / "docs/evaluation-overview.svg"
LOGOS = ROOT / "docs/model-logos"
INK, MUTED, RULE = "#213248", "#64748B", "#E4E9EF"
OURS, BASELINE = "#278577", "#8493A6"
WIDTH, HEIGHT = 1440, 820
LOGO_X, LOGO_SIZE = 86, 42
FIRST_Y, ROW_STEP = 218, 57
LABELS = {
    "JevAny-Gemma-4B-LoRA": "JevAny · Gemma 4B",
    "JevAny-Qwen3.5-4B-LoRA": "JevAny · Qwen3.5 4B",
    "JevAny-Qwen3.5-4B-Direct-Token-LoRA": "JevAny · Qwen3.5 4B",
    "JevAny-Qwen3.8-27B-LoRA": "JevAny · Qwen3.8 27B",
    "JevAny-Muse-Glimmer-30B-LoRA": "JevAny · Muse Glimmer 30B",
}
MODEL_LOGOS = {
    "JevAny-Gemma-4B-LoRA": "jevany-gemma.svg",
    "JevAny-Qwen3.5-4B-LoRA": "jevany-qwen.svg",
    "JevAny-Qwen3.5-4B-Direct-Token-LoRA": "jevany-qwen.svg",
    "JevAny-Qwen3.8-27B-LoRA": "jevany-qwen.svg",
    "JevAny-Muse-Glimmer-30B-LoRA": "jevany-muse.svg",
    "Jev 1.13.0": "typesafe.png",
    "Kev-4B": "kev.svg",
    "Kev-27B": "kev.svg",
    "Laya": "laya.svg",
}


def write_checkpoint_logos() -> None:
    """Use the README project mark with a base-model badge at bottom right."""
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    project = ET.parse(ROOT / "docs/title.svg").find(".//*[@id='jevany-logo']")
    for family, filename in (
        ("qwen", "qwen-color.svg"),
        ("gemma", "gemma-color.svg"),
        ("muse", "meta-color.svg"),
    ):
        root = ET.Element(f"{{{ns}}}svg", {
            "viewBox": "0 0 48 48", "width": "48", "height": "48",
        })
        ET.SubElement(root, f"{{{ns}}}title").text = f"JevAny · {family.title()}"
        mark = ET.SubElement(root, f"{{{ns}}}svg", {
            "viewBox": "0 0 512 512", "width": "40", "height": "40",
        })
        geometry = copy.deepcopy(project)
        geometry.attrib.pop("transform")
        mark.append(geometry)
        ET.SubElement(root, f"{{{ns}}}circle", {
            "cx": "37", "cy": "37", "r": "10.5",
            "fill": "#ffffff", "stroke": "#dce3ec", "stroke-width": "0.7",
        })
        base = ET.fromstring((LOGOS / filename).read_text())
        base.attrib.pop("style", None)
        base.attrib.update(x="29", y="29", width="16", height="16")
        root.append(base)
        path = LOGOS / f"jevany-{family}.svg"
        path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def load_models(source: Path = SOURCE) -> list[dict]:
    """Load every release and baseline; reject incomplete accuracy records."""
    release = json.loads(source.read_text())
    models = []
    seen = set()
    for group in ("released_models", "references"):
        for record in release[group]:
            ours = group == "released_models"
            identifier = record["repository"] if ours else record["model"]
            if identifier in seen:
                raise ValueError(f"Duplicate model: {identifier}")
            seen.add(identifier)
            scores = {}
            for short, key in (
                ("transfer", "transfer_v9_accuracy"),
                ("jevbench", "jevbench_public_accuracy"),
            ):
                score = record.get(key)
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError(f"{identifier}: {key} must be a number in [0, 1]")
                scores[short] = score * 100
            models.append({
                "id": identifier,
                "label": LABELS[identifier.split("/")[-1]] if ours else identifier,
                "logo": MODEL_LOGOS[identifier.split("/")[-1]],
                "detail": (
                    f"LoRA · {'Direct-token' if record['readout'] == 'direct-token' else 'Pointer'}"
                    if ours else ""
                ),
                "ours": ours,
                **scores,
                "mean": (scores["transfer"] + scores["jevbench"]) / 2,
            })
    return sorted(models, key=lambda model: -model["mean"])


def draw(models: list[dict]):
    """Draw zero-based bars and their two input scores on a shared row."""
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "svg.hashsalt": "jevany-evaluation-overview",
        "text.color": INK,
    })
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()

    def text(x, y, value, size=20, color=INK, weight="normal", ha="left"):
        return ax.text(
            x, y, value, fontsize=size * 0.72, color=color,
            weight=weight, ha=ha, va="center",
        )

    def line(x1, y1, x2, y2, color=RULE, width=0.8):
        ax.plot([x1, x2], [y1, y2], color=color, lw=width, zorder=0)

    text(48, 49, "Overall decision accuracy", 36, weight="bold")
    text(48, 91, "JevAny checkpoints and baselines", 18, MUTED)
    for x, label, color in (
        (1074, "JevAny", OURS), (1242, "Baselines", BASELINE),
    ):
        ax.add_patch(Rectangle((x, 45), 14, 14, color=color, lw=0))
        text(x + 25, 52, label, 17, MUTED)
    line(48, 119, 1392, 119)

    text(48, 154, "#", 16, MUTED)
    text(136, 154, "Model", 17, weight="bold")
    text(508, 154, "Mean accuracy (%)", 17, weight="bold")
    text(1190, 148, "Kev Transfer-v9", 16, weight="bold", ha="center")
    text(1330, 148, "JevBench", 18, weight="bold", ha="center")
    text(1190, 174, "1,046 decisions", 13, MUTED, ha="center")
    text(1330, 174, "231 public-dev items", 13, MUTED, ha="center")

    bar_x, bar_width = 508, 560
    last_y = FIRST_Y + ROW_STEP * (len(models) - 1)
    ax.add_patch(Rectangle((34, FIRST_Y - 27), 1372, 54, color="#EDF7F4", lw=0, zorder=-1))
    ax.add_patch(Rectangle((34, FIRST_Y - 27), 4, 54, color=OURS, lw=0))
    for tick in range(0, 101, 20):
        x = bar_x + bar_width * tick / 100
        line(x, 190, x, last_y + 31)
        text(x, last_y + 53, str(tick), 14, MUTED, ha="center")
    line(1110, 137, 1110, last_y + 31)

    images = {}
    for filename in {model["logo"] for model in models}:
        path = LOGOS / filename
        data = (
            cairosvg.svg2png(url=str(path), output_width=144, output_height=144)
            if path.suffix == ".svg" else path.read_bytes()
        )
        images[filename] = plt.imread(io.BytesIO(data), format="png")

    for rank, model in enumerate(models, start=1):
        y = FIRST_Y + (rank - 1) * ROW_STEP
        color = OURS if model["ours"] else BASELINE
        weight = "bold" if rank == 1 else "normal"
        text(53, y, str(rank), 17, MUTED, ha="center")
        image = ax.imshow(
            images[model["logo"]],
            extent=(LOGO_X, LOGO_X + LOGO_SIZE, y + LOGO_SIZE / 2, y - LOGO_SIZE / 2),
            aspect="auto", interpolation="lanczos", zorder=3,
        )
        image.set_gid(f"model-logo-{rank}")
        text(136, y - 9 if model["detail"] else y, model["label"], 20, weight=weight)
        if model["detail"]:
            text(136, y + 15, model["detail"], 14, MUTED)
        end = bar_x + bar_width * model["mean"] / 100
        ax.add_patch(Rectangle((bar_x, y - 13), end - bar_x, 26, color=color, lw=0))
        text(end + 12, y, f'{model["mean"]:.2f}', 21, weight="bold")
        for key, x in (("transfer", 1190), ("jevbench", 1330)):
            best = model[key] == max(item[key] for item in models)
            text(x, y, f'{model[key]:.2f}', 21,
                 weight="bold" if best else "normal", ha="center")

    line(48, 755, 1392, 755)
    text(48, 788, "Mean = (Kev Transfer-v9 + JevBench) / 2", 18)
    return fig


def embed_vector_logos(svg: Path, models: list[dict]) -> None:
    """Retain the source SVG geometry and gradients in the editable export."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(svg)
    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    for rank, model in enumerate(models, start=1):
        source = LOGOS / model["logo"]
        if source.suffix != ".svg":
            continue
        original = root.find(f".//*[@id='model-logo-{rank}']")
        vector = ET.fromstring(source.read_text())
        identifiers = {
            node.attrib["id"]: f"logo-{rank}-{node.attrib['id']}"
            for node in vector.iter() if "id" in node.attrib
        }
        for node in vector.iter():
            for key, value in list(node.attrib.items()):
                if key == "id":
                    node.set(key, identifiers[value])
                else:
                    for old, new in identifiers.items():
                        value = value.replace(f"url(#{old})", f"url(#{new})")
                    node.set(key, value)
        y = FIRST_Y + (rank - 1) * ROW_STEP
        vector.attrib.pop("style", None)
        vector.attrib.update(
            id=f"model-logo-{rank}", x=str(LOGO_X * 0.72),
            y=str((y - LOGO_SIZE / 2) * 0.72),
            width=str(LOGO_SIZE * 0.72), height=str(LOGO_SIZE * 0.72),
        )
        parent = parents[original]
        index = list(parent).index(original)
        parent.remove(original)
        parent.insert(index, vector)
    tree.write(svg, encoding="utf-8", xml_declaration=True)


def main() -> None:
    models = load_models()
    write_checkpoint_logos()
    fig = draw(models)
    description = (
        "All models in results/model-family-v2.json, ranked by the equal-weight "
        "mean of Kev Transfer-v9 and JevBench public-development accuracy. "
        + " ".join(f'{m["label"]} ({m["detail"]}): {m["mean"]:.2f}.' for m in models)
    )
    fig.savefig(
        OUT,
        metadata={"Date": None, "Title": "Overall decision accuracy", "Description": description},
    )
    embed_vector_logos(OUT, models)
    OUT.write_text("\n".join(line.rstrip() for line in OUT.read_text().splitlines()) + "\n")
    plt.close(fig)
    for model in models:
        print(f'{model["mean"]:.2f}  {model["label"]} ({model["detail"]})')


if __name__ == "__main__":
    main()
