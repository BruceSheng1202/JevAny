#!/usr/bin/env python3
"""Render a draft model-family comparison from the unrounded release results.

Requires matplotlib. Run from any directory with:
    python scripts/plot_evaluation_overview.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/model-family-v2.json"
OUT = ROOT / "docs/drafts/evaluation-overview"
INK, MUTED, RULE = "#213248", "#64748B", "#E4E9EF"
OURS, BASELINE = "#278577", "#8493A6"
WIDTH, HEIGHT = 1440, 860
LABELS = {
    "JevAny-Gemma-4B-LoRA": "JevAny · Gemma 4B",
    "JevAny-Qwen3.5-4B-LoRA": "JevAny · Qwen3.5 4B",
    "JevAny-Qwen3.5-4B-Direct-Token-LoRA": "JevAny · Qwen3.5 4B",
    "JevAny-Qwen3.8-27B-LoRA": "JevAny · Qwen3.8 27B",
    "JevAny-Muse-Glimmer-30B-LoRA": "JevAny · Muse Glimmer 30B",
}


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
                "detail": (
                    f"LoRA · {'Direct-token' if record['readout'] == 'direct-token' else 'Pointer'}"
                    if ours else "Baseline"
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
        "pdf.fonttype": 42,
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

    ours_count = sum(model["ours"] for model in models)
    text(48, 49, "Overall decision accuracy", 36, weight="bold")
    text(
        48, 91,
        f"{ours_count} JevAny checkpoints · {len(models) - ours_count} baselines · 2 evaluation suites",
        18, MUTED,
    )
    for x, label, color in (
        (1074, "JevAny", OURS), (1242, "Baselines", BASELINE),
    ):
        ax.add_patch(Rectangle((x, 45), 14, 14, color=color, lw=0))
        text(x + 25, 52, label, 17, MUTED)
    line(48, 119, 1392, 119)

    text(48, 154, "#", 16, MUTED)
    text(90, 154, "Model", 17, weight="bold")
    text(508, 154, "Mean accuracy (%)", 17, weight="bold")
    text(1190, 148, "Transfer", 18, weight="bold", ha="center")
    text(1330, 148, "JevBench", 18, weight="bold", ha="center")
    text(1190, 174, "1,046 decisions", 13, MUTED, ha="center")
    text(1330, 174, "231 public-dev items", 13, MUTED, ha="center")

    first_y, step, bar_x, bar_width = 218, 57, 508, 560
    last_y = first_y + step * (len(models) - 1)
    ax.add_patch(Rectangle((34, first_y - 27), 1372, 54, color="#EDF7F4", lw=0, zorder=-1))
    ax.add_patch(Rectangle((34, first_y - 27), 4, 54, color=OURS, lw=0))
    for tick in range(0, 101, 20):
        x = bar_x + bar_width * tick / 100
        line(x, 190, x, last_y + 31)
        text(x, last_y + 53, str(tick), 14, MUTED, ha="center")
    line(1110, 137, 1110, last_y + 31)

    for rank, model in enumerate(models, start=1):
        y = first_y + (rank - 1) * step
        color = OURS if model["ours"] else BASELINE
        weight = "bold" if rank == 1 else "normal"
        text(53, y, str(rank), 17, MUTED, ha="center")
        text(90, y - 9, model["label"], 21, weight=weight)
        text(90, y + 15, model["detail"], 14, MUTED)
        end = bar_x + bar_width * model["mean"] / 100
        ax.add_patch(Rectangle((bar_x, y - 13), end - bar_x, 26, color=color, lw=0))
        text(end + 12, y, f'{model["mean"]:.2f}', 21, weight="bold")
        for key, x in (("transfer", 1190), ("jevbench", 1330)):
            best = model[key] == max(item[key] for item in models)
            text(x, y, f'{model[key]:.2f}', 21,
                 weight="bold" if best else "normal", ha="center")

    line(48, 755, 1392, 755)
    text(48, 788, "Mean = 50% Transfer + 50% JevBench. Higher is better.", 18)
    text(
        48, 820,
        "Draft summary of the evaluated models. JevBench uses public development items; "
        "Jev 1.13.0 uses its published JevBench result.",
        14, MUTED,
    )
    return fig


def write_preview(models: list[dict]) -> None:
    template = (OUT / "preview.template.html").read_text()
    (OUT / "preview.html").write_text(template.replace(
        "/* MODEL_DATA */ []", json.dumps(models, ensure_ascii=False, allow_nan=False),
    ))


def main() -> None:
    models = load_models()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = draw(models)
    description = (
        "All models in results/model-family-v2.json, ranked by the equal-weight "
        "mean of Transfer and JevBench public-development accuracy. "
        + " ".join(f'{m["label"]} ({m["detail"]}): {m["mean"]:.2f}.' for m in models)
    )
    fig.savefig(
        OUT / "overview.svg",
        metadata={"Date": None, "Title": "Overall decision accuracy", "Description": description},
    )
    svg = OUT / "overview.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUT / "overview.png", dpi=200, metadata={"Description": description})
    fig.savefig(
        OUT / "overview.pdf",
        metadata={"CreationDate": None, "ModDate": None, "Title": "Overall decision accuracy"},
    )
    plt.close(fig)
    write_preview(models)
    for model in models:
        print(f'{model["mean"]:.2f}  {model["label"]} ({model["detail"]})')


if __name__ == "__main__":
    main()
