"""Plot one Optuna study for Lab Report 2 (template item C3).

Usage:
    python scripts/make_study_figure.py runs/optuna-2000 --reference runs/main-2000
        # -> reports/figures/<study>_trials.{pdf,png}

Shows the validation loss of every training run in the study, the best value so far, and,
optionally, the validation loss of a reference run on the same data and split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures"
INK, MUTED, BLUE, ORANGE = "#1f2328", "#57606a", "#2a78d6", "#eb6834"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", help="study folder with trials.json")
    parser.add_argument("--reference", help="run folder whose eval.json loss is drawn as a line")
    args = parser.parse_args()

    study = Path(args.study)
    trials = json.loads((study / "trials.json").read_text())["trials"]
    numbers = [t["number"] for t in trials]
    values = [t["value"] for t in trials]
    best = [min(values[:i + 1]) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.scatter(numbers, values, s=18, color=BLUE, zorder=3, label="training run")
    ax.step(numbers, best, where="post", color=INK, lw=1.2, label="best so far")
    if args.reference:
        ref = json.loads((Path(args.reference) / "eval.json").read_text())
        ax.axhline(ref["loss"], color=ORANGE, ls=(0, (4, 3)), lw=1.1,
                   label=f"{Path(args.reference).name} (untuned)")
    ax.set_yscale("log")
    ticks = [0.2, 0.3, 0.4, 0.6, 0.8]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.19, 0.9)
    ax.set_xlabel("training run in study", fontsize=8)
    ax.set_ylabel("validation loss (log scale)", fontsize=8)
    ax.set_xticks(numbers)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{study.name}_trials.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / (study.name + '_trials.pdf')} and .png")


if __name__ == "__main__":
    main()
