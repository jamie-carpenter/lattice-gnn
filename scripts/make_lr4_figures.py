"""Plots for the Lab Report 4 demo deck, from the run folders (no new training).

Usage:
    python scripts/make_lr4_figures.py            # every figure whose runs exist -> reports/figures/lr4/*.png
    python scripts/make_lr4_figures.py probes     # one figure

Figures:
  probes.png   validation loss per epoch of the four learning-rate probes (runs/probe-lr), with
               each run's final non-PD rate; the evidence behind the collapse and two_causes cases
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import lattice_gnn.report  # noqa: E402,F401  (sets the shared plot style)

OUT = ROOT / "reports" / "figures" / "lr4"
RUNS = ROOT / "runs"
# Same categorical palette and order as the Lab Report 3 deck (scripts/make_demo_figures.py).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#1f2328", "#57606a"


def probes():
    runs = [("lr1e-1", "lr 1e-1", BLUE), ("lr3e-2", "lr 3e-2", ORANGE),
            ("lr5e-3", "lr 5e-3", AQUA), ("lr1e-5", "lr 1e-5", YELLOW)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    nonpd = []
    for folder, label, color in runs:
        run = RUNS / "probe-lr" / folder
        hist = json.loads((run / "history.json").read_text())["epochs"]
        ev = json.loads((run / "eval.json").read_text())
        x = [h["epoch"] for h in hist]
        y = [h["val_loss"] for h in hist]
        ax.plot(x, y, color=color, lw=2, label=label)
        if folder in ("lr5e-3", "lr1e-5"):
            ax.annotate(f"{label}: non-PD {ev['pd']['non_pd_rate']:.3f}", (x[-1], y[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
        else:
            nonpd.append(ev["pd"]["non_pd_rate"])
    ax.text(42, 1.075, f"lr 1e-1, 3e-2: collapsed, stopped early; non-PD {max(nonpd):.3f}",
            fontsize=8, color=INK)
    ax.axhline(0.22, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(102, 0.245, "good runs, Lab Report 3", fontsize=8, color=MUTED)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (standardized MSE)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.15)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=4, bbox_to_anchor=(0.0, 0.02))
    ax.set_title("Collapsed runs (loss near 1) give no non-PD predictions", fontsize=10, loc="left")
    return fig


FIGURES = {"probes": probes}


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names or FIGURES:
        fig = FIGURES[name]()
        fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {OUT / name}.png")


if __name__ == "__main__":
    main(sys.argv[1:])
