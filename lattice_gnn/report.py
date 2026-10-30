"""Describe one training run as tables and figures, without interpreting it.

Reads runs/<id>/config.json, history.json and eval.json and writes runs/<id>/report/:
  report.md         settings, losses, per-target, PD and per-family tables
  loss.png          train and validation loss per epoch, and the learning rate
  targets.png       R² and typical error per target
  families.png      median modulus error per validation family

Usage:
    python -m lattice_gnn.report runs/<id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Light-mode reference palette (dataviz skill): two categorical slots and the text inks.
SERIES = ("#2a78d6", "#eb6834")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SETTINGS = ("paths", "split_file", "split", "epochs", "patience", "batch_size", "lr",
            "weight_decay", "hidden", "layers", "augment", "seed", "git_commit", "torch")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "font.size": 9,
})


def load_run(run: Path) -> tuple[dict, dict, dict]:
    read = lambda name: json.loads((run / name).read_text())  # noqa: E731
    return read("config.json"), read("history.json"), read("eval.json")


def plot_loss(history: dict, out: Path):
    epochs = history["epochs"]
    x = [e["epoch"] for e in epochs]
    best = history["best_epoch"]
    fig, (ax, ax_lr) = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    for key, label, color in (("train_loss", "train", SERIES[0]), ("val_loss", "validation", SERIES[1])):
        y = [e[key] for e in epochs]
        ax.plot(x, y, color=color, label=label)
        ax.annotate(f"{label} {y[-1]:.3f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.axvline(best, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate(f"best epoch {best}", (best, 1), xycoords=("data", "axes fraction"),
                xytext=(3, -10), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_ylabel("loss (standardized MSE, log scale)")
    ax.set_title("Loss per epoch")
    ax.legend(loc="lower left")
    ax_lr.plot(x, [e["lr"] for e in epochs], color=SERIES[0])
    ax_lr.set_yscale("log")
    ax_lr.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0e"))
    ax_lr.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_lr.set_yticks(sorted({e["lr"] for e in epochs}))
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _hbars(ax, names, values, fmt: str, xlabel: str, title: str):
    y = range(len(names))
    ax.barh(y, values, color=SERIES[0], height=0.6)
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in zip(y, values):
        ax.annotate(fmt.format(v), (v, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.margins(x=0.15)


def plot_targets(result: dict, out: Path):
    targets = result["targets"]
    names = list(targets)
    moduli = [n for n in names if "median_percent_error" in targets[n]]
    couplings = [n for n in names if "median_abs_error" in targets[n]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1, 1]})
    _hbars(axes[0], names, [targets[n]["R2"] for n in names], "{:.3f}", "R²", "R² per target")
    _hbars(axes[1], moduli, [targets[n]["median_percent_error"] for n in moduli], "{:.1f} %",
           "median error (%)", "Moduli: median % error")
    _hbars(axes[2], couplings, [targets[n]["median_abs_error"] for n in couplings], "{:.3f}",
           "median absolute error", "Couplings: median abs error")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_families(result: dict, out: Path):
    fams = result["families"]
    names = sorted(fams, key=lambda f: fams[f]["median_percent_error_moduli"], reverse=True)
    values = [fams[f]["median_percent_error_moduli"] for f in names]
    labels = [f"{f}  (n={fams[f]['n']})" for f in names]
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.22 * len(names)))
    _hbars(ax, labels, values, "{:.0f} %", "median modulus error (%)",
           f"Median modulus error per validation family ({len(names)} families)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _table(header, rows) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def markdown(run: Path, config: dict, history: dict, result: dict) -> str:
    epochs = history["epochs"]
    best = epochs[history["best_epoch"] - 1]
    last = epochs[-1]
    pd = result["pd"]
    lines = [f"# Run `{run.name}`", "",
             "Generated by `python -m lattice_gnn.report` from config.json, history.json and eval.json.",
             "Loss is the standardized MSE over the 9 targets; 1.0 equals predicting the "
             "training-set mean for every lattice.", "", "## Settings", ""]
    # With a split file, `split` and the fractions are unused defaults; showing them would misstate the split.
    unused = {"split", "val_frac", "test_frac"} if config.get("split_file") else {"split_file"}
    lines += _table(["setting", "value"], [(k, f"`{config[k]}`") for k in SETTINGS
                                           if k in config and k not in unused])
    lines += ["", "## Training", ""]
    lines += _table(["", "epoch", "train loss", "val loss", "learning rate"], [
        ("best (saved model)", best["epoch"], f"{best['train_loss']:.4f}", f"{best['val_loss']:.4f}",
         f"{best['lr']:.1e}"),
        ("last", last["epoch"], f"{last['train_loss']:.4f}", f"{last['val_loss']:.4f}", f"{last['lr']:.1e}"),
    ])
    lines += ["", f"Epochs run: {len(epochs)} of {config.get('epochs', '?')}.", "",
              "![loss](loss.png)", "", f"## Validation set ({result['part']})", "",
              f"{result['n_lattices']} lattices in {result['n_families']} families. "
              f"Loss of the saved model: {result['loss']:.4f}.", ""]
    lines += _table(["target", "R²", "typical error"], [
        (n, f"{m['R2']:.3f}", f"{m['median_percent_error']:.1f} % (median)" if "median_percent_error" in m
         else f"{m['median_abs_error']:.3f} (median abs)") for n, m in result["targets"].items()])
    o = result["overall"]
    lines += ["", f"All moduli together: median error {o['median_percent_error_moduli']:.1f} %. "
              f"All couplings together: median absolute error {o['median_abs_error_couplings']:.3f}.", "",
              "![targets](targets.png)", "", "## Positive definiteness", ""]
    lines += _table(["", "count", "rate"], [
        ("predictions not positive definite", pd["non_pd"], f"{pd['non_pd_rate']:.1%}"),
        ("true labels not positive definite", pd["true_non_pd"],
         f"{pd['true_non_pd'] / max(result['n_lattices'], 1):.1%}")])
    lines += ["", f"Threshold: smallest eigenvalue of the normalized compliance block > {pd['threshold']:.2e}.",
              "", "## Families", "", "Sorted by median modulus error, highest first.", "",
              "![families](families.png)", ""]
    fams = result["families"]
    order = sorted(fams, key=lambda f: fams[f]["median_percent_error_moduli"], reverse=True)
    lines += _table(["family", "n", "median modulus error", "median coupling abs error", "non-PD"], [
        (f, fams[f]["n"], f"{fams[f]['median_percent_error_moduli']:.1f} %",
         f"{fams[f]['median_abs_error_couplings']:.3f}", fams[f]["non_pd"]) for f in order])
    return "\n".join(lines) + "\n"


def build_report(run: str | Path) -> Path:
    run = Path(run)
    config, history, result = load_run(run)
    out = run / "report"
    out.mkdir(exist_ok=True)
    plot_loss(history, out / "loss.png")
    plot_targets(result, out / "targets.png")
    plot_families(result, out / "families.png")
    (out / "report.md").write_text(markdown(run, config, history, result), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="Tables and figures for one training run.")
    parser.add_argument("run", help="run folder, e.g. runs/<id>")
    args = parser.parse_args()
    out = build_report(args.run)
    print(f"Wrote {out / 'report.md'}, loss.png, targets.png, families.png")


if __name__ == "__main__":
    main()
