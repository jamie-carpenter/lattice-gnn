"""Check plots for the Lab Report 3 demo deck, from the run folders (no new training).

Usage:
    python scripts/make_demo_figures.py        # -> reports/figures/demo/*.png

Figures:
  evaluate_parity.png   predicted vs FE label on the validation families (main-full), non-PD in orange
  pd_check.png          smallest eigenvalue of the normalized compliance block, and non-PD per family
  training_curves.png   train / validation loss per epoch for main-full, main-full-pd, pilot best
  pilot_trials.png      validation loss of every pilot run, by round, with the agent's decisions
  pilot_targets.png     median modulus error per target: main-full-pd vs the pilot's best run
  outcome_check.png     seed spread behind the thresholds, and each pilot decision's loss drop
  comparison.png        optuna-sub vs rules-sub vs agent-sub (subset, 20 runs each), whatever has run so far

    python scripts/make_demo_figures.py comparison   # only the comparison (while the studies run)

Validation predictions are cached in runs/<run>/val_pred.npz (runs/ is not in git).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import lattice_gnn.report  # noqa: E402,F401  (sets the shared plot style)
from lattice_gnn.evaluate import N_MODULI, normalized_block  # noqa: E402
from lattice_gnn.graph import DISPLAY_NAMES  # noqa: E402

OUT = ROOT / "reports" / "figures" / "demo"
RUNS = ROOT / "runs"
# Reference categorical palette, slots 1-4 in fixed order (dataviz skill, light mode).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#c9c8c2"
PILOT = RUNS / "agent-pilot"
PILOT_BEST = PILOT / "round-3" / "trial-019"


def predictions(run: Path):
    """(pred, true, names) on the run's validation families, cached."""
    cache = run / "val_pred.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["pred"], z["true"], list(z["names"])
    from lattice_gnn.data import load_dataset
    from lattice_gnn.evaluate import predict
    from lattice_gnn.model import LatticeGNN
    from lattice_gnn.splits import apply_split, load_split
    config = json.loads((run / "config.json").read_text())
    model, _ = LatticeGNN.load(run / "model.pt")
    _, val, _ = apply_split(load_dataset([str(ROOT / p) for p in config["paths"]]), load_split(run / "split.json"))
    pred, true, _ = predict(model, val)
    pred, true, names = pred.numpy(), true.numpy(), [g.name for g in val]
    np.savez(cache, pred=pred, true=true, names=np.array(names, dtype=object))
    return pred, true, names


def eval_json(run: Path) -> dict:
    return json.loads((run / "eval.json").read_text())


def min_eig(targets: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(normalized_block(targets[:, N_MODULI:].astype(np.float64)))[:, 0]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def evaluate_parity():
    pred, true, _ = predictions(RUNS / "main-full")
    ev = eval_json(RUNS / "main-full")
    non_pd = min_eig(pred) <= ev["pd"]["threshold"]
    names = list(DISPLAY_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, key in zip(axes, ("Ex", "Gxy", "Poisson xy")):
        k = names.index(key)
        x, y = true[:, k], pred[:, k]
        if k < N_MODULI:                      # log10 moduli -> moduli relative to E_s
            x, y = 10 ** x, 10 ** y
            ax.set_xscale("log"), ax.set_yscale("log")
            err = f"median error {ev['targets'][key]['median_percent_error']:.0f}%"
            unit = " / $E_s$"
        else:
            err = f"median abs. error {ev['targets'][key]['median_abs_error']:.3f}"
            unit = ""
        ax.scatter(x[~non_pd], y[~non_pd], s=6, alpha=0.35, color=BLUE, lw=0, label="PD prediction")
        ax.scatter(x[non_pd], y[non_pd], s=8, alpha=0.8, color=ORANGE, lw=0, label="not PD")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls=(0, (4, 3)))
        ax.set_title(f"{key}: R² {ev['targets'][key]['R2']:.2f}, {err}", fontsize=9)
        ax.set_xlabel(f"FE label{unit}"), ax.set_ylabel(f"prediction{unit}")
    axes[2].legend(loc="upper left", markerscale=2.5)
    fig.suptitle("main-full: 2,032 validation lattices in 67 unseen families; "
                 f"{int(non_pd.sum())} predictions not PD (orange)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "evaluate_parity.png")


def pd_check():
    p0, true, names = predictions(RUNS / "main-full")
    p1, _, _ = predictions(RUNS / "main-full-pd")
    tol = eval_json(RUNS / "main-full")["pd"]["threshold"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    bins = np.linspace(-0.25, 1.0, 51)
    for vals, colour, label in ((min_eig(true), INK_2, "FE labels"), (min_eig(p0), ORANGE, "main-full (no PD layer)"),
                                (min_eig(p1), BLUE, "main-full-pd (PD layer)")):
        n_bad = int((vals <= tol).sum())
        a.hist(np.clip(vals, bins[0], bins[-1]), bins=bins, histtype="step", lw=1.8, color=colour,
               label=f"{label}: {n_bad} not PD")
    a.axvline(0, color=INK, lw=1)
    a.text(-0.02, a.get_ylim()[1] * 0.45, "← not PD", ha="right", va="center", fontsize=8, color=INK)
    a.set_xlabel("smallest eigenvalue of the normalized 3×3 compliance block")
    a.set_ylabel("validation lattices")
    a.set_title("Positive-definiteness check (eval.json 'pd')")
    a.legend(loc="upper right")

    ev0, ev1 = eval_json(RUNS / "main-full"), eval_json(RUNS / "main-full-pd")
    fams = sorted(ev0["families"], key=lambda f: -ev0["families"][f].get("non_pd", 0))[:8][::-1]
    y = np.arange(len(fams))
    b.barh(y + 0.2, [ev0["families"][f]["non_pd"] for f in fams], height=0.38, color=ORANGE, label="main-full")
    b.barh(y - 0.2, [ev1["families"][f]["non_pd"] for f in fams], height=0.38, color=BLUE, label="main-full-pd")
    for yi, f in zip(y, fams):
        b.text(ev0["families"][f]["non_pd"] + 0.4, yi + 0.2, str(ev0["families"][f]["non_pd"]), va="center",
               fontsize=8, color=INK_2)
        b.text(0.4, yi - 0.2, "0", va="center", fontsize=8, color=INK_2)
    b.set_yticks(y, [f"{f} (n={ev0['families'][f]['n']})" for f in fams])
    b.set_xlabel("non-PD predictions")
    b.set_title("Families with the most non-PD predictions")
    b.legend(loc="lower right")
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "pd_check.png")


def training_curves():
    runs = ((RUNS / "main-full", ORANGE, "main-full (default, no PD layer)"),
            (RUNS / "main-full-pd", BLUE, "main-full-pd (default + PD layer)"),
            (PILOT_BEST, AQUA, "pilot best (width 256, 8 layers, PD layer)"))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for run, colour, label in runs:
        h = json.loads((run / "history.json").read_text())
        ep = [e["epoch"] for e in h["epochs"]]
        ax.plot(ep, [e["val_loss"] for e in h["epochs"]], color=colour, lw=2, label=f"{label}: validation")
        ax.plot(ep, [e["train_loss"] for e in h["epochs"]], color=colour, lw=1.2, ls=(0, (3, 2)), alpha=0.8)
        best = h["best_epoch"]
        bv = h["epochs"][best - 1]["val_loss"]
        ax.plot([best], [bv], "o", color=colour, ms=8, mec="white", mew=2)
    ax.plot([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)), label="training loss (dashed)")
    ax.plot([], [], "o", color=INK_2, ms=7, label="best epoch")
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.10", "0.20", "0.50"])
    ax.minorticks_off()
    ax.set_xlabel("epoch (cap 100)")
    ax.set_ylabel("loss (standardized MSE)")
    ax.set_title("Training curves on the full catalog: all three still improving at the epoch cap")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "training_curves.png")


def pilot_trials():
    trials = json.loads((PILOT / "trials.json").read_text())["trials"]
    decisions = [json.loads(l) for l in (PILOT / "decisions.jsonl").read_text().splitlines() if l.strip()]
    ref = eval_json(RUNS / "main-full-pd")["loss"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colours = (BLUE, ORANGE, AQUA, YELLOW)
    for r in range(4):
        t = [x for x in trials if x["round"] == r]
        for wide in (False, True):
            s = [x for x in t if (x["params"].get("hidden") == 512) == wide]
            ax.scatter([x["number"] for x in s], [x["value"] for x in s], s=70 if wide else 55,
                       marker="X" if wide else "o", color=colours[r], edgecolor="white", lw=1.2, zorder=3,
                       label=f"round {r}" if not wide else None)
    best, line = np.inf, []
    for x in trials:
        best = min(best, x["value"])
        line.append(best)
    ax.step([x["number"] for x in trials], line, where="post", color=INK, lw=1.5, label="best so far")
    ax.axhline(ref, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(19.4, ref * 1.03, f"main-full-pd (default) {ref:.3f}", ha="right", fontsize=8, color=INK_2)
    ax.scatter([], [], marker="X", s=70, color=INK_2, label="width 512")
    short = {0: "pd_head on", 1: "retune: width {256, 512}", 2: "retune: layers 5–8"}
    for d in decisions:
        xpos = 5 * d["round"] + 4.5
        ax.axvline(xpos, color=GRID, lw=1.2, zorder=1)
        ax.text(xpos + 0.15, 0.72, f"D{d['round'] + 1}: {short[d['round']]}\n→ {d['outcome']}",
                transform=ax.get_xaxis_transform(), fontsize=8, va="top", color=INK)
    ax.set_yscale("log")
    ax.set_yticks([0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8], ["0.08", "0.10", "0.15", "0.20", "0.30", "0.50", "0.80"])
    ax.minorticks_off()
    ax.set_xticks(range(20))
    ax.set_xlabel("training run (5 per round; the agent decides between rounds)")
    ax.set_ylabel("validation loss")
    ax.set_title("agent-pilot: 20 full-catalog runs, 3 agent decisions")
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, "pilot_trials.png")


def pilot_targets():
    a, b = eval_json(RUNS / "main-full-pd"), eval_json(PILOT_BEST)
    keys = [k for k in DISPLAY_NAMES][:N_MODULI]
    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    va = [a["targets"][k]["median_percent_error"] for k in keys]
    vb = [b["targets"][k]["median_percent_error"] for k in keys]
    ax.barh(y + 0.2, va, height=0.38, color=BLUE, label=f"main-full-pd (loss {a['loss']:.3f})")
    ax.barh(y - 0.2, vb, height=0.38, color=AQUA, label=f"pilot best (loss {b['loss']:.3f})")
    for yi, v1, v2 in zip(y, va, vb):
        ax.text(v1 + 0.4, yi + 0.2, f"{v1:.1f}", va="center", fontsize=8, color=INK_2)
        ax.text(v2 + 0.4, yi - 0.2, f"{v2:.1f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, keys)
    ax.set_xlabel("median % error on the validation families")
    ax.set_title("Moduli error per target (couplings: "
                 f"{a['overall']['median_abs_error_couplings']:.3f} → {b['overall']['median_abs_error_couplings']:.3f})",
                 fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "pilot_targets.png")


def outcome_check():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    groups = [("full, no PD", "main-full", "main-full-s1"), ("full, PD", "main-full-pd", "main-full-pd-s1"),
              ("subset, no PD", "main-2000", "main-2000-s1"), ("subset, PD", "main-2000-pd", "main-2000-pd-s1")]
    for i, (label, s0, s1) in enumerate(groups):
        v = [eval_json(RUNS / s0)["loss"], eval_json(RUNS / s1)["loss"]]
        colour = BLUE if label.startswith("full") else ORANGE
        a.plot([i, i], v, color=colour, lw=2)
        a.scatter([i, i], v, color=colour, s=50, edgecolor="white", lw=1.2, zorder=3)
        a.text(i + 0.1, np.mean(v), f"gap {abs(v[1] - v[0]):.4f}", fontsize=8, va="center", color=INK_2)
    a.set_xticks(range(4), [g[0] for g in groups], fontsize=8)
    a.set_xlim(-0.4, 3.9)
    a.set_ylabel("validation loss, seeds 0 and 1")
    a.set_title("Seed spread → thresholds 0.01 (full), 0.02 (subset)")

    decisions = [json.loads(l) for l in (PILOT / "decisions.jsonl").read_text().splitlines() if l.strip()]
    drops = [d["outcome_metrics"]["value_before"] - d["outcome_metrics"]["value_after"] for d in decisions]
    labels = ["D1 pd_head", "D2 retune", "D3 retune"]
    colours = [AQUA if d["outcome"] == "improved" else ORANGE for d in decisions]
    b.bar(range(3), drops, color=colours, width=0.55)
    b.axhline(0.01, color=INK, lw=1.2, ls=(0, (4, 3)), label="threshold 0.01 (seed spread)")
    b.legend(loc="upper right")
    for i, (d, dec) in enumerate(zip(drops, decisions)):
        b.text(i, max(d, 0.0102) + 0.0008, f"{d:.4f}\n{dec['outcome']}", ha="center", va="bottom", fontsize=8, color=INK)
    b.text(0, drops[0] / 2, "also non-PD\n0.088 → 0", ha="center", va="center", fontsize=7.5, color="white")
    b.set_xticks(range(3), labels)
    b.set_ylim(0, max(drops) * 1.35)
    b.set_ylabel("drop in best validation loss")
    b.set_title("Pilot decisions: loss drop in the next round")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "outcome_check.png")


CONDITIONS = (("optuna-sub", "Optuna alone", BLUE), ("rules-sub", "rule table in the loop", ORANGE),
              ("agent-sub", "LLM agent in the loop", AQUA))


def comparison():
    """The three subset studies at equal budget: best-so-far and final best (whatever has run so far)."""
    studies = [(RUNS / name, label, colour) for name, label, colour in CONDITIONS
               if (RUNS / name / "trials.json").exists()]
    if not studies:
        print("comparison: no subset study yet")
        return
    ref = eval_json(RUNS / "main-2000-pd")["loss"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.7, 1]})
    finals = []
    for folder, label, colour in studies:
        trials = json.loads((folder / "trials.json").read_text())["trials"]
        x = np.arange(1, len(trials) + 1)
        v = np.array([t["value"] for t in trials])
        best = np.minimum.accumulate(v)
        a.scatter(x, v, s=22, color=colour, alpha=0.45, lw=0)
        a.step(x, best, where="post", color=colour, lw=2.2, label=f"{label} ({len(trials)}/20 runs)")
        i = int(np.argmin(v))
        finals.append((label, colour, trials[i], len(trials)))
    for xd in (5.5, 10.5, 15.5):
        a.axvline(xd, color=GRID, lw=1, zorder=0)
    a.text(5.6, 0.97, "decision points\n(loops only)", transform=a.get_xaxis_transform(), fontsize=7.5,
           va="top", color=INK_2)
    a.axhline(ref, color=INK_2, lw=1, ls=(0, (4, 3)))
    a.text(20.3, ref, f"default + PD layer\n{ref:.3f}", fontsize=7.5, va="center", color=INK_2)
    a.set_yscale("log")
    a.set_yticks([0.15, 0.2, 0.3, 0.5, 0.8], ["0.15", "0.20", "0.30", "0.50", "0.80"])
    a.minorticks_off()
    a.set_xlim(0.5, 20.5)
    a.set_xticks([1, 5, 10, 15, 20])
    a.set_xlabel("training runs used (same budget, split and sampler seed)")
    a.set_ylabel("validation loss (dots: runs; line: best so far)")
    a.set_title("2,000-lattice subset, 20 runs per condition")
    a.legend(loc="upper right", fontsize=8)

    y = np.arange(len(finals))[::-1]
    for yi, (label, colour, t, n) in zip(y, finals):
        b.barh(yi, t["value"], color=colour, height=0.55)
        note = f"{t['value']:.3f}; non-PD {t['non_pd_rate']:.3f}; moduli " \
               f"{t['overall']['median_percent_error_moduli']:.0f}%" + ("" if n == 20 else f" ({n}/20)")
        b.text(0.005, yi - 0.4, note, fontsize=7.5, va="center", color=INK_2)
    b.set_yticks(y, [f[0] for f in finals], fontsize=8)
    b.set_ylim(-0.8, len(finals) - 0.4)
    b.set_xlabel("best validation loss")
    b.set_title("Best run per condition")
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "comparison.png")


def main():
    if sys.argv[1:] == ["comparison"]:
        comparison()
        return
    evaluate_parity()
    pd_check()
    training_curves()
    pilot_trials()
    pilot_targets()
    outcome_check()
    comparison()


if __name__ == "__main__":
    main()
