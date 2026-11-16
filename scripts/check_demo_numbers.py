"""Recompute every number quoted in the demo deck's tables from the saved run files.

Read-only: it opens JSON files under runs/ and config/ and prints, for each claim, the value
the deck states, the value in the file, and the file it came from. Nothing is retrained or
rewritten. Run from the repo root:

    python scripts/check_demo_numbers.py
"""
import json
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
rows = []


def load(path):
    return json.loads((ROOT / path).read_text())


def check(slide, claim, stated, actual, source, tol=None):
    """tol: allowed |stated - actual|; default is half a unit in the last stated digit."""
    if tol is None:
        s = repr(stated)
        tol = 0.5 * 10 ** -(len(s.split(".")[1]) if "." in s else 0)
    ok = abs(stated - actual) <= tol + 1e-12
    rows.append((slide, claim, stated, actual, source, ok))


# Evaluate
e = load("runs/main-full/eval.json")
src = "runs/main-full/eval.json"
check("Evaluate", "validation lattices", 2032, e["n_lattices"], src)
check("Evaluate", "validation families", 67, e["n_families"], src)
check("Evaluate", "loss", 0.112, e["loss"], src)
check("Evaluate", "moduli median error (%)", 30.8, e["overall"]["median_percent_error_moduli"], src)
check("Evaluate", "couplings median error", 0.046, e["overall"]["median_abs_error_couplings"], src)
check("Evaluate", "non-PD predictions", 194, e["pd"]["non_pd"], src)
check("Evaluate", "non-PD rate (%)", 9.5, 100 * e["pd"]["non_pd_rate"], src)
check("Evaluate", "non-PD FE labels", 0, e["pd"]["true_non_pd"], src)
check("Evaluate", "ort_Z03.7 size", 173, e["families"]["ort_Z03.7"]["n"], src)
check("Evaluate", "ort_Z03.7 non-PD (no PD layer)", 25, e["families"]["ort_Z03.7"]["non_pd"], src)

p = load("runs/main-full-pd/eval.json")
src = "runs/main-full-pd/eval.json"
check("Evaluate", "loss with PD layer", 0.110, p["loss"], src)
check("Evaluate", "non-PD with PD layer", 0, p["pd"]["non_pd"], src)
check("Evaluate", "ort_Z03.7 non-PD (PD layer)", 0, p["families"]["ort_Z03.7"]["non_pd"], src)

s = load("runs/main-2000/eval.json")
src = "runs/main-2000/eval.json"
f = s["families"]["ort_Z11.0"]
check("Evaluate", "ort_Z11.0 lattices", 1, f["n"], src)
check("Evaluate", "ort_Z11.0 moduli error (%)", 199, f["median_percent_error_moduli"], src)
check("Evaluate", "ort_Z11.0 in worst families (1 = yes)", 1, int("ort_Z11.0" in s["worst_families"]), src)

# PD layer
check("PD layer", "moduli error with PD layer (%)", 28.5, p["overall"]["median_percent_error_moduli"], "runs/main-full-pd/eval.json")
check("PD layer", "coupling error with PD layer", 0.042, p["overall"]["median_abs_error_couplings"], "runs/main-full-pd/eval.json")
loss = {r: load(f"runs/{r}/eval.json")["loss"] for r in ("main-full", "main-full-pd", "main-full-s1", "main-full-pd-s1")}
check("PD layer", "loss gain, seed 0", 0.002, loss["main-full"] - loss["main-full-pd"], "runs/main-full{,-pd}/eval.json")
check("PD layer", "loss gain, seed 1", 0.010, loss["main-full-s1"] - loss["main-full-pd-s1"], "runs/main-full{,-pd}-s1/eval.json")
gap = max(abs(loss["main-full"] - loss["main-full-s1"]), abs(loss["main-full-pd"] - loss["main-full-pd-s1"]))
check("PD layer", "largest seed-to-seed gap", 0.012, gap, "runs/main-full*/eval.json")

# Diagnostician and Outcome check: the decision log
d = [json.loads(line) for line in (RUNS / "agent-pilot/decisions.jsonl").read_text().splitlines() if line.strip()]
src = "runs/agent-pilot/decisions.jsonl"
m = [x["outcome_metrics"] for x in d]
check("Diagnostician", "D1 non-PD before (%)", 8.8, 100 * m[0]["non_pd_before"], src)
check("Diagnostician", "D1 non-PD after (%)", 0, 100 * m[0]["non_pd_after"], src)
check("Diagnostician", "D1 loss before", 0.129, m[0]["value_before"], src)
check("Diagnostician", "D1 loss after", 0.119, m[0]["value_after"], src)
check("Diagnostician", "D2 loss after", 0.085, m[1]["value_after"], src)
check("Outcome", "D3 loss after", 0.077, m[2]["value_after"], src)
check("Outcome", "D1 non-PD drop", 0.088, m[0]["non_pd_before"] - m[0]["non_pd_after"], src)
check("Outcome", "D2 loss drop", 0.034, m[1]["value_before"] - m[1]["value_after"], src)
check("Outcome", "D3 loss drop", 0.0076, m[2]["value_before"] - m[2]["value_after"], src)
for i, want in enumerate(["improved", "improved", "failed"]):
    check("Outcome", f"D{i + 1} verdict is '{want}' (1 = yes)", 1, int(d[i]["outcome"] == want), src)
th = yaml.safe_load((ROOT / "config/loop.yaml").read_text())["outcome"]
check("Outcome", "loss threshold", 0.01, th["min_loss_drop"], "config/loop.yaml")
check("Outcome", "non-PD threshold", 0.04, th["min_non_pd_drop"], "config/loop.yaml")

# Width-512 runs in the pilot: loss and wall-clock time (config.json written at the start, eval.json
# after the last epoch; model.pt is saved at the best epoch, so it is not the end). Standby
# pauses are included).
t = load("runs/agent-pilot/trials.json")["trials"]
wide = [x for x in t if x["params"]["hidden"] == 512]
check("Diagnostician", "width-512 runs", 5, len(wide), "runs/agent-pilot/trials.json")
print("Width-512 runs (loss, hours from config.json to eval.json):")
for x in wide:
    folder = RUNS / "agent-pilot" / x["run_id"]
    hours = ((folder / "eval.json").stat().st_mtime - (folder / "config.json").stat().st_mtime) / 3600
    print(f"  {x['run_id']:20s} loss {x['value']:.3f}  {hours:5.1f} h")
print()

# Optuna, earlier subset study
o = load("runs/optuna-2000/trials.json")
src = "runs/optuna-2000/trials.json"
check("Optuna", "runs before stopping", 13, len(o["trials"]), src)
check("Optuna", "best loss", 0.224, o["best"]["value"], src)
check("Optuna", "untuned subset loss", 0.223, s["loss"], "runs/main-2000/eval.json")
best = next(x for x in o["trials"] if x["number"] == o["best"]["number"])
check("Optuna", "best run non-PD (%)", 10, 100 * best["non_pd_rate"], src)

# Subset study (comparison slide, rule-table and Optuna slides, report Table 2)
for run, loss_, npd, mod, cpl, n in [("optuna-sub", 0.225, 0.084, 54.0, 0.081, 20),
                                      ("rules-sub", 0.218, 0, 52.9, 0.080, 20),
                                      ("agent-sub", 0.229, 0, 56.8, 0.085, 15)]:
    src = f"runs/{run}/trials.json"
    tr = load(src)
    b = next(x for x in tr["trials"] if x["number"] == tr["best"]["number"])
    check("Study", f"{run} best loss", loss_, b["value"], src)
    check("Study", f"{run} non-PD rate", npd, b["non_pd_rate"], src)
    check("Study", f"{run} moduli error (%)", mod, b["overall"]["median_percent_error_moduli"], src)
    check("Study", f"{run} coupling error", cpl, b["overall"]["median_abs_error_couplings"], src)
    check("Study", f"{run} runs used", n, len(tr["trials"]), src)
seeds = [load(f"runs/{r}/eval.json")["loss"] for r in ("main-2000-pd", "main-2000-pd-s1")]
check("Study", "default + PD layer, seed 0", 0.216, seeds[0], "runs/main-2000-pd/eval.json")
check("Study", "default + PD layer, seed 1", 0.239, seeds[1], "runs/main-2000-pd-s1/eval.json")
ot = load("runs/optuna-sub/trials.json")["trials"]
check("Study", "Optuna runs with the PD layer", 6, sum(x["params"]["pd_head"] for x in ot), "runs/optuna-sub/trials.json")
check("Study", "Optuna best PD-layer run", 0.290, min(x["value"] for x in ot if x["params"]["pd_head"]), "runs/optuna-sub/trials.json")
for run, want in [("rules-sub", ["improved", "failed", "failed"]), ("agent-sub", ["improved", "failed", None])]:
    dd = [json.loads(line) for line in (RUNS / run / "decisions.jsonl").read_text().splitlines() if line.strip()]
    check("Study", f"{run} verdicts {want} (1 = yes)", 1, int([x.get("outcome") for x in dd] == want), f"runs/{run}/decisions.jsonl")
dd = [json.loads(line) for line in (RUNS / "agent-sub/decisions.jsonl").read_text().splitlines() if line.strip()]
check("Study", "agent's last decision is escalate (1 = yes)", 1, int(dd[-1]["decision"]["decision"] == "escalate"), "runs/agent-sub/decisions.jsonl")
ra, ag = load("runs/rules-sub/trials.json")["trials"][:10], load("runs/agent-sub/trials.json")["trials"][:10]
check("Study", "rounds 0-1 identical settings (1 = yes)", 1, int(all(x["params"] == y["params"] for x, y in zip(ra, ag))), "runs/{rules,agent}-sub/trials.json")
check("Study", "largest loss gap on identical settings", 0.024, max(abs(x["value"] - y["value"]) for x, y in zip(ra, ag)), "runs/{rules,agent}-sub/trials.json")

# Print
w = max(len(r[1]) for r in rows)
current = None
for slide, claim, stated, actual, source, ok in rows:
    if slide != current:
        print(f"== {slide} ==")
        current = slide
    print(f"  {'OK  ' if ok else 'DIFF'} {claim:{w}s}  deck {stated!s:>7}  file {actual:10.4f}  ({source})")
bad = sum(not r[5] for r in rows)
print(f"\n{len(rows) - bad} of {len(rows)} numbers match the files" + (f"; {bad} differ" if bad else "."))
