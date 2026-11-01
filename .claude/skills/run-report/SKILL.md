---
name: run-report
description: Turn one lattice-GNN training run (a runs/<id>/ folder with config.json, history.json and eval.json) into a readable Markdown report with tables and PNG plots - loss curves, per-target errors, positive-definiteness counts and per-family errors - without interpreting the numbers. Use this whenever the user asks to show, summarize, plot, tabulate or "make readable" the results of a training run in this repo, asks what a run's numbers are, or wants figures of a run for a lab report, even if they don't say "report".
---

# Run report

Lay out the results of one training run so a person can read them. The point is a neutral
record: the user (and later the diagnostician agent) draws the conclusions, so this skill must
not. A report that says "the model overfits" or "family X is the main problem" has already
made a judgement the user did not ask for and may disagree with.

## Steps

1. **Find the run.** The user names a run id or folder; if not, list `runs/` and use the
   folder they point to. Don't pick one yourself if several exist; ask. A run folder needs
   `config.json`, `history.json` and `eval.json` (written by `lattice_gnn.training.train`).
   If one is missing, say which and stop.

2. **Build the report** with the bundled script in the repo (it is deterministic, so every
   report of the same run is identical):

   ```bash
   .venv/Scripts/python -m lattice_gnn.report runs/<id>
   ```

   This writes `runs/<id>/report/`: `report.md`, `loss.png`, `targets.png`, `families.png`.
   Don't write your own plotting code or change the script for a one-off; if the user wants
   a different plot, say so and ask whether to add it to `lattice_gnn/report.py`.

3. **Look at the figures** (open the PNGs) to check they rendered: no overlapping labels, no
   empty panels. Report a rendering problem instead of describing a broken figure.

4. **Reply to the user** with:
   - the path to `runs/<id>/report/report.md` and the three figures,
   - a short factual summary of what the report contains, using the numbers in
     `report.md`: best epoch and epochs run, train and validation loss at the best epoch,
     validation lattices and families, R² and typical error per target (or their range),
     non-PD count and rate, and the families with the highest and lowest median modulus
     error, with their `n`.

## Describe, don't judge

State values, counts, orderings and changes over epochs. Leave out evaluation, cause and
advice. For example:

| Write | Not |
|---|---|
| Train loss 0.079, validation loss 0.247 at epoch 91. | The model overfits. |
| Validation loss changed by less than 0.01 over the last 20 epochs. | Training has converged / more epochs won't help. |
| 12 of 184 predictions (6.5%) are not positive definite. | Too many non-PD predictions; add a PD head. |
| trig_Z07.9 has the highest median modulus error, 432% (n=2). | trig_Z07.9 is the worst family / is an outlier. |
| R² ranges from 0.66 to 0.82 across the 9 targets. | The results are good. |

Words to avoid because they carry a verdict: good, bad, poor, strong, weak, worst, best (except
"best epoch", which is the name of the saved checkpoint), problem, issue, overfit, converged,
outlier, should, need, recommend, improve. Sorting families by error is fine; it is an order,
not a ranking of quality.

If the user then asks what the numbers mean or what to do next, that is a separate question:
answer it in its own reply and make clear it is interpretation, not part of the report.
