# Running log

## 2026-11-05

**Done**
- Diagnostician effort level: high (Jamie). Pinned with the model in the new `agent` section of
  `config/loop.yaml` (`claude-sonnet-5-5`, effort high, 3 turns, 1 retry).
- Diagnostician (TODO section 9): `.claude/agents/diagnostician.md` (no tools) and
  `python -m agents.diagnostician <run or study folder>`. It writes the input (state, eval.json
  with the run's settings, optimizer history, decision log) to `agent/prompt-<n>.md`, calls
  `claude -p` with `--json-schema`, validates with `lattice_gnn.validate`, retries once with the
  errors, then forces an escalation. Logs `decisions.jsonl` and `run_info.json` (model requested
  and used, Claude Code version, effort, tokens, cost estimate). `tests/test_diagnostician.py`
  uses a fake `claude`; 66 tests pass.
- Fix: Claude Code rejects a `$schema` key for draft 2020-12, so `DECISION_SCHEMA` has none.
- First real decision, on `runs/main-full` (not committed): valid on the first call; 9.5 s,
  2 turns, 11,161 input tokens (cache creation) and 1,156 output, $0.056 client estimate,
  model used `claude-sonnet-5-5`, Claude Code 2.1.316.
  - Decision: change_structure, `pd_head` = true. Reason: 194 / 2,032 non-PD with 0 non-PD
    labels, spread across families; `augment` already on.
  - Prediction: non-PD 194 -> 0 in every family; coupling error (0.046) about the same or
    slightly better; moduli error (30.8%) within a few points; failed if non-PD > 0 or coupling
    error rises by > 0.02.
  - Checked against `runs/main-full-pd` (2026-11-04): non-PD 0, couplings 0.042, moduli 28.5%.
    The prediction holds; one seed, and the run existed before the decision.
  - One wrong number in its evidence: "15 of 20 runs remain" (19 remain after 1 run). The
    validator checks the budget itself, so this did not matter here, but evidence text is not
    checked.

- The loop (TODO section 10), `agents/diagnostician/loop.py`:
  `python -m agents.diagnostician.loop --config config/loop.yaml --name <name> [--approve ask|all|none]`.
  - Each round: Optuna runs 5 training runs over the current search ranges with the structure
    fixed (the optimizer never sets menu switches here, only the agent does); a new Optuna study
    per round, sampler seed + round. Then check_target, one agent decision, the human gate, apply.
  - Human gate: retunes run without approval; `ask` asks at the terminal for every
    change_structure and on escalation (continue or stop). `all` / `none` approve / refuse every
    change_structure unattended, and stop on escalation (counts as not solved).
  - After the next round, code fills in the decision's `outcome` (improved | failed): the round's
    best run against the best before it, beyond `outcome.min_loss_drop` (objective) or
    `outcome.min_non_pd_drop`. Both are placeholders (0) until the seed spread is known.
  - Writes `runs/<name>/`: loop.json, round-<r>/trial-<n>/, trials.json, decisions.jsonl,
    run_info.json, agent/, summary.json.
  - `tests/test_loop.py`: 6 end-to-end tests on the sample file with a fake `claude` (retune,
    structure change approved / refused, escalation stop / continue, failed option -> forced
    escalation). 72 tests pass.
- Human gate for the unattended studies: `--approve all` (Jamie); recorded in the worksheet,
  section 7.
- Seed-spread runs: `runs/main-full-s1`, `runs/main-full-pd-s1` (seed 1, else as seed 0). Same
  2,032 validation lattices:

  | Run | Seed | Loss | Non-PD (rate) | Moduli median % error | Couplings median abs. error |
  |---|---|---|---|---|---|
  | main-full | 0 | 0.1119 | 194 (0.095) | 30.8 | 0.046 |
  | main-full-s1 | 1 | 0.1239 | 110 (0.054) | 33.3 | 0.051 |
  | main-full-pd | 0 | 0.1099 | 0 | 28.5 | 0.042 |
  | main-full-pd-s1 | 1 | 0.1134 | 0 | 30.3 | 0.045 |

  - Seed-to-seed gap in loss: 0.012 without pd_head, 0.0035 with it (pooled s.d. about 0.006).
    Gap in non-PD rate without pd_head: 0.041.
  - pd_head lowers the loss with both seeds (by 0.002 and 0.010; mean 0.006). That is inside the
    seed spread, so two seeds do not establish an accuracy gain. The removal of non-PD
    predictions is unambiguous (0 with both seeds).
  - Every run is still improving at epoch 100 (best epochs 94-99).

**Remaining**
- Seed spread -> `outcome` thresholds in `config/loop.yaml`.
- `family_balanced_loss` on the menu before the studies start.
- Decision consistency on frozen reports (TODO section 11), e.g. `main-full` x 10 calls.

## 2026-11-04

**Done**
- `pd_head` switch (TODO section 7): `LatticeGNN(pd_head=True)`, `train(config)` key `pd_head`,
  `train.py --pd-head`, and `pd_head: [false, true]` in the `config/loop.yaml` menu.
  - The 3 coupling outputs are mapped through canonical partial correlations (tanh) to a
    correlation matrix R = L L^T, then (1 - 1e-5) R + 1e-5 I, so the normalized compliance
    block is positive definite for every input, and every PD block can still be reached.
  - Margin 1e-5: on the full catalog the labels' smallest block eigenvalue has median 0.037,
    1st percentile 0.002, and only 4 of 17,222 labels are below 1e-4.
  - `tests/test_pd_head.py`: 0 non-PD on random and extreme inputs, finite gradients when
    saturated, inversion on random PD blocks, save/load, a short `train` run. 22 tests pass.
  - Check on `subset_300.txt`, `splits/main.json`, 30 epochs: non-PD 0 / 40 with the head,
    2 / 40 without; validation loss 0.632 vs 0.651 (too small a set to compare accuracy).

- Agent output schema and validator (TODO section 9), `lattice_gnn/validate.py`:
  - `DECISION_SCHEMA`: `decision`, `reason`, `evidence`, plus `ranges` (retune),
    `menu_option` + `prediction` (change_structure) or `would_try` (escalate). Which fields each
    decision needs is checked in code, so the schema has no conditionals for `claude -p`.
  - `validate(decision, state) -> {ok, errors, force_escalate}`. Rules: well-formed; retune
    ranges inside `limits`; menu option on the menu, a real change, under the cap of 3
    structural changes; enough budget for a round of 5 runs. Proposing an option that already
    failed forces escalation (no retry). Escalate is always allowed if well-formed.
  - `config/loop.yaml`: new `limits` (hard bounds on the agent's ranges, wider than `search`),
    `budget.round_runs: 5`, `budget.max_structural_changes: 3`. `jsonschema` pinned.
  - `tests/test_validate.py`: 37 tests, at least one per rule; 59 tests pass in total.
- `runs/main-full-pd` (not committed): `main-full`'s settings plus `--pd-head`, ~1.5 h,
  `python train.py Unit_Cell_Catalog/Unit_Cell_Catalog.txt --split-file splits/main.json --epochs 100 --patience 40 --pd-head --run-id main-full-pd`.
  Same 2,032 validation lattices in 67 families as `main-full`.
  - Non-PD predictions: 0 / 2,032 (`main-full`: 194, 9.5%).
  - Best epoch 99 (`main-full`: 97); validation loss 0.110 (0.112); both still falling at 100.
  - Median error over the moduli 28.5% (30.8%), over the couplings 0.042 (0.046). Per target:
    Gxy 31.0% (38.8%), Gxz 31.0% (34.3%); Ey 25.8% (24.8%) is the one modulus slightly worse.
    R² 0.89-0.93 on the moduli, 0.83-0.84 on the couplings (same ranges as `main-full`).
  - The 5 families with the most non-PD in `main-full` (15-25 each): 0 non-PD, coupling error
    equal or lower in 4 of 5.
  - One seed each: the accuracy differences are within what seed-to-seed spread may explain
    (not measured yet). The non-PD result holds by construction.

**Remaining**
- Seed spread for the full-catalog runs (needed anyway to judge the agent's decisions).
- Check the `limits` values (proposed, not yet decided).
- The diagnostician: agent spec, input file, one `claude -p` call (Lab Report 3).

## 2026-11-01

**Done**
- Decisions (Jamie), recorded in `docs/agent_design_worksheet.md` and `config/loop.yaml`:
  - Agent and baseline studies train on the full catalog, on the laptop CPU (~1.5 h per run;
    ~270 h for 3 repeats x 2 conditions x 20 runs plus the planted failures).
  - Thresholds: non-PD target 0; no accuracy target from a spec, so the accuracy thresholds are 0
    and studies stop on budget or no progress. Runs-to-target uses the baseline's best validation
    error.
  - Each decision is checked in code afterwards (targeted families' error and non-PD rate; a drop
    counts only beyond seed-to-seed spread). Model ID and `--effort` pinned (level still to choose).
- Lab Report 2 drafted and trimmed to 5 pages (`reports/lab_report_2.tex`). Module demo: the
  Labs 4 and 5 literature agent in `capstone-agent` (permissions, arXiv routine, retriever;
  hit@5 0.26-0.37, 6/6 abstentions on out-of-corpus questions).
- Jamie read every cited paper and checked each cited page.

**Remaining**
- Optuna citation; rebuild A2 with the retriever.
- `pd_head`, the validator and the diagnostician (Lab Report 3).
- Choose the `--effort` level.

**Bottlenecks**
- CPU time: the full-catalog studies need ~11 of the 14 days between Lab Reports 3 and 4.

## 2026-10-31

**Done**
- Overnight runs on the laptop CPU (19:53-23:23):
  - `runs/optuna-2000`: Optuna study on the 2,000-lattice subset, patience 20. Stopped after 13
    of 20 runs (no new best in 10). Best: trial 2 (6 layers, width 128, batch 32, lr 4.8e-4),
    validation loss 0.224 vs 0.223 for the untuned `main-2000`; 10% non-PD. Trial 6: 0% non-PD
    but 334% median modulus error.
  - `runs/main-full`: default model on the full training split, 100 epochs, patience 40, ~1.5 h.
    Best epoch 97 (validation loss 0.112, still falling); 2,032 validation lattices in 67
    families; RÂ² 0.89-0.93 (moduli), 0.83-0.85 (couplings); median errors 25-39% and
    0.041-0.048; 194 non-PD predictions (9.5%), 0 non-PD labels.
- `scripts/make_study_figure.py` plots a study's trials.

**Remaining**
- Decide which data the agent and baseline studies use (decided 2026-11-01: full catalog).

**Bottlenecks**
- A full-catalog run takes ~1.5 h on the CPU, against ~10 min on the subset.

## 2026-10-30

**Done**
- Optimizer baseline (Baseline 2) pipeline, TODO sections 3, 4, 6 and 8:
  `lattice_gnn/splits.py` (frozen family splits), `lattice_gnn/evaluate.py` (per-target and
  per-family errors, PD check on the 3 couplings), `lattice_gnn/training.py` (`train(config)`,
  seeded, writes `runs/<id>/`), `lattice_gnn/optimize.py` + `config/loop.yaml` (Optuna study,
  `check_target`). Tests: `tests/test_baseline.py`.
- `family()` fix: it only stripped `_E<n>`, so each RCSR `_R<n>` lattice was its own family.
  The catalog has 456 families. 135 duplicates appear under an `_E` and an `_R` name; 6 of them
  carry a different Z in the two names, and the split now keeps those families together.
- `scripts/make_subset.py`: 17,222 complete records (40 without properties).
- Draft family split `splits/draft.json` (seed 0, 10/10%): 13,741 / 1,745 / 1,776 lattices.
- Tiny study (300-lattice subset, 6 runs x 5 epochs) runs end to end in ~30 s; numbers are
  meaningless at that size.
- First real run, `runs/first-real-2000` (not committed; `runs/` is ignored):
  `python train.py data/raw/subset_2000.txt --split-file splits/draft.json --epochs 100 --run-id first-real-2000`.
  1,603 train / 184 val lattices (36 val families), default model (4 layers, width 128,
  581k parameters), seed 0. 100 epochs in 10.5 min on the laptop CPU (6.3 s/epoch); best epoch 91.
  - Loss (standardized MSE; 1.0 = predicting the training mean): train 0.079, val 0.247.
    Val was still creeping down at epoch 100; the train/val gap is large.
  - Validation: R2 0.72-0.82 on the moduli, 0.66-0.69 on the couplings. Median error 53% on
    the moduli (43-76% per modulus, Gxy worst), 0.071 abs on the couplings.
  - Non-PD predictions: 12 / 184 (6.5%); all true labels are PD. No PD head yet.
  - Worst families: trig_Z07.9 (432%, n=2), tet_Z08.7 (304%, n=1), trig_Z11.1 (257%, n=1),
    mon_Z05.7 (183%, n=6). Several have only 1-2 val lattices, so per-family numbers are noisy.
  - Study cost at this speed: ~10 min per default-size run, so 4-10 h for a 40-run study.
- Locked split frozen: `splits/main.json` (seed 0, 10% val / 10% test, `--max-family 200
  --stratify`). Family = crystal system + Z (456 families). In the uncapped draft, one family
  (tet_Z04.0, 1,000 lattices) was 56% of test, so families over 200 lattices (18 families,
  6,062 lattices) always go to train, and val/test are filled within each crystal system.
  Result: 13,295 / 2,039 / 1,928 lattices in 325 / 67 / 64 families; largest test family 7%
  of test; each crystal system has 10-16% of its lattices in test. The test set measures new
  small-to-medium families. `tests/test_baseline.py` checks the file regenerates unchanged.
- Rerun on the locked split, `runs/main-2000` (not committed):
  `python train.py data/raw/subset_2000.txt --split-file splits/main.json --epochs 100 --run-id main-2000`.
  Same settings as `first-real-2000`. 1,537 train / 239 val lattices (38 of the 67 val
  families appear in the subset). 100 epochs in 10 min; best epoch 76.
  - Loss at epoch 76: train 0.097, val 0.223. At epoch 100: train 0.059, val 0.230.
    Learning rate halved at epochs 68, 87 and 98.
  - Validation: R2 0.78-0.82 on the moduli, 0.67-0.75 on the couplings. Median error 58% on
    the moduli (54-62% per modulus), 0.085 abs on the couplings.
  - Non-PD predictions: 19 / 239 (7.9%); all true labels are PD.
  - Highest median modulus error: ort_Z11.0 199% (n=1), mon_Z05.1 181% (n=4),
    mon_Z05.4 168% (n=2). Lowest: hex_Z06.3 20% (n=1).
  - Report: `python -m lattice_gnn.report runs/main-2000`.
  - Not comparable one-to-one with `first-real-2000`: different validation families.
- `run-report` learning-rate ticks now show exact values (they rounded 2.5e-4 to 3e-4).

**Remaining**
- Set the `config/loop.yaml` thresholds from the spec (placeholders now).
- Menu switches (TODO section 7) for the study to search, starting with the PD head.
- Decide the study budget (40 runs x 100 epochs is 4-10 h per study on the laptop).

## 2026-10-29

**Done**
- Lab Report 2 to-do list (`docs/TODO.md`).
- `Z_avg` check: the catalog's Z_avg is an averaging convention, not a graph bug. It averages
  the degree over every image of each node in the closed cell (corner x8, edge x4, face x2) and
  leaves out 2-connected nodes. The file sometimes omits images (e.g. 6 of 8 corners), so the
  images are counted from the positions. Added `PeriodicGraph.catalog_z_avg`; `check_data.py`
  uses it.
- Full catalog (`check_data.py`): Z_avg mismatches 257 / 17,262 (1.5%), C not positive definite
  4, no properties 40, Poisson asymmetry 0.

**Remaining**
- The 257 remaining Z_avg mismatches: a third have nodes just off a cell face (1e-4 to 2e-3).
  Explain them or exclude them.
- The 4 records whose stiffness matrix is not positive definite (e.g. `tet_Z03.7_E6457`).

## 2026-10-18

**Done**
- AWS environment check on the course WorkSpace (Amazon WorkSpaces, m8i-flex.large, Windows Server
  2022, 2 vCPU, 7.7 GB RAM).
  - PASS: GitHub push (dry run), Python 3.12 with the pinned packages, `lattice_gnn` import and the
    data check on the samples, Claude Code sign-in (Caltech enterprise seat), network and
    literature tools.
  - FAIL: GPU, Docker, Abaqus (not installed; license server reachable).
  - NOT TESTED: persistence after restart, S3, cost alerts.
- Installed Python 3.12.10 from the NuGet package, because MSI installers are blocked on the
  WorkSpace.
- Added a command-line route for the literature agent's tools (`agents/literature/cli.py`), now the
  default for `python -m agents.literature`. Bash is restricted to that one command; tested that
  everything else is refused.
- Literature agent runs (`claude-sonnet-5-5`):
  - Attempt 1 (`literature_review/main_attempt1/`) stopped after 32 turns when a tool call with a
    formula was refused by Claude Code's shell-safety check.
  - Attempt 2 (`literature_review/main/`): 81 turns, 22 papers, `lit_review.md`. Page citations for
    Indurkar 2022 and Grega 2024 checked against the PDFs: 10 of 11 hold up; 1 is a paraphrase.
- Lab Report 1 written (`reports/lab_report_1.md`).

**Remaining**
- Read the dataset paper's supplementary information (how C and n were fitted, exclusions);
  reconcile 17,262 / 17,221 / 17,201 / 17,087.
- Baselines and a first CPU GNN training run.
- Literature run on defects; FE solver choice.

**Bottlenecks**
- Abaqus not installed, and AWS licence use not yet confirmed. Workaround: CalculiX or OpenSeesPy.
  Plan: confirm with Caltech IT by 1 Nov.
- Not treated as bottlenecks:
  - No GPU: training runs on the CPU.
  - No Docker: the WorkSpace itself is the isolated environment for code-running agents.
  - MSI installs blocked: worked around with NuGet Python. The venv should still be rebuilt on it.
  - No S3 or billing access.
