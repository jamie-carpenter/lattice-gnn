# Implementation to-do

Small tasks in build order. Each section depends on the ones above it.
Design: `docs/agent_design_worksheet.md`.

## 1. Repo setup

- [x] Merge `agents/literature-review` into the working branch (brings in `LOG.md`, `reports/`,
      `literature_review/main/`)
- [x] `.gitignore` covers `runs/`, `data/processed/`, `checkpoints/` (subset files live outside the repo)
- [x] Add `pytest` to `requirements.txt` (pinned)

## 2. Data correctness

- [x] Finish the `Z_avg` investigation. The catalog's Z_avg is a different averaging
      convention, not a graph bug: average the degree over every image of each node in the
      closed cell [0, 1]^3 (corner x8, edge x4, face x2) and leave out 2-connected nodes.
      The file sometimes omits images (e.g. `mon_Z05.7_E1265` lists 6 of 8 corners), which is
      why counting the file's own node copies left 8% unexplained. Every 8th record: 99.2%
      match (was 91%). Looser merge/face tolerances (5e-4 to 3e-3) make it worse, so 1e-4 stays.
- [x] Update the `Z_avg` check in `check_data.py` to the catalog's convention
      (`PeriodicGraph.catalog_z_avg`; `z_avg` stays the true average degree, used as a feature)
- [ ] Run `check_data.py` on the full catalog; 0 unexplained mismatches. Now: 257 / 17,262
      (1.5%), all crystal systems (tet 83, trig 54, ort 53, cub 53, mon 11, hex 3); tolerance
      is not the cause (248 still differ by > 0.015, median 0.12). A third of them have nodes
      just off a face (1e-4 to 2e-3), vs 8% of all records; e.g. `trig_Z03.5_E3993` has
      near-duplicate nodes 1e-4 apart. Decide: explain the rest, or accept and list them
      (`problems["Z_avg mismatch"]`) as excluded / flagged
- [ ] Find out why 40 records lack properties (LR1 said 41); note it in `LOG.md`
- [ ] Count lattices per family (`family()` in `train.py`); note the smallest/largest families
- [x] Write `scripts/make_subset.py`: every k-th complete record -> subset file (catalog is
      sorted by cell size, so never take the first N)

## 3. Splits

- [x] Move `family()` and `split_dataset()` out of `train.py` into `lattice_gnn/splits.py`.
      Fixed on the way: `family()` only stripped `_E<n>`, so every RCSR `_R<n>` lattice was its
      own family (456 families, not ~3,100). 6 duplicate pairs have different Z in their two
      names; `family_split` keeps their families in one part (`read_duplicates`)
- [x] Save a split to `splits/<name>.json` (family names per train / val / test) and load it back
- [x] `train.py --split-file splits/<name>.json` so every run uses the same frozen split
- [ ] Freeze the locked test split once; never change it afterwards

## 4. Evaluation tool

- [x] `lattice_gnn/evaluate.py`: errors per target (move `metrics()` out of `train.py`)
- [x] Add errors per family (median % error on moduli, abs error on couplings)
- [x] Add PD check: rebuild the 6x6 stiffness from predictions (`props_from_target` ->
      `stiffness_matrix`), count non-positive-definite cases per family
- [x] Derive the PD threshold from the math (smallest eigenvalue > 0, tolerance from float
      precision), write the reasoning in a comment
- [x] `evaluate.py` writes `runs/<id>/eval.json` (the report the agent will read)
- [x] Unit test: evaluate on the sample file returns all fields

## 5. First real run

- [x] Strided 2,000-lattice subset, frozen split, train 100 epochs (draft split; rerun on
      `splits/main.json` once it is frozen)
- [x] Save `runs/<id>/config.json` (all arguments, seed, git commit) and `eval.json`
- [x] Log the results in `LOG.md`

## 6. Training as a tool

- [x] Refactor `train.py` so training is a function `train(config) -> run_id` (keep the CLI)
- [x] Every run writes `runs/<id>/` (config, metrics, checkpoint path) by itself
- [x] Fix all seeds (torch, numpy, random, DataLoader) from `config["seed"]`
- [x] Check: two runs with the same seed give near-identical metrics

## 7. Menu config and switches

- [ ] `config/loop.yaml`: menu options, hyperparameter ranges, thresholds, budget
- [ ] Switch 1 `pd_head`: predict couplings so the 3x3 normalized compliance block is always
      positive definite (e.g. Cholesky-style); add to `LatticeGNN`
- [ ] Test: with `pd_head` on, PD failures are 0 on any input
- [ ] Switch 2 `physics_residual`: compute a stretch-only stiffness estimate from bar
      n⊗n terms; model predicts log10(true / estimate)
- [ ] Switch 3 `family_balanced_loss`: weight samples by 1 / family size
- [ ] Each switch readable from `config/loop.yaml` and settable via `train(config)`

## 8. Baseline 2 (Optuna)

- [x] Add `optuna` to `requirements.txt` (pinned)
- [x] `lattice_gnn/optimize.py`: Optuna study over hyperparameter ranges + menu switches from
      `config/loop.yaml`, fixed sampler seed, budget = number of training runs
- [x] Save trial history to `runs/<study>/trials.json`
- [x] `check_target(eval, thresholds, budget) -> continue | stop(reason)` in plain code
- [x] Run a tiny study (e.g. 6 runs, few epochs) end to end

## 9. Agent

- [ ] Output schema (JSON): `decision` (retune | change_structure | escalate), `reason`,
      `evidence`, plus `ranges` / `menu_option` + `prediction` depending on the decision
- [ ] `lattice_gnn/validate.py`: reject options not on the menu or already failed, ranges out
      of limits, over budget; unit tests for each rule
- [ ] Agent spec `.claude/agents/diagnostician.md`: role, inputs, options, schema, pinned model,
      no shell / no file edits
- [ ] Build the agent's input: `eval.json` + trial history + decision log -> one prompt file
- [ ] Call the agent once on a saved report via `claude -p`; parse and validate its output
- [ ] Decision log: append each decision + outcome to `runs/<id>/decisions.jsonl`
- [ ] Record model ID, Claude Code version, tokens per call in `runs/<id>/run_info.json`

## 10. The loop

- [ ] Orchestrator (LangGraph or plain Python): optimize -> evaluate -> check -> agent ->
      validate -> human gate -> apply -> repeat
- [ ] Human gate: pause for approval on `change_structure` and `escalate`
- [ ] Stopping rules from `config/loop.yaml` (success, budget, no progress, max 3 structural
      changes)
- [ ] Retry once on invalid agent output, then force escalate
- [ ] One command runs the whole loop: `python -m agents.diagnostician --config config/loop.yaml`

## 11. Evaluation harness

- [ ] Planted failures as config presets: PD head removed + couplings stressed, learning-rate
      range excludes good values, one family's labels corrupted
- [ ] Script: run agent loop and Baseline 2 on the locked test families, N repeats, same budget
- [ ] Script: decision consistency (frozen reports x 10 calls, count outcomes)
- [ ] Results table: test error, non-PD rate, runs to target, correct remedies, escalations

## 12. Reproducibility

- [ ] `tests/test_smoke.py`: sample file -> check, split, 1 tiny train, evaluate, 1 validated
      agent decision (agent step skippable without Claude Code)
- [ ] GitHub Actions CI running the smoke test (agent step skipped)
- [ ] Pre-commit secret scan
- [ ] README: setup, one command, where results go
- [ ] Cloud: allow `download.pytorch.org` in the environment, or document the PyPI install
- [ ] Fresh-clone check: clone, install, one command, smoke test passes
