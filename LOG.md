# Running log

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
    families; R² 0.89-0.93 (moduli), 0.83-0.85 (couplings); median errors 25-39% and
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
