# Running log

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

**Remaining**
- Rerun `first-real-2000` on `splits/main.json` (it used the draft split).
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
