# LR3 literature note: isolating the LLM's contribution in a training/tuning loop

Literature-reviewer agent run, 2026-11-13 (narrow search for Lab Report 3). Saved by the main
session because the agent had no file-writing tools.

**Verification status:** the PDFs came back unreadable, so every claim was read from the arXiv
HTML versions and is cited by section/table, not page. `[HTML-body]` = read in the full HTML
text; `[abstract/snippet]` = search snippet only. Nothing here is PDF-verified. Jamie must read a
paper and convert its section references to pages before it is cited in a report.

Seven papers (2025-2026) are relevant; only the first three bear directly on "LLM or loop
structure". No paper was found that runs an LLM against a deterministic rule table inside the same
loop on planted compound faults.

## 1. Ablations and baselines that isolate the LLM

**Rodrigues et al. 2026 (arXiv:2606.21641).** Budget-matched study on tabular data: an LLM
advisor (LLM-OptFlow) vs random search, Optuna-TPE, GP-BO, successive halving.
- Random, TPE and GP-BO run 40 trials, the LLM 12 iterations. The LLM is warm-started with the
  default configuration as its first evaluation; classical controls get the same seed
  (Sec 4, 5.1) [HTML-body].
- Against seeded random search the advisor leads by +0.20 pp at 2 evaluations (p = 0.03), ties
  at 5, trails by -0.37 pp at 12 (95% CI [-0.82, -0.02]). Unseeded, it ties with TPE and GP-BO at
  12 (Sec 5.1) [HTML-body].
- "vehicle" dataset, final test accuracy: random 82.4 vs LLM 73.3 (default 73.5) (Table 2,
  Sec 5.2): "every classical method beats both the default and the LLM." [HTML-body]
- Title: the gain comes from the warm-start default configuration, not the model.
- Relevance 5: the only seed- and budget-matched control found.

**Li 2026 (arXiv:2603.15916).** Convergence analysis of about 10,000 LLM-proposed experiments,
three baselines (Table 2, Sec 5.2) [HTML-body]: random search from the LLM-curated pool, TPE on
that pool, random search from scratch (172 configurations).
- Pool-based baselines beat the LLM at every checkpoint (AP@50: 0.9876, 0.9874 vs LLM 0.9852);
  only from-scratch random search is worse (0.9648 at N = 50).
- The author attributes this to the LLM having pre-curated the pool: a loop-structure
  explanation. AP's definition not established.
- Relevance 3 (single author, architecture search, pool reuses LLM output).

Ferreira et al. (arXiv:2603.24647, already cited) is consistent with Rodrigues [abstract/snippet].

## 2. LLM agents diagnosing training failures

**AI Training Manager (arXiv:2606.29871; authors not retrieved).** Bounded closed-loop LLM
controller on induced failures [HTML-body].
- Overfitting induced by cutting TinyStories to 2k stories: final validation loss Overfit
  1.971 ± 0.331, with Manager 0.900 ± 0.096, healthy baseline 0.663 ± 0.095 (Table 1).
- RL: Aggressive regime 0.121 -> 0.764 deterministic safe success, Conservative 0.413 -> 0.705,
  healthy 0.695 (Table 2).
- Intervention-history ablation (single run): 0.656 vs 0.764 ± 0.037; without history the Manager
  "repeatedly reduced entropy and increased the collision penalty" before earlier changes could
  be evaluated.
- Limitations: "We also do not compare against an exhaustive handcrafted controller"; early
  stopping "could also have mitigated" the overfitting failure but was not tested.
- Relevance 4.

**Kim et al. 2025 (arXiv:2506.03396; Kim, Humbatova, Jahangirova, Yoo, Tonella).** Fault
localisation and repair for DL systems with LLMs [HTML-body].
- 25 artificial faults (DeepCrime) across six models (Sec 3.2, Table 2) plus 9 real Stack
  Overflow faults (Sec 3.3, Table 3).
- Non-LLM baselines: DeepFD, DeepDiagnosis, UMLAUT, Neuralint (localisation); Random, HEBO, BOHB,
  AutoTrainer (repair).
- GPT-4 localisation recall 1 on artificial faults vs at most 0.31 for other tools (Sec 6.1,
  Table 5); after neutrality analysis total recall 0.91, real-fault recall 0.79 (Sec 6.2, Table 7).
- Median repair improvement rate 1.0 (GPT-4, GPT-4T), 0.55 (Random, HEBO), 0.18 (AutoTrainer)
  (Sec 6.5, Fig 3); "Random performs surprisingly well"; repair more stable for GPTs (Sec 6.6).
- Counterpoint in the LLM's favour, but no rule table, and single-cause faults.
- Relevance 4. Abstract's 44% / 82% figures: [abstract/snippet].

**DeepFix (arXiv:2603.14099).** No baselines, no planted faults; one qualitative industrial case
(Sec III); an "LLM reasoning vs rule-based heuristics" ablation is future work [HTML-body].
Relevance 2 (shows the gap only).

## 3. Compute-aware budgets

**Guo et al. 2026, ASAP (arXiv:2606.25207).** Per-iteration evaluation "silently ignores" serial
LLM and tool overhead [HTML-body].
- Primary metric: regret vs end-to-end wall-clock (Sec 5.1).
- Normalized regret about 0.026 vs 0.05-0.16 for baselines (GP, TPE, Optuna, SKOPT, TuRBO, SMAC,
  DNGO, LLAMBO; best LLAMBO about 0.05) (Sec 5.2); 28 task-model combinations on HPOBench and
  PD1, 3 seeds, GPT-3.5-turbo (Sec 5.1). Without the integration step: 0.051 (Table 2).
- Its proposer is a pool of classical optimizers, so it does not isolate the LLM. Relevance 3.

**LLMZero (arXiv:2606.18388; authors not retrieved).** LLM agent vs static, grid, random and
skill-agent baselines on RL post-training [HTML-body].
- Beats random by small margins (e.g. ChemCoT 40.5 vs 39.4) and grid by 6-15% relative
  (Table 1, Sec 4.2). No rule-based baseline or explicit no-LLM ablation.
- Reward-only ablation (no diagnostic signals): test mean 61.5% -> 49.5% (App E.1).
- Relevance 3.

## What this implies for our comparison

The closest evidence (Rodrigues; Li) suggests much of an LLM loop's apparent gain comes from
structure (warm start, curated pool, sequential commitment), which seed- and budget-matched
classical controls erase. The evidence favouring the LLM on diagnosis (Kim et al.; the AI Training
Manager history ablation) uses single-cause or induced faults, where a rule table would also do
well. The rule-table control is therefore well motivated; the informative cases are compound or
ambiguous faults. If compute matters, report wall-clock and LLM cost next to run count (ASAP).

## Gaps

- No page numbers: open each PDF and convert section references to pages.
- Authors of 2606.29871 and 2606.18388 not retrieved.
- Unread: last 40% of Kim et al., last 28% of Guo et al., last 43% of LLMZero.
- Not checked whether these cite Ferreira, Huai or Osaro (Rodrigues confirmed not to cite Ferreira).
- No study found with planted compound/ambiguous faults comparing an LLM against a rule table.

Sources: arxiv.org/html/2606.21641, /2603.15916, /2606.29871, /2506.03396v1, /2606.25207v1,
/2606.18388, /2603.14099v1
