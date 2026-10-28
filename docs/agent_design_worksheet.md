# Agent Design Worksheet (Lab Report 2)

Use this to redesign the agentic system before writing the Lab Report 2 proposal. Each section
asks one design question, says where the requirement comes from, and shows what Lab Report 1
(LR1) said. Fill in the **Answer** blocks yourself; the analysis notes are a starting point to
argue with, not decisions.

Sources:
- **TA**: LR1 grading comment (technical depth 26/30): *discuss the architecture: the flow of the
  agents and their assigned goal toward "building a reliable surrogate"; is the agents' goal to
  speed up hyperparameter search? Designing the full agent loop with its tools and protocol is
  essential.*
- **Debrief**: "Lab Report 1 debrief: using agentic AI tools well" (Week 4).
- **LR2 brief**: course announcement on Lab Report 2 (due Sun Nov 1, 11:59 pm).

---

## 1. Research question

> One sentence a reviewer could answer yes/no or with a number at the end of the term.
> The question must contain the agent's decision (Section 3); if it only mentions the GNN,
> the agent is outside the research loop (Debrief: "Which box is the project").

LR1: "build a GNN that predicts effective elastic properties and generalizes to unseen unit
cells." This is a surrogate-modelling question; no agent appears in it.

**Answer:**

```
Research question:
  With the same training budget, does an LLM agent that diagnoses why a lattice-stiffness GNN
  fails on held-out lattice families, and chooses whether to retune, change the model's
  structure or escalate, reach lower held-out error than hyperparameter optimization alone?

What is measured (metric, held-out set):
  Error on locked test families (never seen during development); rate of non-positive-definite
  stiffness predictions; budget used.

What result would count as a negative result:
  The agent only ever chooses "retune", so the optimizer alone matches it.

Data split (three groups):
  1. Training lattices: the GNN learns from these.
  2. Validation families: held out from training; agent and optimizer may look at them.
  3. Test families: locked; used once at the end to compare agent and baseline.
  OPEN: what counts as a "family" (same crystal net? similar connectivity?).
```

---

## 2. Cover-up test on the LR1 architecture

> Cover the word "agent" on each box: could ~20 lines of Python produce the same output from
> the same input? If yes, it is a tool. (Debrief)

Analysis of LR1 Table 1:

| LR1 agent  | What it actually does                           | Scriptable? | Becomes                                      |
|------------|-------------------------------------------------|-------------|----------------------------------------------|
| Literature | Judges paper relevance, extracts page-cited claims | No (real judgment) | Support infrastructure, outside the research loop; keep quote/page/DOI verification |
| Data       | Parse catalog, check data, make splits          | Yes         | Tools: `check_data`, `make_splits`           |
| Training   | Run `train.py` with a config                    | Yes         | Tool: `train(config) -> metrics`             |
| Evaluation | RMSE, R², positive-definiteness count           | Yes         | Tool: `evaluate(run) -> report`              |
| Defects    | Run FE on defect specs                          | Yes         | Tool: FE solver wrapper                      |

Result: LR1 has five boxes and, inside the research loop, zero agent decisions. This is the
same gap the TA comment points at.

**Answer** (your own pass; disagree with the table where you think it is wrong):

```
Boxes that stay tools:

Boxes where a real decision remains (state the decision):
```

---

## 3. The agent's decision

> An agent belongs where there is a decision under uncertainty, whose outcome changes what
> happens next, and that a fixed rule cannot encode in advance. Name **one** decision; that is
> what the project evaluates. (Debrief)

**On the TA's question about hyperparameter search:** the debrief is explicit that an LLM
proposing the next parameter set is "the agent as optimizer": an optimizer (Optuna, Bayesian
optimization, grid search) does that better and cheaper. Hyperparameter search goes in the tool
layer. The agent's job is what the optimizer cannot do: formulate the search, diagnose why it
fails, and change structure.

Candidate decisions for this project (pick one, combine, or write your own):

| # | Decision the agent makes | Inputs it sees | Non-agent baseline | Risk |
|---|--------------------------|----------------|--------------------|------|
| A | **Surrogate diagnostician.** When the GNN misses the target on held-out lattice families (or predicts non-positive-definite stiffness), decide whether the cause is (a) hyperparameters → rerun optimizer with new bounds, (b) model structure → change features/architecture/output layer, (c) data coverage → request labels for specific lattices, or (d) something it cannot resolve → escalate to human | Per-family errors, PD-failure cases, optimizer history, run log | Optimizer alone on fixed architecture, same compute budget | If the agent only ever picks (a), the optimizer matches it: a clean negative result |
| B | **Data acquisition for defects.** Choose which defected lattices to simulate with FE next so the surrogate improves fastest | Current model errors/uncertainty, defect design space, FE budget | Random or uncertainty sampling, same FE budget | Depends on an FE solver running reliably (Abaqus licence is open) |
| C | **Trust-or-simulate.** For a new lattice, decide whether the surrogate's prediction can be used or FE must be run | Prediction, ensemble spread, similarity to training families | Threshold on ensemble variance | May fail the cover-up test: a threshold might do as well |
| D | **Operator / auditor.** Monitor training and FE campaigns, detect failed runs and outliers, decide rerun / reindex / escalate, keep log and dataset consistent | Run log, dataset, scheduler | Cron job plus rule-based alerts; evaluated on planted failures | Valid fallback if A–C turn out scriptable |

**Answer:**

```
The decision: candidate A, surrogate diagnostician. After each optimizer round, diagnose why
the GNN still fails on the validation families and pick one remedy. (Defect data acquisition,
B, is a possible later extension; it depends on a working FE solver.)

Input (all produced by tools; the agent computes nothing itself):
  - evaluation report: error per validation family and per target, non-PD rate, distance from
    target
  - optimizer history: settings tried, scores, whether the best result sits at a range edge
  - decision log: its earlier decisions and what happened after each
  - state: current model structure, menu options already tried, remaining budget

Options (enumerated, so they can be logged and counted):
  (a) retune: when the optimizer has not searched enough (best result at a range edge, scores
      still improving). Output: which settings to vary and the new ranges.
  (b) change structure: when retuning has stopped helping and the failure pattern points to a
      structural cause. Output: one option from a fixed menu written in advance (no free code
      edits), the failure it targets, and a prediction of what should improve.
  (c) escalate: see below.
  "Get more data" is dropped: the catalog is fixed and test families must stay untouched.
  OPEN: the contents of the structural menu (about 4-6 options).

Stopping (checked by code, not the agent; values in <> are placeholders):
  - success: every validation family below <target error> and non-PD rate <= <threshold>
  - budget: <40> training runs used
  - no progress: <2> rounds in a row without improving by more than <X>
  - cap of <3> structural changes

Escalation:
  - agent chooses it when no untried menu option fits the failure, the evidence points to a
    data or label problem, the fix needs something outside its permissions (code edits, more
    budget, new data), or it cannot tell which cause is more likely
  - code forces it when the output fails the schema check twice, or the agent proposes an
    option that already failed
  - agent writes: reason, evidence, what it would try if allowed
  - human replies: approve a new option, change the spec, or stop
  - in the final comparison an escalation counts as "not solved"

Checks on the agent's choice:
  1. Before it runs (code): output well-formed; option on the menu and not already failed;
     retune ranges within allowed limits; enough budget left. One retry with the error
     message, then escalate.
  2. Before it runs (human): structural changes need approval.
  3. After it runs (code): next round's evaluation shows whether the targeted family's error
     and the non-PD rate dropped, and whether the (b) prediction came true; result goes into
     the decision log.
  4. End of project: planted failures with known causes, scored on whether the agent names
     the correct remedy, e.g.
       - PSD output layer removed -> change structure, add PSD layer
       - optimizer range excludes good learning rates -> retune with wider bounds
       - one family's labels corrupted -> escalate, data problem

Why a fixed rule cannot make it:
  The optimizer only adjusts numbers inside a fixed structure; deciding that a failure pattern
  (e.g. one family with large errors and non-PD predictions that retuning did not fix) calls
  for a structural change, and which one, requires interpreting the evidence.

Stochasticity:
  The agent's choice can vary between runs even with a fixed menu. Seed training and Optuna so
  remaining variation comes mainly from the agent; repeat each condition ~5 times and report
  spread; measure decision consistency by asking the agent to decide on frozen evaluation
  reports ~10 times each and counting the outcomes.
  OPEN: Claude Code may not allow setting temperature; settle in Section 10.
```

---

## 4. Roles, state and memory

> Required by the project guideline and missing from a third of LR1 reports. (Debrief)

**Answer:**

| Role | LLM or code | Model ID (pinned) | Reads (state/memory) | Writes | Can call tools |
|------|-------------|-------------------|----------------------|--------|----------------|
|      |             |                   |                      |        |                |
|      |             |                   |                      |        |                |
| Human | —          | —                 |                      |        |                |

Questions to settle:
- Where does state live (LangGraph state object, JSON files in the repo, a run database)?
- Is the running log machine-readable so agents can read and append to it (Debrief)?
- What does the agent remember between rounds, and what is reset?

---

## 5. Tool interfaces

> Build and smoke-test each deterministic tool first, wrap it as a typed tool (Python function
> or MCP server), run the scripted baseline on it, then add the agent. Validate every call
> against a schema before it executes. (Debrief)

Starting point from the current repo (check the status column yourself):

| Tool | Signature (inputs → outputs) | Backed by | Status |
|------|------------------------------|-----------|--------|
| `check_data` | catalog path → data report | `check_data.py` | exists, script form |
| `make_splits` | catalog, split type (random / held-out family) → split file | — | to build |
| `train` | config → run id, metrics, checkpoint | `train.py` | exists, script form |
| `evaluate` | run id, split → RMSE, R² per target and per family, PD-failure count | `predict.py` + PD check | partly |
| `optimize` | search space, budget → best config, trial history | Optuna (to add) | to build |
| `fe_solve` | lattice (+ defect spec) → stiffness tensor | Abaqus / CalculiX / OpenSeesPy | blocked on solver choice |
| literature tools | (existing MCP server) | `agents/literature/server.py` | built |

**Answer** (final list, with the schema for each tool the agent calls):

```
```

---

## 6. The loop: hand-offs and stopping conditions

> Draw the loop the TA asked for: who acts, in what order, what is passed at each hand-off,
> and when it stops.

**Answer:**

```
Setup (once, human): freeze the three groups, set the budget (e.g. 40 training runs), set
thresholds from the math.

Each round:
1. Optimizer (tool): Optuna runs ~10 training runs on the fixed structure -> best config,
   trial history.
2. Evaluate (tool): error per validation family and per target, non-PD count.
3. Check (code): target met / budget gone / no progress -> stop.
4. Agent (LLM, only step): reads evaluation report, optimizer history, its past decisions ->
   retune | change structure | escalate, as schema-checked structured output with reasoning.
5. Human gate: approve structural changes; retunes within budget may run without approval.
6. Apply (tools), back to 1.
End: run agent loop and baseline on the locked test families, ~5 repeats each.

Baseline: no agent; Optuna searches the settings and the structural menu together, with the
same budget (see Section 8).

Hand-off payloads (what exactly is passed, in what format):

Stopping conditions (all checkable by code): see Section 3.
```

---

## 7. Human gates

> Anything that touches a cluster job, a budget or an external service needs explicit approval.
> (Debrief)

**Answer:**

```
Approve before:
Review after:
Escalations the human handles:
```

---

## 8. Baseline and evaluation

> Define evaluation before building: held-out tasks the agent never saw during development,
> repeated runs with a pinned model ID and fixed temperature, and abstention accounted for so
> declining cannot inflate accuracy. (Debrief)

LR1 already had the right test set idea: held-out lattice families rather than a random split.

**Answer:**

```
Baseline (same tools and budget, no agent):
  Optuna searches over the hyperparameters AND the structural menu, treating each menu option
  as a categorical setting (e.g. PSD layer yes/no, edge features A/B, layers 3/4/5). Same
  budget (<40> training runs), same locked test families, same repeats. This is the fair
  comparison: the optimizer has access to every option the agent has, so any advantage must
  come from the agent's diagnosis, e.g. reaching a good model in fewer runs.
  (Not planned: optimizer on a fixed structure, too weak; hand-written rules, optional extra
  rung if time allows.)

Held-out tasks (which families / targets; frozen before development):

Repeats per condition: ~5        Model ID:          Temperature:

Metrics (agent vs baseline):
  - error on the locked test families at the end of the budget
  - non-PD rate
  - number of training runs needed to reach the target
  - on planted failures: whether the correct remedy was picked (Section 3, check 4)

How abstention / escalation is scored: an escalation counts as "not solved".
```

---

## 9. Thresholds from the spec, not by eye

> Every threshold in a deterministic check must come from the spec or the math. (Debrief)

**Answer:**

| Check | Threshold | Derived from |
|-------|-----------|--------------|
| Positive definiteness | smallest eigenvalue > ? | |
| "Target met" accuracy | | |
| Data sanity | | |

---

## 10. Implementation stack

Decide and justify; the debrief encourages the course stack:
- LangGraph state machine for the main loop (explicit state, memory, stopping conditions)?
- MCP server(s) for the deterministic tools (typed boundary, call log, portable across agents)?
  Note the debrief says local MCP servers are blocked by policy for some students; the
  literature server already ran for LR1, so check the same holds where the new tools run.
- Claude access: course Claude Code seat (no API key). Check which stack works with that.
- Trace and cost logging: prompts, tool calls, results, tokens per run.

**Answer:**

```
```

---

## 11. Scope and risks (nine weeks)

> A small system with careful evaluation and honest limitations beats a broad demonstration
> that cannot be verified. (Debrief)

**Answer:**

```
In scope:
Explicitly out of scope:
Top risks and mitigation:
Fallback if the agent adds nothing (negative result plan):
```

---

## 12. Lab Report 2 checklist

From the LR2 brief (supersedes the debrief's "one-page proposal"):
- [ ] Project proposal following the course proposal template, max 5 pages including figures
      and references
- [ ] Figure of the agentic system design (Sections 4–7 of this worksheet)
- [ ] Module demo: one working module (e.g. a typed deterministic tool + scripted baseline,
      or the agent making one tool call)
- [ ] Progress log summary; confirm repo and log are actively maintained
- [ ] AI use acknowledgement (APS guidelines example linked in the brief); separate
      agent-suggested references from ones you read

From the debrief's "Before Sunday, Nov 1" and the Notion task:
- [ ] Research question, agent's decision, domain tool, baseline, evaluation, risks, team
- [ ] Repo public, README current, log active, model access verified, one tool call through
      the agent
- [ ] Updated related-work note, every agent-found claim checked
- [ ] Team composition finalized with the TA
- [ ] Repo URL in plain text pointing at the repository
