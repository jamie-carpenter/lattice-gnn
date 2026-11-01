# Prompt: write Lab Report 2 (AM/ME/CE 295)

Open a new Claude Code session in the `lattice-gnn` repo folder, on branch
`docs/lab-report-2-design`, and paste everything below the line.

---

Help me write **Lab Report 2 for AM/ME/CE 295 (AI Agents for Scientific Discovery), due Sunday
Nov 1, 2026, 11:59 pm**. It is a project proposal plus a module demo, a progress log summary and
an AI use acknowledgement. I make the design decisions; you draft from what is in the repo and
from my answers. Never invent a number, result, citation or design choice: if something is not in
the repo or my answers, ask me or leave it out. Ask me before anything that uses significant
Claude usage, pushes to GitHub, or runs training for more than a few minutes.

**Work in steps with checkpoints.** At each **CHECKPOINT**, stop, show me what it asks for, and
wait for my reply. Don't move on by yourself, even if everything looks fine. Explain git and
other tooling in plain words.

**CHECKPOINT 0:** read the sources below, then tell me in a few lines your plan for the session
and list the open design questions from Step 1 that you need me to answer.

## What Lab Report 2 requires (course announcement; links written out)

> Lab Report 2 is due at 11:59pm this Sunday (Nov 1). Please submit the following:
> * Project proposal: Follow the proposal template (kindly prepared by Lauren), including a
>   figure of your agentic system design. Maximum 5 pages, including figures and references.
> * Module demo: Show a working module of your agentic system. You can either build the in-class
>   demos into your project (e.g. lab 4 personal agent, lab 5 retrieval agent), or build part of
>   your own project (e.g. a data pipeline, baseline model, deterministic tools built with agents).
> * Progress log: Summarize your progress and confirm that your repo and log are being actively
>   maintained.
> * AI use acknowledgement: An example of guidelines for such acknowledgements used in journals
>   can be found here: https://journals.aps.org/authors/appropriate-use-ai-tools

**Hard limit: 5 pages including figures and references.** Check the compiled page count.

## Sources (read all of them before Checkpoint 0)

Course material (outside the repo; ask me for access if you can't read them):
- `C:\Users\Jamie\Downloads\Lab_Report_Template_Agent_Project.pdf`: the template. **The report
  follows its sections and numbering exactly:**
  - Header: name, team (I have none unless I say otherwise), GitHub repo link in plain text
    (https://github.com/jamie-carpenter/lattice-gnn, the repository, not my profile)
  - A Project description & goal: A1 problem statement and goal (max 5 sentences); A2 related
    work, domain and agent literature
  - B Why an agent: B1 why not a fixed script, an optimizer or a single prompt
  - C Agent design: C1 figure of agents, tools, data and information flow; C2 why each agent
    role and tool is needed (what breaks if removed); C3 current working module / demo
  - D Evaluation: D1 the non-LLM check that the output is right and what counts as success;
    D2 the simpler baseline with the same tools and no agent
  - E Plan and risks: E1 what works by Lab Report 3 (Nov 15) and Lab Report 4 (Nov 29); E2
    biggest risk or question for the TAs
  - AI usage disclosure statement (the template stresses this)
- `C:\Users\Jamie\Downloads\Lab1_Class_Debrief_Recommendations.pdf`: what the TAs want: the
  cover-up test, one named agent decision in the research loop, optimizer in the tool layer,
  thresholds from the spec, held-out evaluation with repeats, human gates, negative result as a
  valid outcome. Its worked example (version B figure) is the model for C1.
- `C:\Users\Jamie\Downloads\LabReport1.pdf`: my Lab Report 1 (source: `reports/lab_report_1.tex`).
  TA grade comment (technical depth 26/30): *discuss the architecture: the flow of the agents and
  their assigned goal toward "building a reliable surrogate"; is the agents' goal to speed up
  hyperparameter search? Designing the full agent loop with its tools and protocol is
  essential.* LR2 must answer this directly.

In the repo:
- `docs/agent_design_worksheet.md`: **the main design source.** Sections 1, 3, 6 (loop) and 8
  (baseline) have my answers; 4, 5, 7, 9, 10, 11 are partly or fully blank. Section 12 is the LR2
  checklist; tick it off at the end.
- `LOG.md` (running log), `docs/TODO.md` (build order and status): the progress section and all
  numbers come from here.
- `literature_review/main/lit_review.md` and `papers.json`: related work, page-cited. Only cite
  papers listed there or that I confirm I have read; mark which claims I checked against the PDF
  (LOG.md says 10 of 11 Indurkar/Grega citations were checked).
- Code for the demo: `lattice_gnn/splits.py`, `evaluate.py`, `training.py`, `optimize.py`,
  `report.py`, `config/loop.yaml`, `tests/test_baseline.py`, and the `run-report` skill
  (`.claude/skills/run-report/`). The literature agent (`agents/literature/`,
  `.claude/agents/literature-reviewer*.md`) is an agent calling typed tools: it counts as "one
  tool call through the agent", but it is support infrastructure, not the system under test.
- Results: `runs/main-2000/` (reference run on the locked split; `runs/` is not in git, so the
  numbers in `LOG.md` are the record). Rebuild its report with
  `.venv/Scripts/python -m lattice_gnn.report runs/main-2000` if you need the figures.

## Step 1: Settle the open design questions with me

From the worksheet, list every blank or `OPEN` item the report needs, with your suggested answer
and where the suggestion comes from (worksheet, debrief, code). At minimum:
- Roles, state and memory (worksheet section 4): who is LLM vs code, pinned model ID, what each
  reads and writes.
- Tool interfaces (section 5): name and signature of each tool (`train(config) -> run_id`,
  `evaluate`, `optimize`, `check_target`, a planned `validate`), and which exist vs are planned.
- Human gates (section 7).
- Thresholds (section 9): success thresholds in `config/loop.yaml` are placeholders; the report
  must say where each one comes from, or say honestly that they are still to be fixed. The PD
  threshold is derived in a comment in `lattice_gnn/evaluate.py`.
- The structural menu: which switches (`pd_head`, `physics_residual`, `family_balanced_loss`,
  augmentation) and which are built (only augmentation so far).
- Implementation stack (section 10): Claude Code seat, no API key; LangGraph or plain Python;
  MCP; trace and cost logging; whether temperature can be fixed.
- Scope, out of scope, risks, negative-result fallback (section 11). Is the defect extension in
  or out for this term? Status of the Abaqus licence question from LR1?
- Study budget: one 40-run study is about 4-10 h on my laptop CPU (`LOG.md`); how many repeats
  per condition are realistic.

**CHECKPOINT 1:** show me the list with your suggestions. I will answer, and you write my answers
back into the worksheet (my words, not yours, where I give them).

## Step 2: Outline and figure

- Outline: one line per template item (A1 to E2 plus AI disclosure) saying what goes there,
  and a page budget per section that fits 5 pages with the figure and references.
- C1 figure: a diagram in the style of the debrief's version B. Tools (training, evaluation,
  Optuna optimizer, target check, validator) as plain boxes; the diagnostician agent as the one
  LLM box with its three decisions (retune / change structure / escalate); human gate dashed;
  the baseline (Optuna over hyperparameters plus the structural menu, same budget, no agent) to
  the side; the literature agent outside the research loop as verified support. Mark what exists
  today vs planned. Make it from a committed script or TikZ so it can be regenerated, readable at
  column width; render it and look at it before showing me.
- C3 demo: what I show and how to rerun it. Suggestion: frozen family split, `train(config)`
  writing `runs/<id>/`, `evaluate` with per-family errors and the PD check, a tiny Optuna study
  with `check_target`, and the `run-report` output for `main-2000` (numbers from `LOG.md`).
  One short table or figure from that run, not all of them.

**CHECKPOINT 2:** show me the outline, the page budget, the rendered figure, and the demo plan.

## Step 3: Draft

Write `reports/lab_report_2.tex`, reusing the LaTeX setup of `reports/lab_report_1.tex`.
- Answer the TA comment in B1/C2: the agents' goal is not to speed up hyperparameter search
  (that is the optimizer, a tool); the agent diagnoses why the surrogate fails on held-out
  families and chooses retune / change structure / escalate.
- Research question and negative result: state both as in worksheet section 1.
- Progress: done / remaining / bottlenecks from `LOG.md`, and say that the repo and log are
  actively maintained (commit history on GitHub, dated log entries).
- AI use acknowledgement, following the APS guideline (fetch the page to check what it asks):
  which tools (Claude Code with model IDs, the literature agent), for what (code written with
  Claude Code as builder, literature search, drafting), and what I verified myself. Separate
  agent-suggested references from the ones I read.
- Plain, concise writing; no filler, no unfilled placeholders. Every number traceable to
  `LOG.md`, an `eval.json`, or the literature review.

**CHECKPOINT 3:** show me the full draft and its compiled page count, plus a table of every
number and citation in it with its source (file and line, or paper and page). Revise until I
approve. If the report needs a fact you could not source, list it instead of writing it.

## Step 4: Check and save

- Fresh-eyes review: give a reviewer subagent with no project context the compiled PDF, the
  template and the debrief, and ask it to check every template item is answered, the 5-page
  limit, the figure is readable, and for leftover placeholders or unsupported claims. Show me
  its findings; fix only what I agree with.
- Tick the worksheet section 12 checklist and tell me what is still not done.
- Add a dated `LOG.md` entry.

**CHECKPOINT 4:** show me the files you will commit (never keys, `.env` or credentials) and the
commit message. Commit after I approve; ask me separately before pushing.
