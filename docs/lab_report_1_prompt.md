# Prompt: complete Lab Report 1 (AM/ME/CE 295)

Paste everything below the line into a new Claude Code session. If you run it inside the AWS
workspace, first clone the repo there and open the session in the repo folder.

---

Help me finish **Lab Report 1 for AM/ME/CE 295 (AI Agents for Scientific Discovery), due today,
Sunday Oct 18, 2026**. We have limited time, so prioritize a complete, honest report over solving
problems. Ask me before anything that costs money, pushes to GitHub, or changes AWS resources.

## What Lab Report 1 requires (from the course Lab Guideline)
Goal: show I understand the problem space and that my tooling is getting ready. No working demo needed.
1. **Related-work note:** 2-3 papers/approaches relevant to my project, with a short synthesis.
2. **Draft project proposal:** problem statement and goal.
3. Optional: repo and environment set up; running log started.
Every report also has: **progress (done / remaining / bottlenecks)** and **next steps for the coming
two weeks**. Graded on technical depth 30%, reproducibility 20%, results and analysis 25%,
writing 15%, participation 10% (GitHub commit activity).
Project rules that matter: sandbox code execution, human approval before consequential actions,
never commit API keys, pinned model IDs, reproducible from the repo.

## My project
A multi-agent system that helps build, tune and stress-test a graph neural network predicting the
effective elastic properties of periodic strut-lattice unit cells, later including manufacturing
defects. Repo: `lattice-gnn`, branch `agents/literature-review`.
- `lattice_gnn/`: parser, periodic graph builder, GNN model (message passing on nodes and struts).
  Targets: log10 of Ex, Ey, Ez, Gyz, Gxz, Gxy (normalized by base-material modulus Es) plus 3
  Poisson couplings. `train.py`, `predict.py`, `check_data.py`, `plot_lattice.py`.
- `Unit_Cell_Catalog/Unit_Cell_Catalog.txt`: full dataset, 17,262 lattices, apparently from
  Lumpe & Stankovic, PNAS 2021 (doi 10.1073/pnas.2003504118). 41 records have no properties
  (excluded by the authors). Solving E/Es = C*rho^n from each record gives median relative
  density 1.00% (98% within 0.90-1.12%), so the labels appear to be at 1% relative density.
- `agents/literature/`: literature-review agent built on the Claude Agent SDK (pinned
  `claude-agent-sdk==0.2.168`, model `claude-sonnet-5-5`). Tools: arXiv search, OpenAlex search,
  citation snowballing, page-by-page open-access PDF reading, and record/write tools. It has no
  shell or file-edit access; it only writes `papers.json` and `lit_review.md` into its output folder.
  Claims are cited to PDF pages. Spending cap $10 per run (`--budget`).
  Run: `python -m agents.literature` (needs ANTHROPIC_API_KEY, or Bedrock; see below).
  Its search and PDF tools have been tested against the live APIs; a full agent run has NOT been
  done yet (no model access configured).
- Planned agents: architecture/training/tuning agents (write and run PyTorch, so they need a
  sandbox: AWS) and a defects agent that needs finite element analysis (Abaqus if the license allows
  cloud use; otherwise CalculiX or OpenSeesPy).

## Step 1: AWS environment check (about 45 minutes max; do NOT fix problems, only record them)
The course requires us to confirm everything we need can run on AWS. Run each check, record
PASS / FAIL / NOT TESTED with the key output, and for each failure write a bottleneck entry:
*what failed, impact on the project, workaround for now, plan and owner to resolve, by when*.
If this session is not running on the AWS machine, give me the commands to run there and I will
paste the output back.
1. Environment: what kind of AWS machine this is (EC2 instance or Amazon WorkSpaces desktop),
   OS, CPU cores, RAM, free disk, GPU (`nvidia-smi`). Do files persist after stop/restart?
2. GitHub: can the repo be cloned and pushed from here (credentials)?
3. Python: is 3.12 available; create a venv; install `requirements.txt` (GPU build of PyTorch if
   there is a GPU, CPU build otherwise) and `agents/requirements.txt`; run a quick import of
   `lattice_gnn` and a short data check on `data/samples/sample_lattices.txt`.
4. Model access: Is Amazon Bedrock available on this account (`aws sts get-caller-identity`, then
   whether Claude models are enabled in the region)? Check the current Claude Agent SDK docs for
   how to point it at Bedrock and note the exact Bedrock model ID we would pin. Otherwise, is an
   Anthropic API key available? Never print, log or commit a key.
5. Network: can the machine reach export.arxiv.org, api.openalex.org and arxiv.org PDFs? Run the
   literature tools' free smoke test (no Claude calls), e.g. call `_arxiv_search`, `_oa_search`
   and `_pdf_pages` from `agents/literature/tools.py`. Note: OpenAlex's free quota is shared per IP,
   which can be a problem on cloud IPs.
6. Sandbox: is Docker available and usable without root?
7. Finite element analysis: is Abaqus installed (`abaqus information=release`), and is a license
   server reachable? If not, record it. Workaround: open-source FE (CalculiX/OpenSeesPy) and
   confirm with Caltech IT whether the Abaqus license allows AWS use.
8. Storage and cost: S3 access; can I see costs / set a budget alert?

## Step 2: Run the literature agent (ask me first, it costs money)
Run it wherever model access works: AWS if Step 1 passed, otherwise my laptop. It does not execute
code, so it does not need the sandbox. Before the full run, show me the command and expected cost.
If no model access is available today, say so and record it as the main bottleneck; then help me
write the related-work note from papers you can verify with the literature tools directly (still
with page-cited claims), and say in the report that the agent run is pending.

## Step 3: Write the report
Create `reports/lab_report_1.md` with:
1. **Project summary and draft proposal**: problem statement, intended users, why an agent is
   appropriate (and what a non-agent baseline would be), planned agents/tools/architecture, how
   success will be measured, major risks.
2. **Related-work note**: the 2-3 most relevant papers (expect the dataset paper and GNN surrogates
   for lattice elasticity), each with what it does, results with units and normalization, and
   page-cited relevance to my project; then a short synthesis and the gap my project targets.
   Use only papers confirmed by the literature tools; never invent citations.
3. **Progress**: done / remaining / bottlenecks. The bottleneck table from Step 1 goes here, each
   with its workaround and resolution plan. Make it clear they don't block Lab Report 2.
4. **Next steps (weeks 3-4)**: lock in the research question and the domain tool (FE solver
   choice), resolve AWS bottlenecks, full literature run, team/TA confirmation.
5. **AI use note**: which AI tools helped and what I verified myself.
Also start a running log `LOG.md` (dated entries: done / remaining / bottlenecks), since later
reports draw on it. Keep the report concise (about 2 pages). Then show me the report to review.

## Step 4: Save
Commit the report, log and any results (never keys or `.env`), then ask me before pushing to GitHub.
