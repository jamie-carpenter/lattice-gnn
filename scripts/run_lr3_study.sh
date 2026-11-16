#!/usr/bin/env bash
# Lab Report 3 comparison on the 2,000-lattice subset (config/loop_subset.yaml): the same
# 20-run budget, split, search ranges and sampler seed for each condition, run one after another.
#   1. optuna-sub  Optuna over hyperparameters and menu switches together (no decider)
#   2. rules-sub   the loop with the rule table deciding (agents/diagnostician/rules.py)
#   3. agent-sub   the loop with the LLM diagnostician deciding
# Usage (repo root, laptop plugged in): bash scripts/run_lr3_study.sh
# Log: runs/logs/lr3-study.log (start/end time and exit code per condition).
set -u
PY=.venv/Scripts/python
CFG=config/loop_subset.yaml
LOG=runs/logs/lr3-study.log
mkdir -p runs/logs

step() {
  local name=$1; shift
  echo "$name start $(date)" >> "$LOG"
  "$@" > "runs/logs/$name.log" 2>&1
  echo "$name exit $? at $(date)" >> "$LOG"
}

step optuna-sub $PY -m lattice_gnn.optimize --config $CFG --study optuna-sub
step rules-sub  $PY -m agents.diagnostician.loop --config $CFG --decider rules --approve all --name rules-sub
step agent-sub  $PY -m agents.diagnostician.loop --config $CFG --decider agent --approve all --name agent-sub
echo "all done $(date)" >> "$LOG"
