#!/usr/bin/env bash
# Lab Report 4 planted-failure study (config/planted.yaml, base config/loop_lr4.yaml, 22 CPU-hours
# per loop). Steps run one after another, the known cases first:
#   known cases      full loop with the agent, then 10 frozen agent calls (and the rule table) on
#                    that loop's round 0, then the full loop with the rule table
#   ambiguous cases  one round of training, then 10 frozen agent calls and the rule table
# Roughly 4 h of laptop time per full loop (22 CPU-hours at about 6 CPU-hours per hour on 8
# threads), so about a day in all. Keep the laptop plugged in: on battery, Modern Standby pauses
# training whenever the screen turns off.
# Usage (repo root): bash scripts/run_lr4_study.sh            # everything
#                    bash scripts/run_lr4_study.sh collapse   # one case
# Log: runs/logs/lr4-study.log (start/end time and exit code per step); results:
#   .venv/Scripts/python -m agents.diagnostician.planted score runs/planted-*
set -u
PY=.venv/Scripts/python
P="$PY -m agents.diagnostician.planted"
LOG=runs/logs/lr4-study.log
mkdir -p runs/logs

step() {
  local name=$1; shift
  if [ -e "runs/$name" ]; then echo "$name skipped (runs/$name exists) $(date)" >> "$LOG"; return; fi
  echo "$name start $(date)" >> "$LOG"
  "$@" > "runs/logs/$name.log" 2>&1
  echo "$name exit $? at $(date)" >> "$LOG"
}

known() {
  local c=$1
  step "planted-$c-agent"         $P loop --case "$c" --decider agent
  step "planted-$c-agent-repeats" $P repeat --from "runs/planted-$c-agent" --repeats 10
  step "planted-$c-rules"         $P loop --case "$c" --decider rules
}

ambiguous() {
  step "planted-$1-frozen" $P frozen --case "$1" --repeats 10
}

if [ $# -gt 0 ]; then
  for c in "$@"; do
    case $c in control|hidden_noise|two_causes) ambiguous "$c" ;; *) known "$c" ;; esac
  done
else
  for c in collapse units_low_lr width_trap; do known "$c"; done
  for c in control hidden_noise two_causes; do ambiguous "$c"; done
fi
echo "all done $(date)" >> "$LOG"
